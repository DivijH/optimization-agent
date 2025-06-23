import asyncio
import json
import random
import sys
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import click

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from shopping_agent import EtsyShoppingAgent

def _load_persona_file(file_path: Path) -> Tuple[str, str]:
    """Load a single persona JSON file and return (task, persona_text).
    The expected schema is:
        {
            "persona": "<long persona text>",
            "intent": "<shopping intent>",
            ...
        }
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading persona file {file_path}: {e}")

    task = data.get("intent", None)
    persona_text = data.get("persona", None)
    if persona_text and task:
        return task, persona_text
    else:
        raise ValueError(f"Persona file {file_path} is missing either 'intent' or 'persona' field.")


async def _run_single_agent(
    *,
    agent_id: int,
    group: str,
    persona_file: Path,
    max_steps: int,
    headless: bool,
    model_name: str,
    debug_root: Path,
):
    """Create and execute a single `EtsyShoppingAgent` instance.

    Args:
        agent_id: Sequential number of the agent (used for debug path naming).
        group: Either "control" or "target".
        persona_file: Path to JSON file describing the persona.
        max_steps: Maximum number of steps the agent is allowed to take.
        headless: Whether the browser should run in headless mode.
        model_name: Name of the LLM model to use.
        record_video: Whether to capture a video of the session.
        debug_root: Base directory under which per-agent debug folders are
            created.
    """

    task, persona = _load_persona_file(persona_file)

    # We intentionally do NOT create the debug directory beforehand.  Doing so would
    # trigger the interactive `click.confirm` prompt inside `EtsyShoppingAgent.__post_init__`
    # that asks for permission to delete the existing directory.  By ensuring the
    # path does not yet exist we avoid any blocking user interaction in batch runs.

    debug_path = debug_root / f"{group}_agent_{agent_id}_{int(time.time())}"
    user_data_dir = debug_path / "browser_profile"

    agent = EtsyShoppingAgent(
        task=task,
        persona=persona,
        headless=headless,
        max_steps=max_steps,
        model_name=model_name,
        debug_path=str(debug_path),
        user_data_dir=str(user_data_dir),
    )

    print(
        f"[Agent {agent_id} | {group}] Persona file: '{persona_file.name}', model: {model_name}, task: {task}"
    )
    try:
        await agent.run()
    except Exception as e:
        print(f"[Agent {agent_id} | {group}] ⚠️  Encountered an error: {e}")


###############################################################################
# CLI entry-point
###############################################################################

@click.command()
@click.option(
    "--n-agents",
    type=int,
    default=4,
    show_default=True,
    help="Total number of agents to spawn. Half will be assigned to the control group and half to the target group.",
)
@click.option(
    "--personas-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=CURRENT_DIR.parent / "data" / "personas",
    show_default=True,
    help="Directory containing JSON persona files.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for persona selection. If provided, the same personas will be selected for each run with the same seed.",
)
@click.option(
    "--control-model",
    type=str,
    default="gpt-4o-mini",
    show_default=True,
    help="Model name to use for the *control* group.",
)
@click.option(
    "--target-model",
    type=str,
    default="gpt-4o-mini",
    show_default=True,
    help="Model name to use for the *target* group.",
)
@click.option(
    "--max-steps",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of steps each agent is allowed to take.",
)
@click.option("--headless/--no-headless", default=True, show_default=True, help="Run browsers in headless mode.")
@click.option(
    "--concurrency",
    type=int,
    default=2,
    show_default=True,
    help="Maximum number of agents to run concurrently. Lower this if you run out of system resources.",
)
@click.option(
    "--debug-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=CURRENT_DIR / "debug_run_ab",
    show_default=True,
    help="Root directory under which per-agent debug folders will be created.",
)
def cli(
    n_agents: int,
    personas_dir: Path,
    seed: Optional[int],
    control_model: str,
    target_model: str,
    max_steps: int,
    headless: bool,
    concurrency: int,
    debug_root: Path,
):
    """Run a simple A/B test by spawning multiple shopping agents.

    Half of the agents will be assigned to the *control* group and will use the
    `--control-model`, while the other half will form the *target* group and
    use the `--target-model`.  Persona JSON files are sampled randomly (without
    replacement if possible) from *personas-dir*.
    """

    if seed is not None:
        random.seed(seed)
        print(f"Using random seed {seed} for persona selection.")

    if n_agents < 2:
        raise click.BadParameter("--n-agents must be greater than 2.")

    if not personas_dir.exists() or not personas_dir.is_dir():
        raise click.BadParameter(f"--personas-dir '{personas_dir}' does not exist or is not a directory.")

    persona_files: List[Path] = sorted(personas_dir.glob("*.json"))
    if not persona_files:
        raise click.BadParameter(f"No .json persona files found in '{personas_dir}'.")

    # Sample personas without replacement if there are enough; otherwise fall back to sampling with replacement
    if len(persona_files) >= n_agents:
        selected_personas = random.sample(persona_files, k=n_agents)
    else:
        selected_personas = [random.choice(persona_files) for _ in range(n_agents)]

    if debug_root.exists():
        if click.confirm(f'Debug path {debug_root} already exists. Do you want to remove it and all its contents?'):
            shutil.rmtree(debug_root)
        else:
            print('Aborting.')
            sys.exit(1)
    debug_root.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each agent
    agent_specs = []
    for idx, persona_file in enumerate(selected_personas):
        group = "control" if idx < n_agents / 2 else "target"
        model_name = control_model if group == "control" else target_model
        agent_specs.append((idx, group, persona_file, model_name))

    async def _runner():
        # Use a semaphore to limit the number of concurrent agents so we don't exhaust system resources
        sem = asyncio.Semaphore(concurrency)

        async def _wrap(spec):
            async with sem:
                agent_id, group, persona_path, model_name_local = spec
                await _run_single_agent(
                    agent_id=agent_id,
                    group=group,
                    persona_file=persona_path,
                    max_steps=max_steps,
                    headless=headless,
                    model_name=model_name_local,
                    debug_root=debug_root,
                )

        await asyncio.gather(*[_wrap(spec) for spec in agent_specs])

    asyncio.run(_runner())


if __name__ == "__main__":
    cli()
