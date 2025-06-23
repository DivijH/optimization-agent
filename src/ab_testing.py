import asyncio
import json
import random
import sys
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import io
from contextlib import redirect_stdout, redirect_stderr

import click

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from shopping_agent import EtsyShoppingAgent

print_lock = asyncio.Lock()

class StatusDisplay:
    """Manages a clean, updating multi-line status display in the terminal."""
    def __init__(self, agent_specs: List[Tuple]):
        self.agent_specs = agent_specs
        self.statuses = {spec[0]: "Waiting to start..." for spec in agent_specs}
        self._num_lines = 0

    async def update(self, agent_id: int, status: str):
        """Update the status for a given agent and redraw the display."""
        async with print_lock:
            self.statuses[agent_id] = status
            
            # Move cursor up to overwrite previous status lines
            if self._num_lines > 0:
                sys.stdout.write(f"\x1b[{self._num_lines}A")

            # Print all statuses
            for spec_id, group, _, _ in self.agent_specs:
                # \x1b[2K clears the entire line before writing
                sys.stdout.write(f"\x1b[2K[Agent {spec_id} | {group:^7}] {self.statuses[spec_id]}\n")
            
            sys.stdout.flush()

    async def print_initial_statuses(self):
        """Prints the initial waiting status for all agents."""
        async with print_lock:
            for agent_id, group, _, _ in self.agent_specs:
                 sys.stdout.write(f"[Agent {agent_id} | {group:^7}] {self.statuses[agent_id]}\n")
            self._num_lines = len(self.agent_specs)
            sys.stdout.flush()


def _setup_file_logger(name: str, log_file: Path) -> logging.Logger:
    """Sets up a logger that writes to a file."""
    logger = logging.getLogger(name)
    # Prevent logger from propagating to the root logger
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
    return logger

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
) -> str:
    """Create and execute a single `EtsyShoppingAgent` instance."""

    task, persona = _load_persona_file(persona_file)

    debug_path = debug_root / f"{group}_agent_{agent_id}_{int(time.time())}"
    debug_path.mkdir(parents=True, exist_ok=True)
    log_file = debug_path / "agent.log"
    logger = _setup_file_logger(f"agent_{agent_id}", log_file)

    user_data_dir = debug_path / "browser_profile"

    agent = EtsyShoppingAgent(
        task=task,
        persona=persona,
        headless=headless,
        max_steps=max_steps,
        model_name=model_name,
        debug_path=str(debug_path),
        user_data_dir=str(user_data_dir),
        logger=logger,
        non_interactive=True,
    )

    try:
        # Redirect any unexpected stdout/stderr during agent execution to the log file
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            await agent.run()

        captured_stdout = stdout_capture.getvalue()
        if captured_stdout:
            logger.info("--- Captured stdout ---\n%s", captured_stdout)

        captured_stderr = stderr_capture.getvalue()
        if captured_stderr:
            logger.warning("--- Captured stderr ---\n%s", captured_stderr)

        return "Finished."
    except Exception:
        # Log the full exception to the agent's dedicated log file
        logger.error("Agent run failed with an exception.", exc_info=True)
        # Keep error message to a single line for clean display
        return f"⚠️  Failed. See log for details: {log_file}"


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
@click.option(
    "--headless/--no-headless",
    default=True,
    show_default=True,
    help="Run browsers in headless mode.",
)
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

    # Silence all existing loggers to prevent library logs from flooding stdout.
    # We will set up our own file-based loggers for each agent.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.setLevel(logging.CRITICAL + 1)

    # Specifically target and silence the browser_use loggers which are the source of the noise
    for logger_name in ['browser_use', 'browser_use.telemetry', 'browser_use.telemetry.service']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        logger.disabled = True
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    if seed is not None:
        random.seed(seed)

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

    status_display = StatusDisplay(agent_specs)

    async def _runner():
        # Use a semaphore to limit the number of concurrent agents so we don't exhaust system resources
        sem = asyncio.Semaphore(concurrency)
        await status_display.print_initial_statuses()

        async def _wrap(spec):
            agent_id, group, persona_path, model_name_local = spec
            async with sem:
                # Update status to Running BEFORE starting the agent
                await status_display.update(agent_id, "Running...")
                # Small delay to allow status update to complete
                await asyncio.sleep(0.1)
                
                try:
                    final_status = await _run_single_agent(
                        agent_id=agent_id,
                        group=group,
                        persona_file=persona_path,
                        max_steps=max_steps,
                        headless=headless,
                        model_name=model_name_local,
                        debug_root=debug_root,
                    )
                except Exception as e:
                    final_status = f"⚠️  Failed: {str(e)}"
                    
                await status_display.update(agent_id, final_status)

        tasks = []
        for spec in agent_specs:
            tasks.append(_wrap(spec))
        await asyncio.gather(*tasks)

    asyncio.run(_runner())


if __name__ == "__main__":
    cli()
