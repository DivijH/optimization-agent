import asyncio
import json
import random
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import click
import subprocess
import os

CURRENT_DIR = Path(__file__).resolve().parent
project_root = str(CURRENT_DIR.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up OpenAI credentials
try:
    key_path = CURRENT_DIR / "keys" / "litellm.key"
    os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()
except FileNotFoundError:
    raise Exception(
        "litellm.key file not found. It is expected in optimization-agent/src/keys/litellm.key"
    )
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"


from src.shopping_agent.agent import EtsyShoppingAgent
from src.processing_results import process_agent_results

DEFAULT_TASK = "Dunlap pocket knife"

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
            for spec_id, _, _, _ in self.agent_specs:
                # \x1b[2K clears the entire line before writing
                sys.stdout.write(f"\x1b[2K[Agent {spec_id}] {self.statuses[spec_id]}\n")
            
            sys.stdout.flush()

    async def print_initial_statuses(self):
        """Prints the initial waiting status for all agents."""
        async with print_lock:
            for agent_id, _, _, _ in self.agent_specs:
                 sys.stdout.write(f"[Agent {agent_id}] {self.statuses[agent_id]}\n")
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

def _load_persona(file_path: Path) -> str:
    """Load a single persona JSON file and return persona_text.
    The expected schema is:
        {
            "persona": "<long persona text>",
            ...
        }
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading persona file {file_path}: {e}")

    persona_text = data.get("persona", None)
    if persona_text:
        return persona_text
    else:
        raise ValueError(f"Persona file {file_path} is missing a 'persona' field.")


async def _run_single_agent(
    *,
    agent_id: int,
    task: str,
    curr_query: Optional[str],
    persona_file: Path,
    max_steps: int,
    headless: bool,
    model_name: str,
    debug_root: Path,
    width: int,
    height: int,
    final_decision_model_name: Optional[str],
    temperature: float,
    record_video: bool,
    save_local: bool,
    save_gcs: bool,
    gcs_bucket: str,
    gcs_prefix: str,
) -> str:
    """Create and execute a single `EtsyShoppingAgent` instance."""

    persona = _load_persona(persona_file)

    debug_path = debug_root / f"agent_{agent_id}"
    debug_path.mkdir(parents=True, exist_ok=True)
    log_file = debug_path / "agent.log"
    logger = _setup_file_logger(f"agent_{agent_id}", log_file)

    user_data_dir = debug_path / "browser_profile"

    agent = EtsyShoppingAgent(
        task=task,
        curr_query=curr_query if curr_query is not None else task,
        persona=persona,
        headless=headless,
        max_steps=max_steps,
        model_name=model_name,
        debug_path=str(debug_path),
        user_data_dir=str(user_data_dir),
        logger=logger,
        non_interactive=True,
        viewport_width=width,
        viewport_height=height,
        final_decision_model_name=final_decision_model_name,
        temperature=temperature,
        record_video=record_video,
        save_local=save_local,
        save_gcs=save_gcs,
        gcs_bucket_name=gcs_bucket,
        gcs_prefix=gcs_prefix,
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
    finally:
        # Ensure that the agent's shutdown sequence (saving memory, closing browser etc.)
        # is always called, regardless of whether the run succeeded or failed.
        await agent.shutdown()


def _start_main_recording(
    debug_root: Path, width: int, height: int
) -> Optional[subprocess.Popen]:
    """Spawn an ffmpeg process to capture the entire primary display for the run."""
    output_path = debug_root / "main_session.mp4"

    # Build ffmpeg command depending on OS
    if sys.platform == "darwin":
        # macOS – capture primary display (id 1).
        input_spec = "1:none"  # video id 1, no audio
        cmd = ["ffmpeg", "-y", "-f", "avfoundation", "-framerate", "30", "-i", input_spec, "-pix_fmt", "yuv420p", str(output_path)]
    elif sys.platform.startswith("linux"):
        # Linux – capture :0 X11 display.
        display = os.environ.get("DISPLAY", ":0")
        resolution = f"{width}x{height}"
        cmd = ["ffmpeg", "-y", "-f", "x11grab", "-s", resolution, "-i", f"{display}", "-r", "30", "-pix_fmt", "yuv420p", str(output_path)]
    elif sys.platform == "win32":
        # Windows – capture desktop using gdigrab.
        cmd = ["ffmpeg", "-y", "-f", "gdigrab", "-framerate", "30", "-i", "desktop", "-pix_fmt", "yuv420p", str(output_path)]
    else:
        print("⚠️  Screen recording not supported on this OS. Skipping video capture.")
        return None

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🎥 Central screen recording started → {output_path}")
        return proc
    except FileNotFoundError:
        print("❌ ffmpeg not found. Install ffmpeg to enable screen recording.")
        return None

def _stop_main_recording(proc: Optional[subprocess.Popen]):
    """Terminate the main ffmpeg process gracefully."""
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
        print("🎬 Central screen recording saved.")
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        proc = None


###############################################################################
# CLI entry-point
###############################################################################

@click.command()
@click.option(
    "--task",
    type=str,
    default=DEFAULT_TASK,
    show_default=True,
    help="The shopping task for all agents to perform.",
)
@click.option(
    "--curr-query",
    type=str,
    default=None,
    help="The current query for the agents to search. If not provided, defaults to the task value.",
)
@click.option(
    "--n-agents",
    type=int,
    default=4,
    show_default=True,
    help="Total number of agents to spawn.",
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
    "--model-name",
    type=str,
    default="global-gemini-2.5-flash",
    show_default=True,
    help="Model name to use for the agents.",
)
@click.option(
    "--final-decision-model",
    type=str,
    default=None,
    show_default=True,
    help="Model name for the final decision in each agent. Defaults to the main model if not set.",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    show_default=True,
    help="Model name to use for generating summary. Defaults to the main model if not set.",
)
@click.option(
    "--max-steps",
    type=int,
    default=None,
    show_default=True,
    help="Maximum number of steps each agent is allowed to take.",
)
@click.option(
    "--width",
    type=int,
    default=1920,
    show_default=True,
    help="The width of the browser viewport.",
)
@click.option(
    "--height",
    type=int,
    default=1080,
    show_default=True,
    help="The height of the browser viewport.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    show_default=True,
    help="Sampling temperature for the language model (0.0-2.0).",
)
@click.option(
    "--record-video",
    is_flag=True,
    default=False,
    show_default=True,
    help="Record a video of each agent's session.",
)
@click.option(
    "--save-local/--no-save-local",
    default=True,
    show_default=True,
    help="Save data to local directory.",
)
@click.option(
    "--save-gcs/--no-save-gcs",
    default=True,
    show_default=True,
    help="Save data to Google Cloud Storage.",
)
@click.option(
    "--gcs-bucket",
    type=str,
    default="training-dev-search-data-jtzn",
    show_default=True,
    help="GCS bucket name for data storage.",
)
@click.option(
    "--gcs-prefix",
    type=str,
    default="smu-agent-optimizer",
    show_default=True,
    help="GCS prefix for data storage.",
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
    default=CURRENT_DIR / "debug_run",
    show_default=True,
    help="Root directory under which per-agent debug folders will be created.",
)
def cli(
    task: str,
    curr_query: Optional[str],
    n_agents: int,
    personas_dir: Path,
    seed: Optional[int],
    model_name: str,
    summary_model: Optional[str],
    max_steps: int,
    headless: bool,
    concurrency: int,
    debug_root: Path,
    width: int,
    height: int,
    final_decision_model: Optional[str],
    temperature: float,
    record_video: bool,
    save_local: bool,
    save_gcs: bool,
    gcs_bucket: str,
    gcs_prefix: str,
):
    """Run multiple shopping agents with different personas for the same query.

    This script spawns N agents that will all perform the same shopping task,
    but each will be assigned a different, randomly selected persona from the
    `--personas-dir`. This is useful for evaluating how different personas
    affect the agent's behavior for a single task.
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

    # If recording is enabled, start it for the whole duration of the test.
    recording_proc = None
    if record_video:
        recording_proc = _start_main_recording(debug_root, width, height)

    # Prepare arguments for each agent
    agent_specs = []
    for idx, persona_file in enumerate(selected_personas):
        agent_specs.append(
            (idx, persona_file, model_name, final_decision_model)
        )

    status_display = StatusDisplay(agent_specs)

    async def _runner():
        # Use a semaphore to limit the number of concurrent agents so we don't exhaust system resources
        sem = asyncio.Semaphore(concurrency)
        await status_display.print_initial_statuses()

        async def _wrap(spec):
            (
                agent_id,
                persona_path,
                model_name_local,
                final_decision_model_name_local,
            ) = spec
            async with sem:
                # Update status to Running BEFORE starting the agent
                await status_display.update(agent_id, "Running...")
                # Small delay to allow status update to complete
                await asyncio.sleep(0.1)
                
                try:
                    final_status = await _run_single_agent(
                        agent_id=agent_id,
                        task=task,
                        curr_query=curr_query,  # Will default to task if None
                        persona_file=persona_path,
                        max_steps=max_steps,
                        headless=headless,
                        model_name=model_name_local,
                        debug_root=debug_root,
                        width=width,
                        height=height,
                        final_decision_model_name=final_decision_model_name_local,
                        temperature=temperature,
                        record_video=False,  # Agents should not handle recording
                        save_local=save_local,
                        save_gcs=save_gcs,
                        gcs_bucket=gcs_bucket,
                        gcs_prefix=gcs_prefix,
                    )
                except Exception as e:
                    final_status = f"⚠️  Failed: {str(e)}"
                    
                await status_display.update(agent_id, final_status)

        tasks = []
        for spec in agent_specs:
            tasks.append(_wrap(spec))
        await asyncio.gather(*tasks)

    try:
        asyncio.run(_runner())
    finally:
        # Ensure the main recording process is stopped when the test finishes.
        if recording_proc:
            _stop_main_recording(recording_proc)

    summary_model = summary_model if summary_model else model_name
    process_agent_results(debug_root, summary_model, temperature)


if __name__ == "__main__":
    cli()
