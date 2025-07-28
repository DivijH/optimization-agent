import asyncio
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
        self.statuses = {spec[0]: f"Waiting to start... (temp={spec[3]:.3f})" for spec in agent_specs}
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



async def _run_single_agent(
    *,
    agent_id: int,
    task: str,
    curr_query: Optional[str],
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
    gcs_project: str,
) -> str:
    """Create and execute a single `EtsyShoppingAgent` instance."""

    debug_path = debug_root / f"agent_{agent_id}"
    debug_path.mkdir(parents=True, exist_ok=True)
    log_file = debug_path / "agent.log"
    logger = _setup_file_logger(f"agent_{agent_id}", log_file)

    user_data_dir = debug_path / "browser_profile"

    agent = EtsyShoppingAgent(
        task=task,
        curr_query=curr_query if curr_query is not None else task,
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
        gcs_project=gcs_project,
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


async def run_analyze_query(
    task: str,
    curr_query: Optional[str],
    n_agents: int,
    model_name: str,
    summary_model: Optional[str],
    max_steps: Optional[int],
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
    gcs_project: str,
    skip_confirmation: bool = False,
) -> None:
    """
    Run multiple shopping agents for the same query.
    This function spawns N agents that will all perform the same shopping task.
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

    if debug_root.exists():
        if not skip_confirmation:
            if not click.confirm(f'Debug path {debug_root} already exists. Do you want to remove it and all its contents?'):
                raise FileExistsError(f"Debug path {debug_root} already exists and user declined to remove it.")
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)

    # If recording is enabled, start it for the whole duration of the test.
    recording_proc = None
    if record_video:
        recording_proc = _start_main_recording(debug_root, width, height)

    # Prepare arguments for each agent with different temperatures
    agent_specs = []
    for idx in range(n_agents):
        # Calculate temperature evenly distributed between 0 and 1
        if n_agents == 1:
            agent_temperature = 0.0
        else:
            agent_temperature = idx / (n_agents - 1)
        
        agent_specs.append(
            (idx, model_name, final_decision_model, agent_temperature)
        )

    status_display = StatusDisplay(agent_specs)

    async def _runner():
        # Use a semaphore to limit the number of concurrent agents so we don't exhaust system resources
        sem = asyncio.Semaphore(concurrency)
        await status_display.print_initial_statuses()

        async def _wrap(spec):
            (
                agent_id,
                model_name_local,
                final_decision_model_name_local,
                agent_temperature,
            ) = spec
            async with sem:
                # Update status to Running BEFORE starting the agent
                await status_display.update(agent_id, f"Running... (temp={agent_temperature:.3f})")
                # Small delay to allow status update to complete
                await asyncio.sleep(0.1)
                
                try:
                    final_status = await _run_single_agent(
                        agent_id=agent_id,
                        task=task,
                        curr_query=curr_query,  # Will default to task if None
                        max_steps=max_steps,
                        headless=headless,
                        model_name=model_name_local,
                        debug_root=debug_root,
                        width=width,
                        height=height,
                        final_decision_model_name=final_decision_model_name_local,
                        temperature=agent_temperature,  # Use agent-specific temperature
                        record_video=False,  # Agents should not handle recording
                        save_local=save_local,
                        save_gcs=save_gcs,
                        gcs_bucket=gcs_bucket,
                        gcs_prefix=gcs_prefix,
                        gcs_project=gcs_project,
                    )
                except Exception as e:
                    final_status = f"⚠️  Failed: {str(e)}"
                    
                await status_display.update(agent_id, f"{final_status} (temp={agent_temperature:.3f})")

        tasks = []
        for spec in agent_specs:
            tasks.append(_wrap(spec))
        await asyncio.gather(*tasks)

    try:
        await _runner()
    finally:
        # Ensure the main recording process is stopped when the test finishes.
        if recording_proc:
            _stop_main_recording(recording_proc)

    summary_model = summary_model if summary_model else model_name
    process_agent_results(debug_root, summary_model, temperature)


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
    "--gcs-project",
    type=str,
    default="etsy-search-ml-dev",
    show_default=True,
    help="GCS project name for client initialization.",
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
    gcs_project: str,
):
    """Run multiple shopping agents for the same query.

    This script spawns N agents that will all perform the same shopping task
    using analytical thinking and common sense to judge semantic relevance.
    This is useful for evaluating agent behavior consistency across multiple runs.
    """


    try:
        asyncio.run(run_analyze_query(
            task=task,
            curr_query=curr_query,
            n_agents=n_agents,
            model_name=model_name,
            summary_model=summary_model,
            max_steps=max_steps,
            headless=headless,
            concurrency=concurrency,
            debug_root=debug_root,
            width=width,
            height=height,
            final_decision_model=final_decision_model,
            temperature=temperature,
            record_video=record_video,
            save_local=save_local,
            save_gcs=save_gcs,
            gcs_bucket=gcs_bucket,
            gcs_prefix=gcs_prefix,
            gcs_project=gcs_project,
            skip_confirmation=False,
        ))
    except FileExistsError:
        print('Aborting.')
        sys.exit(1)


if __name__ == "__main__":
    cli()
