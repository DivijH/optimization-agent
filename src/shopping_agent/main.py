import asyncio
import signal
import sys
import os
import click

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shopping_agent.agent import EtsyShoppingAgent
from src.shopping_agent.config import DEFAULT_TASK


# Set up LiteLLM credentials
try:
    os.environ["OPENAI_API_KEY"] = open("../keys/litellm.key").read().strip()
except FileNotFoundError:
    raise Exception(
        "litellm.key file not found. It is expected in optimization-agent/src/keys/litellm.key"
    )
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"


async def async_main(agent: EtsyShoppingAgent):
    """Runs the agent and handles graceful shutdown within a single event loop."""
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    async def shutdown_handler():
        print("\n👋 Interrupted by user. Shutting down...")
        # Prevent further interruptions during shutdown
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            print(
                "\nShutting down gracefully. Please wait, this may take a moment and cannot be interrupted..."
            )
            await agent.shutdown()
        finally:
            # Restore the original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)
            print("\nShutdown complete.")

    def handle_sigint(sig, frame):
        asyncio.create_task(shutdown_handler())

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        await agent.run()
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        await agent.shutdown()


@click.command()
@click.option("--task", default=DEFAULT_TASK, help="The shopping task for the agent.")
@click.option("--curr-query", default=None, help="The current query for the agent. Defaults to the task.")
@click.option("--manual", is_flag=True, help="Wait for user to press Enter after each agent action.")
@click.option("--headless", is_flag=True, help="Run the browser in headless mode.")
@click.option("--max-steps", default=None, type=int, help="The maximum number of steps the agent will take. If not provided, the agent will continue until no more products are left to analyze.")
@click.option("--debug-path", type=click.Path(), default="debug_run", help="Path to save debug artifacts, such as screenshots.")
@click.option("--width", default=1920, help="The width of the browser viewport.")
@click.option("--height", default=1080, help="The height of the browser viewport.")
@click.option("--model", "model_name", default="global-gemini-2.5-flash", help="Model name to use.")
@click.option("--final-decision-model", "final_decision_model_name", default=None, help="Model name for the final decision. Defaults to the main model.")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature for the language model (0-2).")
@click.option("--record-video", is_flag=True, help="Record the agent's browser session and save it to the debug path.")
@click.option("--user-data-dir", type=click.Path(), help="Path to user data directory for the browser.")
@click.option("--save-local/--no-save-local", default=True, help="Save data to local directory (default: True).")
@click.option("--save-gcs/--no-save-gcs", default=True, help="Save data to Google Cloud Storage (default: True).")
@click.option("--gcs-bucket", default="training-dev-search-data-jtzn", help="GCS bucket name for data storage.")
@click.option("--gcs-prefix", default="mission-understanding/optimization-agent", help="GCS prefix for data storage.")
@click.option("--gcs-project", default="etsy-search-ml-dev", help="GCS project name for client initialization.")
def cli(
    task,
    curr_query,
    manual,
    headless,
    max_steps,
    debug_path,
    width,
    height,
    model_name,
    temperature,
    record_video,
    user_data_dir,
    final_decision_model_name,
    save_local,
    save_gcs,
    gcs_bucket,
    gcs_prefix,
    gcs_project,
):
    """A command-line interface to run the EtsyShoppingAgent."""

    agent = EtsyShoppingAgent(
        task=task,
        curr_query=curr_query,
        manual=manual,
        headless=headless,
        max_steps=max_steps,
        debug_path=debug_path,
        viewport_width=width,
        viewport_height=height,
        model_name=model_name,
        temperature=temperature,
        record_video=record_video,
        user_data_dir=user_data_dir,
        final_decision_model_name=final_decision_model_name,
        save_local=save_local,
        save_gcs=save_gcs,
        gcs_bucket_name=gcs_bucket,
        gcs_prefix=gcs_prefix,
        gcs_project=gcs_project,
    )
    asyncio.run(async_main(agent))


if __name__ == "__main__":
    cli() 