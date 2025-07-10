# Shopping Agent

This directory contains the core logic for the AI-powered shopping agent.

## Files

-   `main.py`: The main entry point for running the shopping agent from the command line.
-   `agent.py`: Contains the core implementation of the shopping agent, including its decision-making loop.
-   `agent_actions.py`: Defines the specific actions that the agent can perform on a webpage, such as clicking, typing, and scrolling.
-   `browser_utils.py`: Provides utility functions for controlling and interacting with the web browser.
-   `config.py`: Handles configuration management for the agent, loading settings from files or environment variables.
-   `gcs_utils.py`: Contains functions for interacting with Google Cloud Storage (GCS), used for saving agent run data.
-   `prompts.py`: Stores the various prompts used to guide the language model's behavior for different tasks.
-   `token_utils.py`: Includes utilities for tracking and managing token usage for Large Language Model (LLM) API calls.
-   `__init__.py`: Initializes the `shopping_agent` directory as a Python package.
