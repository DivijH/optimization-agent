# Core Components

This directory contains the core logic for the Optimization Agent, including the shopping agent, memory management, and other utilities.

## Directory and File Descriptions

-   **`shopping_agent/`**: This directory contains the main shopping agent, which can autonomously browse e-commerce websites and analyze products. For more details, see the [Shopping Agent README](./shopping_agent/README.md).

-   **`feature_suggestion.py`**: A script for generating product feature suggestions.

-   **`memory.py`**: This file implements the agent's memory system, which is used to store and retrieve information about analyzed products and search history.

-   **`semantic_relevance_match.py`**: This script is a utility to analyze the semantic relevance scores assigned by the shopping agent. It reads the `_memory.json` file from a debug run, and for each product, it calls a live semantic relevance model to compare the agent's score with a fresh prediction. This is useful for evaluating the agent's understanding of product relevance for a given query.

-   **`keys/`**: This directory is used to store API keys. You will need to create a `litellm.key` file in this directory with your API key.

## How It Works

1.  **Task Interpretation**: The agent breaks down shopping tasks into specific search strategies.
2.  **Product Discovery**: It searches e-commerce sites and identifies relevant product listings.
3.  **Visual Selection**: It uses screenshots and a language model to choose the most promising products to look at.
4.  **Product Analysis**: It scrolls through product pages and extracts detailed information.
5.  **Memory Management**: It stores analysis results and avoids re-analyzing the same products.
6.  **Decision Making**: It generates final purchase recommendations based on the collected data. 