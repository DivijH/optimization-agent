# Core Components

This directory contains the core logic for the Optimization Agent, including the shopping agent, A/B testing framework, memory management, prompts, and feature suggestion utilities.

## Files Overview

| File | Description |
|------|-------------|
| `ab_testing.py` | A/B testing framework for comparing different LLM models |
| `shopping_agent.py` | Main shopping agent that browses Etsy and analyzes products |
| `feature_suggestion.py` | Utility for generating product feature suggestions |
| `memory.py` | Memory management system for storing product analysis |
| `prompts.py` | LLM prompts for product analysis and decision making |
| `keys/litellm.key` | API key file (create this manually) |

## Shopping Agent (`shopping_agent.py`)

An AI-powered Etsy shopping agent that autonomously browses and analyzes products based on a given task and persona. The agent uses browser automation to search for products, analyze product pages, and make informed shopping decisions.

### Key Features
- **Autonomous Browsing**: Navigates Etsy search results and product pages automatically
- **Visual Product Selection**: Uses screenshots and multimodal LLM to choose which products to analyze
- **Comprehensive Analysis**: Extracts product information including price, pros, cons, and relevance scores
- **Memory System**: Tracks analyzed products and search history to avoid duplicates
- **Final Recommendations**: Makes data-driven purchase decisions based on collected analysis
- **Debug Artifacts**: Saves detailed logs, screenshots, and analysis results
- **Cost Tracking**: Monitors LLM token usage and associated costs in real-time
- **Video Recording**: Optional screen recording of browser sessions (requires ffmpeg)

### Usage

#### All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | "indoor frisbee" | The shopping task for the agent |
| `--persona` | Samantha (default persona) | The persona for the agent |
| `--config-file` | *None* | Path to a JSON file containing `intent` and `persona`. Overrides flags |
| `--manual` | False | Wait for user to press Enter after each agent action |
| `--headless` | False | Run the browser in headless mode |
| `--max-steps` | *None* | Maximum number of steps the agent will take |
| `--debug-path` | "debug_run" | Path to save debug artifacts (screenshots, logs) |
| `--width` | 3024 | Browser viewport width |
| `--height` | 1964 | Browser viewport height |
| `--model` | "gpt-4o-mini" | LLM model name (e.g., gpt-4o, gpt-4o-mini, o3-mini) |
| `--final-decision-model` | "gpt-4o" | Model for final decision (defaults to main model) |
| `--temperature` | 0.7 | LLM sampling temperature (0-2) |
| `--record-video` | False | Record browser session video (requires ffmpeg) |
| `--user-data-dir`| *None* | Path to browser profile directory for session persistence |

#### Examples

**Basic shopping task**:
```bash
python shopping_agent.py --task "find a handmade ceramic mug"
```

**With custom persona**:
```bash
python shopping_agent.py --task "buy a birthday gift" --persona "Sarah, a 28-year-old graphic designer who loves minimalist design"
```

**Debug mode with manual control**:
```bash
python shopping_agent.py --task "find vintage jewelry" --manual --debug-path "my_debug_run"
```

**Headless mode with video recording**:
```bash
python shopping_agent.py --task "buy art supplies" --headless --record-video --max-steps 20
```

**Custom model and temperature**:
```bash
python shopping_agent.py --task "find a wedding gift" --model "gpt-4o" --temperature 0.3
```

**Using a JSON config file**:
```bash
python shopping_agent.py --config-file my_shopping_task.json
```

---

## Semantic Relevance Match (`semantic_relevance_match.py`)

This script is a utility to analyze the semantic relevance scores assigned by the shopping agent. It reads the `_memory.json` file from a debug run, and for each product, it calls a live semantic relevance model to compare the agent's score with a fresh prediction.

This is useful for evaluating the agent's understanding of product relevance for a given query.

### How it Works

1.  **Reads Memory**: Loads the `_memory.json` file from `debug_run/_memory.json`.
2.  **Calls Model**: For each product in the memory, it calls a semantic relevance model endpoint.
3.  **Compares Scores**: It compares the `semantic_score` stored by the agent with the live model's classification.
4.  **Saves Results**: Saves a detailed comparison for each product in `debug_run/_semantic_comparison_results.json`.

### Quick Start

First, ensure you have a `debug_run` directory with a `_memory.json` file from a previous `shopping_agent.py` run. Then:

```bash
python semantic_relevance_match.py
```

The script will print a summary of matches and mismatches to the console.

---
## How It Works

1. **Task Breakdown**: Breaks your shopping task into specific search queries
2. **Product Discovery**: Searches Etsy and identifies relevant products
3. **Visual Selection**: Uses screenshots + LLM to choose the most promising listings
4. **Product Analysis**: Summarises pros/cons for each product page
5. **Memory Management**: Remembers analysed products to avoid duplicates
6. **Decision Making**: Generates the final purchase recommendation based on memory

## Output

- **Console**: Real-time progress and decision logs
- **Memory**: `_memory.json` with all analysed products & search history
- **Final Decision**: `_final_purchase_decision.json` with the LLM's recommendation
- **Debug Files**: Step-by-step JSON logs and screenshots (if `--debug-path` is set)
- **Video**: `session.mp4` recording (if `--record-video` is enabled)

## Key Features

- **Task Decomposition**: Automatically breaks high-level shopping goals into smaller sub-tasks using an LLM.
- **Visual Product Selection**: Captures screenshots of search results and lets a multimodal LLM decide which listing to open next.
- **In-Depth Product Analysis**: Scrolls through product pages, extracts text & images, and asks the LLM to summarise pros, cons, and a short description.
- **Long-Term Memory**: Stores analysed products and search queries to avoid duplicates and provide context for future decisions.
- **Final Recommendation**: After exploring, the agent uses its memory to output a JSON purchase recommendation.
- **Rich Debug Artefacts**: Saves JSON logs and annotated/plain screenshots for every step to the folder specified by `--debug-path`.
- **Screen Recording (optional)**: Record the entire browser session to MP4 with `--record-video` (requires ffmpeg).
- **Interactive Mode**: Add `--manual` to pause after every action so you can inspect what the agent is doing.

### Prerequisites for Screen Recording

The `--record-video` flag relies on **[ffmpeg](https://ffmpeg.org/)** being installed and available in your `PATH`. On macOS you can install it with Homebrew:

```bash
brew install ffmpeg
```

⚠️  `--record-video` and `--headless` cannot be used together – the script will exit with an error if both flags are supplied.

## Using a JSON Config File

Passing large persona descriptions via the command line can get messy. Instead, place the shopping `intent` and `persona` in a small JSON file:

```json
{
  "intent": "buy a pair of compression socks for women",
  "persona": "Persona: Sophia\n\nBackground:\nSophia is a dedicated community college professor..."
}
```

---

## How It Works

1. **Task Interpretation**: The agent breaks down shopping tasks into specific search strategies
2. **Product Discovery**: Searches Etsy and identifies relevant product listings
3. **Visual Selection**: Uses screenshots and LLM to choose the most promising products
4. **Product Analysis**: Scrolls through product pages and extracts detailed information
5. **Memory Management**: Stores analysis results and avoids duplicate work
6. **Decision Making**: Generates final purchase recommendations based on collected data

## Prerequisites

- **FFmpeg** (for video recording): Install via package manager
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/)
- **LiteLLM API Key**: Configure in `keys/litellm.key`
- **Python Dependencies**: Install via `pip install -r ../requirements.txt` 