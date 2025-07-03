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

### Output Files

The agent generates the following files in the debug directory:
- `_memory.json` - All analyzed products and search history
- `_final_purchase_decision.json` - LLM's final purchase recommendations
- `_token_usage.json` - Token usage and cost breakdown by model
- `_semantic_scores.json` - Distribution of product relevance scores
- `screenshot_step_N_*.png` - Screenshots from each step
- `debug_step_N.json` - Detailed logs for each step
- `_session.mp4` - Video recording (if `--record-video` enabled)

---

## A/B Testing Framework (`ab_testing.py`)

The A/B testing script runs multiple shopping agents in parallel to compare the performance of two different models (a "control" and a "target"). It assigns unique personas to each agent and runs them concurrently with isolated logging.

### Key Features
- **Model Comparison**: Test control vs target models with statistical validity
- **Concurrent Execution**: Run multiple agents simultaneously with configurable limits
- **Persona Sampling**: Randomly assign personas from the dataset (with or without replacement)
- **Isolated Environments**: Each agent gets its own debug directory and browser profile
- **Real-time Monitoring**: Live status updates for all running agents
- **Comprehensive Logging**: Separate log files for each agent with error handling

### Usage

#### All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | "toddler-sized driving wheel toy" | Shopping task for all agents |
| `--n-agents` | 4 | Total agents to run (half control, half target) |
| `--personas-dir` | "../data/personas" | Directory with JSON persona files |
| `--seed` | *None* | Random seed for persona selection |
| `--control-model`| "gpt-4o-mini" | Model name for the control group |
| `--target-model`| "gpt-4o-mini" | Model name for the target group |
| `--max-steps` | 10 | Maximum number of steps per agent |
| `--headless` | True | Run browsers in headless mode |
| `--concurrency` | 4 | Max number of agents to run concurrently |
| `--debug-root` | "debug_run_ab" | Root directory for per-agent debug folders |
| `--width` | 1920 | Browser viewport width |
| `--height` | 1080 | Browser viewport height |
| `--control-final-decision-model` | "gpt-4o" | Final decision model for control group |
| `--target-final-decision-model` | "gpt-4o" | Final decision model for target group |
| `--temperature` | 0.7 | LLM sampling temperature |
| `--record-video` | False | Record video of the entire test session |

#### Examples

**Basic A/B test**:
```bash
python ab_testing.py --n-agents 4 --control-model "gpt-4o-mini" --target-model "gpt-4o"
```

**Large-scale comparison**:
```bash
python ab_testing.py --n-agents 12 --concurrency 6 --control-model "gpt-4o-mini" --target-model "o3-mini"
```

**Controlled experiment with seed**:
```bash
python ab_testing.py --n-agents 8 --seed 42 --max-steps 15
```

---

## Memory System (`memory.py`)

The memory module provides persistent storage for product analysis and search history. It prevents duplicate analysis and enables informed decision-making.

### Classes

#### `ProductMemory`
Stores information about analyzed products:
- `product_name`: Name of the product
- `url`: Product page URL
- `price`: Product price (float or None)
- `pros`: List of positive aspects
- `cons`: List of negative aspects  
- `summary`: Brief analysis summary
- `semantic_score`: Relevance rating ("HIGHLY RELEVANT", "SOMEWHAT RELEVANT", "NOT RELEVANT")

#### `MemoryModule`
Manages collections of products and search queries:
- `add_product()`: Add product to memory (prevents duplicates)
- `add_search_query()`: Track search queries
- `get_product_by_url()`: Retrieve product by URL
- `is_product_in_memory()`: Check if product exists
- `save_to_json()`: Persist memory to file

---

## Prompts (`prompts.py`)

Contains the LLM prompts used for product analysis and decision-making.

### `PRODUCT_ANALYSIS_PROMPT`
Analyzes individual product pages and returns structured JSON with:
- Pros and cons relative to the persona and search query
- Price extraction
- Summary of analysis
- Semantic relevance score

### `FINAL_DECISION_PROMPT`
Makes final purchase recommendations based on analyzed products:
- Compares all relevant products
- Provides reasoning for recommendations
- Calculates total cost

---

## Feature Suggestion (`feature_suggestion.py`)

A utility script that suggests new features for e-commerce websites to boost sales. Currently configured to analyze a sample Etsy page and generate feature recommendations using an LLM.

### Usage
```bash
python feature_suggestion.py
```

**Note**: This script currently references a hardcoded HTML file path that may not exist. Update the `webpage_html` path to point to a valid HTML file before running.

---

## Configuration

### API Keys
Create the API key file:
```bash
echo "your-api-key-here" > keys/litellm.key
```

### Supported Models
The system tracks costs for these models:
- `gpt-4o-mini`: $0.15/$0.60 per million input/output tokens
- `gpt-4o`: $2.50/$10.00 per million input/output tokens
- `o3-mini`: $1.10/$4.40 per million input/output tokens
- `openai/o4-mini`: $1.10/$4.40 per million input/output tokens

### JSON Config File Format
For complex personas, use a JSON file:
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