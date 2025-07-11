# Core Components

This directory contains the core logic for the Optimization Agent, including the shopping agent, memory management, and analysis utilities.

## Directory and File Descriptions

### `shopping_agent/`
Contains the main shopping agent implementation that can autonomously browse e-commerce websites and analyze products. For detailed information about each component, see the [Shopping Agent README](./shopping_agent/README.md).

### `analyze_query.py`
Batch testing framework for running multiple shopping agents with different personas on the same query.

**Default Values:**
- `--task`: "Dunlap pocket knife"
- `--n-agents`: 4 (number of agents to spawn)
- `--personas-dir`: `data/personas/`
- `--seed`: None (random persona selection)
- `--model-name`: "openai/o4-mini"
- `--final-decision-model`: None (uses main model)
- `--summary-model`: None (uses main model)
- `--max-steps`: None (unlimited)
- `--width`: 1920
- `--height`: 1080
- `--temperature`: 0.7
- `--record-video`: False
- `--headless`: True (runs in background)
- `--concurrency`: 2 (max parallel agents)
- `--debug-root`: `src/debug_run`

**Features:**
- Runs multiple agents concurrently with different personas
- Generates comparative analysis across all agents
- Tracks token usage and costs per agent
- Creates consolidated summary reports

### `processing_results.py`
Post-processing utility that analyzes results from multiple agent runs.

**Key Functions:**
- Aggregates memory data from all agents
- Calculates semantic relevance statistics
- Generates purchase decision summaries
- Creates consolidated cost reports
- Produces markdown summary files

**Output Files:**
- `_summary.txt`: Consolidated analysis report
- `_semantic_scores.json`: Aggregated relevance scores
- `_final_purchase_decision.json`: Combined recommendations

### `feature_suggestion.py`
Script for generating product feature suggestions and query rewrites.

**Functionality:**
- Analyzes shopping queries
- Suggests alternative search terms
- Generates feature recommendations
- Helps optimize search strategies

### `semantic_relevance_match.py`
Utility to validate the agent's semantic relevance scoring.

**Purpose:**
- Reads `_memory.json` from debug runs
- Compares agent scores with live semantic model
- Validates relevance assessment accuracy
- Useful for quality assurance

**Usage:**
```bash
python src/semantic_relevance_match.py --debug-path debug_run/agent_0
```

### `summary_prompt.txt`
Template file containing the prompt structure used for generating agent summaries. This template is used by `processing_results.py` to create consistent summary reports across different agent runs.

### `keys/`
Directory for storing API keys. Required files:
- `litellm.key`: Your LiteLLM API key for accessing language models

**Setup:**
```bash
echo "your-api-key-here" > src/keys/litellm.key
```

## How It Works

1. **Task Interpretation**: The agent breaks down shopping tasks into specific search strategies
2. **Product Discovery**: Searches e-commerce sites and identifies relevant product listings
3. **Visual Selection**: Uses screenshots and language models to choose promising products
4. **Product Analysis**: Scrolls through product pages and extracts detailed information
5. **Memory Management**: Stores analysis results and avoids re-analyzing products
6. **Decision Making**: Generates final purchase recommendations based on collected data

## Usage Examples

### Single Agent Run
```bash
# Basic usage
python -m src.shopping_agent.main --task "vintage camera"

# With custom persona and settings
python -m src.shopping_agent.main \
    --task "camping tent" \
    --model gpt-4o \
    --max-steps 15 \
    --headless
```

### Batch Testing Multiple Agents
```bash
# Run 4 agents with random personas
python src/analyze_query.py --task "winter jacket" --n-agents 4

# Run with specific settings
python src/analyze_query.py \
    --task "coffee maker" \
    --n-agents 6 \
    --concurrency 3 \
    --model gpt-4o-mini \
    --record-video
```

### Analyzing Results
```bash
# Check semantic relevance accuracy
python src/semantic_relevance_match.py --debug-path debug_run/agent_0

# Generate feature suggestions
python src/feature_suggestion.py --query "outdoor furniture"
```

## Environment Variables

- `OPENAI_API_KEY`: Set automatically from `src/keys/litellm.key`
- `OPENAI_API_BASE`: Points to LiteLLM proxy endpoint
- `GOOGLE_APPLICATION_CREDENTIALS`: For GCS authentication (optional) 
