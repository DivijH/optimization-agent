# Core Components

This directory contains the core logic for the Optimization Agent, including the shopping agent, genetic algorithms, batch processing, and analysis utilities.

## File Overview

| File | Purpose | Entry Point |
|------|---------|-------------|
| [`shopping_agent/`](./shopping_agent/README.md) | Main shopping agent implementation | Yes (`main.py`) |
| `analyze_query.py` | Batch testing with multiple agents | Yes |
| `genetic_query_optimizer.py` | Single query genetic optimization | Yes |
| `batch_genetic_optimizer.py` | Batch genetic optimization for CSV | Yes |
| `processing_results.py` | Results aggregation and analysis | Yes |
| `visualize_optimization.py` | Optimization results visualization | Yes |
| `semantic_relevance_match.py` | Semantic scoring validation | Yes |
| `genetic_prompts.py` | Prompts for genetic algorithm | No |
| `summary_prompt.txt` | Template for agent summaries | No |
| `keys/` | API keys directory | No |

## Detailed File Descriptions

### `shopping_agent/`
Contains the main shopping agent implementation that can autonomously browse e-commerce websites and analyze products. For detailed information about each component, see the [Shopping Agent README](./shopping_agent/README.md).

### `analyze_query.py`
Batch testing framework for running multiple shopping agents with different personas on the same query.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | "Dunlap pocket knife" | Shopping task for agents |
| `--curr-query` | str | None | Current query (defaults to task) |
| `--n-agents` | int | 4 | Number of agents to spawn |
| `--personas-dir` | Path | "data/personas/" | Directory with persona files |
| `--seed` | int | None | Random seed for persona selection |
| `--model-name` | str | "global-gemini-2.5-flash" | LLM model name |
| `--final-decision-model` | str | None | Model for final decisions |
| `--summary-model` | str | None | Model for summaries |
| `--max-steps` | int | None | Max steps per agent (unlimited) |
| `--width` | int | 1920 | Browser viewport width |
| `--height` | int | 1080 | Browser viewport height |
| `--temperature` | float | 0.7 | LLM sampling temperature |
| `--record-video` | flag | False | Record browser sessions |
| `--headless` | flag | True | Run browsers in headless mode |
| `--concurrency` | int | 2 | Max parallel agents |
| `--debug-root` | Path | "debug_run" | Debug output directory |
| `--save-local` | flag | True | Save results locally |
| `--save-gcs` | flag | True | Save results to GCS |
| `--gcs-bucket` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project |

**Features:**
- Runs multiple agents concurrently with different personas
- Generates comparative analysis across all agents
- Tracks token usage and costs per agent
- Creates consolidated summary reports

### `genetic_query_optimizer.py`
Genetic algorithm implementation for automatically optimizing shopping queries.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | str | Required | The original query to optimize |
| `--population-size` | int | 5 | Query variations per generation |
| `--generations` | int | 4 | Number of evolution cycles |
| `--mutation-rate` | float | 0.1 | Probability of mutation (0.0-1.0) |
| `--crossover-rate` | float | 0.7 | Probability of crossover (0.0-1.0) |
| `--n-agents` | int | 5 | Agents per query evaluation |
| `--max-steps` | int | None | Max steps per agent (unlimited) |
| `--debug-root` | Path | "debug_ga" | Debug output directory |
| `--page1-semantic-weight` | float | 0.4 | Weight for page 1 semantic relevance |
| `--top10-semantic-weight` | float | 0.5 | Weight for top 10 semantic relevance |
| `--purchase-weight` | float | 0.1 | Weight for purchase decision rate |
| `--headless` | flag | True | Run browsers in headless mode |
| `--model-name` | str | "global-gemini-2.5-flash" | Model for genetic operations |
| `--save-gcs` | flag | False | Upload results to GCS |
| `--gcs-bucket-name` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project |

**Process:**
1. Creates variations of your original query using AI
2. Tests each variation with real shopping agents 
3. Measures performance based on semantic relevance and purchase decisions
4. Evolves better queries over multiple generations
5. Returns the best performing query

### `batch_genetic_optimizer.py`
Batch wrapper for genetic optimization that processes queries from `final_queries.csv`.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start-index` | int | 0 | Starting CSV index (0-based, inclusive) |
| `--end-index` | int | 999 | Ending CSV index (0-based, inclusive) |
| `--population-size` | int | 5 | Population size per generation |
| `--generations` | int | 4 | Number of generations |
| `--n-agents` | int | 5 | Agents per query evaluation |
| `--max-steps` | int | None | Max steps per agent (unlimited) |
| `--csv-path` | Path | "data/final_queries.csv" | Path to query CSV file |
| `--debug-root` | Path | "batch_run" | Debug output directory |
| `--headless` | flag | True | Run browsers in headless mode |
| `--save-gcs` | flag | True | Upload results to GCS |
| `--gcs-bucket-name` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project |

**Features:**
- Processes large datasets of queries systematically
- Supports range selection (start-index to end-index)
- Automatically uploads results to GCS
- Generates comprehensive batch summary reports

### `processing_results.py`
Post-processing utility that analyzes results from multiple agent runs.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--debug-root` | Path | "debug_run" | Directory containing agent results |
| `--save-gcs` | flag | True | Upload results to GCS |
| `--gcs-bucket` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |

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
### `visualize_optimization.py`
Analysis and visualization tool for genetic algorithm optimization results.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results-path` | Path | "debug_ga/optimization_results.json" | Path to optimization results file |
| `--analysis` | str | "all" | Analysis type: all, evolution, diversity, genetics, semantic, best, summary |

**Analysis Types:**
- `summary`: Quick overview of optimization results
- `evolution`: Shows fitness trends across generations
- `diversity`: Population diversity metrics
- `genetics`: Crossover and mutation analysis
- `semantic`: Semantic relevance breakdown
- `best`: Top performing queries
- `all`: Complete analysis report

**Usage Examples:**
```bash
# Quick Summary
python src/visualize_optimization.py --analysis summary

# Evolution Trends
python src/visualize_optimization.py --analysis evolution

# Best Queries Found
python src/visualize_optimization.py --analysis best
```

### `semantic_relevance_match.py`
Utility to validate the agent's semantic relevance scoring against live semantic models.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--debug-path` | Path | "debug_run" | Path to agent debug directory |

**Purpose:**
- Reads `_memory.json` from debug runs
- Compares agent scores with live semantic model
- Validates relevance assessment accuracy
- Useful for quality assurance and model validation

**Usage:**
```bash
python src/semantic_relevance_match.py --debug-path debug_run/agent_0
```

### `genetic_prompts.py`
Contains prompts used by the genetic algorithm for generating query variations. This file defines the system prompts for:
- Initial population generation
- Crossover operations between queries
- Mutation operations for query variation

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

### Single Agent Workflow
1. **Task Interpretation**: The agent breaks down shopping tasks into specific search strategies
2. **Product Discovery**: Searches e-commerce sites and identifies relevant product listings
3. **Visual Selection**: Uses screenshots and language models to choose promising products
4. **Product Analysis**: Scrolls through product pages and extracts detailed information
5. **Memory Management**: Stores analysis results and avoids re-analyzing products
6. **Decision Making**: Generates final purchase recommendations based on collected data

### Genetic Query Optimization Workflow
1. **Initial Population**: Creates diverse variations of the original query using AI
2. **Fitness Evaluation**: Tests each query variation with multiple shopping agents
3. **Performance Measurement**: Calculates fitness based on semantic relevance and purchase decisions
4. **Selection & Crossover**: Combines successful queries to create new variations
5. **Mutation**: Introduces random modifications to maintain diversity
6. **Evolution**: Repeats the process over multiple generations to find optimal queries

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

### Genetic Query Optimization
```bash
# Basic optimization
python src/genetic_query_optimizer.py --query "vintage jewelry"

# Quick test (faster)
python src/genetic_query_optimizer.py \
    --query "coffee mug" \
    --population-size 4 \
    --generations 2

# Thorough optimization
python src/genetic_query_optimizer.py \
    --query "handmade scarf" \
    --population-size 10 \
    --generations 5 \
    --n-agents 3
```

### Analyzing Results
```bash
# Check semantic relevance accuracy
python src/semantic_relevance_match.py --debug-path debug_run/agent_0

# Analyze optimization results
python src/visualize_optimization.py --analysis summary
python src/visualize_optimization.py --analysis evolution
```

## Environment Variables

- `OPENAI_API_KEY`: Set automatically from `src/keys/litellm.key`
- `OPENAI_API_BASE`: Points to LiteLLM proxy endpoint
- `GOOGLE_APPLICATION_CREDENTIALS`: For GCS authentication (optional) 
