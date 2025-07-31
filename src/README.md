# Core Components

This directory contains the core logic for the Optimization Agent, including the analysis agent, genetic algorithms, batch processing, and utilities.

## File Overview

| File | Purpose | Entry Point |
|------|---------|-------------|
| [`shopping_agent/`](./shopping_agent/README.md) | Main analysis agent implementation | Yes (`main.py`) |
| `analyze_query.py` | Analysis with multiple agents | Yes |
| `genetic_query_optimizer.py` | Single query genetic optimization | Yes |
| `batch_genetic_optimizer.py` | Batch genetic optimization from CSV | Yes |
| `processing_results.py` | Results aggregation and analysis | Yes |
| `visualize_optimization.py` | Optimization results visualization | Yes |
| `genetic_prompts.py` | Prompts for genetic algorithm | No |
| `summary_prompt.txt` | Template for agent summaries | No |
| `keys/` | API keys directory | No |

## Detailed File Descriptions

### `shopping_agent/`
Contains the main shopping agent implementation that can autonomously browse e-commerce websites and analyze products. For detailed information about each component, see the [Shopping Agent README](./shopping_agent/README.md).

### `analyze_query.py`
Testing framework for running multiple shopping agents with different temperatures on the same query.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | "Dunlap pocket knife" | Shopping task for agents |
| `--curr-query` | str | None | Current query (defaults to task) |
| `--n-agents` | int | 4 | Number of agents to spawn |
| `--model-name` | str | "global-gemini-2.5-flash" | LLM model name |
| `--final-decision-model` | str | None | Model for final decisions (defaults to model-name) |
| `--summary-model` | str | None | Model for summaries (defaults to model-name) |
| `--max-steps` | int | None | Max steps per agent (default unlimited) |
| `--width` | int | 1920 | Browser viewport width |
| `--height` | int | 1080 | Browser viewport height |
| `--temperature` | float | 0.7 | LLM sampling temperature |
| `--record-video` | flag | False | Record browser sessions |
| `--save-local/--no-save-local` | flag | True | Save logs on local machine |
| `--save-gcs/--no-save-gcs` | flag | True | Save logs in Google Cloud Storage |
| `--gcs-bucket` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project |
| `--headless/--no-headless` | flag | True | Run browsers in headless mode |
| `--concurrency` | int | 2 | Max parallel agents |
| `--debug-root` | Path | "debug_run" | Output directory for saving logs |

**Features:**
- Runs multiple agents concurrently with different temperatures, distributed equally from 0 to 1
- Generates comparative analysis across all agents
- Tracks token usage and costs per agent
- Creates consolidated summary reports

### `genetic_query_optimizer.py`
Genetic algorithm implementation for automatically optimizing shopping queries.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | str | (Required) | The user query to optimize |
| `--population-size` | int | 5 | Query variations per generation |
| `--generations` | int | 4 | Number of evolution cycles |
| `--mutation-rate` | float | 0.1 | Probability of mutation (0.0-1.0) |
| `--crossover-rate` | float | 0.7 | Probability of crossover (0.0-1.0) |
| `--n-agents` | int | 5 | Agents per query evaluation |
| `--max-steps` | int | None | Max steps per agent (None for unlimited) |
| `--debug-root` | Path | "debug_ga" | Debug output directory |
| `--page1-semantic-weight` | float | 0.4 | Fitness weight for page 1 semantic relevance |
| `--top10-semantic-weight` | float | 0.5 | Fitness weight for top 10 semantic relevance |
| `--purchase-weight` | float | 0.1 | Fitness weight for amount of money spent |
| `--headless/--no-headless` | flag | True | Run browsers in headless mode |
| `--model-name` | str | "global-gemini-2.5-flash" | Model for genetic operations |
| `--save-gcs/--no-save-gcs` | flag | False | Upload results to GCS |
| `--gcs-bucket-name` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project |

**Process:**
- Creates variations of your original query using LLM-powered Genetic Algorithms
- Tests each variation with shopping agents 
- Measures performance based on semantic relevance and purchase decisions
- Evolves better queries over multiple generations
- Returns the best performing query

### `batch_genetic_optimizer.py`
Wrapper for genetic optimization that processes queries from [`final_queries.csv`](../data/final_queries.csv).

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start-index` | int | 0 | Starting CSV index (0-based, inclusive) |
| `--end-index` | int | 999 | Ending CSV index (0-based, inclusive) |
| `--population-size` | int | 5 | Population size per generation |
| `--generations` | int | 4 | Number of generations |
| `--n-agents` | int | 5 | Agents per query evaluation |
| `--max-steps` | int | None | Max steps per agent (None for unlimited) |
| `--csv-path` | Path | "data/final_queries.csv" | Path to query CSV file |
| `--debug-root` | Path | "batch_run" | Debug output directory |
| `--headless` | flag | True | Run browsers in headless mode |
| `--save-gcs` | flag | True | Upload results to GCS |
| `--gcs-bucket-name` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project |

**Features:**
- Processes large datasets of queries sequentially
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
# Full Analysis
python src/visualize_optimization.py

# Quick Summary
python src/visualize_optimization.py --analysis summary

# Evolution Trends
python src/visualize_optimization.py --analysis evolution

# Best Queries Found
python src/visualize_optimization.py --analysis best
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