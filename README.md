# Optimization Agent for Query Rewriting

[![GitHub repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/DivijH/optimization-agent)

This repository contains an optimization agent for query rewriting leveraging multiple AI-powered shopping agents that can autonomously browse e-commerce websites like Etsy, analyze products, and make purchasing decisions based on a given query. The system includes genetic algorithms to optimize shopping queries and comprehensive tools for batch testing and analysis.

For more information, please see the overall project documentation: [Google Doc](https://docs.google.com/document/d/1ORWmq6GQMyoQZR7_b2S9Hs7l2A-e0Ce9f6EKy-pQ69Q/edit?tab=t.0#heading=h.4wbqtehjjc4)

## Key Features

- **Autonomous Product Analysis**: Browses e-commerce websites and analyzes products without human intervention
- **Visual Analysis**: Uses product images and language models to analyze product pages
- **Memory Management**: Tracks analyzed products and avoids duplicates
- **Batch Testing**: Run multiple agents with different temperatures for comparative analysis
- **Genetic Query Optimization**: Automatically evolve shopping queries to find better results using genetic algorithms
- **Batch Genetic Optimization**: Process large datasets of queries through genetic optimization
- **Cost Tracking**: Monitors token usage and API costs

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/DivijH/optimization-agent.git
    cd optimization-agent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API key**:
    ```bash
    echo "your-api-key-here" > src/keys/litellm.key
    ```

## Docker Deployment

To build and deploy the Docker image for AMD64 architecture:

```bash
docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/etsy-mlinfra-dev/etsy-llm/optimization-agent:latest .

docker push us-central1-docker.pkg.dev/etsy-mlinfra-dev/etsy-llm/optimization-agent:latest
```

## GCS Authentication

The agent uses the Google Cloud Storage client library which automatically detects credentials from:
1. Environment variable `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account key file
2. Default service account credentials if running on Google Cloud Platform
3. User credentials from `gcloud auth application-default login`

For local development, run:
```bash
gcloud auth application-default login
```

## Core Components

### Single Agent Analysis
Test individual shopping scenarios with customizable personas:
```bash
python -m src.shopping_agent.main --task "large, inflatable spider decoration for halloween"
```

### Batch Testing (Multiple Agents Analysis)
Run multiple agents simultaneously with different personas to compare behaviors:
```bash
python src/analyze_query.py --task "winter jacket" --n-agents 4
```

### Genetic Query Optimization
Evolve individual queries using genetic algorithms to find the most effective search terms:
```bash
python src/genetic_query_optimizer.py --query "vintage jewelry"
```

### Batch Genetic Query Optimization
Process a dataset of queries through genetic optimization for improvements:
```bash
python src/batch_genetic_optimizer.py --start-index 0 --end-index 99 --population-size 5 --generations 4
```

## Documentation

- **[Core Components (`src/`)](./src/README.md)**: Detailed documentation on the shopping agent's core components, utilities, and default values.
- **[Shopping Agent (`src/shopping_agent/`)](./src/shopping_agent/README.md)**: In-depth documentation for the main analysis shopping agent including all configuration options.
- **[Data (`data/`)](./data/README.md)**: Information about the query dataset used by the agent.

## Directory Structure

```
.
├── data/
│   ├── final_queries.csv            # 1000 queries for evaluation
│   └── README.md                    # Query documentation
├── src/
│   ├── shopping_agent/              # Core shopping agent logic (analysis agent)
│   │   ├── main.py                  # CLI entry point
│   │   ├── agent.py                 # Main agent class
│   │   ├── agent_actions.py         # Browser actions
│   │   ├── browser_utils.py         # Browser utilities
│   │   ├── config.py                # Configuration and defaults
│   │   ├── gcs_utils.py             # Google Cloud Storage Utils
│   │   ├── memory.py                # Memory management
|   |   ├── page_parser.py           # Parsing webpage for analysis
│   │   ├── prompts.py               # LLM prompts
│   │   ├── token_utils.py           # Token tracking
│   │   └── README.md                # Analysis agent documentation
│   ├── analyze_query.py             # CLI entry point for multiple analysis agents
│   ├── processing_results.py        # Results aggregation for multiple analysis agents
│   ├── genetic_query_optimizer.py   # Genetic algorithm for query optimization
│   ├── visualize_optimization.py    # Analysis and visualization of optimization results
│   ├── genetic_prompts.py           # Prompts for genetic algorithm
│   ├── batch_genetic_optimizer.py   # Wrapper for sequentially running optimization for all queries
│   ├── summary_prompt.txt           # LLM prompt for generating trends from all agents (used in GA)
│   ├── visualize_optimization.py    # Gathers all the results for an optimization
│   ├── keys/                        # API keys directory
│   │   └── litellm.key              # LiteLLM API key (create this!)
│   └── README.md                    # Core components documentation
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```
