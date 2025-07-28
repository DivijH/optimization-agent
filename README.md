# Optimization Agent

[![GitHub repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/DivijH/optimization-agent)

This repository contains an AI-powered shopping agent that can autonomously browse e-commerce websites like Etsy, analyze products, and make purchasing decisions based on a given task and persona.

For more information, please see the overall project documentation: [Google Doc](https://docs.google.com/document/d/1ORWmq6GQMyoQZR7_b2S9Hs7l2A-e0Ce9f6EKy-pQ69Q/edit?tab=t.0#heading=h.4wbqtehjjc4)

## Key Features

- **Autonomous Product Analysis**: Browses e-commerce websites and analyzes products without human intervention
- **Visual Analysis**: Uses screenshots and language models to understand product pages
- **Memory Management**: Tracks analyzed products and avoids duplicates
- **Batch Testing**: Run multiple agents with different personas for comparative analysis
- **Genetic Query Optimization**: Automatically evolve shopping queries to find better results using genetic algorithms
- **Cost Tracking**: Monitors token usage and API costs
- **Decision Making**: Generates purchase recommendations based on persona preferences

### GCS Authentication

The agent uses the Google Cloud Storage client library which automatically detects credentials from:
1. Environment variable `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account key file
2. Default service account credentials if running on Google Cloud Platform
3. User credentials from `gcloud auth application-default login`

For local development, run:
```bash
gcloud auth application-default login
```

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

## Quick Start

### Test the System
First, verify everything works by running a quick optimization:
```bash
cd optimization-agent
python src/genetic_query_optimizer.py --query "test" --population-size 2 --generations 1 --n-agents 1 --max-steps 2
```

### Single Agent Run
```bash
python -m src.shopping_agent.main --task "large, inflatable spider decoration for halloween"
```

### Batch Testing (Multiple Agents)
```bash
python src/analyze_query.py --task "winter jacket" --n-agents 4
```

### Genetic Query Optimization
```bash
python src/genetic_query_optimizer.py --query "vintage jewelry"
```

### Analyze Results
```bash
python src/visualize_optimization.py --analysis summary
```

## Documentation

- **[Core Components (`src/`)](./src/README.md)**: Detailed documentation on the shopping agent's core components, utilities, and default values.
- **[Shopping Agent (`src/shopping_agent/`)](./src/shopping_agent/README.md)**: In-depth documentation for the main shopping agent including all configuration options.
- **[Data (`data/`)](./data/README.md)**: Information about the virtual customer personas used by the agent.

## Directory Structure

```
.
├── data/
│   ├── personas/                    # Virtual customer personas for agents
│   │   ├── ... (1000 persona files)
│   └── README.md                    # Persona documentation
├── src/
│   ├── shopping_agent/              # Core shopping agent logic
│   │   ├── main.py                  # CLI entry point
│   │   ├── agent.py                 # Main agent class
│   │   ├── agent_actions.py         # Browser actions
│   │   ├── browser_utils.py         # Browser utilities
│   │   ├── config.py                # Configuration and defaults
│   │   ├── gcs_utils.py             # Google Cloud Storage
│   │   ├── memory.py                # Memory management
│   │   ├── prompts.py               # LLM prompts
│   │   ├── token_utils.py           # Token tracking
│   │   └── README.md                # Agent documentation
│   ├── analyze_query.py             # Batch testing framework
│   ├── processing_results.py        # Results aggregation
│   ├── genetic_query_optimizer.py   # Genetic algorithm for query optimization
│   ├── visualize_optimization.py    # Analysis and visualization of optimization results
│   ├── genetic_prompts.py           # Prompts for genetic algorithm
│   ├── semantic_relevance_match.py  # Relevance validation
│   ├── summary_prompt.txt           # LLM prompt for generating trends from all agents
│   ├── keys/                        # API keys directory
│   │   └── litellm.key              # LiteLLM API key (create this)
│   └── README.md                    # Core components documentation
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```


See individual component READMEs for detailed output file descriptions.
