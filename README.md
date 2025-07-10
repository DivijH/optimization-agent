# Optimization Agent

[![GitHub repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/DivijH/optimization-agent)

This repository contains an AI-powered shopping agent that can autonomously browse e-commerce websites like Etsy, analyze products, and make purchasing decisions based on a given task and persona.

For more information, please see the overall project documentation: [Google Doc](https://docs.google.com/document/d/1ORWmq6GQMyoQZR7_b2S9Hs7l2A-e0Ce9f6EKy-pQ69Q/edit?tab=t.0#heading=h.4wbqtehjjc4)


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

```bash
python -m src.shopping_agent.main --task "large, inflatable spider decoration for halloween"
```

## Documentation

- **[Core Components (`src/`)](./src/README.md)**: Detailed documentation on the shopping agent's core components and utilities.
- **[Shopping Agent (`src/shopping_agent/`)](./src/shopping_agent/README.md)**: Documentation for the main shopping agent.
- **[Data (`data/`)](./data/README.md)**: Information about the virtual customer personas used by the agent.

## Directory Structure

```
.
├── data/
│   ├── personas/                    # Virtual customer personas for agents
│   │   ├── ... (persona files)
│   └── README.md                    # Information about the data
├── src/
│   ├── shopping_agent/              # Core logic for the shopping agent
│   │   ├── main.py
│   │   ├── agent.py
│   │   ├── agent_actions.py
│   │   ├── browser_utils.py
│   │   ├── config.py
│   │   ├── gcs_utils.py
│   │   ├── memory.py
│   │   ├── prompts.py
│   │   └── token_utils.py
│   ├── feature_suggestion.py        # Code for suggesting new query rewrites
│   ├── semantic_relevance_match.py  # Semantic relevance analysis
│   ├── keys/                        # Directory for API keys
│   │   └── litellm.key              # LiteLLM API key (create this file)
│   └── README.md                    # Core components documentation
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```