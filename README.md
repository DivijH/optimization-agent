# Optimization Agent

[![GitHub repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/DivijH/optimization-agent)

This repository contains an AI-powered shopping agent that can autonomously browse e-commerce websites like Etsy, analyze products, and make purchasing decisions based on a given task and persona. It also includes an A/B testing framework to compare the performance of different language models.

For more information, please see the overall project documentation: [Google Doc](https://docs.google.com/document/d/1ORWmq6GQMyoQZR7_b2S9Hs7l2A-e0Ce9f6EKy-pQ69Q/edit?tab=t.0#heading=h.4wbqtehjjc4)

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

### Single Agent
```bash
cd src
python shopping_agent.py --task "buy a large, inflatable spider decoration for halloween"
```

### A/B Test
```bash
cd src
python ab_testing.py --n-agents 4
```

## Documentation

- **[Core Components (`src/`)](./src/README.md)**: Detailed documentation on the shopping agent, A/B testing framework, and their configurations.
- **[Data (`data/`)](./data/README.md)**: Information about the virtual customer personas used by the agent.

## Directory Structure

```
.
├── data/
│   ├── etsy_pages/           # Cached HTML of Etsy search results and product pages
│   ├── personas/             # Virtual customer personas for agents
│   │   ├── virtual customer 0.json
│   │   ├── virtual customer 1.json
│   │   └── ...
│   └── README.md             # Information about the data
├── src/
│   ├── ab_testing.py         # Script for A/B testing
│   ├── shopping_agent.py     # Core logic for the shopping agent
│   ├── feature_suggestion.py # Code for suggesting new features for products
│   ├── memory.py             # Agent's memory implementation
│   ├── prompts.py            # Prompts used by LLM
│   ├── etsy_environment/     # Code for scraping and hosting Etsy environment
│   │   ├── batch_scraper.py
│   │   ├── get_search_queries.py
│   │   ├── hosting_webpages.py
│   │   └── webpage_downloader.py
│   ├── keys/                 # Directory for API keys
│   │   └── litellm.key
│   └── README.md             # Core components documentation
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```