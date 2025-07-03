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

### A/B Test
```bash
cd src
python ab_testing.py --n-agents 4
```

### Single Agent
```bash
cd src
python shopping_agent.py --task "buy a large, inflatable spider decoration for halloween"
```

## Documentation

- **[Core Components (`src/`)](./src/README.md)**: Detailed documentation on the shopping agent, A/B testing framework, and their configurations.
- **[Data (`data/`)](./data/README.md)**: Information about the virtual customer personas used by the agent.

## Directory Structure

```
.
├── data/
│   ├── personas/             # Virtual customer personas for agents
│   │   ├── virtual customer 0.json
│   │   ├── virtual customer 1.json
│   │   └── ... (200+ persona files)
│   └── README.md             # Information about the data
├── src/
│   ├── ab_testing.py         # Script for A/B testing different models
│   ├── shopping_agent.py     # Core logic for the shopping agent
│   ├── feature_suggestion.py # Code for suggesting new features for products
│   ├── memory.py             # Agent's memory implementation
│   ├── prompts.py            # Prompts used by LLM
│   ├── keys/                 # Directory for API keys
│   │   └── litellm.key       # LiteLLM API key (create this file)
│   └── README.md             # Core components documentation
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Key Features

### Shopping Agent (`shopping_agent.py`)
- **Task Decomposition**: Automatically breaks high-level shopping goals into search queries
- **Visual Product Selection**: Uses screenshots and multimodal LLM to choose products to analyze
- **In-Depth Product Analysis**: Analyzes product pages for pros, cons, and relevance scores
- **Memory System**: Remembers analyzed products to avoid duplicates and inform decisions
- **Final Purchase Recommendations**: Makes data-driven purchase decisions based on analysis
- **Debug Artifacts**: Saves screenshots, logs, and analysis results for debugging
- **Screen Recording**: Optional video recording of browser sessions (requires ffmpeg)

### A/B Testing Framework (`ab_testing.py`)
- **Model Comparison**: Compare performance between different LLMs
- **Concurrent Execution**: Run multiple agents in parallel with configurable concurrency
- **Persona Sampling**: Randomly assign unique personas from the dataset
- **Isolated Logging**: Each agent gets its own debug directory and logs
- **Status Monitoring**: Real-time status updates for all running agents

### Memory and Analysis
- **Product Memory**: Stores analyzed products with prices, pros/cons, and relevance scores
- **Search History**: Tracks all search queries performed
- **Semantic Scoring**: Rates products as "HIGHLY RELEVANT", "SOMEWHAT RELEVANT", or "NOT RELEVANT"
- **Cost Tracking**: Monitors LLM token usage and associated costs

## Configuration

### Models and Pricing
The system supports multiple LLM models with automatic cost tracking:
- **gpt-4o-mini**: $0.15/$0.60 per million input/output tokens
- **gpt-4o**: $2.50/$10.00 per million input/output tokens  
- **o3-mini**: $1.10/$4.40 per million input/output tokens

### Browser Configuration
- Configurable viewport size (default: 1920x1080)
- Headless or visible browser modes
- Optional user data directory for session persistence
- Smooth scrolling for better visual continuity

### Debug and Monitoring
- Comprehensive debug artifacts including screenshots and JSON logs
- Real-time token usage and cost tracking
- Memory persistence across sessions
- Video recording support (requires ffmpeg installation)

## Usage Examples

### Basic Shopping Task
```bash
cd src
python shopping_agent.py --task "find a handmade ceramic mug" --max-steps 15
```

### Custom Persona and Model
```bash
cd src
python shopping_agent.py \
  --task "buy vintage jewelry" \
  --persona "Sarah, a 28-year-old graphic designer who loves minimalist design" \
  --model "gpt-4o" \
  --temperature 0.3
```

### A/B Testing Different Models
```bash
cd src
python ab_testing.py \
  --n-agents 6 \
  --control-model "gpt-4o-mini" \
  --target-model "gpt-4o" \
  --concurrency 3 \
  --max-steps 12
```

### Debug Mode with Video Recording
```bash
cd src
python shopping_agent.py \
  --task "find art supplies" \
  --manual \
  --record-video \
  --debug-path "my_debug_session"
```

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- FFmpeg (optional, for video recording)
- LiteLLM API key configured in `src/keys/litellm.key`

## Output Files

When running the shopping agent, the following files are generated in the debug directory:

- `_memory.json` - All analyzed products and search history
- `_final_purchase_decision.json` - LLM's final purchase recommendations  
- `_token_usage.json` - Token usage and cost breakdown
- `_semantic_scores.json` - Distribution of product relevance scores
- `screenshot_step_N_*.png` - Screenshots from each step
- `debug_step_N.json` - Detailed logs for each step
- `_session.mp4` - Video recording (if enabled)
- `agent.log` - Complete agent execution log