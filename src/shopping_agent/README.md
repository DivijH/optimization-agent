# Shopping Agent

This directory contains the core logic for the AI-powered shopping agent that can autonomously browse e-commerce websites, analyze products, and make purchasing decisions.

## Core Files

### `main.py`
The main entry point for running the shopping agent from the command line.

**Default Values:**
- `--task`: "silver, vintage-style metal belt buckle"
- `--persona`: Evelyn (retired educator persona - see config.py for full text)
- `--manual`: False (automatic mode)
- `--headless`: False (browser visible)
- `--max-steps`: None (continues until no more products)
- `--debug-path`: "debug_run"
- `--width`: 1920
- `--height`: 1080
- `--model`: "gpt-4o-mini"
- `--final-decision-model`: None (uses main model)
- `--temperature`: 0.7
- `--record-video`: False
- `--save-local`: True
- `--save-gcs`: True
- `--gcs-bucket`: "training-dev-search-data-jtzn"
- `--gcs-prefix`: "smu-agent-optimizer"

### `agent.py`
Contains the core implementation of the `EtsyShoppingAgent` class, including:
- Browser session management
- Main decision-making loop
- Memory management
- Token usage tracking
- Debug information saving

**Key Features:**
- Automatic Etsy navigation and search
- Product discovery and analysis
- Screenshot capture for visual analysis
- Final purchase decision making

### `agent_actions.py`
Defines specific browser actions the agent can perform:
- Clicking elements
- Typing text
- Scrolling pages
- Taking screenshots
- Opening/closing tabs

### `browser_utils.py`
Provides utility functions for browser interaction:
- `find_search_bar()`: Locates search input elements
- `extract_product_name_from_url()`: Parses product names from URLs
- `scroll_and_collect()`: Captures page content via scrolling
- `analyze_product_page()`: Performs product analysis using LLM
- `choose_product_from_search()`: Selects products from search results
- `make_final_purchase_decision()`: Generates final recommendations

### `config.py`
Configuration and default values:

**Model Pricing** (per million tokens):
- `gpt-4o-mini`: $0.15 input, $0.60 output
- `gpt-4o`: $2.50 input, $10.00 output
- `o3-mini`: $1.10 input, $4.40 output
- `openai/o4-mini`: $1.10 input, $4.40 output

**Token Analysis Breakdown:**
- `IMAGE_TOKEN_PERCENTAGE`: 0.484 (48.4% of analysis tokens)
- `TEXT_TOKEN_PERCENTAGE`: 0.516 (51.6% of analysis tokens)

**Default Task and Persona:**
- Contains the default shopping task and Evelyn persona used when not specified

### `memory.py`
Memory management system with two main classes:

**`ProductMemory`**: Stores information about analyzed products
- product_name
- url
- price
- pros/cons lists
- summary
- semantic_score

**`MemoryModule`**: Manages agent's memory
- Tracks search queries
- Stores analyzed products
- Prevents duplicate analysis
- Provides memory summaries
- Saves/loads from JSON

### `gcs_utils.py`
Google Cloud Storage integration for saving:
- Debug information
- Screenshots
- Memory data
- Final decisions

**Authentication**: Uses Application Default Credentials (ADC)

### `prompts.py`
Contains all LLM prompts used by the agent:
- Product analysis prompts
- Decision-making prompts
- Search strategy prompts

### `token_utils.py`
Token usage tracking utilities:
- Calculates token costs per model
- Tracks image vs text tokens
- Provides usage summaries
- Calculates total session costs

## Usage Example

Basic usage:
```bash
python -m src.shopping_agent.main --task "halloween decorations"
```

With custom settings:
```bash
python -m src.shopping_agent.main \
    --task "vintage leather jacket" \
    --model gpt-4o \
    --max-steps 20 \
    --headless \
    --record-video
```

Using a config file:
```bash
python -m src.shopping_agent.main --config-file persona_config.json
```

## Output Structure

The agent creates a debug directory with:
- `agent.log`: Detailed execution log
- `screenshot_*.png`: Page screenshots
- `debug_step_*.json`: Step-by-step debug info
- `_memory.json`: Product analysis results
- `_semantic_scores.json`: Relevance scoring
- `_final_purchase_decision.json`: Purchase recommendations
- `_token_usage.json`: API usage and costs
- `_session.mp4`: Screen recording (if enabled)
