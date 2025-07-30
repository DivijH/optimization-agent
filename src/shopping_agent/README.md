# Shopping Agent

This directory contains the core logic for the AI-powered shopping agent that can autonomously browse e-commerce websites, analyze products, and make purchasing decisions.

## File Overview

| File | Purpose | Entry Point |
|------|---------|-------------|
| `main.py` | CLI entry point for single agent runs | Yes |
| `agent.py` | Core EtsyShoppingAgent implementation | No |
| `agent_actions.py` | Browser action definitions | No |
| `browser_utils.py` | Browser interaction utilities | No |
| `config.py` | Configuration and default values | No |
| `memory.py` | Memory management system | No |
| `gcs_utils.py` | Google Cloud Storage integration | No |
| `prompts.py` | LLM prompts for agent operations | No |
| `token_utils.py` | Token usage tracking utilities | No |

## Entry Point

### `main.py`
The main entry point for running the shopping agent from the command line.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | "silver, vintage-style metal belt buckle" | Shopping task for the agent |
| `--curr-query` | str | None | Current query (defaults to task) |
| `--manual` | flag | False | Wait for user input after each action |
| `--headless` | flag | False | Run browser in headless mode |
| `--max-steps` | int | None | Max steps (unlimited if None) |
| `--debug-path` | Path | "debug_run" | Debug artifacts directory |
| `--width` | int | 1920 | Browser viewport width |
| `--height` | int | 1080 | Browser viewport height |
| `--model` | str | "global-gemini-2.5-flash" | LLM model name |
| `--final-decision-model` | str | None | Model for final decisions |
| `--temperature` | float | 0.7 | LLM sampling temperature (0-2) |
| `--record-video` | flag | False | Record browser session |
| `--user-data-dir` | Path | None | Browser user data directory |
| `--save-local` | flag | True | Save data locally |
| `--save-gcs` | flag | True | Save data to GCS |
| `--gcs-bucket` | str | "training-dev-search-data-jtzn" | GCS bucket name |
| `--gcs-prefix` | str | "smu-agent-optimizer" | GCS prefix |
| `--gcs-project` | str | "etsy-search-ml-dev" | GCS project name |

**Usage Examples:**
```bash
# Basic usage
python -m src.shopping_agent.main --task "vintage camera"

# With custom settings
python -m src.shopping_agent.main \
    --task "camping tent" \
    --model gpt-4o \
    --max-steps 15 \
    --headless
```

## Core Files

### `agent.py`
Contains the core implementation of the `EtsyShoppingAgent` class, the main orchestrator for shopping sessions.

**Key Components:**
- Browser session management and initialization
- Main decision-making loop with step-by-step execution
- Memory management integration
- Token usage tracking and cost monitoring
- Debug information saving and GCS uploads
- Screenshot capture and visual analysis coordination

**Core Methods:**
- `run()`: Main execution loop
- `take_screenshot()`: Captures current browser state
- `save_debug_info()`: Stores step-by-step debugging data
- `upload_to_gcs()`: Handles cloud storage uploads

### `agent_actions.py`
Defines specific browser actions that the agent can perform during shopping sessions.

**Available Actions:**
- `click`: Click on page elements
- `type`: Enter text into input fields
- `scroll`: Navigate through page content
- `screenshot`: Capture visual state
- `open_tab`: Open new browser tabs
- `close_tab`: Close existing tabs
- `go_to_url`: Navigate to specific URLs

### `browser_utils.py`
Provides utility functions for browser interaction and content analysis.

**Key Functions:**
- `find_search_bar()`: Locates search input elements on pages
- `extract_product_name_from_url()`: Parses product names from URLs
- `scroll_and_collect()`: Captures page content via scrolling
- `analyze_product_page()`: Performs detailed product analysis using LLM
- `choose_product_from_search()`: Selects products from search results
- `make_final_purchase_decision()`: Generates final purchase recommendations

### `config.py`
Configuration file containing default values, model pricing, and system constants.

**Model Pricing** (per million tokens):
- `global-gemini-2.5-flash`: Variable pricing with vendor discounts
- `gpt-4o-mini`: $0.15 input, $0.60 output
- `gpt-4o`: $2.50 input, $10.00 output
- `o3-mini`: $1.10 input, $4.40 output

**Token Analysis Breakdown:**
- `IMAGE_TOKEN_PERCENTAGE`: 0.484 (48.4% of analysis tokens are image-based)
- `TEXT_TOKEN_PERCENTAGE`: 0.516 (51.6% of analysis tokens are text-based)

**Default Configuration:**
- Contains the default shopping task and Evelyn persona specifications
- Defines browser dimensions, timeouts, and behavior settings

### `memory.py`
Memory management system with classes for tracking analyzed products and search history.

**`ProductMemory` Class:** Stores individual product information
- `product_name`: Product title and identifier
- `url`: Product page URL
- `price`: Product pricing information
- `pros`: List of product advantages
- `cons`: List of product disadvantages
- `summary`: Product analysis summary
- `semantic_score`: Relevance scoring for the search query

**`MemoryModule` Class:** Manages agent's overall memory
- Tracks search queries and their results
- Stores analyzed products to prevent duplicate analysis
- Provides memory summaries and statistics
- Handles save/load operations to/from JSON files
- Calculates semantic relevance statistics

### `gcs_utils.py`
Google Cloud Storage integration for persistent data storage and sharing.

**Upload Capabilities:**
- Debug information and execution logs
- Screenshots and visual analysis data
- Memory data and product analysis results
- Final purchase decisions and recommendations
- Token usage and cost tracking data

**Authentication:** Uses Application Default Credentials (ADC) for secure access

### `prompts.py`
Contains all LLM prompts used by the agent for various decision-making processes.

**Prompt Categories:**
- Product analysis prompts for detailed evaluation
- Decision-making prompts for purchase recommendations
- Search strategy prompts for query optimization
- Visual analysis prompts for screenshot interpretation

### `token_utils.py`
Token usage tracking utilities for cost monitoring and optimization.

**Key Features:**
- Calculates token costs per model with current pricing
- Tracks image vs text token distribution
- Provides detailed usage summaries per session
- Calculates total session costs including vendor discounts
- Monitors token efficiency and usage patterns

## Output Structure

The agent creates a comprehensive debug directory structure:

```
debug_run/
├── agent.log                           # Detailed execution log
├── screenshot_*.png                    # Page screenshots at each step
├── debug_step_*.json                   # Step-by-step debug information
├── _memory.json                        # Product analysis results
├── _semantic_scores.json               # Relevance scoring data
├── _final_purchase_decision.json       # Purchase recommendations
├── _token_usage.json                   # API usage and costs
└── _session.mp4                        # Screen recording (if enabled)
```

## Integration with Other Components

The shopping agent integrates seamlessly with:
- **Genetic Query Optimizer**: Provides fitness evaluation for query evolution
- **Batch Testing Framework**: Runs multiple instances with different personas
- **Results Processing**: Contributes data for comparative analysis
- **GCS Storage**: Enables cloud-based result sharing and archival
