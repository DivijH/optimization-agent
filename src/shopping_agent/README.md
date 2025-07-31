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
| `gcs_utils.py` | Google Cloud Storage integration | No |
| `memory.py` | Memory management system | No |
| `page_parser.py` | Python parser for Etsy Listing page | No |
| `prompts.py` | LLM prompts for agent operations | No |
| `token_utils.py` | Token usage tracking utilities | No |

## Entry Point

### `main.py`
The main entry point for running the shopping agent from the command line.

**Command Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | "silver, vintage-style metal belt buckle" | Original user query that is being optimized |
| `--curr-query` | str | None | Query candidate being tested (defaults to task) |
| `--manual` | flag | False | Wait for user input after each action (for debugging) |
| `--headless` | flag | False | Run browser in headless mode |
| `--max-steps` | int | None | Max steps (unlimited if None) |
| `--debug-path` | Path | "debug_run" | Debug artifacts directory |
| `--width` | int | 1920 | Browser viewport width |
| `--height` | int | 1080 | Browser viewport height |
| `--model` | str | "global-gemini-2.5-flash" | LLM model name |
| `--final-decision-model` | str | None | Model for final decisions (defaults to model) |
| `--temperature` | float | 0.7 | LLM sampling temperature (0-2) |
| `--record-video` | flag | False | Record browser session |
| `--user-data-dir` | Path | None | Browser user data directory |
| `--save-local/--no-save-local` | flag | True | Save data locally |
| `--save-gcs/--no-save-gcs` | flag | True | Save data to GCS |
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

**Core Methods:**
- `run()`: Main execution loop
- `handle_search_page()`: Extracts all listings from the search page, and analyze them
- `handle_listing_page()`: Analyze the given listing with respect to the original query given by the user
- `shutdown()`: Gracefully terminates the agents
- `calculate_scores()`: Calculates the semantic relevance for all listings

### `agent_actions.py`
Defines specific browser actions that the agent can perform during shopping sessions.

**Available Actions:**
- `type`: Enter text into input fields
- `screenshot`: Capture visual state
- `go_to_url`: Navigate to specific URLs

### `browser_utils.py`
Provides utility functions for browser interaction and content analysis.

**Key Functions:**
- `download_and_resize_image()`: Downloads the product image and downsizes it for lower token usage
- `find_search_bar()`: Locates search input elements on pages
- `extract_product_name_from_url()`: Parses product names from URLs
- `analyze_product_page()`: Performs detailed product analysis using LLM
- `choose_product_from_search()`: Selects products from search results
- `extract_all_listings_from_search()`: Parses the webpage and extracts all listings for processing
- `make_final_purchase_decision()`: Generates final purchase recommendations

### `config.py`
Configuration file containing default values, model pricing, and system constants.

### `gcs_utils.py`
Google Cloud Storage integration for persistent data storage and sharing.

**Upload Capabilities:**
- Debug information and execution logs
- Screenshots and visual analysis data
- Memory data and product analysis results
- Final purchase decisions and recommendations
- Token usage and cost tracking data

**Authentication:** Uses Application Default Credentials (ADC) for secure access

### `memory.py`
Memory management system with classes for tracking analyzed products and search history.

**`ProductMemory` Class:** Stores individual product information
- `product_name`: Product title and identifier
- `url`: Product page URL
- `price`: Product pricing information
- `summary`: Product analysis summary containins pros and cons
- `semantic_score`: Relevance scoring for the search query

**`MemoryModule` Class:** Manages agent's overall memory
- Tracks search queries and their results
- Stores analyzed products to prevent duplicate analysis
- Provides memory summaries and statistics
- Handles save/load operations to/from JSON files
- Calculates semantic relevance statistics

### `page_parser.py`
Extracts the following from the listing page using Beautiful Soup.
- Product name
- Product image
- Product price
- Seller name
- Reviews and ratings
- Shipping information

### `prompts.py`
Contains all LLM prompts used by the agent for various decision-making processes.

- Product analysis prompt for semantic score of a listing
- Final purchase decision prompt for selecting products that are relevant to the searched query

### `token_utils.py`
Token usage tracking utilities for cost monitoring and optimization.

**Key Features:**
- Calculates token costs per model with current pricing
- Provides detailed usage summaries per session
- Calculates total session costs including vendor discounts (if applicable)
- Monitors token efficiency and usage patterns

## Output Structure

The agent creates a comprehensive debug directory structure:

```
debug_run/
├── agent.log                           # Detailed execution log
├── product_image_*.png                 # Product image at each step
├── debug_step_*.json                   # Step-by-step debug information
├── _extracted_listings.json            # All listings that are present on the search page
├── _final_purchase_decision.json       # Purchase recommendations
├── _memory.json                        # Product analysis results
├── _semantic_scores.json               # Relevance scoring data
├── _token_usage.json                   # API usage and costs
└── _session.mp4                        # Screen recording (if enabled)
```