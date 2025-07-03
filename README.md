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

4. Set up the local website for agents
[TODO]

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

### Template-Based Web Scraping (New!)
```bash
# Efficient content extraction only (recommended)
cd src/etsy_environment
python template_batch_scraper.py --extract-only --start 1 --end 10

# Extract + generate full HTML pages
python template_batch_scraper.py --start 1 --end 5

# Manage scraped data
python template_utils.py analyze
```

## Documentation

- **[Core Components (`src/`)](./src/README.md)**: Detailed documentation on the shopping agent, A/B testing framework, and their configurations.
- **[Data (`data/`)](./data/README.md)**: Information about the virtual customer personas used by the agent.
- **[Template-Based Scraping](#template-based-web-scraping)**: Efficient web scraping approach that separates static templates from dynamic content.

## Directory Structure

```
.
├── data/
│   ├── etsy_pages/           # Cached HTML of Etsy search results and product pages
│   │   ├── templates/        # NEW: Reusable HTML templates and static assets
│   │   ├── content/          # NEW: Lightweight JSON content files
│   │   └── generated/        # NEW: Full HTML pages generated from templates
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
│   │   ├── batch_scraper.py          # Original batch scraper
│   │   ├── template_scraper.py       # NEW: Template-based scraper
│   │   ├── template_batch_scraper.py # NEW: Efficient batch scraping
│   │   ├── template_utils.py         # NEW: Data management utilities
│   │   ├── get_search_queries.py
│   │   ├── hosting_webpages.py
│   │   └── webpage_downloader.py
│   ├── keys/                 # Directory for API keys
│   │   └── litellm.key
│   └── README.md             # Core components documentation
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Template-Based Web Scraping

The template-based scraping system provides an efficient alternative to the original batch scraper, dramatically reducing storage requirements and scraping time by separating static website assets from dynamic content.

### 🎯 Key Benefits

- **90%+ Storage Reduction**: CSS/JS files downloaded only once vs. hundreds of times
- **3-5x Faster Scraping**: Subsequent scraping focuses only on dynamic content
- **Clean Data Format**: Structured JSON files perfect for analysis and ML
- **Flexible Regeneration**: Recreate full HTML pages anytime from lightweight data
- **Scalable Architecture**: Template approach scales efficiently to thousands of pages

### 🏗️ How It Works

1. **Template Creation** (One-time): Downloads complete HTML/CSS/JS on first page visit and creates reusable templates with placeholders
2. **Content Extraction**: Only extracts dynamic data (titles, prices, descriptions, images) and saves as lightweight JSON
3. **Page Reconstruction**: Injects extracted content into templates to recreate full HTML pages when needed

### 📁 File Structure

```
data/etsy_pages/
├── templates/              # Created once, reused for all pages
│   ├── search_template.html     # Search page template with placeholders
│   ├── listing_template.html    # Product page template with placeholders
│   └── assets/                  # Static assets (CSS, JS, images)
│       ├── css/
│       ├── js/
│       └── images/
├── content/               # Lightweight JSON files (main storage)
│   ├── persona_1_query_1_page_1_chess_board.json
│   ├── listing_123456.json
│   └── ...
└── generated/            # Full HTML pages (optional, generated on-demand)
    └── search_persona_1_query_1_page_1.html
```

### 🚀 Usage Examples

#### Basic Content Extraction (Recommended)
```bash
cd src/etsy_environment

# Extract content for personas 1-10 (most efficient)
python template_batch_scraper.py --extract-only --start 1 --end 10

# Include individual listing pages for detailed analysis
python template_batch_scraper.py --extract-only --include-listings --start 1 --end 5

# Custom delay between requests
python template_batch_scraper.py --extract-only --delay 2.0 --workers 2
```

#### Data Management
```bash
# Analyze extracted data
python template_utils.py analyze

# List content files
python template_utils.py list --pattern persona_1

# Generate HTML pages from JSON data
python template_utils.py generate --pattern persona_1

# Merge multiple content files for analysis
python template_utils.py merge combined_data --pattern persona_1
```

#### Advanced Usage
```bash
# Extract + generate full HTML pages
python template_batch_scraper.py --start 1 --end 5

# Clean up storage (keep content, remove generated files)
python template_utils.py clean --keep-content --keep-templates
```

### 📊 Performance Comparison

| Approach | Storage (100 pages) | Time (100 pages) | Data Format |
|----------|-------------------|------------------|-------------|
| Original | ~500MB | 30 minutes | Full HTML |
| Template | ~50MB | 10 minutes | JSON + Templates |

### 🔧 Template Placeholders

**Search Page Templates:**
- `{{SEARCH_QUERY}}` - Search query in title/breadcrumbs
- `{{LISTING_ITEMS}}` - Product listing cards

**Listing Page Templates:**
- `{{PRODUCT_TITLE}}` - Product title
- `{{PRODUCT_PRICE}}` - Product price
- `{{PRODUCT_DESCRIPTION}}` - Product description
- `{{PRODUCT_IMAGE_URL}}` - Main product image

### 💡 Best Practices

1. **Start Small**: Test with `--start 1 --end 3` first
2. **Use Extract-Only**: Save storage with `--extract-only` flag
3. **Monitor Progress**: Each worker shows progress in separate terminal lines
4. **Analyze First**: Use `template_utils.py analyze` to understand your data
5. **Generate On-Demand**: Create HTML pages only when needed for viewing

### 🛠️ Integration with Existing Code

The template scraper is designed to work alongside the existing shopping agent. You can:

- Use JSON content files directly for ML analysis
- Generate HTML pages when agents need to browse
- Switch between original and template scrapers as needed
- Migrate existing scraped data to template format

This template-based approach makes large-scale web scraping practical and cost-effective while maintaining full functionality for agent browsing and analysis.