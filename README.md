# Optimization Agent

📋 **Overall Project Documentation**: [Google Doc](https://docs.google.com/document/d/1ORWmq6GQMyoQZR7_b2S9Hs7l2A-e0Ce9f6EKy-pQ69Q/edit?tab=t.0#heading=h.4wbqtehjjc4)

## Shopping Agent

An AI-powered Etsy shopping agent that autonomously browses and analyzes products based on a given task and persona. The agent uses browser automation to search for products, analyze product pages, and make informed shopping decisions. The personas, gathered from the paper [UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design](https://arxiv.org/abs/2502.12561), are located at `data/personas`.

## Quick Start

```bash
cd src
python shopping_agent.py --task "buy a large, inflatable spider decoration for halloween"
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   ```bash
   echo "your-api-key-here" > src/keys/litellm.key
   ```

## Usage

### Basic Usage

```bash
python shopping_agent.py --task "your shopping task here"
```

### All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | `"buy a large, inflatable spider decoration for halloween"` | The shopping task for the agent |
| `--persona` | Michael (42yo marketing manager) | The persona for the agent |
| `--config-file` | *None* | Path to a JSON file containing `task` and `persona`. Values in the file override the flags above |
| `--manual` | False | Wait for user to press Enter after each agent action |
| `--headless` | False | Run the browser in headless mode |
| `--max-steps` | 10 | Maximum number of steps the agent will take |
| `--debug-path` | `"debug_run"` | Path to save debug artifacts (screenshots, logs) |
| `--width` | 1920 | Browser viewport width |
| `--height` | 1080 | Browser viewport height |
| `--n-products` | 3 | Number of products to analyze (-1 for all) |
| `--model` | `"gpt-4o-mini"` | LLM model name (e.g., gpt-4o, gpt-4o-mini) |
| `--temperature` | 0.7 | LLM sampling temperature (0-2) |
| `--record-video` | False | Record browser session video (requires ffmpeg) |

### Examples

**Simple task**:
```bash
python shopping_agent.py --task "find a handmade ceramic mug"
```

**With custom persona**:
```bash
python shopping_agent.py --task "buy a birthday gift" --persona "Sarah, a 28-year-old graphic designer who loves minimalist design"
```

**Debug mode with manual control**:
```bash
python shopping_agent.py --task "find vintage jewelry" --manual --debug-path "my_debug_run"
```

**Headless mode with video recording**:
```bash
python shopping_agent.py --task "buy art supplies" --headless --record-video --max-steps 20
```

**Custom model and temperature**:
```bash
python shopping_agent.py --task "find a wedding gift" --model "gpt-4o" --temperature 0.3
```

**Using a JSON config file**:
```bash
python shopping_agent.py --config-file my_shopping_task.json
```

## How It Works

1. **Task Breakdown**: Breaks your shopping task into specific search queries
2. **Product Discovery**: Searches Etsy and identifies relevant products
3. **Visual Selection**: Uses screenshots + LLM to choose the most promising listings
4. **Product Analysis**: Summarises pros/cons for each product page
5. **Memory Management**: Remembers analysed products to avoid duplicates
6. **Decision Making**: Generates the final purchase recommendation based on memory

## Output

- **Console**: Real-time progress and decision logs
- **Memory**: `_memory.json` with all analysed products & search history
- **Final Decision**: `_final_purchase_decision.json` with the LLM's recommendation
- **Debug Files**: Step-by-step JSON logs and screenshots (if `--debug-path` is set)
- **Video**: `session.mp4` recording (if `--record-video` is enabled)

## Key Features

- **Task Decomposition**: Automatically breaks high-level shopping goals into smaller sub-tasks using an LLM.
- **Visual Product Selection**: Captures screenshots of search results and lets a multimodal LLM decide which listing to open next.
- **In-Depth Product Analysis**: Scrolls through product pages, extracts text & images, and asks the LLM to summarise pros, cons, and a short description.
- **Long-Term Memory**: Stores analysed products and search queries to avoid duplicates and provide context for future decisions.
- **Final Recommendation**: After exploring, the agent uses its memory to output a JSON purchase recommendation.
- **Rich Debug Artefacts**: Saves JSON logs and annotated/plain screenshots for every step to the folder specified by `--debug-path`.
- **Screen Recording (optional)**: Record the entire browser session to MP4 with `--record-video` (requires ffmpeg).
- **Interactive Mode**: Add `--manual` to pause after every action so you can inspect what the agent is doing.

### Prerequisites for Screen Recording

The `--record-video` flag relies on **[ffmpeg](https://ffmpeg.org/)** being installed and available in your `PATH`. On macOS you can install it with Homebrew:

```bash
brew install ffmpeg
```

⚠️  `--record-video` and `--headless` cannot be used together – the script will exit with an error if both flags are supplied.

## Using a JSON Config File

Passing large persona descriptions via the command line can get messy. Instead, place them in a small JSON file:

```json
{
  "intent": "buy a pair of Jobst zipper compression socks for women.",
  "persona": "Persona: Sophia\n\nBackground:\nSophia is a dedicated community college professor with a deep passion for education and empowering underserved students. She has spent the past two decades sharing her expertise and inspiring young minds to reach their full potential.\n\nDemographics:\nAge: 54\nGender: Female\nEducation: Master's degree in Education\nProfession: Community College Professor\nIncome: $65,000\n\nFinancial Situation:\nSophia's income as a community college professor provides her with a comfortable, yet modest, living. She is financially responsible and manages her budget carefully, prioritizing her personal and professional goals.\n\nShopping Habits:\nSophia is a practical shopper who focuses on finding high-quality, durable items that will serve her needs for the long term. She enjoys browsing local thrift stores and online marketplaces for unique finds, but she is also willing to invest in essential items that will last. Sophia values sustainability and often looks for eco-friendly or ethically sourced products.\n\nProfessional Life:\nAs a community college professor, Sophia takes great pride in her work and the impact she has on her students' lives. She is known for her engaging teaching style, her deep subject matter expertise, and her genuine care for the well-being and success of her students. Sophia is actively involved in curriculum development and mentoring programs, constantly seeking ways to improve the educational experience.\n\nPersonal Style:\nSophia's personal style reflects her practical and comfortable approach to life. She often wears classic pieces, such as button-down shirts, cardigans, and well-fitted trousers, that allow her to move freely and feel confident in the classroom. She also enjoys adding personal touches, like colorful scarves or statement jewelry, to express her own sense of style."
}
```

Run the agent with:

```bash
python shopping_agent.py --config-file my_shopping_task.json
```

The values in the JSON file take precedence over any of the corresponding CLI flags.
