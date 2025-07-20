# Data

This directory contains data used by the shopping agent, primarily the virtual customer personas that guide agent behavior and decision-making.

## Personas (`personas/`)

The `personas/` directory contains around 1,000 JSON files, each representing a unique virtual customer persona. These personas are used by the shopping agent to inform its browsing and purchasing decisions, simulating a wide range of user behaviors, preferences, and demographics.

The personas were gathered from the paper [UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design](https://arxiv.org/abs/2502.12561).

### Persona File Format

Each JSON file in this directory defines a comprehensive persona with shopping intent, demographic information, and suggested search queries. Here is the complete structure:

```json
{
  "persona": "Detailed persona description with background, demographics, financial situation, shopping habits, and personal style",
  "intent": "Specific shopping task or goal",
  "age": 63,
  "age_group": "55-64",
  "gender": "male",
  "income": [0, 30000],
  "income_group": "0-30000",
  "search_queries": [
    "Suggested search query 1",
    "Suggested search query 2",
    "Suggested search query 3",
    "Suggested search query 4",
    "Suggested search query 5"
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `persona` | string | Detailed character description including background, demographics, financial situation, shopping habits, professional life, and personal style |
| `intent` | string | Specific shopping task or product the persona wants to buy |
| `age` | integer | Exact age of the persona |
| `age_group` | string | Age range classification (e.g., "55-64", "25-34") |
| `gender` | string | Gender identity ("male", "female", etc.) |
| `income` | array | Income range as [min, max] in dollars |
| `income_group` | string | Income bracket classification (e.g., "0-30000", "60000-100000") |
| `search_queries` | array | List of 5 suggested search queries related to the intent |

### Example Persona

```json
{
  "persona": "Persona: Frank\n\nBackground:\nFrank is a retiree who spends his time pursuing his hobbies and taking care of his family. After working blue-collar jobs for most of his life, he's now enjoying his golden years with a modest income.\n\nDemographics:\nAge: 63\nGender: Male\nEducation: High school diploma\nProfession: Retired\nIncome: $25,000\n\nFinancial Situation:\nFrank lives on a fixed income from his pension and Social Security benefits. He is careful with his spending, prioritizing necessities and leaving little room for frivolous purchases.\n\nShopping Habits:\nFrank tends to be a cautious shopper, taking the time to research and compare prices before making a purchase. He values quality and durability over trends, preferring to invest in items that will last. Frank often shops at local thrift stores or discount retailers to stretch his budget.\n\nPersonal Life:\nIn his retirement, Frank enjoys spending time with his grandchildren, going for walks in the park, and working on DIY projects around the house. He takes pride in his ability to fix things and often shares his skills with his family and neighbors.",
  "intent": "buy a Traxxas Slash body.",
  "age": 63,
  "age_group": "55-64",
  "gender": "male",
  "income": [0, 30000],
  "income_group": "0-30000",
  "search_queries": [
    "Traxxas Slash body options",
    "Traxxas Slash body prices", 
    "where to find Traxxas Slash body",
    "Traxxas Slash replacement body",
    "Traxxas Slash body for sale"
  ]
}
```

### Usage in Shopping Agent

The shopping agent uses these personas in several ways:

1. **Single Agent Mode**: 
   - Default persona: "Evelyn" (retired educator) - defined in `src/shopping_agent/config.py`
   - To use a different persona, modify the `DEFAULT_TASK` or update the persona text directly in code.

2. **Batch Testing Mode** (`src/analyze_query.py`):
   - By default, randomly samples the required number of personas from this directory
   - Total agents controlled by the `--n-agents` flag (default: 4)
   - If there are fewer personas than agents, sampling occurs with replacement
