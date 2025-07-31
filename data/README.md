# Data

This directory contains data used by the optimization agent.

## Final Queries CSV (`final_queries.csv`)

The `final_queries.csv` file contains 1000 sampled shopping queries used by real users on Etsy for testing and optimization. These queries are reviewed manually to remove any personally identifiable information and unethical content. The queries are distributed across different categories and languages:

- **250 head queries**: High-frequency, popular search terms 
- **250 torso queries**: Medium-frequency search terms
- **250 tail queries**: Low-frequency, niche search terms
- **100 fandom queries**: Queries related to specific fandoms, franchises, or cultural interests
- **150 multi-lingual queries**: Non-English queries covering various languages

### CSV Structure

| Column | Description |
|--------|-------------|
| `Query` | The search query text |
| `Frequency` | Query frequency category (Head, Torso, Tail) |
| `Language` | Query language (English or specific language for multi-lingual queries) |
| `Fandom` | Associated fandom/franchise (for fandom queries) or N/A |

This dataset is used by `batch_genetic_optimizer.py` to systematically test and improve search query effectiveness across different categories and languages.