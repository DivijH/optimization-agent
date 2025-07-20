GENERATE_INITIAL_POPULATION_PROMPT = """
You are an expert at creating variations of shopping queries for e-commerce platforms like Etsy.
Given an original query, generate semantically similar but diverse variations that could potentially find better or different relevant products.

Guidelines:
- Keep variations relevant to the original intent
- Use synonyms, related terms, and different phrasings
- Consider different product attributes (size, color, style, material)
- Include both broader and more specific versions
- Each variation should be a single line, natural search query
- Variations should be 2-8 words typically
""".strip()