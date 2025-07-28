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
- Return the variations as a JSON list of strings.
""".strip()

CROSSOVER_PROMPT = """
You are an expert at combining shopping queries to create new, potentially better variations.
Given two parent queries, create a new query that combines the best aspects of both while maintaining search relevance.
        
Guidelines:
- The result should be a natural, searchable query
- Combine meaningful elements from both parents
- Keep it concise (2-8 words typically)
- Maintain the original search intent
- Just return the new query, no other text.
""".strip()

MUTATION_PROMPT = """
You are an expert at creating subtle variations of shopping queries while maintaining their core meaning and search intent. You will be given a query and a summarized feedback for this query. Return the revised query, addressing the feedback.
        
Guidelines:
- Keep the same general length (don't make it significantly longer or shorter)
- Maintain the original search intent
- Make small but meaningful changes (synonyms, reordering, slight modifications)
- The result should still be a natural, searchable query
- Just return the revised query, no other text.
- Try to address the feedback in the revised query.
""".strip()