# Pricing per million tokens for OpenAI models
MODEL_PRICING = {
    # model_name: {"input": cost_per_million, "output": cost_per_million}
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "openai/o4-mini": {"input": 1.10, "output": 4.40},
    "global-gemini-2.5-flash-lite-preview-06-17": {"input": 0.10, "output": 0.40},  ### The cheapest model (no thinking)
    "global-gemini-2.5-flash": {"input": 0.30, "output": 2.50},  ### The cheapest model (with thinking)
    # "search_vertex_ai/gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60}
}

# Constants for token usage breakdown
IMAGE_TOKEN_PERCENTAGE = 0.5  # 50% of analysis tokens are from images
# TEXT_TOKEN_PERCENTAGE = 0.5   # 50% of analysis tokens are from text
VENDOR_DISCOUNT_GEMINI = 0.35  # 35% discount on Gemini models

# Default task for the agent
DEFAULT_TASK = """
silver, vintage-style metal belt buckle
""".strip()