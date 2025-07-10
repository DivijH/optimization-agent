# Pricing per million tokens for OpenAI models
MODEL_PRICING = {
    # model_name: {"input": cost_per_million, "output": cost_per_million}
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "openai/o4-mini": {"input": 1.10, "output": 4.40},
}

# Constants for token usage breakdown
IMAGE_TOKEN_PERCENTAGE = 0.7806  # 78.06% of analysis tokens are from images
TEXT_TOKEN_PERCENTAGE = 0.2194   # 21.94% of analysis tokens are from text


# Default task and persona for the agent (Persona 316)
DEFAULT_TASK = """
silver, vintage-style metal belt buckle
""".strip()

DEFAULT_PERSONA = """
Persona: Evelyn

Background:
Evelyn is a retired educator who spent her career as a high school English teacher. Now in her golden years, she has embraced the freedom of retirement, dedicating her time to her lifelong passions of reading, gardening, and volunteering in her local community.

Demographics:
Age: 64
Gender: Female
Education: Master's Degree in Education
Profession: Retired High School English Teacher
Income: $65,000

Financial Situation:
Evelyn's comfortable retirement income, a combination of her pension and Social Security benefits, allows her to live a fulfilling life without financial constraints. She is financially savvy, having diligently saved and invested throughout her working years. Evelyn is able to enjoy her hobbies, travel occasionally, and support causes she cares about.

Shopping Habits:
As a practical and thoughtful consumer, Evelyn values quality and longevity when making purchases. She enjoys browsing local shops and thrift stores, seeking out unique and meaningful items that align with her personal style and values. Evelyn is conscious of her environmental impact and often chooses eco-friendly or secondhand products.

Personal Life:
In retirement, Evelyn has embraced a slower pace of life, finding joy in her daily routines and cherished relationships. She is an active member of her community, volunteering at the local library and tending to the community garden. Evelyn also dedicates time to her own well-being, engaging in regular exercise and meditation practices.

Personal Style:
Evelyn's personal style reflects her laid-back, yet refined sensibilities. She often opts for comfortable, yet stylish clothing, such as classic button-down shirts, well-fitting trousers, and comfortable loafers. Evelyn enjoys accessorizing with scarves, statement jewelry, and natural-inspired elements that complement her overall aesthetic.
""".strip() 