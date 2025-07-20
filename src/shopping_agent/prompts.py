"""
Prompts for the Etsy Shopping Agent.
"""

# Prompt for analyzing a product page
PRODUCT_ANALYSIS_PROMPT = """
You are a product analyst for Etsy, an online shopping platform. Based on the provided images from the product page and the searched query, give me a detailed analysis of the product. Analyze the product using analytical thinking and common sense to determine its semantic relevance to the searched query.

Please provide your analysis in a JSON format with the following keys:
- "pros": A list of reasons why this product is a good fit for the searched query. As many pros as you can find.
- "cons": A list of reasons why this product might not be a good fit for the searched query. As many cons as you can find.
- "summary": A brief summary of your overall opinion.
- "price": The price of the product as a float. If the price is not available, return null.
- "semantic_score": Select either "HIGHLY RELEVANT", "SOMEWHAT RELEVANT", or "NOT RELEVANT" based on how well the product matches the searched query.

**EXAMPLE INPUT 1:**

Searched Query: healthy energy drink

Current Date: May 21

**OUTPUT:**
{{
    "pros": ["The product is affordable at $1.00.", "The seller has a high rating with 1,470 reviews.", "Can be delivered on May 22, which is 1 day delivery."],
    "cons": ["The product is a healthy snack not a drink."],
    "summary": "While the product is affordable and has a high rating, it doesn't fully meet the specific search for a healthy energy drink. The product is a healthy snack not a drink, which is not what was searched for.",
    "price": 1.0,
    "semantic_score": "NOT RELEVANT"
}}

**EXAMPLE INPUT 2:**

Searched Query: House of Staunton Chess Set

Current Date: June 13

**OUTPUT:**
{{
    "pros": ["Made of high-quality wood.", "Can be personalized with a name.", "Delivery time is June 15, 2 days from now."],
    "cons": ["There are just 3 reviews for this product.", "The seller has a no return policy.", "The price is a bit high.", "The board is not from House of Staunton."],
    "summary": "The product is a high-quality wooden chess set that can be personalized, with reasonable delivery time. However, it has limited reviews, no return policy, a bit pricy, and most importantly, it is not from House of Staunton brand as specifically searched for. It is a chess set, but not exactly what was searched for.",
    "price": 123.99,
    "semantic_score": "SOMEWHAT RELEVANT"
}}

**EXAMPLE INPUT 3:**

Searched Query: renaissance-style necklace

Current Date: September 10

**OUTPUT:**
{{
    "pros": ["The product has a high rating with 500 reviews.", "The necklace matches the renaissance style aesthetic."],
    "cons": ["The shipping date is September 20, which is more than a week."],
    "summary": "This is a high-quality renaissance-style necklace with excellent reviews and craftsmanship. The shipping time is longer than ideal, but the product strongly matches the searched query.", 
    "price": 100.0,
    "semantic_score": "HIGHLY RELEVANT"
}}

**EXAMPLE INPUT 4:**

Searched Query: sweater

Current Date: August 2

**OUTPUT:**
{{
    "pros": ["The product is a sweater aligning with the searched query.", "The product is made of high-quality material.", "The product is stylish and fashionable.", "It is from a luxury brand."],
    "cons": ["The product is marketed as a gift item, which wasn't specified in the search.", "The product is designed for a female recipient, which wasn't specified in the search."],
    "summary": "The sweater is from a luxury brand with high-quality materials and fashionable design. However, it is specifically marketed as a gift item for women, which adds constraints not present in the generic sweater search.",
    "price": 100.0,
    "semantic_score": "SOMEWHAT RELEVANT"
}}
""".strip()

# Prompt for the final purchase decision
FINAL_DECISION_PROMPT = """
You are making a purchase decision based on the given query and the products. Analyze the products and make a purchase decision based on analytical thinking and common sense.

Inputs you will receive:
- The searched query.
- A list of products and their price and a short summary.

Your job:
1. Critically compare the products based on their semantic relevance to the searched query.
2. Consider factors like price, quality, reviews, delivery time, and exact match to the search terms.
3. Decide which product(s) (one or more) should be purchased based on analytical reasoning.
4. Provide a short, logical justification for each recommended product.
5. If none of the products should be purchased, explain why and return an empty list of recommendations.
6. Finally, calculate the total cost of all recommended products.

Return ONLY a valid JSON object with the following structure:
{
  "reasoning": "<explanation of why you chose or rejected the products>",
  "recommendations": [
    {
      "product_name": "<name_1>",
      "reasoning": "<short logical explanation for product_1>"
    },
    {
      "product_name": "<name_2>",
      "reasoning": "<short logical explanation for product_2>"
    },
    ...
  ],
  "total_cost": <total cost of all recommended products as a float>
}
""".strip()