import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Configure the OpenAI/LiteLLM endpoint just like in shopping_agent.py
os.environ["OPENAI_API_KEY"] = open("keys/litellm.key").read().strip()
os.environ["OPENAI_API_BASE"] = "https://litellm.litellm.kn.ml-platform.etsy-mlinfra-dev.etsycloud.com"

# Load the webpage HTML that will be provided to the model
webpage_html = open("../data/etsy_pages/search_inflatable_halloween_spider_page_1/index.html").read().strip()

messages = [
    SystemMessage(
        content=(
            "You are a helpful assistant that suggest a new feature for "
            "boosting sales on the website. You will be given a webpage along "
            "with some previously tried attempts and you will need to suggest "
            "a **NEW** single feature. Be detailed in your suggestion giving "
            "detailed reasoning for your suggestion."
        )
    ),
    HumanMessage(
        content=f"""
The webpage is as follows:
{webpage_html}

The previous attempts are as follows:
No previous attempts yet.

You will need to suggest a single new feature that will boost sales on the website. Do NOT suggest any features that are very difficult to implement (like adding 3D models for every product).
""".strip()
    ),
]

# Instantiate the LLM (mirroring defaults from shopping_agent.py)
llm = ChatOpenAI(model_name="gemini-2.0-flash", temperature=0)

# Synchronously invoke the model and print the assistant's suggestion
response = llm.invoke(messages)
print(response.content)