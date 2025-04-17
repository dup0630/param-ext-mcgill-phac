"""
gpt_client.py

Provides a wrapper for querying Azure OpenAI's GPT models using the Chat Completions API.

This module initializes an AzureOpenAI client using credentials stored in environment variables and exposes a single function, `ask_GPT`, which sends a list of messages to a specified model and returns the response text.

Environment Variables Required:
- OPENAI_KEY: API key for Azure OpenAI
- OPENAI_ENDPOINT: Endpoint URL
- OPENAI_VERSION: API version (e.g., "2023-05-15")

Dependencies:
- openai
- python-dotenv
"""

from openai import AzureOpenAI
from dotenv import load_dotenv
import os

def ask_GPT(prompt: list[dict], deployment_name: str = "gpt-4o-mini") -> str:
    """
    Sends a prompt to an Azure OpenAI chat model and returns the generated response.
    """
    # Initialize Azure client
    load_dotenv()
    key = os.getenv("OPENAI_KEY")
    endpoint = os.getenv("OPENAI_ENDPOINT")
    version = os.getenv("OPENAI_VERSION")

    if not key or not endpoint:
        raise ValueError("OPENAI_KEY and/or OPENAI_ENDPOINT not set in environment variables.")

    client = AzureOpenAI(
      azure_endpoint = endpoint, 
      api_key=key,  
      api_version=version
    )

    # Ask ChatGPT
    response = client.chat.completions.create(
        model = deployment_name,
        messages = prompt
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage:
    sys = input("Enter system prompt: ")
    usr = input("Enter user prompt: ")
    prompt = [
        {"role": "developer", "content": sys},
        {"role": "user", "content": usr}
    ]
    print(ask_GPT(prompt))