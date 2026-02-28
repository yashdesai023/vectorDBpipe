import logging
import requests
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class AnthropicLLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to the Anthropic Messages API.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("Anthropic API Key is required for generation.")
            
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        logger.info(f"Initialized Anthropic LLM Client for model: {self.model_name}")

    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        
        final_prompt = (
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer based on the context, just say that you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {user_query}\n"
        )
        
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": final_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Failed to generate response using Anthropic: {e}")
            raise
