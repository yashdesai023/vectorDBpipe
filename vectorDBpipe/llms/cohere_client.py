import logging
import requests
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class CohereLLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to the Cohere V2 API.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("Cohere API Key is required for generation.")
            
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.cohere.ai/v1/chat"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        logger.info(f"Initialized Cohere LLM Client for model: {self.model_name}")

    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        
        final_prompt = (
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer based on the context, just say that you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {user_query}\n"
        )
        
        payload = {
            "model": self.model_name,
            "message": final_prompt,
            "preamble": system_prompt,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("text", "Error: No response generated from Cohere API.")
        except Exception as e:
            logger.error(f"Failed to generate response using Cohere: {e}")
            raise
