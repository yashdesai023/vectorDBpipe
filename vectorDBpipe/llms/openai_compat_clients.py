import logging
import requests
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class MistralLLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to Mistral AI API.
    Mistral uses an OpenAI-compatible endpoint structure.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("Mistral API Key is required for generation.")
            
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Mistral LLM Client for model: {self.model_name}")

    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        
        final_prompt = (
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer based on the context, just say that you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {user_query}\n"
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate response using Mistral: {e}")
            raise

class DeepSeekLLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to DeepSeek API.
    DeepSeek uses an OpenAI-compatible endpoint structure.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("DeepSeek API Key is required for generation.")
            
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized DeepSeek LLM Client for model: {self.model_name}")

    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        
        final_prompt = (
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer based on the context, just say that you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {user_query}\n"
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate response using DeepSeek: {e}")
            raise
