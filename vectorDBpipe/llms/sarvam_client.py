import logging
import requests
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class SarvamLLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to Sarvam AI.
    Excellent for Indic languages.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("Sarvam API Key is required for generation.")
            
        valid_models = ["sarvam-m", "sarvam-30b", "sarvam-105b"]
        if model_name not in valid_models:
            logger.warning(f"Invalid Sarvam model '{model_name}'. Defaulting to 'sarvam-m'.")
            self.model_name = "sarvam-m"
        else:
            self.model_name = model_name
        
        self.api_key = api_key
        self.url = "https://api.sarvam.ai/v1/chat/completions" # Sarvam uses an OpenAI compatible structure
        self.headers = {
            "api-subscription-key": self.api_key, # Sarvam might use a custom header or Bearer auth
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Sarvam LLM Client for model: {self.model_name}")

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
                {"role": "user", "content": f"{system_prompt}\n\n{final_prompt}"}
            ]
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            if response.status_code == 401:
                logger.error(f"Sarvam API Auth Error: Check your API key. {response.text}")
                raise RuntimeError("Sarvam API unauthorized.")
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate response using Sarvam: {e}")
            raise
