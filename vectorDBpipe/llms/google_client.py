import logging
import requests
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GoogleLLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to the Google Gemini API.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("Google API Key is required for generation.")
            
        self.model_name = model_name
        self.api_key = api_key
        # Gemini expects URL format based on model name
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Google LLM Client for model: {self.model_name}")

    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        
        final_prompt = (
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer based on the context, just say that you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {user_query}\n"
        )
        
        payload = {
            "system_instruction": {
                 "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": final_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3
            }
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return "Error: No response generated from Google API."
        except Exception as e:
            logger.error(f"Failed to generate response using Google Gemini: {e}")
            raise
