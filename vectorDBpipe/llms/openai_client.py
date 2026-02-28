import logging
import requests
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class OpenAILLMProvider(BaseLLMProvider):
    """
    LLM generation interface connecting to the OpenAI Chat Completions API.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("OpenAI API Key is required for generation.")
            
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized OpenAI LLM Client for model: {self.model_name}")

    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        
        # Combine the user query with the RAG knowledge wrapper
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
            "temperature": 0.3 # Keep generation highly factual based on context
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate response using OpenAI: {e}")
            raise
