from typing import List
from vectorDBpipe.embeddings.base import BaseEmbeddingProvider
import logging
import requests

logger = logging.getLogger(__name__)

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Cloud embedding provider using the OpenAI REST API.
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        if not api_key:
            raise ValueError("API Key is required for OpenAI Cloud Embeddings.")
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized OpenAI Embedding Client for model: {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "input": texts,
            "model": self.model_name
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"OpenAI API Error: {response.text}")
            raise RuntimeError(f"Failed to fetch OpenAI embeddings: {response.text}")

        data = response.json()
        # Ensure vectors are returned in the exact input text order
        # data['data'] is a list of dicts: {'index': 0, 'embedding': [...] }
        sorted_results = sorted(data["data"], key=lambda x: x["index"])
        return [res["embedding"] for res in sorted_results]


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Cloud embedding provider using the Google Gemini / PaLM REST API.
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        # Default typical model: text-embedding-004
        self.model_name = model_name 
        if not api_key:
            raise ValueError("API Key is required for Google Cloud Embeddings.")
        self.api_key = api_key
        # Gemini REST URL structure
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Google Embedding Client for model: {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Google Batch API requires a specific structure
        requests_payload = [{"model": f"models/{self.model_name}", "content": {"parts": [{"text": txt}]}} for txt in texts]
        payload = {"requests": requests_payload}
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Google API Error: {response.text}")
            raise RuntimeError(f"Failed to fetch Google embeddings: {response.text}")

        data = response.json()
        embeddings = []
        for embedding_obj in data.get("embeddings", []):
            embeddings.append(embedding_obj.get("values", []))
        return embeddings


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """
    Cloud embedding provider using the Cohere REST API.
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        # Default typically: embed-v4.0 or embed-english-v3.0
        self.model_name = model_name 
        if not api_key:
            raise ValueError("API Key is required for Cohere Cloud Embeddings.")
        self.api_key = api_key
        self.url = "https://api.cohere.ai/v1/embed"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        logger.info(f"Initialized Cohere Embedding Client for model: {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "texts": texts,
            "model": self.model_name,
            "input_type": "search_document"
        }
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Cohere API Error: {response.text}")
            raise RuntimeError(f"Failed to fetch Cohere embeddings: {response.text}")

        data = response.json()
        return data.get("embeddings", [])
