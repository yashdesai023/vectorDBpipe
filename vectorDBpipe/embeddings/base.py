from abc import ABC, abstractmethod
from typing import List, Union

class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for all embedding model providers (Local and Cloud).
    """

    @abstractmethod
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        """
        Initialize the embedding client.
        :param model_name: Name of the model to use (e.g., 'text-embedding-3-small' or 'all-MiniLM-L6-v2')
        :param api_key: Secret key for cloud providers. None for local models.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single string.
        :param text: Input text content.
        :return: List of floats representing the vector.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of strings efficiently.
        :param texts: List of input text contents.
        :return: List of vector embeddings.
        """
        pass
