from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorDatabase(ABC):
    """
    Abstract base class for all Vector Databases supported by vectorDBpipe.
    """

    @abstractmethod
    def __init__(self, collection_name: str, mode: str = "local", api_key: str = None, **kwargs):
        """
        Initialize the database client connection.
        :param collection_name: Name of the collection/index to use.
        :param mode: 'local' (offline) or 'cloud' (API-based).
        :param api_key: Secret key for cloud DBs (e.g., Pinecone, Weaviate Cloud).
        """
        pass

    @abstractmethod
    def add(self, embeddings: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]], ids: List[str] = None):
        """
        Add generated vectors and their corresponding document chunks to the database.
        :param embeddings: List of numerical float vectors.
        :param documents: Original text chunks corresponding to the vectors.
        :param metadata: List of dicts containing source metadata (page numbers, filename, etc).
        :param ids: (Optional) List of unique identifiers for each entry.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a semantic similarity search retrieving the most relevant chunks.
        :param query_embedding: The numerical vector of the user's search query.
        :param top_k: The number of results to return.
        :return: A list of dicts containing {'id', 'document', 'metadata', 'score'}.
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retrieve generic statistics about the current collection (e.g., entity count).
        """
        pass
