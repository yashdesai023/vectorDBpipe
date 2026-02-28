import logging
import chromadb
from typing import List, Dict, Any
from vectorDBpipe.vectordb.base import BaseVectorDatabase

logger = logging.getLogger(__name__)

class ChromaDatabase(BaseVectorDatabase):
    """
    Vector Database implementation using ChromaDB.
    Supports both local persistence and cloud host connections.
    """

    def __init__(self, collection_name: str, mode: str = "local", api_key: str = None, save_dir: str = "./data", **kwargs):
        self.collection_name = collection_name
        
        if mode == "local":
            import os
            abs_path = os.path.abspath(save_dir)
            os.makedirs(abs_path, exist_ok=True)
            logger.info(f"Connecting to Local ChromaDB at {abs_path}")
            
            # To prevent thread-locking bugs on default_tenant, explicitly assign settings
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(
                path=abs_path,
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
        elif mode == "cloud":
            logger.info("Connecting to Cloud ChromaDB via HTTP")
            # In a real cloud setup, we'd pass host/port/auth params
            # This is a stub for the architecture 
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", "8000")
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'local' or 'cloud'.")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add(self, embeddings: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]], ids: List[str] = None):
        if not ids:
            # Generate deterministic string IDs if none provided
            ids = [f"doc_{i}_{hash(doc)}" for i, doc in enumerate(documents)]
            
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(embeddings)} vectors to Chroma collection {self.collection_name}.")
        except Exception as e:
            logger.error(f"Failed to add vectors to ChromaDB: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        # Chroma query returns a dict of lists
        if results and results.get("ids", [[]])[0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i] if "documents" in results else None,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                    "score": results["distances"][0][i] if "distances" in results else None
                })
        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "total_vectors": self.collection.count(),
            "provider": "chromadb"
        }
