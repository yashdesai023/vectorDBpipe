import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
from vectorDBpipe.vectordb.base import BaseVectorDatabase
import uuid

logger = logging.getLogger(__name__)

class QdrantDatabase(BaseVectorDatabase):
    """
    Vector Database implementation using Qdrant.
    Supports both local persistence and cloud host connections.
    """

    def __init__(self, collection_name: str, mode: str = "local", api_key: str = None, save_dir: str = "./data/qdrant", dimension: int = 384, **kwargs):
        self.collection_name = collection_name
        self.dimension = dimension
        
        if mode == "local":
            logger.info(f"Connecting to Local Qdrant at {save_dir}")
            self.client = QdrantClient(path=save_dir)
        elif mode == "cloud":
            url = kwargs.get("url")
            if not url or not api_key:
                raise ValueError("Both 'url' and 'api_key' are required for Qdrant Cloud mode.")
            logger.info(f"Connecting to Cloud Qdrant at {url}")
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'local' or 'cloud'.")

        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection: {self.collection_name} with dimension {self.dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def add(self, embeddings: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]], ids: List[str] = None):
        if not ids:
            # Qdrant requires IDs to be UUIDs or integers. We'll generate a UUID for each.
            ids = [str(uuid.uuid4()) for _ in documents]
            
        points = []
        for i in range(len(embeddings)):
            meta = metadata[i] if metadata and len(metadata) > i else {}
            # store the text chunk in the metadata payload
            meta["text"] = documents[i] 
            
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload=meta
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(embeddings)} vectors to Qdrant collection {self.collection_name}.")
        except Exception as e:
            logger.error(f"Failed to add vectors to Qdrant: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "id": hit.id,
                "document": hit.payload.get("text") if hit.payload else None,
                "metadata": hit.payload,
                "score": hit.score
            })
        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "total_vectors": info.vectors_count,
            "provider": "qdrant"
        }
