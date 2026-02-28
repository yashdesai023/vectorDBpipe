import logging
import pinecone
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from vectorDBpipe.vectordb.base import BaseVectorDatabase

logger = logging.getLogger(__name__)

class PineconeDatabase(BaseVectorDatabase):
    """
    Vector Database implementation using Pinecone.
    Pinecone is entirely cloud-native, so 'local' mode is not supported.
    """

    def __init__(self, collection_name: str, mode: str = "cloud", api_key: str = None, dimension: int = 1024, **kwargs):
        self.collection_name = collection_name
        self.dimension = int(dimension)
        
        self.metric = kwargs.get("metric", "cosine")
        self.cloud = kwargs.get("cloud", "aws")
        self.region = kwargs.get("region", kwargs.get("environment", "us-east-1"))
        self.capacity_mode = kwargs.get("capacity_mode", "serverless")
        
        if mode != "cloud":
            logger.warning("Pinecone only supports 'cloud' mode. Forcing mode='cloud'.")
            mode = "cloud"

        if not api_key:
            raise ValueError("Pinecone API Key is strictly required.")

        logger.info(f"Connecting to Pinecone Cloud (Region: {self.region})")
        self.pc = Pinecone(api_key=api_key)
        
        self._ensure_index()
        self.index = self.pc.Index(self.collection_name)

    def _ensure_index(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.collection_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.collection_name} | {self.dimension}d | {self.metric} | {self.capacity_mode}")
            
            if str(self.capacity_mode).lower() == "serverless":
                spec = ServerlessSpec(cloud=self.cloud, region=self.region)
            else:
                from pinecone import PodSpec
                # Default for pod-based, although in modern Pinecone free tier is serverless
                spec = PodSpec(environment=self.region)
                
            self.pc.create_index(
                name=self.collection_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=spec
            )

    def add(self, embeddings: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]], ids: List[str] = None):
        if not ids:
            ids = [f"doc_{hash(doc)}" for doc in documents]
            
        vectors_to_upsert = []
        for i in range(len(embeddings)):
            meta = metadata[i] if metadata and len(metadata) > i else {}
            meta["text"] = documents[i] 
            vectors_to_upsert.append(
                {"id": str(ids[i]), "values": embeddings[i], "metadata": meta}
            )

        try:
            # Upsert in batches of 100 to respect Pinecone limits
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
            logger.info(f"Added {len(embeddings)} vectors to Pinecone index {self.collection_name}.")
        except Exception as e:
            logger.error(f"Failed to add vectors to Pinecone: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        formatted_results = []
        for match in results.get("matches", []):
            formatted_results.append({
                "id": match.get("id"),
                "document": match.get("metadata", {}).get("text"),
                "metadata": match.get("metadata"),
                "score": match.get("score")
            })
        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        stats = self.index.describe_index_stats()
        return {
            "name": self.collection_name,
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": self.dimension,
            "provider": "pinecone"
        }
