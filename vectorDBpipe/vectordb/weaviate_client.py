import logging
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from typing import List, Dict, Any
from vectorDBpipe.vectordb.base import BaseVectorDatabase
import uuid

logger = logging.getLogger(__name__)

class WeaviateDatabase(BaseVectorDatabase):
    """
    Vector Database implementation using Weaviate (v4 client).
    Supports both local persistence (Docker/Local) and Weaviate Cloud (WCD).
    """

    def __init__(self, collection_name: str, mode: str = "local", api_key: str = None, save_dir: str = None, **kwargs):
        # Weaviate collections must start with a capital letter
        self.collection_name = collection_name.capitalize()
        
        if mode == "local":
            logger.info("Connecting to Local Weaviate instance.")
            # Typically connects to localhost:8080 assuming a local docker container
            self.client = weaviate.connect_to_local()
        elif mode == "cloud":
            url = kwargs.get("url")
            if not url or not api_key:
                raise ValueError("Both 'url' and 'api_key' are required for Weaviate Cloud mode.")
            logger.info(f"Connecting to Weaviate Cloud at {url}")
            self.client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key),
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'local' or 'cloud'.")

        self._ensure_collection()
        self.collection = self.client.collections.get(self.collection_name)

    def _ensure_collection(self):
        if not self.client.collections.exists(self.collection_name):
            logger.info(f"Creating Weaviate collection: {self.collection_name}")
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source_meta", data_type=DataType.TEXT), # Store JSON string here for simplicity
                ],
            )

    def add(self, embeddings: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]], ids: List[str] = None):
        import json
        
        with self.collection.batch.dynamic() as batch:
            for i in range(len(embeddings)):
                custom_id = uuid.UUID(ids[i]) if ids and len(ids) > i else uuid.uuid4()
                meta = metadata[i] if metadata and len(metadata) > i else {}
                
                batch.add_object(
                    properties={
                        "text": documents[i],
                        "source_meta": json.dumps(meta) # Flatten metadata
                    },
                    vector=embeddings[i],
                    uuid=custom_id
                )
                
        logger.info(f"Added {len(embeddings)} vectors to Weaviate collection {self.collection_name}.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        import json
        results = self.collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        formatted_results = []
        for obj in results.objects:
            meta_dict = {}
            if "source_meta" in obj.properties:
                try:
                    meta_dict = json.loads(obj.properties["source_meta"])
                except Exception:
                    pass
                    
            formatted_results.append({
                "id": str(obj.uuid),
                "document": obj.properties.get("text"),
                "metadata": meta_dict,
                "score": obj.metadata.distance
            })
        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        count_result = self.collection.aggregate.over_all(total_count=True)
        return {
            "name": self.collection_name,
            "total_vectors": count_result.total_count,
            "provider": "weaviate"
        }

    def __del__(self):
        # Weaviate client requires closing
        try:
            self.client.close()
        except:
            pass
