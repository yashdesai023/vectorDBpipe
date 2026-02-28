import logging
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from vectorDBpipe.vectordb.base import BaseVectorDatabase

logger = logging.getLogger(__name__)

class FaissDatabase(BaseVectorDatabase):
    """
    Local Vector Database implementation using Facebook AI Similarity Search (FAISS).
    Fast, offline, and purely memory-based (with disk persistence).
    """

    def __init__(self, collection_name: str, mode: str = "local", api_key: str = None, dimension: int = 384, save_dir: str = "./data", **kwargs):
        self.collection_name = collection_name
        self.save_dir = save_dir
        self.dimension = dimension
        
        self.index_path = os.path.join(self.save_dir, f"{collection_name}.index")
        self.meta_path = os.path.join(self.save_dir, f"{collection_name}_meta.pkl")

        os.makedirs(self.save_dir, exist_ok=True)
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata_store = pickle.load(f)
        else:
            logger.info(f"Initializing new FAISS index of dimension {self.dimension}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata_store = [] # List storing dictionaries containing document and metadata

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
        logger.debug(f"Saved FAISS index to {self.index_path}")

    def add(self, embeddings: List[List[float]], documents: List[str], metadata: List[Dict[str, Any]], ids: List[str] = None):
        if not embeddings:
            return

        vector_array = np.array(embeddings).astype("float32")
        self.index.add(vector_array)
        
        # Store metadata aligned with FAISS internal index
        for i in range(len(embeddings)):
            self.metadata_store.append({
                "document": documents[i],
                "metadata": metadata[i] if len(metadata) > i else {},
                "id": ids[i] if ids and len(ids) > i else None
            })
            
        self.save()
        logger.info(f"Added {len(embeddings)} vectors to FAISS collection {self.collection_name}.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        # FAISS expects 2D array: (1, dim)
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata_store): # -1 means not enough results
                meta_info = self.metadata_store[idx]
                results.append({
                    "id": meta_info.get("id"),
                    "document": meta_info.get("document"),
                    "metadata": meta_info.get("metadata"),
                    "score": float(distances[0][i]) # L2 distance (lower is better)
                })
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "total_vectors": getattr(self.index, 'ntotal', 0),
            "dimension": self.dimension,
            "provider": "faiss"
        }
