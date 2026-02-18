# vectordb/store.py

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ==========================
# Base Abstract Vector Store
# ==========================
class BaseVectorStore(ABC):
    """Abstract class for all vector database operations."""

    @abstractmethod
    def insert_vectors(self, vectors, metadata=None):
        """Insert embeddings and metadata into vector DB."""
        pass

    @abstractmethod
    def search_vectors(self, query_vector, top_k=5):
        """Search for similar vectors."""
        pass

    @abstractmethod
    def update_vector(self, vector_id, new_vector, metadata=None):
        """Update existing vector."""
        pass

    @abstractmethod
    def delete_vector(self, vector_id):
        """Delete a vector from DB."""
        pass


# =======================
# Local ChromaDB Handling (new PersistentClient API)
# =======================
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_directory: str = "data/chroma_store"):
        try:
            from chromadb import PersistentClient
        except Exception as e:
            raise RuntimeError("chromadb is required for ChromaVectorStore: pip install chromadb") from e

        # Use PersistentClient with path to persist_directory
        self.client = PersistentClient(path=persist_directory)
        # collection name
        self.collection = self.client.get_or_create_collection(name="text_embeddings")
        logger.info(f"Initialized local ChromaDB at {persist_directory}")

    def insert_vectors(self, vectors, metadata=None):
        # Expect vectors: list of lists (embeddings)
        ids = [f"id_{i}" for i in range(len(vectors))]
        metadatas = metadata if metadata is not None else [{} for _ in range(len(vectors))]
        # chroma expects embeddings as list of vectors
        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=[m.get("text", "") for m in metadatas])
        logger.info(f"Inserted {len(vectors)} vectors into ChromaDB")

    def search_vectors(self, query_vector, top_k=5):
        # chroma returns a dict-like structure; we return it as-is
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        return results

    def update_vector(self, vector_id, new_vector, metadata=None):
        # Chroma supports update by id
        self.collection.update(ids=[vector_id], embeddings=[new_vector], metadatas=[metadata or {}])
        logger.info(f"Updated vector ID: {vector_id}")

    def delete_vector(self, vector_id):
        self.collection.delete(ids=[vector_id])
        logger.info(f"Deleted vector ID: {vector_id}")


# ======================
# Cloud Pinecone Handler
# ======================
class PineconeVectorStore(BaseVectorStore):
    def __init__(self, api_key: str, index_name: str, dimension: int = 1536):
        try:
            from pinecone import Pinecone
        except ImportError:
            try:
                # Fallback for older versions just in case, or clearer error
                import pinecone
            except Exception as e:
                raise RuntimeError("pinecone-client is required: pip install pinecone-client") from e

        # Initialize Pinecone client (v3+)
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Check if index exists
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            # Create index with serverless spec or default (simplified for now)
            # Note: create_index in v3 requires spec. For now, we assume user created it or we need more config.
            # But to keep it simple and compatible with simple usage:
            try:
                from pinecone import ServerlessSpec
                self.pc.create_index(
                    name=index_name, 
                    dimension=dimension, 
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1") # Defaulting to common region
                )
            except Exception:
                logger.warning("Could not create Pinecone index automatically. Ensure it exists.")

        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

    def insert_vectors(self, vectors, metadata=None):
        items = [(f"id_{i}", vectors[i], metadata[i] if metadata else {}) for i in range(len(vectors))]
        self.index.upsert(vectors=items)
        logger.info(f"Inserted {len(vectors)} vectors into Pinecone")

    def search_vectors(self, query_vector, top_k=5):
        return self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    def update_vector(self, vector_id, new_vector, metadata=None):
        self.index.upsert(vectors=[(vector_id, new_vector, metadata or {})])
        logger.info(f"Updated vector ID: {vector_id}")

    def delete_vector(self, vector_id):
        self.index.delete(ids=[vector_id])
        logger.info(f"Deleted vector ID: {vector_id}")


# =====================
# Local FAISS Handler
# =====================
class FaissVectorStore(BaseVectorStore):
    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = index_path
        self.metadata_path = index_path + "_metadata.pkl"
        self.dimension = None
        self.index = None
        self.metadata_store = {} # Map id -> metadata

        try:
            import faiss
            import numpy as np
            import pickle
            import os
        except ImportError as e:
            raise RuntimeError("faiss-cpu, numpy, and pickle are required: pip install faiss-cpu numpy") from e

        self.faiss = faiss
        self.np = np
        self.pickle = pickle
        self.os = os

        if self.os.path.exists(self.index_path) and self.os.path.exists(self.metadata_path):
             self.load_index()
        else:
             logger.info(f"Initialized new FAISS store (will be created at {self.index_path})")

    def load_index(self):
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = self.faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata_store = self.pickle.load(f)

    def save_index(self):
        if self.index:
            self.faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                self.pickle.dump(self.metadata_store, f)
            logger.info(f"Saved FAISS index to {self.index_path}")

    def insert_vectors(self, vectors, metadata=None):
        if not vectors:
            return

        # Ensure vectors are numpy array
        vectors_np = self.np.array(vectors).astype('float32')
        
        # Initialize index if not exists (dimension based on first insert)
        if self.index is None:
            self.dimension = vectors_np.shape[1]
            # IDMap needed to map custom IDs, otherwise just sequential
            # For simplicity, we use IndexFlatL2
            self.index = self.faiss.IndexFlatL2(self.dimension)
            # If we want to support ID mapping properly, we'd need IndexIDMap
            # self.index = self.faiss.IndexIDMap(self.index) 

        # Add to FAISS
        # Note: IndexFlatL2 adds sequentially. We need to track IDs manually if we don't use IndexIDMap with explicit IDs.
        # Current logic: FAISS internal ID = current total + index in batch
        start_id = self.index.ntotal
        self.index.add(vectors_np)
        
        # Store metadata
        for i, meta in enumerate(metadata or []):
            internal_id = start_id + i
            # meta is dict, add user-friendly ID if not present
            if "id" not in meta:
                meta["id"] = f"id_{internal_id}"
            self.metadata_store[internal_id] = meta

        self.save_index()
        logger.info(f"Inserted {len(vectors)} vectors into FAISS")

    def search_vectors(self, query_vector, top_k=5):
        if not self.index:
            return []

        # Ensure query is numpy array 2D
        query_np = self.np.array([query_vector]).astype('float32')
        
        distances, indices = self.index.search(query_np, top_k)
        
        results = []
        # indices[0] is the list of nearest neighbor IDs for the first query
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # No enough results
            
            meta = self.metadata_store.get(idx, {})
            score = float(distances[0][i]) # L2 distance (lower is better)
            
            # Normalize score for consistency implies 0-1 usually, but L2 is unbounded. 
            # 1 / (1 + distance) is a common normalization for distance -> similarity
            sim_score = 1 / (1 + score)

            results.append({
                "id": meta.get("id", str(idx)),
                "score": sim_score,
                "metadata": meta
            })
        
        return results

    def update_vector(self, vector_id, new_vector, metadata=None):
        logger.warning("Update not fully supported in simple FAISS implementation yet.")
        pass

    def delete_vector(self, vector_id):
        logger.warning("Delete not fully supported in simple FAISS implementation yet.")
        pass


# =====================
# Store Factory Utility
# =====================
def get_vector_store(store_type: str = "chroma", config: dict = None):
    """
    Factory method to select vector store.
    :param store_type: "chroma", "pinecone", or "faiss"
    :param config: dict with credentials or paths
    """
    config = config or {}
    store_type = store_type.lower()
    
    if store_type == "pinecone":
        return PineconeVectorStore(
            api_key=config.get("api_key"),
            index_name=config.get("index_name"),
            dimension=config.get("dimension", 1536)
        )
    elif store_type == "chroma" or store_type == "chromadb":
        return ChromaVectorStore(
            persist_directory=config.get("persist_directory", "data/chroma_store")
        )
    elif store_type == "faiss":
        return FaissVectorStore(
            index_path=config.get("index_path", "data/faiss_index")
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
