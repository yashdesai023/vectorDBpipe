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
# Local ChromaDB Handling
# =======================
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_directory="data/chroma_store"):
        from chromadb import Client
        from chromadb.config import Settings

        self.client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self.collection = self.client.get_or_create_collection("text_embeddings")
        logger.info(f"Initialized local ChromaDB at {persist_directory}")

    def insert_vectors(self, vectors, metadata=None):
        ids = [f"id_{i}" for i in range(len(vectors))]
        self.collection.add(embeddings=vectors, metadatas=metadata, ids=ids)
        logger.info(f"Inserted {len(vectors)} vectors into ChromaDB")

    def search_vectors(self, query_vector, top_k=5):
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        return results

    def update_vector(self, vector_id, new_vector, metadata=None):
        self.collection.update(ids=[vector_id], embeddings=[new_vector], metadatas=[metadata])
        logger.info(f"Updated vector ID: {vector_id}")

    def delete_vector(self, vector_id):
        self.collection.delete(ids=[vector_id])
        logger.info(f"Deleted vector ID: {vector_id}")


# ======================
# Cloud Pinecone Handler
# ======================
class PineconeVectorStore(BaseVectorStore):
    def __init__(self, api_key, index_name):
        import pinecone
        pinecone.init(api_key=api_key)
        self.index_name = index_name

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536, metric="cosine")

        self.index = pinecone.Index(index_name)
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
# Store Factory Utility
# =====================
def get_vector_store(store_type="chroma", config=None):
    """
    Factory method to select vector store.
    :param store_type: "chroma" or "pinecone"
    :param config: dict with credentials or paths
    """
    if store_type == "pinecone":
        return PineconeVectorStore(
            api_key=config.get("api_key"),
            index_name=config.get("index_name")
        )
    elif store_type == "chroma":
        return ChromaVectorStore(
            persist_directory=config.get("persist_directory", "data/chroma_store")
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
