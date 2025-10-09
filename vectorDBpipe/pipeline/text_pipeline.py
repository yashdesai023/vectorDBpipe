# pipeline/text_pipeline.py

import logging
from vectorDBpipe.config.config_manager import ConfigManager
from vectorDBpipe.data.loader import DataLoader
from vectorDBpipe.embeddings.embedder import Embedder
from vectorDBpipe.vectordb.store import get_vector_store
from vectorDBpipe.utils.common import clean_text, chunk_text
from vectorDBpipe.logger.logging import setup_logger

logger = logging.getLogger(__name__)

from typing import List, Optional, Union
import os

# Optional imports for embedding and vector storage
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None

# -----------------------------
# TextPipeline Class Definition
# -----------------------------
class TextPipeline:
    def __init__(
        self,
        embedding_model: Optional[object] = None,
        vector_store: Optional[object] = None
    ):
        """
        Initialize the pipeline with embedding model and vector store.
        """
        self.embedding_model = embedding_model or self.default_embedding_model()
        self.vector_store = vector_store or self.default_vector_store()

    # -----------------------------
    # Stage 1: Data Loading
    # -----------------------------
    def load_data(self, source: Union[str, List[str]]) -> List[str]:
        """
        Load text from a file, URL, or directly from a list of strings.
        """
        texts = []
        if isinstance(source, list):
            texts = source
        elif os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as f:
                texts = f.readlines()
        else:
            # For URLs (basic example, requires requests)
            import requests
            resp = requests.get(source)
            resp.raise_for_status()
            texts = [resp.text]
        return texts

    # -----------------------------
    # Stage 2: Text Preprocessing
    # -----------------------------
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """
        Basic text cleaning: lowercasing, stripping, optional advanced cleaning.
        """
        cleaned = [t.strip().lower() for t in texts if t.strip()]
        return cleaned

    # -----------------------------
    # Stage 3: Embedding Generation
    # -----------------------------
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts to vector embeddings using the embedding model.
        """
        if not self.embedding_model:
            raise ValueError("No embedding model provided.")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings

    # -----------------------------
    # Stage 4: Vector Storage
    # -----------------------------
    def store_vectors(self, embeddings: List[List[float]], metadata: Optional[List[dict]] = None):
        """
        Insert embeddings into the vector store.
        """
        if not self.vector_store:
            raise ValueError("No vector store provided.")
        self.vector_store.add(embeddings, metadata)

    # -----------------------------
    # Stage 5: Retrieval / QA
    # -----------------------------
    def search(self, query: str, top_k: int = 5):
        """
        Retrieve top-k similar documents for a given query.
        """
        if not self.vector_store:
            raise ValueError("No vector store provided.")
        query_embedding = self.embedding_model.encode([query])[0]
        return self.vector_store.search(query_embedding, top_k=top_k)

    # -----------------------------
    # Default Components
    # -----------------------------
    def default_embedding_model(self):
        if SentenceTransformer:
            return SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ImportError("Install sentence-transformers to use default embedding model.")

    def default_vector_store(self):
        if faiss:
            return FAISSVectorStore()
        else:
            raise ImportError("Install faiss to use default vector store.")


# -----------------------------
# Example Vector Store Wrapper
# -----------------------------
class FAISSVectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embeddings, metadata=None):
        import numpy as np
        self.index.add(np.array(embeddings).astype("float32"))
        self.data.extend(metadata or [{}] * len(embeddings))

    def search(self, query_embedding, top_k=5):
        import numpy as np
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.data):
                results.append(self.data[idx])
        return results
