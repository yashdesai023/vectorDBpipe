from typing import List
from vectorDBpipe.embeddings.base import BaseEmbeddingProvider
import logging

logger = logging.getLogger(__name__)

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using `sentence-transformers` for offline vector generation.
    Supports models like 'all-MiniLM-L6-v2', 'BAAI/bge-m3', etc.
    """

    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.model_name = model_name
        logger.info(f"Initializing Local Embedding Model: {self.model_name}")
        try:
            # We attempt to load the model. It will download if not present locally.
            # Lazy load large PyTorch library to prevent blocking parent threads
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load local model '{self.model_name}'. Ensure sentence-transformers is installed. Error: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding batch of texts: {e}")
            raise
