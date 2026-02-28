# Lazy load SentenceTransformer inside __init__ to prevent Torch DLL crashes
import logging
from typing import List, Union
import numpy as np
from vectorDBpipe.logger.logging import setup_logger

logger = setup_logger("Embedder")


class Embedder:
    """
    Handles text embeddings via HuggingFace sentence-transformers.
    Provides compatibility methods used by tests and pipeline.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        logger.info(f"Initializing embedder with model: {model_name}")
        self.model_name = model_name
        self.batch_size = batch_size
        
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        from sentence_transformers import SentenceTransformer
        # Force CPU device to prevent 'meta tensor' ThreadPoolExecutor crashes in FastAPI
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for single or multiple texts.
        Returns a numpy ndarray.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        return np.array(embeddings)

    # Backwards-compatible alias used by tests
    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode(texts)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of strings efficiently.
        """
        return self.encode(texts).tolist()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single string.
        """
        return self.encode(text)[0].tolist()

    @staticmethod
    def get_supported_models() -> List[str]:
        """
        Return list of recommended open-source models.
        """
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5"
        ]
