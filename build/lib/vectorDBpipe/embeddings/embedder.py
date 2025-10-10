from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from vectorDBpipe.logger.logging import setup_logger

logger = setup_logger("Embedder")


class Embedder:
    """
    Handles text embeddings via HuggingFace sentence-transformers.
    Provides compatibility methods used by tests and pipeline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        # default model name short form (tests use "all-MiniLM-L6-v2")
        if model_name.startswith("sentence-transformers/"):
            model_ref = model_name
        else:
            # allow either full HF path or short name
            model_ref = f"sentence-transformers/{model_name}" if "sentence-transformers" not in model_name else model_name

        logger.info(f"Initializing embedder with model: {model_ref}")
        self.model_name = model_ref
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_ref)

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
