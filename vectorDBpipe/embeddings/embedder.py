from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from vectorDBpipe.logger.logging import setup_logger

logger = setup_logger("Embedder")


class Embedder:
    """
    Handles text embeddings via HuggingFace sentence-transformers.
    Future support: OpenAI / Gemini / Custom Models
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        logger.info(f"Initializing embedder with model: {model_name}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for single or multiple texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        return np.array(embeddings)

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
