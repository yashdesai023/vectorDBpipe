import unittest
from pipeline.text_pipeline import TextPipeline, FAISSVectorStore

class TestTextPipeline(unittest.TestCase):
    
    def setUp(self):
        # Initialize pipeline with default FAISS + embedding model
        self.pipeline = TextPipeline(
            embedding_model=None,  # will use default SentenceTransformer
            vector_store=FAISSVectorStore(dim=384)
        )
        # Sample text data
        self.texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial Intelligence is transforming the world.",
            "Python is a versatile programming language."
        ]

    # -----------------------------
    # Test: Data Loading
    # -----------------------------
    def test_preprocess_text(self):
        cleaned = self.pipeline.preprocess_text(self.texts)
        self.assertTrue(all(isinstance(t, str) for t in cleaned))
        self.assertTrue(all(t == t.lower() for t in cleaned))
        self.assertEqual(len(cleaned), len(self.texts))

    # -----------------------------
    # Test: Embedding Generation
    # -----------------------------
    def test_generate_embeddings(self):
        cleaned = self.pipeline.preprocess_text(self.texts)
        embeddings = self.pipeline.generate_embeddings(cleaned)
        self.assertEqual(len(embeddings), len(cleaned))
        self.assertEqual(len(embeddings[0]), 384)  # embedding dimension check

    # -----------------------------
    # Test: Vector Storage
    # -----------------------------
    def test_store_and_search_vectors(self):
        cleaned = self.pipeline.preprocess_text(self.texts)
        embeddings = self.pipeline.generate_embeddings(cleaned)
        metadata = [{"text": t} for t in cleaned]
        self.pipeline.store_vectors(embeddings, metadata)

        # Search test
        query = "AI technology"
        results = self.pipeline.search(query, top_k=2)
        self.assertTrue(len(results) <= 2)
        self.assertTrue(all("text" in r for r in results))

    # -----------------------------
    # Test: Pipeline End-to-End
    # -----------------------------
    def test_pipeline_end_to_end(self):
        # Load, preprocess, embed, store, and search
        cleaned = self.pipeline.preprocess_text(self.texts)
        embeddings = self.pipeline.generate_embeddings(cleaned)
        metadata = [{"text": t} for t in cleaned]
        self.pipeline.store_vectors(embeddings, metadata)

        query = "Programming in Python"
        results = self.pipeline.search(query, top_k=3)
        self.assertTrue(any("python" in r["text"] for r in results))

if __name__ == "__main__":
    unittest.main()
