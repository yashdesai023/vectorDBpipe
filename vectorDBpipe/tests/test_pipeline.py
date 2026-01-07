import pytest
from vectorDBpipe.pipeline.text_pipeline import TextPipeline

class MockStore:
    def __init__(self, *args, **kwargs):
        pass  # Accept any arguments like api_key, index_name, etc.
    
    def insert_vectors(self, *args, **kwargs):
        return True

    def search_vectors(self, *args, **kwargs):
        return [{"score": 0.99, "text": "Test document"}]


def test_pipeline_run(monkeypatch):
    # Mock dependencies to avoid external API calls
    monkeypatch.setattr("vectorDBpipe.data.loader.DataLoader.load_data", lambda self: [{"content": "Test document", "source": "test_src"}])
    monkeypatch.setattr("vectorDBpipe.embeddings.embedder.Embedder.encode", lambda self, x: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr("vectorDBpipe.embeddings.embedder.Embedder.embed_texts", lambda self, x: [[0.1, 0.2, 0.3]])

    # Mock vector store (before pipeline initialization)
    monkeypatch.setattr("vectorDBpipe.vectordb.store.get_vector_store", lambda *a, **kw: MockStore())
    monkeypatch.setattr("vectorDBpipe.vectordb.store.PineconeVectorStore", MockStore)

    pipeline = TextPipeline()
    pipeline.run()

    results = pipeline.search("Test document")
    assert results[0]["score"] > 0.9
    assert "Test document" in results[0]["text"]
