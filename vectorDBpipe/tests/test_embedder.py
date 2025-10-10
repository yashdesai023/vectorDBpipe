from vectorDBpipe.embeddings.embedder import Embedder

def test_embedding_shape():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    text = ["Artificial intelligence"]
    vectors = embedder.embed_texts(text)
    assert len(vectors) == 1
    assert len(vectors[0]) in [384, 768]
