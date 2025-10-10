from vectorDBpipe.vectordb.store import get_vector_store

def test_chroma_insert_and_search(tmp_path):
    store = get_vector_store("chroma", {"persist_directory": str(tmp_path)})
    vectors = [[0.1, 0.2, 0.3]]
    metadata = [{"source": "test"}]
    store.insert_vectors(vectors, metadata)
    result = store.search_vectors([0.1, 0.2, 0.3], top_k=1)
    assert len(result) > 0
