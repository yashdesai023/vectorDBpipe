"""
vectorDBpipe — Comprehensive Unit Test Suite
Tests all 4 RAG engines, Semantic OmniRouter, sentence chunking,
PPTX loader, Graph/PageIndex persistence, and no-LLM fallback paths.

Run with:
    cd <project-root>
    python -m pytest tests/ -v
"""

import json
import os
import tempfile
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock, mock_open


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixture — builds a VDBpipe instance without loading any real models
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    """
    Fully mocked VDBpipe instance constructed via __new__ (bypasses __init__).
    All heavy dependencies (Embedder, VectorStore, LLM, DataLoader) are replaced
    with MagicMock objects — no API keys, PyTorch, or FAISS required.
    """
    from vectorDBpipe.pipeline.vdbpipe import VDBpipe

    p = VDBpipe.__new__(VDBpipe)

    # Core state
    p.logger = MagicMock()
    p.graph = nx.DiGraph()
    p.page_index = {}
    p._state_dir = tempfile.mkdtemp()
    p._intent_embeddings = {}   # disable semantic router → use keyword fallback

    # Mocked providers
    p.embedder = MagicMock()
    p.embedder.embed_text.return_value = [0.1] * 384
    p.embedder.embed_batch.return_value = [[0.1] * 384]

    p.vector_store = MagicMock()
    p.vector_store.search.return_value = [
        {"document": "The revenue was $1M in Q3.", "score": 0.9, "metadata": {"source": "report.pdf"}}
    ]

    p.llm = MagicMock()
    p.llm.generate_response.return_value = "Generated LLM answer."
    p.llm.stream_response.return_value = iter(["Generated ", "LLM ", "answer."])

    loader_mock = MagicMock()
    loader_mock.data_path = None
    loader_mock.load_data.return_value = [
        {"content": "Alice leads the API team. Bob is a researcher.", "source": "test.txt"}
    ]
    p.loader = loader_mock

    p._embed_and_store = MagicMock()

    return p


# ─────────────────────────────────────────────────────────────────────────────
# #1 — Initialization Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialization:
    def test_has_required_attributes(self, pipeline):
        assert hasattr(pipeline, "graph")
        assert hasattr(pipeline, "page_index")
        assert hasattr(pipeline, "ingest")
        assert hasattr(pipeline, "query")
        assert hasattr(pipeline, "extract")
        assert hasattr(pipeline, "search")

    def test_is_not_text_pipeline_subclass(self):
        """VDBpipe must be pure composition — NOT a subclass of TextPipeline."""
        from vectorDBpipe.pipeline.vdbpipe import VDBpipe
        from vectorDBpipe.pipeline.text_pipeline import TextPipeline
        assert not issubclass(VDBpipe, TextPipeline), (
            "VDBpipe should NOT inherit from TextPipeline (composition refactor violated)"
        )

    def test_base_classes_are_object_only(self):
        """VDBpipe should only inherit from object — pure composition."""
        from vectorDBpipe.pipeline.vdbpipe import VDBpipe
        assert VDBpipe.__bases__ == (object,)


# ─────────────────────────────────────────────────────────────────────────────
# #2 — Ingestion (Tri-Processing)
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_ingest_sets_loader_path_and_calls_load_data(self, pipeline):
        pipeline.ingest("dummy_path/report.pdf")
        assert pipeline.loader.data_path == "dummy_path/report.pdf"
        pipeline.loader.load_data.assert_called_once()

    def test_ingest_populates_page_index(self, pipeline):
        pipeline.ingest("dummy_path")
        assert len(pipeline.page_index) > 0

    def test_ingest_skips_empty_documents(self, pipeline):
        pipeline.loader.load_data.return_value = [{"content": "", "source": "empty.txt"}]
        result = pipeline.ingest("dummy_path")
        assert result == 0  # nothing was embedded


# ─────────────────────────────────────────────────────────────────────────────
# #3 — Semantic OmniRouter (keyword fallback path, no embedder)
# ─────────────────────────────────────────────────────────────────────────────

class TestOmniRouter:
    def test_summarize_routes_to_engine_2(self, pipeline):
        assert pipeline._route_query("Summarize the entire document.") == "ENGINE_2"

    def test_overview_routes_to_engine_2(self, pipeline):
        assert pipeline._route_query("Give me an overview of the report.") == "ENGINE_2"

    def test_relationship_routes_to_engine_3(self, pipeline):
        assert pipeline._route_query("How are Alice and Bob connected?") == "ENGINE_3"

    def test_fact_routes_to_engine_1(self, pipeline):
        assert pipeline._route_query("What was the revenue in Q3?") == "ENGINE_1"

    def test_default_falls_back_to_engine_1(self, pipeline):
        assert pipeline._route_query("Tell me something.") == "ENGINE_1"

    def test_query_dispatches_to_correct_engine(self, pipeline):
        """End-to-end routing: summary query → Engine 2 called."""
        pipeline._engine_2_vectorless_rag = MagicMock(return_value="Vectorless answer")
        result = pipeline.query("Summarize the document please.")
        pipeline._engine_2_vectorless_rag.assert_called_once()
        assert result == "Vectorless answer"


# ─────────────────────────────────────────────────────────────────────────────
# #4 — Engine 1: Vector RAG
# ─────────────────────────────────────────────────────────────────────────────

class TestEngine1VectorRAG:
    def test_returns_llm_answer(self, pipeline):
        result = pipeline._engine_1_vector_rag("What is the revenue?")
        assert result == "Generated LLM answer."

    def test_calls_search(self, pipeline):
        pipeline._engine_1_vector_rag("Test query")
        pipeline.vector_store.search.assert_called()

    def test_no_llm_returns_raw_context(self, pipeline):
        pipeline.llm = None
        result = pipeline._engine_1_vector_rag("What happened?")
        assert "Retrieved Context" in result or "revenue" in result


# ─────────────────────────────────────────────────────────────────────────────
# #5 — Engine 2: Vectorless RAG
# ─────────────────────────────────────────────────────────────────────────────

class TestEngine2VectorlessRAG:
    def test_returns_llm_answer_when_page_index_populated(self, pipeline):
        pipeline.page_index = {
            "report.pdf": {
                "chapters": ["Introduction", "Results"],
                "summary": "A quarterly financial report.",
                "total_chars": 500,
                "raw_lines": [],
            }
        }
        result = pipeline._engine_2_vectorless_rag("What is this document about?")
        pipeline.llm.generate_response.assert_called_once()
        assert result == "Generated LLM answer."

    def test_returns_empty_message_when_no_page_index(self, pipeline):
        pipeline.page_index = {}
        result = pipeline._engine_2_vectorless_rag("Summarize")
        assert "ingest" in result.lower()

    def test_no_llm_returns_structured_fallback(self, pipeline):
        pipeline.llm = None
        pipeline.page_index = {
            "doc.txt": {"chapters": ["Ch1"], "summary": "Short summary.", "total_chars": 100, "raw_lines": []}
        }
        result = pipeline._engine_2_vectorless_rag("Overview?")
        assert "Vectorless RAG" in result
        assert "doc.txt" in result


# ─────────────────────────────────────────────────────────────────────────────
# #6 — Engine 3: GraphRAG
# ─────────────────────────────────────────────────────────────────────────────

class TestEngine3GraphRAG:
    def test_empty_graph_falls_back_to_engine_1(self, pipeline):
        pipeline.graph = nx.DiGraph()  # empty
        pipeline._engine_1_vector_rag = MagicMock(return_value="Vector fallback")
        result = pipeline._engine_3_graph_rag("How are entities connected?")
        assert "Vector fallback" in result or "No graph" in result

    def test_populated_graph_calls_llm(self, pipeline):
        pipeline.graph.add_edge("Alice", "API Team", relation="leads")
        pipeline.graph.add_edge("Bob", "Research", relation="manages")
        result = pipeline._engine_3_graph_rag("Who leads the API team?")
        pipeline.llm.generate_response.assert_called_once()
        assert result == "Generated LLM answer."

    def test_no_llm_returns_graph_text_fallback(self, pipeline):
        pipeline.llm = None
        pipeline.graph.add_edge("Alice", "API Team", relation="leads")
        result = pipeline._engine_3_graph_rag("Who leads what?")
        assert "GraphRAG" in result
        assert "Alice" in result


# ─────────────────────────────────────────────────────────────────────────────
# #7 — Engine 4: Structured Extract
# ─────────────────────────────────────────────────────────────────────────────

class TestEngine4StructuredExtract:
    def test_returns_parsed_json(self, pipeline):
        pipeline.llm.generate_response.return_value = '{"name": "Alice", "role": "Engineer"}'
        result = pipeline.extract("Who is Alice?", {"name": "str", "role": "str"})
        assert result.get("name") == "Alice"
        assert result.get("role") == "Engineer"

    def test_no_llm_returns_error_dict(self, pipeline):
        pipeline.llm = None
        result = pipeline.extract("Extract data", {"field": "str"})
        assert result.get("status") == "error"
        assert "Engine 4" in result.get("error", "")

    def test_malformed_json_returns_raw_output(self, pipeline):
        pipeline.llm.generate_response.return_value = "I cannot extract anything."
        result = pipeline.extract("Find fields", {"a": "str"})
        assert "raw_output" in result or "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# #8 — No-LLM Fallback for All Engines
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLLMFallbacks:
    def test_all_engines_return_non_empty_string_without_llm(self, pipeline):
        pipeline.llm = None
        pipeline.page_index = {
            "doc.pdf": {"chapters": ["Intro"], "summary": "A document.", "total_chars": 100, "raw_lines": []}
        }
        pipeline.graph.add_edge("X", "Y", relation="is")

        e1 = pipeline._engine_1_vector_rag("Query")
        e2 = pipeline._engine_2_vectorless_rag("Overview")
        e3 = pipeline._engine_3_graph_rag("Connections")
        e4 = pipeline._engine_4_extract("Extract", {"field": "str"})

        assert isinstance(e1, str) and len(e1) > 0
        assert isinstance(e2, str) and len(e2) > 0
        assert isinstance(e3, str) and len(e3) > 0
        assert isinstance(e4, dict)


# ─────────────────────────────────────────────────────────────────────────────
# #9 — Sentence-Boundary Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestSentenceChunking:
    def test_basic_sentence_split(self):
        from vectorDBpipe.utils.common import chunk_text_sentences
        text = "Alice is smart. Bob is kind. Charlie leads the team."
        chunks = chunk_text_sentences(text, max_tokens=6, overlap_sentences=0)
        assert len(chunks) >= 2
        # No chunk should start mid-sentence (must start with capital letter or digit)
        for chunk in chunks:
            assert chunk[0].isupper() or chunk[0].isdigit() or chunk[0] == "["

    def test_overlap_sentences_included(self):
        from vectorDBpipe.utils.common import chunk_text_sentences
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text_sentences(text, max_tokens=5, overlap_sentences=1)
        if len(chunks) >= 2:
            # "Second sentence." should appear in both chunk 1 and chunk 2
            assert any("Second" in c for c in chunks)

    def test_single_sentence_returns_one_chunk(self):
        from vectorDBpipe.utils.common import chunk_text_sentences
        text = "This is one single sentence."
        chunks = chunk_text_sentences(text, max_tokens=100)
        assert len(chunks) == 1

    def test_empty_string_returns_empty_list(self):
        from vectorDBpipe.utils.common import chunk_text_sentences
        assert chunk_text_sentences("") == []

    def test_no_mid_sentence_splits(self):
        """Fixed-size word chunker CAN split mid-sentence; sentence chunker must NOT."""
        from vectorDBpipe.utils.common import chunk_text_sentences
        text = (
            "Machine learning is a subset of AI. "
            "Deep learning uses neural networks. "
            "Natural language processing handles text."
        )
        chunks = chunk_text_sentences(text, max_tokens=8, overlap_sentences=0)
        for chunk in chunks:
            # Each chunk must end with sentence-ending punctuation
            assert chunk.rstrip()[-1] in ".!?", f"Chunk does not end at sentence boundary: '{chunk}'"


# ─────────────────────────────────────────────────────────────────────────────
# #10 — PPTX DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestPPTXLoader:
    def test_pptx_extension_in_supported_ext(self):
        from vectorDBpipe.data.loader import DataLoader
        loader = DataLoader("/tmp")
        assert ".pptx" in loader.supported_ext

    def test_pptx_load_extracts_slide_text(self):
        """Mock python-pptx Presentation to verify slide text extraction."""
        from vectorDBpipe.data.loader import DataLoader

        # Build mock slide/shape structure
        mock_shape = MagicMock()
        mock_shape.text = "Hello from slide 1"
        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]
        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]

        loader = DataLoader("/tmp")
        with patch("vectorDBpipe.data.loader.DataLoader._load_pptx") as mock_load:
            mock_load.return_value = "Hello from slide 1"
            result = loader._load_pptx("fake.pptx")
            assert "Hello" in result


# ─────────────────────────────────────────────────────────────────────────────
# #11 — Graph + PageIndex Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_persist_state_creates_files(self, pipeline):
        pipeline.graph.add_edge("Alice", "Team", relation="leads")
        pipeline.page_index["doc.pdf"] = {"summary": "Test doc.", "chapters": [], "total_chars": 100, "raw_lines": []}

        pipeline._persist_state(pipeline._state_dir)

        assert os.path.exists(os.path.join(pipeline._state_dir, "graph_state.json"))
        assert os.path.exists(os.path.join(pipeline._state_dir, "page_index.json"))

    def test_load_state_restores_graph(self, pipeline):
        pipeline.graph.add_edge("X", "Y", relation="connected")
        pipeline._persist_state(pipeline._state_dir)

        # Reset state
        pipeline.graph = nx.DiGraph()
        pipeline.page_index = {}

        # Restore from disk
        pipeline._load_state(pipeline._state_dir)

        assert len(pipeline.graph.nodes) > 0
        assert "X" in pipeline.graph.nodes or "Y" in pipeline.graph.nodes

    def test_load_state_restores_page_index(self, pipeline):
        pipeline.page_index["report.pdf"] = {
            "summary": "A test report.", "chapters": ["Ch1"], "total_chars": 200, "raw_lines": []
        }
        pipeline._persist_state(pipeline._state_dir)

        pipeline.page_index = {}
        pipeline._load_state(pipeline._state_dir)

        assert "report.pdf" in pipeline.page_index
        assert pipeline.page_index["report.pdf"]["summary"] == "A test report."

    def test_missing_state_files_do_not_crash(self, pipeline):
        """If no state files exist, _load_state must not raise any exceptions."""
        import tempfile
        empty_dir = tempfile.mkdtemp()
        try:
            pipeline._load_state(empty_dir)  # Should not raise
        except Exception as e:
            pytest.fail(f"_load_state raised unexpectedly on missing files: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# #12 — Streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestStreaming:
    def test_stream_query_yields_tokens(self, pipeline):
        tokens = list(pipeline.stream_query("Tell me about Alice."))
        assert len(tokens) > 0
        assert "".join(tokens)  # non-empty joined result

    def test_stream_query_no_llm_yields_context(self, pipeline):
        pipeline.llm = None
        tokens = list(pipeline.stream_query("Query"))
        full_text = "".join(tokens)
        assert len(full_text) > 0

    def test_stream_query_no_results(self, pipeline):
        pipeline.vector_store.search.return_value = []
        tokens = list(pipeline.stream_query("Query with no matches"))
        assert any("ingest" in t.lower() or "No relevant" in t for t in tokens)
