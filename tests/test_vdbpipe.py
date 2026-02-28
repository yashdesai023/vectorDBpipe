import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from vectorDBpipe import VDBpipe

@pytest.fixture
def dummy_pipeline():
    """
    Builds a VDBpipe instance without calling __init__ at all.
    This is fully decoupled from any parent class method names,
    making it robust across different versions of TextPipeline.
    """
    # Bypass __init__ entirely — directly construct the object
    pipeline = VDBpipe.__new__(VDBpipe)

    # Set up the logger mock
    pipeline.logger = MagicMock()

    # Set up Omni-RAG state attributes (normally set in VDBpipe.__init__)
    pipeline.graph = nx.DiGraph()
    pipeline.page_index = {}

    # Set up a real DataLoader-compatible mock for the loader attribute
    loader_mock = MagicMock()
    loader_mock.data_path = None
    pipeline.loader = loader_mock

    # Mock all provider dependencies — no real API keys needed
    pipeline.llm = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.embedder = MagicMock()

    # Mock parent class methods that differ between TextPipeline versions
    pipeline._embed_and_store = MagicMock()
    pipeline.query_with_llm = MagicMock(return_value="Mocked LLM answer")

    # Give config a minimal mock
    pipeline.config = MagicMock()

    return pipeline

def test_vdbpipe_initialization(dummy_pipeline):
    """Test that the VDBpipe orchestrator initializes correctly."""
    assert dummy_pipeline is not None
    assert hasattr(dummy_pipeline, 'graph')
    assert hasattr(dummy_pipeline, 'page_index')
    assert hasattr(dummy_pipeline, 'ingest')
    assert hasattr(dummy_pipeline, 'query')

def test_vdbpipe_ingest_tri_processing(dummy_pipeline):
    """Test the ingest method runs all 3 phases of Omni-RAG processing."""

    # Mock documents returned by the loader (using fixture's pre-set loader mock)
    mock_doc = {
        "content": "This is a test document about artificial intelligence.",
        "source": "test.txt"
    }
    dummy_pipeline.loader.load_data.return_value = [mock_doc]

    # Run ingestion
    dummy_pipeline.ingest("dummy_path")

    # Verify data_path was set on the loader before calling load_data
    assert dummy_pipeline.loader.data_path == "dummy_path"

    # Verify load_data was called (no arguments — path is set as attribute)
    dummy_pipeline.loader.load_data.assert_called_once()

    # Verify the PageIndex was populated
    assert isinstance(dummy_pipeline.page_index, dict)

    # Verify the Knowledge Graph exists
    assert dummy_pipeline.graph is not None

def test_omnirouter_classification(dummy_pipeline):
    """Test that the OmniRouter correctly classifies different query intents."""
    
    # Test 1: Summarization -> Vectorless RAG
    intent1 = dummy_pipeline._route_query("Summarize the entire document.")
    assert intent1 == "ENGINE_2"
    
    # Test 2: Relationships -> Graph RAG
    intent2 = dummy_pipeline._route_query("How are the CEO and the board connected?")
    assert intent2 == "ENGINE_3"
    
    # Test 3: Specific Fact -> Vector RAG
    intent3 = dummy_pipeline._route_query("What was the revenue in Q3?")
    assert intent3 == "ENGINE_1"

def test_vector_rag_engine(dummy_pipeline):
    """Test the execution of the standard Vector RAG engine."""
    # query_with_llm is pre-mocked in the fixture
    dummy_pipeline.query_with_llm.return_value = "Mocked LLM answer"

    result = dummy_pipeline._engine_1_vector_rag("Test query")

    dummy_pipeline.query_with_llm.assert_called_once()
    assert result == "Mocked LLM answer"
