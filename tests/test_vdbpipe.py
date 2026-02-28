import pytest
import os
from unittest.mock import patch, MagicMock
from vectorDBpipe import VDBpipe

@pytest.fixture
def dummy_pipeline():
    """Returns a VDBpipe instance with mocked LLM/DB/Embeddings to avoid API calls during tests."""
    with patch('vectorDBpipe.pipeline.vdbpipe.TextPipeline._init_embedding_provider'), \
         patch('vectorDBpipe.pipeline.vdbpipe.TextPipeline._init_database_provider'), \
         patch('vectorDBpipe.pipeline.vdbpipe.TextPipeline._init_llm_provider'):
        pipeline = VDBpipe()
        # Mock the underlying components
        pipeline.llm = MagicMock()
        pipeline.vector_store = MagicMock()
        pipeline.embedder = MagicMock()
        return pipeline

def test_vdbpipe_initialization(dummy_pipeline):
    """Test that the VDBpipe orchestrator initializes correctly."""
    assert dummy_pipeline is not None
    assert hasattr(dummy_pipeline, 'graph')
    assert hasattr(dummy_pipeline, 'page_index')
    assert hasattr(dummy_pipeline, 'ingest')
    assert hasattr(dummy_pipeline, 'query')

@patch('vectorDBpipe.data.loader.DataLoader.load_data')
def test_vdbpipe_ingest_tri_processing(mock_loader, dummy_pipeline):
    """Test the ingest method runs the 3 phases of Omni-RAG processing."""
    
    # Mock documents returned by the loader
    mock_doc = {
        "content": "This is a test document about artificial intelligence.",
        "source": "test.txt"
    }
    mock_loader.return_value = [mock_doc]
    
    # Mock the LLM structural and graph extraction responses
    dummy_pipeline.llm.generate.side_effect = [
        '{"title": "Test Doc", "sections": ["AI intro"]}',  # Structural PageIndex Phase
        '{"entities": ["AI"], "relations": []}'           # GraphRAG Phase
    ]
    
    # Run ingestion
    dummy_pipeline.ingest("dummy_path")
    
    # Verify the loader was called with no arguments (data_path is pre-set as attribute)
    mock_loader.assert_called_once()
    
    # Verify data_path was set on the loader before calling load_data
    assert dummy_pipeline.loader.data_path == "dummy_path"
    
    # Verify Structural PageIndex Phase updated state
    assert isinstance(dummy_pipeline.page_index, dict)
    
    # Verify Graph Phase updated state
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
    with patch.object(dummy_pipeline, 'query_with_llm') as mock_query:
        mock_query.return_value = "Mocked LLM answer"
        
        result = dummy_pipeline._engine_1_vector_rag("Test query")
        
        mock_query.assert_called_once()
        assert result == "Mocked LLM answer"
