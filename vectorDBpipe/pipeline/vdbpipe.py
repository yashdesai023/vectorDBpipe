import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional

from vectorDBpipe.config.config_manager import ConfigManager
from vectorDBpipe.data.loader import DataLoader
from vectorDBpipe.utils.common import clean_text, chunk_text
from vectorDBpipe.logger.logging import setup_logger
from vectorDBpipe.pipeline.text_pipeline import TextPipeline  # We'll build on top of the original logic

# LangChain structured output
from langchain_core.prompts import ChatPromptTemplate
from pydantic import create_model, BaseModel

class VDBpipe(TextPipeline):
    """
    VDBpipe: The core Omni-RAG orchestrator for vectorDBpipe.
    Handles Tri-Processing Ingestion and intelligent 4-Engine Routing.
    """
    
    def __init__(self, config_path="config.yaml", config_override=None):
        super().__init__(config_path, config_override)
        self.logger.info("Initializing VDBpipe (Omni-RAG) Architecture...")
        
        # Initialize the Local Knowledge Graph (NetworkX) for GraphRAG
        self.graph = nx.DiGraph()
        self.page_index = {} # Vectorless Document Structure Store
        
    def ingest(self, data_path: str, batch_size: int = 100):
        """
        The Tri-Processing Ingestion Engine.
        Processes data into vectors, structural indexes, and a graph.
        """
        self.logger.info(f"Starting Omni-Ingestion for: {data_path}")
        self.loader.data_path = data_path
        documents = self.loader.load_data()
        
        if not documents:
            self.logger.warning("No documents found to ingest.")
            return

        chunk_batch, docs_batch, meta_batch = [], [], []
        total_chunks = 0
        
        for doc in documents:
            content, source = doc.get("content"), doc.get("source")
            if not content: continue
            
            cleaned = clean_text(content)
            
            # --- OPTIMIZATION: Concurrent Tri-Processing Execution ---
            # To massively boost speed, we run the slow LLM Extractions (Phase 2/3) 
            # in parallel with the CPU-bound Vector Chunking (Phase 1)
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Fire off Phase 2 & 3 (Structural & Graph Extraction) if LLM exists
                extraction_future = None
                if self.llm:
                    extraction_future = executor.submit(self._extract_structure_and_graph, source, cleaned[:2000])
                
                # Phase 1: Traditional Vector Chunking (Runs in parallel)
                chunk_future = executor.submit(chunk_text, cleaned, 512)
                chunks = chunk_future.result()
                
                # Wait for LLM Extractions to finish
                if extraction_future:
                    extraction_future.result()
            
            chunk_batch.extend(chunks)
            docs_batch.extend(chunks)
            meta_batch.extend([{"source": source}] * len(chunks))
            
            if len(chunk_batch) >= batch_size:
                self._embed_and_store(chunk_batch, docs_batch, meta_batch)
                total_chunks += len(chunk_batch)
                chunk_batch, docs_batch, meta_batch = [], [], []
                
        if chunk_batch:
            self._embed_and_store(chunk_batch, docs_batch, meta_batch)
            total_chunks += len(chunk_batch)

        self.logger.info(f"Omni-Ingestion complete! Embedded {total_chunks} chunks. Extracted {len(self.graph.nodes)} Graph Nodes.")
        return total_chunks

    def _extract_structure_and_graph(self, source: str, content_sample: str):
        """Mock extraction of structures and entities to populate the PageIndex and Knowledge Graph."""
        try:
            # 1. Structural Phase (PageIndex)
            self.page_index[source] = {"chapters": ["Introduction", "Main Body", "Conclusion"], "summary": content_sample[:200]}
            
            # 2. Graph Phase (GraphRAG)
            prompt = f"Extract 2 entity relationships from this text formatted rigidly as 'Entity1|Relationship|Entity2'. Text: {content_sample[:500]}"
            response = self.llm.generate_response(system_prompt="You are a data extractor.", user_query=prompt)
            lines = [line for line in response.split("\\n") if "|" in line]
            for line in lines:
                parts = line.split("|")
                if len(parts) == 3:
                    self.graph.add_edge(parts[0].strip(), parts[2].strip(), relation=parts[1].strip())
        except Exception as e:
            self.logger.warning(f"Failed graph extraction on {source}: {e}")

    def query(self, user_query: str) -> str:
        """
        The OmniRouter. Identifies intent, selects engine 1, 2, or 3, and routes the query.
        """
        engine = self._route_query(user_query)
        self.logger.info(f"OmniRouter selected: {engine}")

        if engine == "ENGINE_1":
            return self._engine_1_vector_rag(user_query)
        elif engine == "ENGINE_2":
            return self._engine_2_vectorless_rag(user_query)
        elif engine == "ENGINE_3":
            return self._engine_3_graph_rag(user_query)
        else:
            return self._engine_1_vector_rag(user_query) # Fallback

    def extract(self, query: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """Engine 4: LangChain Extract"""
        self.logger.info("OmniRouter selected: ENGINE_4 (LangChain Extract)")
        return self._engine_4_extract(query, schema)

    def _route_query(self, query: str) -> str:
        """Determines which engine should handle the query based on heuristics or LLM."""
        q = query.lower()
        if "summarize" in q or "overall tone" in q or "chapter" in q:
            return "ENGINE_2"
        elif "connected" in q or "relationship" in q or "how is" in q:
            return "ENGINE_3"
        return "ENGINE_1"

    def _engine_1_vector_rag(self, query: str) -> str:
        """Fast factual lookup using standard Vector DB."""
        return self.query_with_llm(query)

    def _engine_2_vectorless_rag(self, query: str) -> str:
        """Holistic reading bypassing vectors using the PageIndex."""
        if not self.llm: return "LLM not configured."
        index_dump = json.dumps(self.page_index)
        sys_prompt = "You are a Vectorless RAG Agent. Read the provided page structures and answer fundamentally."
        return self.llm.generate_response(sys_prompt, user_query=query, retrieved_context=index_dump)

    def _engine_3_graph_rag(self, query: str) -> str:
        """Traversal over NetworkX Graph for multi-hop reasoning."""
        if not self.llm: return "LLM not configured."
        edges = list(self.graph.edges(data=True))
        graph_dump = "\\n".join([f"{u} -> {d['relation']} -> {v}" for u, v, d in edges[:20]])
        if not graph_dump:
            return "Knowledge graph is currently empty. Run ingestion first."
        sys_prompt = "You are a GraphRAG Detective. Use the provided Knowledge Graph relationships to answer reasoning questions."
        return self.llm.generate_response(sys_prompt, user_query=query, retrieved_context=graph_dump)

    def _engine_4_extract(self, query: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """Structured output generation using pseudo-LangChain formatting."""
        if not self.llm: return {"error": "LLM not configured."}
        sys_prompt = f"Extract information based on query. Return ONLY a valid JSON string matching this schema types: {json.dumps(schema)}"
        try:
            # Note: A pure implementation would use langchain_core.with_structured_output.
            # We are mimicking it here to gracefully support any LLM provider bound to `self.llm`
            response = self.llm.generate_response(sys_prompt, user_query=query)
            # Find JSON block
            import re
            match = re.search(r'\\{.*\\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"raw_output": response}
        except Exception as e:
            return {"error": str(e)}
