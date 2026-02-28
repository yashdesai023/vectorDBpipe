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
        # Store config_override for self-sufficient re-initialization
        self._config_override = config_override or {}

        try:
            super().__init__(config_path, config_override)
        except TypeError:
            # Old parent without config_override support — call with just config_path
            super().__init__(config_path)

        # --- Defensive attribute initialization ---
        if not hasattr(self, 'llm'):
            self.llm = None
        if not hasattr(self, 'embedder'):
            self.embedder = None
        if not hasattr(self, 'vector_store'):
            self.vector_store = None
        if not hasattr(self, 'loader'):
            data_dir = self._config_override.get('paths', {}).get('data_dir', 'data/')
            self.loader = DataLoader(data_dir)

        # Re-initialize any provider that the old parent misconfigured (e.g., embedder with model=None)
        self._safe_reinit()

        self.logger.info("Initializing VDBpipe (Omni-RAG) Architecture...")

        # Initialize the Local Knowledge Graph (NetworkX) for GraphRAG
        self.graph = nx.DiGraph()
        self.page_index = {}  # Vectorless Document Structure Store

    def _safe_reinit(self):
        """
        Re-initializes all providers completely using the Omni-RAG configuration schema
        (embedding, database, llm) since the old TextPipeline parent uses legacy keys
        (model, vectordb) which causes it to misconfigure components (e.g., missing model_name).
        """
        cfg = getattr(self, 'config', None)
        if not cfg:
            return  # Config not ready

        # --- 1. Re-initialize Embedder ---
        provider = (cfg.get('embedding') or {}).get('provider', 'local').lower()
        model_name = (cfg.get('embedding') or {}).get('model_name', 'all-MiniLM-L6-v2')
        if provider in ['local', 'huggingface', '']:
            try:
                from vectorDBpipe.embeddings.embedder import Embedder
                self.embedder = Embedder(model_name=model_name)
                self.logger.info(f"VDBpipe initialized embedder: {model_name}")
            except Exception as e:
                self.logger.warning(f"Embedder init failed: {e}")

        # --- 2. Re-initialize Vector Store ---
        db_cfg = cfg.get('database') or {}
        db_provider = db_cfg.get('provider', 'faiss').lower()
        collection = db_cfg.get('collection_name', 'default_collection')
        mode = db_cfg.get('mode', 'local')
        save_dir = (cfg.get('paths') or {}).get('persistent_db', 'vector_dbs')

        try:
            if db_provider == 'faiss':
                from vectorDBpipe.vectordb.faiss_client import FaissDatabase
                self.vector_store = FaissDatabase(
                    collection_name=collection, mode=mode, save_dir=save_dir)
            elif db_provider in ['chroma', 'chromadb']:
                from vectorDBpipe.vectordb.chroma_client import ChromaDatabase
                self.vector_store = ChromaDatabase(
                    collection_name=collection, mode=mode, save_dir=save_dir)
            self.logger.info(f"VDBpipe initialized vector store: {db_provider}")
        except Exception as e:
            self.logger.warning(f"Vector store init failed: {e}")

        # --- 3. Re-initialize LLM ---
        llm_cfg = cfg.get('llm') or {}
        llm_provider = llm_cfg.get('provider', 'null').lower()
        if llm_provider not in ['null', 'none', '']:
            try:
                llm_model = llm_cfg.get('model_name', 'gpt-4o-mini')
                llm_key = llm_cfg.get('api_key') or os.environ.get('OPENAI_API_KEY')
                if llm_provider == 'openai':
                    from vectorDBpipe.llms.openai_client import OpenAILLMProvider
                    self.llm = OpenAILLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider == 'groq':
                    from vectorDBpipe.llms.groq_client import GroqLLMProvider
                    self.llm = GroqLLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider == 'anthropic':
                    from vectorDBpipe.llms.anthropic_client import AnthropicLLMProvider
                    self.llm = AnthropicLLMProvider(model_name=llm_model, api_key=llm_key)
                self.logger.info(f"VDBpipe initialized LLM: {llm_provider}")
            except Exception as e:
                self.logger.warning(f"LLM init failed: {e}")
        else:
            self.llm = None  # Explicitly disable LLM if null

        # --- 4. Re-initialize Loader ---
        data_dir = (cfg.get('paths') or {}).get('data_dir', 'data/')
        self.loader = DataLoader(data_dir)

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

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Phase 1: Vector Chunking (always runs)
                chunk_future = executor.submit(chunk_text, cleaned, 512)

                # Phase 2 & 3: PageIndex + Graph Extraction (always runs — LLM optional)
                extraction_future = executor.submit(
                    self._extract_structure_and_graph, source, cleaned[:2000])

                chunks = chunk_future.result()
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

    def _embed_and_store(self, chunks, docs, metadata):
        """
        VDBpipe override of _embed_and_store.
        Version-safe: works even if old TextPipeline didn't initialize embedder/vector_store.
        """
        embedder = getattr(self, 'embedder', None)
        vector_store = getattr(self, 'vector_store', None)

        if embedder is None or vector_store is None:
            self.logger.warning("Embedder or vector store not initialized — skipping vector storage.")
            return

        try:
            embeddings = embedder.embed_batch(chunks)
            vector_store.add(embeddings=embeddings, documents=docs, metadata=metadata)
        except Exception as e:
            self.logger.warning(f"embed_and_store failed: {e}")

    def _extract_structure_and_graph(self, source: str, content_sample: str):
        """
        Phase 2: Always builds the PageIndex (no LLM needed).
        Phase 3: Extracts graph relationships using LLM (only if configured).
        """
        try:
            # ── Phase 2: Structural PageIndex (always runs, no LLM needed) ──
            # Split the content into rough sections by newline clusters
            lines = [l.strip() for l in content_sample.split('\n') if l.strip()]
            headings = [l for l in lines if l.startswith('#') or l.isupper()]
            summary = content_sample[:300].replace('\n', ' ')
            self.page_index[source] = {
                "chapters": headings[:5] if headings else ["Section 1", "Section 2"],
                "summary": summary,
                "total_chars": len(content_sample)
            }

            # ── Phase 3: Graph Extraction (only if LLM is configured) ──
            llm = getattr(self, 'llm', None)
            if llm:
                prompt = (
                    f"Extract 2 entity relationships from this text. "
                    f"Format EACH as 'Entity1|Relationship|Entity2' on its own line.\n"
                    f"Text: {content_sample[:500]}"
                )
                response = llm.generate_response(
                    system_prompt="You are a data extractor. Reply only with the formatted lines.",
                    user_query=prompt
                )
                for line in response.split('\n'):
                    parts = line.strip().split('|')
                    if len(parts) == 3:
                        self.graph.add_edge(
                            parts[0].strip(), parts[2].strip(),
                            relation=parts[1].strip()
                        )
        except Exception as e:
            self.logger.warning(f"Extraction failed for {source}: {e}")

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

    def search(self, query: str, top_k: int = 5):
        """
        Version-safe search. Uses embedder + vector_store if available,
        otherwise returns an empty list gracefully.
        """
        embedder = getattr(self, 'embedder', None)
        vector_store = getattr(self, 'vector_store', None)
        if embedder is None or vector_store is None:
            return []
        try:
            query_embedding = embedder.embed_text(query)
            return vector_store.search(query_embedding, top_k=top_k)
        except Exception as e:
            self.logger.warning(f"Search failed: {e}")
            return []

    def query_with_llm(self, user_query: str) -> str:
        """
        Version-safe RAG generation. Overrides parent to ensure it always exists.
        Searches the vector store, builds context, and calls the LLM.
        Falls back to returning the raw retrieved text if no LLM is configured.
        """
        llm = getattr(self, 'llm', None)

        # Retrieve relevant chunks
        results = self.search(user_query, top_k=3)

        if not results:
            return "No relevant information found in the knowledge base. Please run ingest() first."

        # Build context string from results
        context = "\n\n---\n\n".join(
            [r.get('document', '') for r in results if r.get('document')]
        )

        # If no LLM configured, return the raw context
        if llm is None:
            return f"[Retrieved Context — configure an LLM for generated answers]\n\n{context}"

        try:
            system_prompt = (
                "You are an intelligent documentation assistant. "
                "Answer the user's question using only the provided context."
            )
            return llm.generate_response(
                system_prompt=system_prompt,
                user_query=user_query,
                retrieved_context=context
            )
        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}")
            return context

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
