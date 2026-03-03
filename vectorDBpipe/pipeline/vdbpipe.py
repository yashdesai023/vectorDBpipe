import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional

from vectorDBpipe.config.config_manager import ConfigManager
from vectorDBpipe.data.loader import DataLoader
from vectorDBpipe.utils.common import clean_text, chunk_text
from vectorDBpipe.logger.logging import setup_logger

# LangChain structured output
from langchain_core.prompts import ChatPromptTemplate
from pydantic import create_model, BaseModel


class VDBpipe:
    """
    VDBpipe: The core Omni-RAG orchestrator for vectorDBpipe.

    Implements pure composition (no inheritance from TextPipeline).
    Handles Tri-Processing Ingestion and intelligent 4-Engine Routing via
    a Semantic OmniRouter (embedding-cosine similarity intent classification).

    Persistence: graph + page_index are automatically saved to disk after
    every ingest() call and reloaded on construction.
    """

    # Semantic prototype phrases for intent routing (embedded once per instance)
    _INTENT_PROTOTYPES = {
        "ENGINE_2": [
            "summarize the document",
            "give me an overview",
            "what is the overall tone",
            "high level summary",
            "what are the main chapters",
            "tldr of the document",
            "what is the gist",
        ],
        "ENGINE_3": [
            "how are entities connected",
            "what is the relationship between",
            "how is x related to y",
            "find connections between",
            "trace the path from",
            "what links these concepts",
            "multi hop reasoning",
        ],
    }

    def __init__(self, config_path: str = "config.yaml", config_override: Optional[Dict] = None):
        self._config_override = config_override or {}

        # ── Load config ──────────────────────────────────────────────────────
        self.config = ConfigManager(config_path, config_override=config_override)
        cfg = self.config  # alias

        # ── Setup logger ──────────────────────────────────────────────────────
        log_cfg = cfg.get("logging") or {}
        self.logger = setup_logger(
            name="VDBpipe",
            log_dir=cfg.get("paths.logs_dir") or "logs/",
            level=log_cfg.get("level", "INFO"),
        )
        self.logger.info("Initializing VDBpipe (Omni-RAG) — pure composition mode")

        # ── Paths ─────────────────────────────────────────────────────────────
        paths_cfg = cfg.get("paths") or {}
        self._data_dir = self._config_override.get("paths", {}).get("data_dir") or paths_cfg.get("data_dir") or "data/"
        self._state_dir = (
            self._config_override.get("paths", {}).get("persistent_db")
            or paths_cfg.get("persistent_db")
            or "vector_dbs"
        )
        os.makedirs(self._state_dir, exist_ok=True)

        # ── 1. DataLoader ─────────────────────────────────────────────────────
        data_dir = (
            self._config_override.get("paths", {}).get("source_data")
            or self._data_dir
        )
        self.loader = DataLoader(data_dir)

        # ── 2. Embedder ───────────────────────────────────────────────────────
        self.embedder = None
        embed_cfg = (cfg.get("embedding") or {})
        embed_provider = embed_cfg.get("provider", "local").lower()
        embed_model = embed_cfg.get("model_name", "all-MiniLM-L6-v2")

        if embed_provider in ["local", "huggingface", ""]:
            try:
                from vectorDBpipe.embeddings.embedder import Embedder
                self.embedder = Embedder(model_name=embed_model)
                self.logger.info(f"Embedder initialized: {embed_model}")
            except Exception as e:
                self.logger.warning(f"Embedder init failed: {e}")

        # ── 3. Vector Store ───────────────────────────────────────────────────
        self.vector_store = None
        db_cfg = cfg.get("database") or {}
        db_provider = db_cfg.get("provider", "faiss").lower()
        collection = db_cfg.get("collection_name", "default_collection")
        mode = db_cfg.get("mode", "local")
        save_dir = self._state_dir

        try:
            if db_provider == "faiss":
                from vectorDBpipe.vectordb.faiss_client import FaissDatabase
                self.vector_store = FaissDatabase(
                    collection_name=collection, mode=mode, save_dir=save_dir
                )
            elif db_provider in ["chroma", "chromadb"]:
                from vectorDBpipe.vectordb.chroma_client import ChromaDatabase
                self.vector_store = ChromaDatabase(
                    collection_name=collection, mode=mode, save_dir=save_dir
                )
            self.logger.info(f"Vector store initialized: {db_provider}")
        except Exception as e:
            self.logger.warning(f"Vector store init failed: {e}")

        # ── 4. LLM Client ─────────────────────────────────────────────────────
        self.llm = None
        llm_cfg = cfg.get("llm") or {}
        llm_provider = llm_cfg.get("provider", "null").lower()

        if llm_provider not in ["null", "none", ""]:
            try:
                llm_model = llm_cfg.get("model_name", "gpt-4o-mini")
                llm_key = llm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
                if llm_provider == "openai":
                    from vectorDBpipe.llms.openai_client import OpenAILLMProvider
                    self.llm = OpenAILLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider == "groq":
                    from vectorDBpipe.llms.groq_client import GroqLLMProvider
                    self.llm = GroqLLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider == "anthropic":
                    from vectorDBpipe.llms.anthropic_client import AnthropicLLMProvider
                    self.llm = AnthropicLLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider == "sarvam":
                    from vectorDBpipe.llms.sarvam_client import SarvamLLMProvider
                    self.llm = SarvamLLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider in ["google", "gemini"]:
                    from vectorDBpipe.llms.google_client import GoogleLLMProvider
                    self.llm = GoogleLLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider == "cohere":
                    from vectorDBpipe.llms.cohere_client import CohereLLMProvider
                    self.llm = CohereLLMProvider(model_name=llm_model, api_key=llm_key)
                elif llm_provider in ["deepseek", "mistral", "openai-compat"]:
                    from vectorDBpipe.llms.openai_compat_clients import OpenAICompatLLMProvider
                    self.llm = OpenAICompatLLMProvider(
                        provider=llm_provider, model_name=llm_model, api_key=llm_key
                    )
                self.logger.info(f"LLM initialized: {llm_provider} / {llm_model}")
            except Exception as e:
                self.logger.warning(f"LLM init failed: {e}")

        # ── 5. Omni-RAG State ─────────────────────────────────────────────────
        self.graph = nx.DiGraph()
        self.page_index: Dict[str, Any] = {}

        # ── 6. Semantic Router — pre-compute intent embeddings ─────────────────
        self._intent_embeddings: Dict[str, Any] = {}
        self._init_semantic_router()

        # ── 7. Load persisted state if available ─────────────────────────────
        self._load_state(self._state_dir)

    # =========================================================================
    # SEMANTIC OMNIROUTER
    # =========================================================================

    def _init_semantic_router(self):
        """
        Pre-compute mean embeddings for each intent category so routing
        is a single cosine similarity call per query — no LLM needed.
        Falls back gracefully if embedder is unavailable.
        """
        if self.embedder is None:
            self.logger.info("Semantic router disabled — no embedder configured. Using keyword fallback.")
            return

        try:
            import numpy as np
            for engine, phrases in self._INTENT_PROTOTYPES.items():
                vecs = self.embedder.embed_batch(phrases)              # list[list[float]]
                mean_vec = np.mean(np.array(vecs, dtype="float32"), axis=0)
                norm = np.linalg.norm(mean_vec)
                self._intent_embeddings[engine] = mean_vec / norm if norm > 0 else mean_vec
            self.logger.info("Semantic OmniRouter intent embeddings computed.")
        except Exception as e:
            self.logger.warning(f"Semantic router init failed: {e}. Falling back to keyword mode.")
            self._intent_embeddings = {}

    def _route_query(self, query: str) -> str:
        """
        Semantic OmniRouter:
        1. Embeds the user query.
        2. Computes cosine similarity against per-engine intent prototypes.
        3. Routes to the highest-similarity engine (if above threshold=0.35).
        4. Falls back to keyword heuristics if semantic router is unavailable.
        """
        # ── Semantic path (preferred) ──────────────────────────────────────
        if self._intent_embeddings and self.embedder is not None:
            try:
                import numpy as np
                THRESHOLD = 0.35  # min cosine similarity to accept a routing decision

                q_vec = np.array(self.embedder.embed_text(query), dtype="float32")
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                best_engine = "ENGINE_1"
                best_score = -1.0

                for engine, proto_vec in self._intent_embeddings.items():
                    score = float(np.dot(q_vec, proto_vec))
                    self.logger.info(f"[SemanticRouter] {engine} similarity={score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_engine = engine

                if best_score >= THRESHOLD and best_engine != "ENGINE_1":
                    self.logger.info(
                        f"[SemanticRouter] → {best_engine} (score={best_score:.3f}, threshold={THRESHOLD})"
                    )
                    return best_engine
                else:
                    self.logger.info(
                        f"[SemanticRouter] → ENGINE_1 (best={best_score:.3f} < threshold={THRESHOLD})"
                    )
                    return "ENGINE_1"
            except Exception as e:
                self.logger.warning(f"[SemanticRouter] Error during routing: {e}. Using keyword fallback.")

        # ── Keyword fallback (when embedder not available) ─────────────────
        q = query.lower()
        summarize_kw = {"summarize", "summary", "overall tone", "chapter", "overview", "gist", "tldr", "tl;dr"}
        graph_kw = {"connected", "relationship", "how is", "related to", "links", "connection", "path from"}

        if any(kw in q for kw in summarize_kw):
            return "ENGINE_2"
        if any(kw in q for kw in graph_kw):
            return "ENGINE_3"
        return "ENGINE_1"

    # =========================================================================
    # PERSISTENCE — Graph + PageIndex
    # =========================================================================

    def _persist_state(self, save_dir: str):
        """
        Serialize graph (NetworkX → node-link JSON) and page_index (dict → JSON)
        to disk so they survive server restarts.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            # Graph
            graph_path = os.path.join(save_dir, "graph_state.json")
            graph_data = nx.node_link_data(self.graph)
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            # PageIndex
            pi_path = os.path.join(save_dir, "page_index.json")
            with open(pi_path, "w", encoding="utf-8") as f:
                json.dump(self.page_index, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"State persisted → {graph_path} ({len(self.graph.nodes)} nodes) | "
                f"{pi_path} ({len(self.page_index)} docs)"
            )
        except Exception as e:
            self.logger.warning(f"State persistence failed: {e}")

    def _load_state(self, save_dir: str):
        """
        Restore graph and page_index from disk if saved state files exist.
        """
        try:
            graph_path = os.path.join(save_dir, "graph_state.json")
            if os.path.exists(graph_path):
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                self.graph = nx.node_link_graph(graph_data)
                self.logger.info(
                    f"Restored graph from disk: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"
                )

            pi_path = os.path.join(save_dir, "page_index.json")
            if os.path.exists(pi_path):
                with open(pi_path, "r", encoding="utf-8") as f:
                    self.page_index = json.load(f)
                self.logger.info(f"Restored page_index from disk: {len(self.page_index)} documents")
        except Exception as e:
            self.logger.warning(f"State restore failed (starting fresh): {e}")

    # =========================================================================
    # INGESTION — Tri-Processing
    # =========================================================================

    def ingest(self, data_path: str, batch_size: int = 100):
        """
        The Tri-Processing Ingestion Engine.
        Processes data into vectors (Phase 1), structural indexes (Phase 2),
        and a graph (Phase 3) — all in parallel via ThreadPoolExecutor.
        State is automatically persisted to disk after ingestion completes.
        """
        self.logger.info(f"Starting Omni-Ingestion for: {data_path}")
        self.loader.data_path = data_path
        documents = self.loader.load_data()

        if not documents:
            self.logger.warning("No documents found to ingest.")
            return 0

        chunk_batch, docs_batch, meta_batch = [], [], []
        total_chunks = 0

        import concurrent.futures

        for doc in documents:
            content, source = doc.get("content"), doc.get("source")
            if not content:
                continue

            cleaned = clean_text(content)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Phase 1: Vector Chunking (always runs)
                chunk_future = executor.submit(chunk_text, cleaned, 512)

                # Phase 2 & 3: PageIndex + Graph Extraction
                extraction_future = executor.submit(
                    self._extract_structure_and_graph, source, cleaned[:2000]
                )

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

        self.logger.info(
            f"Omni-Ingestion complete! Embedded {total_chunks} chunks. "
            f"Graph: {len(self.graph.nodes)} nodes. PageIndex: {len(self.page_index)} docs."
        )

        # ── Auto-persist state after every ingest ──────────────────────────
        self._persist_state(self._state_dir)

        return total_chunks

    def _embed_and_store(self, chunks, docs, metadata):
        """Embed a batch of text chunks and store in the vector store."""
        if self.embedder is None or self.vector_store is None:
            self.logger.warning("Embedder or vector store not initialized — skipping vector storage.")
            return
        try:
            embeddings = self.embedder.embed_batch(chunks)
            self.vector_store.add(embeddings=embeddings, documents=docs, metadata=metadata)
        except Exception as e:
            self.logger.warning(f"embed_and_store failed: {e}")

    def _extract_structure_and_graph(self, source: str, content_sample: str):
        """
        Phase 2: Builds the PageIndex (always, no LLM needed).
        Phase 3: Extracts graph relationships (LLM if available, regex fallback).
        """
        try:
            # ── Phase 2: Structural PageIndex ──────────────────────────────
            lines = [l.strip() for l in content_sample.split("\n") if l.strip()]
            headings = [l for l in lines if l.startswith("#") or l.isupper()]
            summary = content_sample[:300].replace("\n", " ")
            self.page_index[source] = {
                "chapters": headings[:5] if headings else lines[:3],
                "summary": summary,
                "total_chars": len(content_sample),
                "raw_lines": lines[:15],
            }

            # ── Phase 3: Graph Extraction ──────────────────────────────────
            llm = self.llm
            if llm:
                prompt = (
                    f"Extract up to 5 entity relationships from this text. "
                    f"Format EACH as 'Entity1|Relationship|Entity2' on its own line. "
                    f"No preamble, no explanation.\n"
                    f"Text: {content_sample[:800]}"
                )
                try:
                    response = llm.generate_response(
                        system_prompt="You are a knowledge graph extractor. Reply only with pipe-separated triplets.",
                        user_query=prompt,
                        retrieved_context="",
                    )
                    for line in response.split("\n"):
                        parts = line.strip().split("|")
                        if len(parts) == 3:
                            self.graph.add_edge(
                                parts[0].strip(),
                                parts[2].strip(),
                                relation=parts[1].strip(),
                            )
                except Exception as e:
                    self.logger.warning(f"LLM graph extraction failed for {source}: {e}")
                    self._regex_graph_extract(source, content_sample)
            else:
                self._regex_graph_extract(source, content_sample)
        except Exception as e:
            self.logger.warning(f"Extraction failed for {source}: {e}")

    def _regex_graph_extract(self, source: str, content_sample: str):
        """Fallback graph extraction using regex when LLM is absent."""
        import re
        relation_patterns = [
            (r"([A-Z][a-zA-Z ]{2,25}) is ([A-Z][a-zA-Z ]{2,25})", "is"),
            (r"([A-Z][a-zA-Z ]{2,25}) has ([A-Z][a-zA-Z ]{2,25})", "has"),
            (r"([A-Z][a-zA-Z ]{2,25}) includes ([A-Z][a-zA-Z ]{2,25})", "includes"),
            (r"([A-Z][a-zA-Z ]{2,25}) leads? ([A-Z][a-zA-Z ]{2,25})", "leads"),
            (r"([A-Z][a-zA-Z ]{2,25}) and ([A-Z][a-zA-Z ]{2,25})", "related_to"),
        ]
        added = 0
        for pattern, relation in relation_patterns:
            for match in re.finditer(pattern, content_sample):
                e1, e2 = match.group(1).strip(), match.group(2).strip()
                if e1 and e2 and e1 != e2:
                    self.graph.add_edge(e1, e2, relation=relation)
                    added += 1
                    if added >= 15:
                        return

    # =========================================================================
    # QUERY — OmniRouter + 4 Engines
    # =========================================================================

    def query(self, user_query: str) -> str:
        """
        The OmniRouter entry point. Routes the query to Engine 1, 2, or 3
        using Semantic Cosine Similarity intent classification.
        """
        engine = self._route_query(user_query)
        self.logger.info(f"OmniRouter selected: {engine}")

        if engine == "ENGINE_1":
            return self._engine_1_vector_rag(user_query)
        elif engine == "ENGINE_2":
            return self._engine_2_vectorless_rag(user_query)
        elif engine == "ENGINE_3":
            return self._engine_3_graph_rag(user_query)
        return self._engine_1_vector_rag(user_query)  # fallback

    def extract(self, query: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """Engine 4: Structured JSON extraction."""
        self.logger.info("OmniRouter selected: ENGINE_4 (Structured Extract)")
        return self._engine_4_extract(query, schema)

    # ── Engine 1: Vector RAG ───────────────────────────────────────────────
    def _engine_1_vector_rag(self, query: str) -> str:
        """Fast factual lookup using standard Vector DB similarity search."""
        return self.query_with_llm(query)

    # ── Engine 2: Vectorless RAG ───────────────────────────────────────────
    def _engine_2_vectorless_rag(self, query: str) -> str:
        """Holistic reading via PageIndex — bypasses vector search entirely."""
        if not self.page_index:
            return "PageIndex is empty. Please run ingest() first."
        index_dump = json.dumps(self.page_index, indent=2)
        if not self.llm:
            lines = []
            for src, data in self.page_index.items():
                lines.append(f"Source: {src}")
                lines.append(f"Summary: {data.get('summary', '')}")
                chaps = data.get("chapters", [])
                if chaps:
                    lines.append("Chapters/Sections: " + " | ".join(str(c) for c in chaps))
                lines.append("")
            return "[Vectorless RAG — configure an LLM for AI-generated answers]\n\n" + "\n".join(lines)
        sys_prompt = (
            "You are a Vectorless RAG Agent. The user has NOT done a vector search. "
            "Instead, read the full page index (document structure) provided and answer holistically."
        )
        try:
            return self.llm.generate_response(
                system_prompt=sys_prompt,
                user_query=query,
                retrieved_context=index_dump,
            )
        except Exception as e:
            self.logger.warning(f"Engine 2 LLM call failed: {e}")
            return index_dump

    # ── Engine 3: GraphRAG ─────────────────────────────────────────────────
    def _engine_3_graph_rag(self, query: str) -> str:
        """
        Multi-hop reasoning over the NetworkX Knowledge Graph.
        Filters edges by query relevance; falls back to Engine 1 if graph is empty.
        """
        import re as _re
        edges = list(self.graph.edges(data=True))
        if not edges:
            self.logger.info("Graph empty — falling back to Engine 1 (Vector RAG)")
            return (
                "[GraphRAG] No graph data available yet. Falling back to vector search:\n\n"
                + self._engine_1_vector_rag(query)
            )

        stop_words = {
            "the", "is", "are", "was", "how", "what", "who", "why",
            "when", "where", "a", "an", "to", "of", "in", "and", "or",
            "with", "by", "from", "for", "on", "at", "does",
        }
        keywords = [
            w.lower()
            for w in _re.findall(r"\b\w{3,}\b", query)
            if w.lower() not in stop_words
        ]

        def edge_is_relevant(u, v, d):
            text = f"{u} {d.get('relation', '')} {v}".lower()
            return any(kw in text for kw in keywords)

        relevant_edges = [(u, v, d) for u, v, d in edges if edge_is_relevant(u, v, d)]

        if relevant_edges:
            graph_lines = [
                f"{u}  --[{d.get('relation', 'related_to')}]-->  {v}"
                for u, v, d in relevant_edges[:20]
            ]
            context_note = f"Found {len(relevant_edges)} relevant connections for query: '{query}'"
        else:
            graph_lines = [
                f"{u}  --[{d.get('relation', 'related_to')}]-->  {v}"
                for u, v, d in edges[:20]
            ]
            context_note = (
                f"No direct graph matches for '{query}'. Showing full graph "
                f"({len(edges)} total edges) and supplementing with vector search."
            )

        graph_dump = "\n".join(graph_lines)

        if not self.llm:
            output = "[GraphRAG — configure an LLM for AI-generated answers]\n"
            output += f"{context_note}\n\nKnowledge Graph Connections:\n{graph_dump}\n"
            vector_ctx = self._get_raw_vector_context(query)
            if vector_ctx:
                output += f"\n\nRelated Context (from vector search):\n{vector_ctx}"
            return output

        sys_prompt = (
            "You are a GraphRAG Detective. You are given an entity-relationship knowledge graph "
            "extracted from documents. Use these relationships to answer the user's query. "
            "If the graph contains entities related to the query, trace the connections to answer. "
            "If not directly relevant, say so clearly and reason from what IS available."
        )
        full_context = f"{context_note}\n\nKnowledge Graph:\n{graph_dump}"
        try:
            return self.llm.generate_response(
                system_prompt=sys_prompt,
                user_query=query,
                retrieved_context=full_context,
            )
        except Exception as e:
            self.logger.warning(f"Engine 3 LLM call failed: {e}")
            return f"{context_note}\n\nKnowledge Graph:\n{graph_dump}"

    # ── Engine 4: Structured Extract ───────────────────────────────────────
    def _engine_4_extract(self, query: str, schema: Dict[str, str]) -> Dict[str, Any]:
        """Structured JSON output generation using LLM + schema."""
        if not self.llm:
            return {
                "status": "error",
                "error": "Engine 4 requires an LLM provider.",
                "how_to_enable": (
                    "Initialize VDBpipe with an LLM in config_override:\n"
                    "  pipeline = VDBpipe(config_override={\n"
                    "    'llm': {'provider': 'sarvam', 'model_name': 'sarvam-m', 'api_key': 'YOUR_KEY'}\n"
                    "  })"
                ),
                "schema_expected": schema,
            }
        sys_prompt = (
            f"Extract information based on the user query. "
            f"Return ONLY a valid JSON object (no markdown, no explanation) "
            f"matching exactly this schema: {json.dumps(schema)}"
        )
        try:
            response = self.llm.generate_response(
                system_prompt=sys_prompt,
                user_query=query,
                retrieved_context="",
            )
            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"raw_output": response}
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # SEARCH + RAG helpers
    # =========================================================================

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic similarity search against the vector store.
        Returns list of {document, score, metadata} dicts.
        """
        if self.embedder is None or self.vector_store is None:
            return []
        try:
            query_embedding = self.embedder.embed_text(query)
            return self.vector_store.search(query_embedding, top_k=top_k)
        except Exception as e:
            self.logger.warning(f"Search failed: {e}")
            return []

    def query_with_llm(self, user_query: str) -> str:
        """
        Standard RAG pipeline: search → build context → call LLM.
        Falls back to returning raw retrieved text if no LLM is configured.
        """
        results = self.search(user_query, top_k=3)
        if not results:
            return "No relevant information found in the knowledge base. Please run ingest() first."

        context = "\n\n---\n\n".join(
            [r.get("document", "") for r in results if r.get("document")]
        )

        if self.llm is None:
            return f"[Retrieved Context — configure an LLM for generated answers]\n\n{context}"

        try:
            system_prompt = (
                "You are an intelligent documentation assistant. "
                "Answer the user's question using only the provided context."
            )
            return self.llm.generate_response(
                system_prompt=system_prompt,
                user_query=user_query,
                retrieved_context=context,
            )
        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}")
            return context

    def _get_raw_vector_context(self, query: str, top_k: int = 2) -> str:
        """Helper: Get raw text from vector search without calling LLM."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        return "\n---\n".join(r.get("document", "") for r in results if r.get("document"))

    # =========================================================================
    # STREAMING (delegates to LLM provider's stream_response())
    # =========================================================================

    def stream_query(self, user_query: str):
        """
        Generator that streams Engine 1 (Vector RAG) response token by token.
        Requires an LLM provider that implements stream_response().
        Yields string tokens.
        """
        results = self.search(user_query, top_k=3)
        context = "\n\n---\n\n".join(
            [r.get("document", "") for r in results if r.get("document")]
        )
        if not context:
            yield "No relevant information found. Please run ingest() first."
            return

        if self.llm is None:
            yield "[No LLM configured — raw context below]\n\n"
            yield context
            return

        system_prompt = (
            "You are an intelligent documentation assistant. "
            "Answer the user's question using only the provided context."
        )
        try:
            yield from self.llm.stream_response(
                system_prompt=system_prompt,
                user_query=user_query,
                retrieved_context=context,
            )
        except Exception as e:
            self.logger.warning(f"LLM stream failed: {e}")
            yield context
