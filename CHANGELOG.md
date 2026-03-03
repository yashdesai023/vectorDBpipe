# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.4] — 2026-03-03

### 🏗 Architecture
- **Refactored `VDBpipe` to pure composition** — removed `TextPipeline` inheritance entirely. `VDBpipe` is now a standalone class with all providers (`Embedder`, `VectorStore`, `DataLoader`, `LLM`) as instance attributes. Eliminated the `_safe_reinit()` hack.
- **Replaced `TextPipeline` with `VDBpipe` in the backend** — all pipeline endpoints (`/ingest`, `/chat`, `/retrieve`) now use `VDBpipe`, giving full OmniRouter access to Engines 1–3 via the web dashboard.

### 🧠 Semantic OmniRouter (New)
- **Embedding-based semantic query routing** — replaced keyword matching with cosine-similarity classification. Intent prototype embeddings for Engine 2 (Vectorless RAG) and Engine 3 (GraphRAG) are pre-computed at startup. Queries are embedded once and scored against all prototypes (threshold = 0.35). Falls back to keyword heuristics when no embedder is configured.

### 💾 Persistence (New)
- **Graph + PageIndex auto-persistence** — `_persist_state()` serializes the NetworkX knowledge graph (node-link JSON) and `page_index` (JSON) to disk after every `ingest()` call. `_load_state()` restores them on `VDBpipe.__init__()`. Knowledge graph and document index now survive server/TUI restarts.

### 🌊 Streaming (New)
- **`BaseLLMProvider.stream_response()`** — new method with a safe default implementation (wraps `generate_response()` as a single-chunk generator). All 7 LLM providers get streaming support for free.
- **`OpenAILLMProvider.stream_response()`** — real SSE token streaming using `requests` with `stream=True`. Parses `data: {...}` events and yields delta content tokens.
- **`VDBpipe.stream_query()`** — generator that delegates to `llm.stream_response()` for live token output.
- **`POST /pipelines/chat/stream`** — new SSE backend endpoint (`StreamingResponse`, `text/event-stream`) for token-by-token streaming in the frontend.

### 📄 Data Loading
- **PPTX support** — added `.pptx` to `DataLoader.supported_ext`. New `_load_pptx()` uses `python-pptx` to extract text from all slides. Requires `pip install python-pptx`.
- CSV, JSON, HTML were already supported; confirmed and retained.

### ✂️ Chunking Strategy
- **`chunk_text_sentences(text, max_tokens, overlap_sentences)`** — new sentence-boundary sliding-window chunker in `utils/common.py`. Groups sentences into chunks not exceeding `max_tokens` words with configurable sentence-level overlap. Eliminates mid-sentence splits that the fixed word-level chunker can produce. Old `chunk_text()` kept for backwards compatibility.

### 🧪 Tests
- **Expanded from 4 to 39 unit tests** across 12 test classes.
- New coverage: Engine 2 (Vectorless RAG), Engine 3 (GraphRAG), Engine 4 (Structured Extract), no-LLM fallback paths for all engines, sentence-boundary chunking correctness, PPTX loader, Graph+PageIndex persistence roundtrip, and streaming output.
- All tests use mocked providers — no API keys, GPU, or network required.

### 🖥 TUI
- **System Doctor — real runtime checks**: Replaced hard-coded status badges with 6 live `execSync` checks: Node.js version, Python version (`python`/`python3` fallback), `pip show vectordbpipe`, `config.yaml` existence, internet ping to `8.8.8.8`, VectorDB provider read from YAML. Shows a loading spinner until checks complete.
- **Setup Wizard — error screen fix**: `finishSetup()` now calls `setStep(8)` in the `catch` block. Write failures are no longer silently swallowed.
- **Setup Wizard — API key validation**: New `validateAndSave()` makes a lightweight `GET` request to the LLM provider's `/models` endpoint before writing `config.yaml`. Step 9 shows "Validating API Key..." spinner; Step 10 shows an error screen with the HTTP status code. Network failures allow save with a warning.
- **TUI `postinstall.cjs` — smarter auto-install**: Now resolves Python via `python`/`python3`/`py`, always uses `python -m pip` (avoids broken pip launcher issues), checks if `vectordbpipe` is already installed before re-installing, streams install output live, and prints clear manual instructions on failure.

### 🐛 Bug Fixes
- **File isolation bug**: Backend uploads no longer share a flat `data/` directory. Files are saved to `data/<user_id>/<uuid>_<filename>` (per-user isolation, no collisions).
- **Stale config on backend update**: `PUT /pipelines/{id}/config` now evicts the pipeline cache entry so subsequent requests pick up the new config.

### 📦 Dependencies Added
- `python-pptx>=0.6.23` — PPTX loader
- `networkx>=3.1` — Knowledge Graph (now explicit in `setup.py`)

---

## [0.2.3] — 2026-02-27 (hotfix)
- Fixed missing `llms` subpackage (`__init__.py`) that caused `ImportError` on all LLM providers after PyPI install.
- Pinned `chromadb>=0.5.0` to fix `PersistentClient` API changes.

## [0.2.2] — 2026-02-20
- Added `GroqLLMProvider`, `AnthropicLLMProvider`, `CohereLLMProvider`.
- Backend: JWT authentication, API key vaulting, chat history persistence.

## [0.2.1] — 2026-02-15
- Added TUI (`vectordbpipe-tui` npm package) with Setup Wizard and System Doctor.
- Added `VDBpipe.extract()` — Engine 4 structured JSON extraction.

## [0.2.0] — 2026-02-10
- Initial public release.
- Omni-RAG with 4 engines: Vector RAG, Vectorless RAG, GraphRAG, Structured Extract.
- Tri-Processing Ingestion: vectors + PageIndex + Knowledge Graph.
- FastAPI backend + React frontend.
