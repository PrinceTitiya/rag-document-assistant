# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Retrieval-Augmented Generation (RAG) chatbot for PDF documents, with three layers:

- **`src/`** — the RAG pipeline: loads PDFs (table-aware extraction), chunks them, embeds via a hosted sentence-transformer model, indexes into a persistent Chroma vector store, retrieves via hybrid (dense + BM25) search with LLM query expansion and cross-encoder re-ranking, and generates answers with a Groq-hosted LLM through LangChain.
- **`src/api.py`** — a FastAPI server exposing the pipeline over HTTP, including on-the-fly PDF upload/re-indexing.
- **`frontend/`** — a React + TypeScript (Vite) chat UI that talks to the API.

## Commands

### Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use `python3` (not `python`) to create the venv — on macOS (especially with the python.org installer) there is often no bare `python` command on `PATH` until a venv is activated. Once the venv is activated, `python`/`pip` inside it work fine.

Requires a `.env` file in the project root with:

```bash
GROQ_API_KEY=your_groq_api_key_here
HUGGING_FACE_TOKEN=your_hugging_face_token_here
```

`src/config.py` loads `GROQ_API_KEY` via `python-dotenv`; `src/embedding_manager.py` reads `HUGGING_FACE_TOKEN` directly. Both fail fast with a clear error if missing rather than surfacing a confusing error later when a model is first invoked.

### Running

```bash
python main.py                              # CLI: prompts for a question on stdin, always uses data/raw/
uvicorn src.api:app --reload --port 8000     # API server for the web chat UI
```

```bash
cd frontend && npm install                  # first time only
cd frontend && npm run dev                   # dev server on :5173
cd frontend && npm run build                 # tsc -b && vite build
cd frontend && npm run lint                  # oxlint
```

The frontend's CORS is hardcoded in `src/api.py` to allow only `http://localhost:5173` — if you run the dev server on a different port, update `allow_origins` there.

There is no Python test suite, linter, or build step configured in this repo (only the frontend has `lint`/`build` via oxlint/tsc+vite).

Input PDFs live in `data/raw/`; the default vector store is persisted to `data/vectorstore/` (gitignored) on first run and reused on subsequent runs — deleting that directory forces a full re-index. Same caveat applies to changes in `chunk_size`, `chunk_overlap`, or `EMBEDDING_MODEL_NAME`: they have no effect on an existing persisted store.

## Architecture

### Core pipeline

`RAGPipeline` (`src/pipeline.py`) wires together the following components, in order:

1. **`DocumentLoader`** (`src/document_loader.py`) — loads PDFs page-by-page via `pdfplumber` (not `PyPDFLoader`), detects genuine tables per page (filtering out pdfplumber's frequent false positives on ordinary paragraph text via `_is_real_table()`) and renders real tables as Markdown appended to that page's text, then splits everything into chunks with `RecursiveCharacterTextSplitter` (default `chunk_size=1200`, `chunk_overlap=200`).
2. **`EmbeddingManager`** (`src/embedding_manager.py`) — wraps `HuggingFaceEndpointEmbeddings`: this calls the **hosted** Hugging Face Inference API (authenticated via `HUGGING_FACE_TOKEN`), not a local model download, for the configured model (default `BAAI/bge-base-en-v1.5`).
3. **`VectorStoreManager`** (`src/vectorstore_manager.py`) — creates a new Chroma store (via `langchain_chroma.Chroma`) from chunks or loads an existing one from `persist_dir`. `get_vectorstore()` is the entry point pipeline code should call; it checks `vectorstore_exists()` and dispatches to `create_vectorstore()`/`load_vectorstore()` accordingly. If a store already exists at `persist_dir`, it's loaded as-is — new/changed source PDFs are silently ignored until that directory is deleted and rebuilt.
4. **`src/llm.py`** — builds the single shared `ChatGroq` instance used by both retrieval (query expansion) and generation, fails fast with a clear error if `GROQ_API_KEY` is missing, and defines `retry_on_transient_groq_error` (via `tenacity`): a shared retry-with-backoff policy for rate limits/timeouts/connection errors/5xx responses, applied to every Groq call in the pipeline.
5. **`src/query_expander.py`** (`QueryExpander`) — asks the LLM to generate alternate phrasings/sub-questions for the user's query before retrieval, so retrieval isn't limited to the user's exact wording. Degrades gracefully (logs a warning, falls back to the original query only) if the LLM call fails even after retries.
6. **`RAGRetriever`** (`src/retriever.py`) — hybrid retrieval: for the original question plus every expanded variant, runs both dense (Chroma) similarity search and BM25 keyword search, fuses all resulting ranked lists via Reciprocal Rank Fusion, then re-ranks the fused candidate pool against the *original* question with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for the final top-`k`. Needs the full `chunks` list (not just the vectorstore) to build its BM25 index — kept on `retriever.chunks`.
7. **`RAGGenerator`** (`src/generator.py`) — builds a context string from retrieved `Document`s (with source/page metadata), fills a strict context-only prompt template, and calls the shared `ChatGroq` LLM (default model `llama-3.3-70b-versatile`) with retry-on-transient-error. The prompt is designed to avoid hallucination and instructs the model to answer `"Answer not found in documents"` when the context doesn't support an answer.

All defaults (paths, model names) live centrally in `src/config.py` and flow into `RAGPipeline.__init__`, which accepts overrides for `data_dir`, `persist_dir`, `embedding_model_name`, and `llm_model_name`.

`RAGPipeline` exposes two query methods:
- `query(question, k=None)` — returns the answer string.
- `query_with_sources(question, k=None)` — returns `{"answer": ..., "sources": [{"source", "page"}, ...]}`.

**Token usage note**: this pipeline makes two LLM calls per question (query expansion + generation), roughly double a single-call pipeline. Groq's free/on-demand tier has a fairly small daily token budget (observed: 100,000 tokens/day for `llama-3.3-70b-versatile` on one account) — heavy interactive use or test loops can exhaust it, at which point Groq returns a `RateLimitError` with a `type: tokens` / TPD (tokens-per-day) reason and a wait time that can be many minutes, not a few seconds. `retry_on_transient_groq_error` retries transient rate limits with short backoff, but a genuinely exhausted *daily* quota will still exhaust the retry budget and raise a clear `RuntimeError` — that's expected; it isn't something client-side retries can wait out within a single request.

### API layer (`src/api.py`)

The FastAPI app builds `default_pipeline = RAGPipeline()` once at import time (from `data/raw/` → `data/vectorstore/`), then holds a second, mutable module-level reference, `active_pipeline`, which every `/query` call actually uses:

- **`POST /upload`** (multipart `file` field) saves the PDF to `data/uploads/`, builds a **brand-new** `RAGPipeline` scoped to just that file, and reassigns `active_pipeline` to it. Each upload gets its own UUID-named directory under `data/uploads_vectorstore/<uuid>/` rather than reusing one fixed path — reusing a path here previously corrupted Chroma with `"attempt to write a readonly database"` because a prior upload's `Chroma` client could still be alive in-process when the directory was deleted and recreated. Per-upload directories are *not* cleaned up within a running process (only wiped once at server startup) — acceptable for a local single-user tool, not for heavy repeated use.
- **`POST /reset`** points `active_pipeline` back at `default_pipeline`.
- **`GET /status`** returns the current active source filename (or `null`), used by the frontend to restore UI state on load.
- **`active_pipeline`/`active_source` are process-wide globals**, not per-session — there is no auth or multi-user isolation. Uploading a PDF from one browser tab changes what every client gets answered from.

### Frontend (`frontend/`)

Single-page chat UI in `frontend/src/App.tsx`, styled dark-mode-only in `App.css`/`index.css`. Talks to the API at a hardcoded `http://localhost:8000` base URL (see `API_BASE` in `App.tsx`). Renders a source badge synced from `GET /status` on mount, a message list, and an upload button that posts to `/upload` and clears local chat state on success.

`notebook/RAG_Ingestion.ipynb` is an exploratory/dev notebook, not part of the runtime path.
