# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Retrieval-Augmented Generation (RAG) pipeline for answering questions over PDF documents. It loads PDFs (with table-aware extraction), chunks them, embeds with a sentence-transformer model, indexes into a persistent Chroma vector store, retrieves relevant chunks via hybrid (dense + BM25) search with LLM query expansion and cross-encoder re-ranking, and generates answers with a Groq-hosted open-source LLM through LangChain.

## Setup

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
```

`src/config.py` loads this via `python-dotenv`.

## Running

```bash
python main.py
```

Prompts for a question on stdin and prints the answer. There is no test suite, linter, or build step configured in this repo.

Input PDFs live in `data/raw/`; the vector store is persisted to `data/vectorstore/` (gitignored) on first run and reused on subsequent runs — deleting that directory forces a full re-index.

## Architecture

`RAGPipeline` (`src/pipeline.py`) wires together the following components:

1. **`DocumentLoader`** (`src/document_loader.py`) — loads PDFs page-by-page via `pdfplumber` (not `PyPDFLoader`), detects genuine tables per page (filtering out pdfplumber's frequent false positives on ordinary paragraph text via `_is_real_table()`) and renders real tables as Markdown appended to that page's text, then splits everything into chunks with `RecursiveCharacterTextSplitter` (default `chunk_size=1200`, `chunk_overlap=200`).
2. **`EmbeddingManager`** (`src/embedding_manager.py`) — wraps `HuggingFaceEmbeddings` for the configured sentence-transformer model (default `BAAI/bge-large-en-v1.5`).
3. **`VectorStoreManager`** (`src/vectorstore_manager.py`) — creates a new Chroma store (via `langchain_chroma.Chroma`) from chunks or loads an existing one from `persist_dir`. `get_vectorstore()` is the entry point pipeline code should call; it checks `vectorstore_exists()` and dispatches to `create_vectorstore()`/`load_vectorstore()` accordingly. **Note**: if the vector store already exists on disk, it is loaded as-is and newly added/changed PDFs in `data/raw/` will *not* be re-indexed unless `data/vectorstore/` is deleted first. Changing `chunk_size`, `chunk_overlap`, or `EMBEDDING_MODEL_NAME` likewise has no effect until the store is deleted and rebuilt.
4. **`src/llm.py`** — builds the single shared `ChatGroq` instance used by both retrieval (query expansion) and generation, fails fast with a clear error if `GROQ_API_KEY` is missing, and defines `retry_on_transient_groq_error` (via `tenacity`): a shared retry-with-backoff policy for rate limits/timeouts/connection errors/5xx responses, applied to every Groq call in the pipeline.
5. **`src/query_expander.py`** (`QueryExpander`) — asks the LLM to generate alternate phrasings/sub-questions for the user's query before retrieval, so retrieval isn't limited to the user's exact wording. Degrades gracefully (logs a warning, falls back to the original query only) if the LLM call fails even after retries.
6. **`RAGRetriever`** (`src/retriever.py`) — hybrid retrieval: for the original question plus every expanded variant, runs both dense (Chroma) similarity search and BM25 keyword search, fuses all resulting ranked lists via Reciprocal Rank Fusion, then re-ranks the fused candidate pool against the *original* question with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for the final top-`k`. Needs the full `chunks` list (not just the vectorstore) to build its BM25 index.
7. **`RAGGenerator`** (`src/generator.py`) — builds a context string from retrieved `Document`s (with source/page metadata), fills a strict context-only prompt template, and calls the shared `ChatGroq` LLM (default model `llama-3.3-70b-versatile`) with retry-on-transient-error. The prompt is designed to avoid hallucination and instructs the model to answer `"Answer not found in documents"` when the context doesn't support an answer.

All defaults (paths, model names) live centrally in `src/config.py` and flow into `RAGPipeline.__init__`, which accepts overrides for `data_dir`, `persist_dir`, `embedding_model_name`, and `llm_model_name`.

**Token usage note**: this pipeline makes two LLM calls per question (query expansion + generation), roughly double a single-call pipeline. Groq's free/on-demand tier has a fairly small daily token budget (observed: 100,000 tokens/day for `llama-3.3-70b-versatile` on one account) — heavy interactive use or test loops can exhaust it, at which point Groq returns a `RateLimitError` with a `type: tokens` / TPD (tokens-per-day) reason and a wait time that can be many minutes, not a few seconds. `retry_on_transient_groq_error` retries transient rate limits with short backoff, but a genuinely exhausted *daily* quota will still exhaust the retry budget and raise a clear `RuntimeError` — that's expected; it isn't something client-side retries can wait out within a single request.

`RAGPipeline` exposes two query methods:
- `query(question, k=None)` — returns the answer string.
- `query_with_sources(question, k=None)` — returns `{"answer": ..., "sources": [{"source", "page"}, ...]}`.

`notebook/RAG_Ingestion.ipynb` is a exploratory/dev notebook, not part of the runtime path.
