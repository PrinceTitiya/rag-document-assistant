## RAG Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot for PDF documents. It loads PDFs with table-aware extraction, chunks them, embeds them with a hosted sentence-transformer model, indexes into a persistent Chroma vector store, retrieves with hybrid (dense + BM25) search plus LLM-driven query expansion and cross-encoder re-ranking, and generates grounded answers with a Groq-hosted LLM via LangChain.

The project has three layers:

- **`src/`** — the RAG pipeline itself (loading, embedding, retrieval, generation).
- **`src/api.py`** — a FastAPI server that exposes the pipeline over HTTP, including on-the-fly PDF upload.
- **`frontend/`** — a React + TypeScript (Vite) chat UI that talks to the API.

You can use the pipeline directly from Python (see [Basic Python usage](#basic-python-usage)), from the CLI (`main.py`), or through the web chat UI.

### Features

- **Table-aware PDF loading**: page-by-page extraction via `pdfplumber`; genuine tables (filtered from pdfplumber's frequent false positives) are rendered as Markdown and appended to the page text so row/column structure survives chunking.
- **Hybrid retrieval**: dense (Chroma) similarity search + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF).
- **LLM query expansion**: alternate phrasings/sub-questions are generated before retrieval so search isn't limited to the user's exact wording (degrades gracefully to the original query if the LLM call fails).
- **Cross-encoder re-ranking**: the fused candidate pool is re-ranked against the original question with `cross-encoder/ms-marco-MiniLM-L-6-v2` for the final top-k.
- **Config-driven**: central configuration in `src/config.py`.
- **Persistent vector store**: powered by Chroma in `data/vectorstore/`, built once and reused on later runs.
- **Groq-based answering**: uses `llama-3.3-70b-versatile` via LangChain, with shared retry-with-backoff on transient errors (rate limits, timeouts, 5xx).
- **Context-only prompting**: the generator answers strictly from retrieved context and responds `"Answer not found in documents"` when the context doesn't support an answer.
- **Chat UI with PDF upload**: upload your own PDF from the browser and the whole pipeline (chunk → embed → index) runs on it on the fly; without an upload, the assistant answers from whatever is in `data/raw/`.

## Architecture / Flow

### Core pipeline (`src/`)

```
data/raw/*.pdf
      │
      ▼
DocumentLoader (src/document_loader.py)
  - pdfplumber, page by page
  - detects real tables, renders as Markdown, appends to page text
  - RecursiveCharacterTextSplitter (chunk_size=1200, chunk_overlap=200)
      │
      ▼  chunks: List[Document]
      │
      ├──────────────────────────────────────────────┐
      ▼                                                ▼
EmbeddingManager                                 (chunks kept for BM25 index)
  - HuggingFaceEndpointEmbeddings
    (hosted HF Inference API call, default BAAI/bge-base-en-v1.5)
      │
      ▼
VectorStoreManager (src/vectorstore_manager.py)
  - if persist_dir already exists → load it as-is
  - else → embed chunks, build + persist a new Chroma store
      │
      ▼  vectorstore
      │
      ▼
RAGRetriever (src/retriever.py)      ◄── shared ChatGroq LLM (src/llm.py)
  1. QueryExpander generates N alternate phrasings of the question
  2. for the question + each variant:
       dense search (Chroma)  +  BM25 search  →  ranked lists
  3. Reciprocal Rank Fusion merges all ranked lists
  4. CrossEncoder re-ranks the fused pool against the ORIGINAL question
      │
      ▼  top-k Documents
      │
      ▼
RAGGenerator (src/generator.py)
  - builds a context string (source + page + text per chunk)
  - fills a strict "context-only" prompt template
  - calls shared ChatGroq LLM (llama-3.3-70b-versatile), with retry-on-
    transient-error
      │
      ▼
  answer string  /  {"answer": ..., "sources": [{"source","page"}, ...]}
```

`RAGPipeline` (`src/pipeline.py`) wires all of the above together and is the entry point application code uses — the CLI, the API, or your own scripts.

### Web app (API + frontend)

```
┌────────────────────┐   POST /query    ┌──────────────────────────┐
│                     │ ───────────────▶ │                          │
│  frontend/          │   POST /upload   │  src/api.py (FastAPI)    │
│  React chat UI      │ ───────────────▶ │                          │
│  (localhost:5173)   │   POST /reset    │  (localhost:8000)        │
│                     │ ◀─────────────── │                          │
└────────────────────┘   GET  /status    └────────────┬─────────────┘
                                                        │
                                          holds a global "active_pipeline"
                                                        │
                              ┌─────────────────────────┴─────────────────────────┐
                              ▼                                                   ▼
                    default_pipeline                                   pipeline built fresh
                    RAGPipeline(data/raw,                               per /upload over the
                    data/vectorstore)                                  uploaded PDF, in an
                    built once at startup                               isolated vectorstore
                                                                         dir (data/uploads_
                                                                         vectorstore/<uuid>)
```

- **No PDF uploaded**: `active_pipeline` is `default_pipeline`, built from `data/raw/` at server startup — this is what answers every `/query` call by default.
- **PDF uploaded** (`POST /upload`, multipart form): the file is saved to `data/uploads/`, a brand-new `RAGPipeline` is built over just that file (its own chunking → embedding → Chroma index, in a fresh UUID-named directory under `data/uploads_vectorstore/` so it never collides with a previous upload's still-live Chroma client), and `active_pipeline` is swapped to it. All subsequent `/query` calls answer from the uploaded PDF.
- **`POST /reset`**: swaps `active_pipeline` back to `default_pipeline`.
- **`GET /status`**: returns the currently active source filename (or `null` for default), used by the frontend to restore the badge state on page load.
- This is process-wide, single-user state (no auth/session concept) — appropriate for a local personal tool, not a multi-tenant deployment.

## Project Structure

### Backend (`src/`)

- **`src/config.py`**: central defaults — `DATA_DIR`, `VECTORSTORE_DIR`, `EMBEDDING_MODEL_NAME`, `LLM_MODEL_NAME`, and `GROQ_API_KEY` loading via `python-dotenv`.
- **`src/pipeline.py`**: `RAGPipeline` — orchestrates loader → embedder → vector store → shared LLM → retriever → generator. Exposes `query()` and `query_with_sources()`.
- **`src/document_loader.py`**: `DocumentLoader` — table-aware PDF loading and chunking.
- **`src/embedding_manager.py`**: `EmbeddingManager` — wraps `HuggingFaceEndpointEmbeddings` (Hugging Face's hosted Inference API, authenticated with `HUGGING_FACE_TOKEN` — no local model download).
- **`src/vectorstore_manager.py`**: `VectorStoreManager` — creates or loads a persistent Chroma store at a given `persist_dir`.
- **`src/llm.py`**: builds the single shared `ChatGroq` instance (used by both query expansion and generation) and defines `retry_on_transient_groq_error`, a shared retry-with-backoff policy applied to every Groq call.
- **`src/query_expander.py`**: `QueryExpander` — generates alternate phrasings of the question via the LLM before retrieval.
- **`src/retriever.py`**: `RAGRetriever` — hybrid dense + BM25 search, RRF fusion, cross-encoder re-ranking.
- **`src/generator.py`**: `RAGGenerator` — builds the context-only prompt and calls the LLM to produce the final answer.
- **`src/api.py`**: FastAPI app exposing `/query`, `/upload`, `/reset`, `/status`. Holds the global `active_pipeline` described above.
- **`main.py`**: minimal CLI entry point — prompts for a question on stdin, prints the answer. Always uses `data/raw/` (no upload support).
- **`data/raw/`**: input PDFs to be indexed by default (e.g. `Ethereum-whitepaper.pdf`, `Blockchain_For_Beginners.pdf`).
- **`data/vectorstore/`**: persisted Chroma index for `data/raw/` (created on first run, gitignored).
- **`data/uploads/`**, **`data/uploads_vectorstore/`**: the currently-uploaded PDF and its isolated index (created on first upload, gitignored).

### Frontend (`frontend/`)

React + TypeScript app scaffolded with Vite.

- **`frontend/src/App.tsx`**: the entire chat UI — message list, input box, send button, PDF upload button, and the active-source badge (with a "Reset to default" action). Talks to the API at `http://localhost:8000`.
- **`frontend/src/App.css`**, **`frontend/src/index.css`**: dark-themed styling.
- **`frontend/src/main.tsx`**: React entry point.

## Prerequisites

- **Python**: 3.9+ recommended.
- **Node.js**: 18+ (only needed if you're running the web chat UI).
- **Groq API key** with access to Groq-hosted models.
- **Hugging Face token** with access to the Inference API (for embeddings).

## Installation

### Backend

From the project root:

```bash
python3 -m venv .venv   # On Windows: python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Use `python3` (not `python`) to create the venv — on macOS (especially with the python.org installer) there is often no bare `python` command on `PATH` until a venv is activated.

### Frontend

```bash
cd frontend
npm install
```

## Environment Setup

Create a `.env` file in the project root with:

```bash
GROQ_API_KEY=your_groq_api_key_here
HUGGING_FACE_TOKEN=your_hugging_face_token_here
```

Both are read in `src/config.py` / `src/embedding_manager.py` via `python-dotenv`. If either is missing, the pipeline fails fast with a clear error (`src/llm.py` for the Groq key, `src/embedding_manager.py` for the HF token) rather than surfacing a confusing error later when the model is first invoked.

## Preparing Documents

Place your default PDF files in:

```text
data/raw/
```

Examples in this repo include:

- `data/raw/Ethereum-whitepaper.pdf`
- `data/raw/Blockchain_For_Beginners.pdf`

Any PDFs in `data/raw/` are picked up by `DocumentLoader` — **but only when the vector store is built**. If `data/vectorstore/` already exists, it's loaded as-is and newly added/changed PDFs will *not* be re-indexed until you delete `data/vectorstore/` and rebuild (see [Notes and Tips](#notes-and-tips)). Alternatively, upload a PDF from the chat UI to query it without touching `data/raw/` at all.

## Usage

### Web chat UI (recommended)

Run the API and the frontend in two terminals:

```bash
# terminal 1 — backend
source .venv/bin/activate
uvicorn src.api:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm run dev
```

Open `http://localhost:5173`. On first backend startup it builds/loads the default vector store from `data/raw/`, which can take a while the first time.

- Type a question and hit **Send** (or press Enter) to query the default documents.
- Click **📎** to upload your own PDF — the assistant re-indexes and answers from just that file until you click **Reset to default**.

### CLI

```bash
python main.py
```

Prompts for a question on stdin and prints the answer, always using `data/raw/`. On first run this also builds and persists the vector store, which can take a while (embedding every chunk of every PDF in `data/raw/`).

### Basic Python usage

```python
from src.pipeline import RAGPipeline

# First run: builds embeddings and vector store, which may take some time.
pipeline = RAGPipeline()

question = "What is Ethereum and how does it work?"
answer = pipeline.query(question)

print("Q:", question)
print("A:", answer)
```

### Getting answers with sources

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.query_with_sources("Explain the main idea of a blockchain.")

print("Answer:", result["answer"])
print("Sources:")
for src in result["sources"]:
    print(f"- {src['source']} (page {src['page']})")
```

This uses `RAGGenerator.generate_with_sources` under the hood, which retrieves once and reports the same chunks that were actually sent to the LLM. `src/api.py`'s `/query` endpoint uses this same method.

### API reference

| Method | Path      | Body                          | Description                                                              |
|--------|-----------|--------------------------------|----------------------------------------------------------------------------|
| POST   | `/query`  | `{"question": "..."}`         | Answers using whichever pipeline is currently active.                     |
| POST   | `/upload` | multipart form, field `file`  | Uploads a PDF, indexes it, and makes it the active pipeline.              |
| POST   | `/reset`  | —                              | Switches back to the default (`data/raw/`) pipeline.                      |
| GET    | `/status` | —                              | Returns `{"active_source": "<filename>" | null}`.                        |

## Configuration

Key configuration values live in `src/config.py`:

- **`DATA_DIR`**: directory containing raw documents (`data/raw` by default).
- **`VECTORSTORE_DIR`**: where the default Chroma DB is persisted (`data/vectorstore`).
- **`EMBEDDING_MODEL_NAME`**: Hugging Face model served via the Inference API (default `BAAI/bge-base-en-v1.5`).
- **`LLM_MODEL_NAME`**: Groq model used for query expansion and generation (default `llama-3.3-70b-versatile`).
- **`GROQ_API_KEY`**: loaded from `.env`.

You can override these either by editing `src/config.py` or by passing custom values into `RAGPipeline(...)` (`data_dir`, `persist_dir`, `embedding_model_name`, `llm_model_name`) — this is exactly what `src/api.py`'s `/upload` endpoint does to build an isolated pipeline per uploaded PDF. Retrieval knobs (`k`, `fetch_k`, reranker on/off, number of query expansion variants) and chunking (`chunk_size`, `chunk_overlap`) are constructor args on `RAGRetriever` and `DocumentLoader` respectively — currently hardcoded in `RAGPipeline.__init__`, so change them there if needed.

## Notes and Tips

- **First run cost**: the initial run loads PDFs, detects tables, splits into chunks, embeds everything, and builds the vector store. Subsequent runs against the same `persist_dir` just load the persisted index and are much faster.
- **Stale index**: adding/changing PDFs in `data/raw/`, or changing `chunk_size`, `chunk_overlap`, or `EMBEDDING_MODEL_NAME`, has **no effect** until you delete `data/vectorstore/` and let it rebuild.
- **Two LLM calls per question**: query expansion and answer generation each call Groq, roughly doubling token usage versus a single-call pipeline. Groq's free/on-demand tier has a fairly small daily token budget (observed: 100,000 tokens/day for `llama-3.3-70b-versatile` on one account). `retry_on_transient_groq_error` retries transient rate limits with backoff, but a genuinely exhausted *daily* quota will still exhaust the retry budget and raise a clear `RuntimeError` — that's expected and not something client-side retries can wait out.
- **Context-only answers**: the generator prompt is designed to avoid hallucinations and responds with `"Answer not found in documents"` when the answer isn't supported by the retrieved context.
- **Graceful degradation**: if query expansion fails (e.g. LLM error after retries), retrieval falls back to hybrid search on the original question only rather than failing the request.
- **Upload is single-user, in-memory state**: `active_pipeline` in `src/api.py` is a process-wide global, not per-browser-session — uploading a PDF in one tab changes what every tab queries. This matches the app's scope (a local personal tool), not a multi-tenant deployment.
- **Upload vectorstore cleanup**: per-upload Chroma directories under `data/uploads_vectorstore/` are cleaned up on the next server restart, but accumulate (one per upload) within a single running process — acceptable for occasional local use.
- `notebook/RAG_Ingestion.ipynb` is an exploratory/dev notebook, not part of the runtime path.

## Troubleshooting

- **Missing API key**: if you see authentication errors, ensure `GROQ_API_KEY` and `HUGGING_FACE_TOKEN` are set in `.env` and that the `.env` file is in the project root.
- **Rate limit / quota errors**: a `RateLimitError` with a TPD (tokens-per-day) reason means the daily Groq quota is exhausted for now — wait for it to reset rather than retrying immediately. Transient per-minute rate limits are already retried automatically.
- **No documents loaded**: confirm your PDFs are in `data/raw/` and that `data/vectorstore/` doesn't already exist from a run before those PDFs were added (delete it to force a rebuild).
- **Frontend can't reach the backend**: confirm `uvicorn src.api:app --port 8000` is running and that you're opening the frontend at `http://localhost:5173` — the API's CORS policy only allows that origin.
- **Upload fails with "Couldn't index this PDF"**: the PDF likely has no extractable text (e.g. a pure image scan with no OCR layer) — `pdfplumber` can't chunk what it can't extract.
- **Model or import errors**: reinstall dependencies with `pip install -r requirements.txt` (backend) or `npm install` (frontend), and ensure your Python/Node versions are compatible.
