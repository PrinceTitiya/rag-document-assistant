## RAG Question Answering on PDFs

This repository implements a Retrieval-Augmented Generation (RAG) pipeline over PDF documents (e.g. blockchain/Ethereum whitepapers). It loads PDFs with table-aware extraction, chunks them, embeds with a sentence-transformer model, indexes into a persistent Chroma vector store, retrieves with hybrid (dense + BM25) search plus LLM-driven query expansion and cross-encoder re-ranking, and generates grounded answers with a Groq-hosted LLM via LangChain.

### Features

- **Table-aware PDF loading**: page-by-page extraction via `pdfplumber`; genuine tables (filtered from pdfplumber's frequent false positives) are rendered as Markdown and appended to the page text so row/column structure survives chunking.
- **Hybrid retrieval**: dense (Chroma) similarity search + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF).
- **LLM query expansion**: alternate phrasings/sub-questions are generated before retrieval so search isn't limited to the user's exact wording (degrades gracefully to the original query if the LLM call fails).
- **Cross-encoder re-ranking**: the fused candidate pool is re-ranked against the original question with `cross-encoder/ms-marco-MiniLM-L-6-v2` for the final top-k.
- **Config-driven**: central configuration in `src/config.py`.
- **Persistent vector store**: powered by Chroma in `data/vectorstore/`, built once and reused on later runs.
- **Groq-based answering**: uses `llama-3.3-70b-versatile` via LangChain, with shared retry-with-backoff on transient errors (rate limits, timeouts, 5xx).
- **Context-only prompting**: the generator answers strictly from retrieved context and responds `"Answer not found in documents"` when the context doesn't support an answer.

## Architecture / Flow

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
  - HuggingFaceEmbeddings
    (BAAI/bge-large-en-v1.5)
      │
      ▼
VectorStoreManager (src/vectorstore_manager.py)
  - if data/vectorstore/ exists → load it as-is
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

`RAGPipeline` (`src/pipeline.py`) wires all of the above together and is the only entry point application code needs to use.

## Project Structure

- **`src/config.py`**: central defaults — `DATA_DIR`, `VECTORSTORE_DIR`, `EMBEDDING_MODEL_NAME`, `LLM_MODEL_NAME`, and `GROQ_API_KEY` loading via `python-dotenv`.
- **`src/pipeline.py`**: `RAGPipeline` — orchestrates loader → embedder → vector store → shared LLM → retriever → generator.
- **`src/document_loader.py`**: `DocumentLoader` — table-aware PDF loading and chunking.
- **`src/embedding_manager.py`**: `EmbeddingManager` — wraps `HuggingFaceEmbeddings`.
- **`src/vectorstore_manager.py`**: `VectorStoreManager` — creates or loads the persistent Chroma store.
- **`src/llm.py`**: builds the single shared `ChatGroq` instance (used by both query expansion and generation) and defines `retry_on_transient_groq_error`, a shared retry-with-backoff policy applied to every Groq call.
- **`src/query_expander.py`**: `QueryExpander` — generates alternate phrasings of the question via the LLM before retrieval.
- **`src/retriever.py`**: `RAGRetriever` — hybrid dense + BM25 search, RRF fusion, cross-encoder re-ranking.
- **`src/generator.py`**: `RAGGenerator` — builds the context-only prompt and calls the LLM to produce the final answer.
- **`main.py`**: minimal CLI entry point — prompts for a question on stdin, prints the answer.
- **`data/raw/`**: input PDFs to be indexed (e.g. `Ethereum-whitepaper.pdf`, `Blockchain_For_Beginners.pdf`).
- **`data/vectorstore/`**: persisted Chroma index (created on first run, gitignored).

## Prerequisites

- **Python**: 3.9+ recommended.
- **Groq API key** with access to Groq-hosted models.

## Installation

From the project root:

```bash
python3 -m venv .venv   # On Windows: python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Use `python3` (not `python`) to create the venv — on macOS (especially with the python.org installer) there is often no bare `python` command on `PATH` until a venv is activated.

## Environment Setup

Create a `.env` file in the project root with:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

The key is read in `src/config.py` via `python-dotenv`. If it's missing, `src/llm.py` fails fast with a clear error rather than surfacing a confusing error later when the model is first invoked.

## Preparing Documents

Place your PDF files in:

```text
data/raw/
```

Examples in this repo include:

- `data/raw/Ethereum-whitepaper.pdf`
- `data/raw/Blockchain_For_Beginners.pdf`

Any PDFs in `data/raw/` are picked up by `DocumentLoader` — **but only when the vector store is built**. If `data/vectorstore/` already exists, it's loaded as-is and newly added/changed PDFs will *not* be re-indexed until you delete `data/vectorstore/` and rebuild (see [Notes and Tips](#notes-and-tips)).

## Usage

### CLI

```bash
python main.py
```

Prompts for a question on stdin and prints the answer. On first run this also builds and persists the vector store, which can take a while (embedding every chunk of every PDF in `data/raw/`).

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

This uses `RAGGenerator.generate_with_sources` under the hood, which retrieves once and reports the same chunks that were actually sent to the LLM.

## Configuration

Key configuration values live in `src/config.py`:

- **`DATA_DIR`**: directory containing raw documents (`data/raw` by default).
- **`VECTORSTORE_DIR`**: where the Chroma DB is persisted (`data/vectorstore`).
- **`EMBEDDING_MODEL_NAME`**: sentence-transformer model (default `BAAI/bge-large-en-v1.5`).
- **`LLM_MODEL_NAME`**: Groq model used for query expansion and generation (default `llama-3.3-70b-versatile`).
- **`GROQ_API_KEY`**: loaded from `.env`.

You can override these either by editing `src/config.py` or by passing custom values into `RAGPipeline(...)` (`data_dir`, `persist_dir`, `embedding_model_name`, `llm_model_name`). Retrieval knobs (`k`, `fetch_k`, reranker on/off, number of query expansion variants) and chunking (`chunk_size`, `chunk_overlap`) are constructor args on `RAGRetriever` and `DocumentLoader` respectively — currently hardcoded in `RAGPipeline.__init__`, so change them there if needed.

## Notes and Tips

- **First run cost**: the initial run loads PDFs, detects tables, splits into chunks, embeds everything, and builds the vector store in `data/vectorstore/`. Subsequent runs just load the persisted index and are much faster.
- **Stale index**: adding/changing PDFs in `data/raw/`, or changing `chunk_size`, `chunk_overlap`, or `EMBEDDING_MODEL_NAME`, has **no effect** until you delete `data/vectorstore/` and let it rebuild.
- **Two LLM calls per question**: query expansion and answer generation each call Groq, roughly doubling token usage versus a single-call pipeline. Groq's free/on-demand tier has a fairly small daily token budget (observed: 100,000 tokens/day for `llama-3.3-70b-versatile` on one account). `retry_on_transient_groq_error` retries transient rate limits with backoff, but a genuinely exhausted *daily* quota will still exhaust the retry budget and raise a clear `RuntimeError` — that's expected and not something client-side retries can wait out.
- **Context-only answers**: the generator prompt is designed to avoid hallucinations and responds with `"Answer not found in documents"` when the answer isn't supported by the retrieved context.
- **Graceful degradation**: if query expansion fails (e.g. LLM error after retries), retrieval falls back to hybrid search on the original question only rather than failing the request.
- **Extensibility**: plug this pipeline into a CLI, API, or UI by wrapping calls to `RAGPipeline.query` / `RAGPipeline.query_with_sources`.
- `notebook/RAG_Ingestion.ipynb` is an exploratory/dev notebook, not part of the runtime path.

## Troubleshooting

- **Missing API key**: if you see authentication errors, ensure `GROQ_API_KEY` is set in `.env` and that the `.env` file is in the project root.
- **Rate limit / quota errors**: a `RateLimitError` with a TPD (tokens-per-day) reason means the daily Groq quota is exhausted for now — wait for it to reset rather than retrying immediately. Transient per-minute rate limits are already retried automatically.
- **No documents loaded**: confirm your PDFs are in `data/raw/` and that `data/vectorstore/` doesn't already exist from a run before those PDFs were added (delete it to force a rebuild).
- **Model or import errors**: reinstall dependencies with `pip install -r requirements.txt` and ensure your Python version is compatible.
