## RAG Question Answering on PDFs

This repository implements a Retrieval-Augmented Generation (RAG) pipeline on top of PDF documents (for example, blockchain/Ethereum whitepapers). It uses sentence-transformer embeddings, a Chroma vector store, and Google Gemini as the LLM.

### Features

- **End-to-end RAG pipeline**: load PDFs, split into chunks, embed, index, and query.
- **Config-driven**: central configuration in `src/config.py`.
- **Persistent vector store**: powered by Chroma in `data/vectorstore`.
- **Groq based answering**: uses `llama-3.1-8b-instant` via LangChain.

## Project Structure

- **`main.py`**: primary entrypoint for running the RAG app end-to-end.
- **`src/config.py`**: paths, embedding model, LLM model, and `GOOGLE_API_KEY` loading.
- **`src/pipeline.py`**: `RAGPipeline` orchestration (loader → embedder → vector store → retriever → generator).
- **`src/generator.py`**: `RAGGenerator` that builds prompts and calls Gemini.
- **`notebook/`**: experimental Jupyter notebooks (e.g. `RAG_Ingestion.ipynb`) used for prototyping and analysis only.
- **`data/raw/`**: input documents (PDFs) to be indexed.
- **`data/vectorstore/`**: persisted Chroma index (created on first run).

## Prerequisites

- **Python**: 3.9+ recommended.
- **Groq API key** with access to groq models.(Free tier)

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root (if not already present) with:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

get the key from: https://console.groq.com/keys

The key is read in `src/config.py` via `python-dotenv`.

## Preparing Documents

Place your PDF files in:

```text
data/raw/
```

Examples in this repo include:

- `data/raw/Ethereum-whitepaper.pdf`
- `data/raw/Blockchain_For_Beginners.pdf`

Any PDFs in `data/raw/` will be picked up by the document loader used inside the pipeline.

## Usage

### Running the main app

From the project root, run:

```bash
python3 main.py
```

This will execute the main RAG pipeline using the configuration in `src/config.py`.

### Basic Python usage (library-style)

In a Python shell or another script:

```python
from src.pipeline import RAGPipeline

# First run: builds embeddings and vector store, which may take some time.
pipeline = RAGPipeline()

question = input("Enter the query")
answer = pipeline.query(question)

print("Q:", question)
print("A:", answer)
```

## Configuration

Key configuration values live in `src/config.py`:

- **`DATA_DIR`**: directory containing raw documents (`data/raw` by default).
- **`VECTORSTORE_DIR`**: where the Chroma DB is persisted (`data/vectorstore`).
- **`EMBEDDING_MODEL_NAME`**: sentence-transformer model (default `sentence-transformers/all-MiniLM-L6-v2`).
- **`LLM_MODEL_NAME`**: Groq model used for generation (default `llama-3.1-8b-instant`).
- **`Groq_API_KEY`**: loaded from `.env`.

You can override these either by editing `src/config.py` or by passing custom values into `RAGPipeline` when instantiating it.

## Notes and Tips

- **First run cost**: The initial run will load PDFs, split them into chunks, and build the vector store in `data/vectorstore`. Subsequent runs reuse this index and are much faster.
- **Context-only answers**: The generator prompt is designed to avoid hallucinations and will respond with `"Answer not found in documents"` when the answer is not supported by the retrieved context.
- **Extensibility**: You can plug this pipeline into a CLI, API, or UI by wrapping calls to `RAGPipeline.query` / `query_with_sources`.

## Troubleshooting

- **Missing API key**: If you see authentication errors, ensure `GOOGLE_API_KEY` is set in `.env` and that the `.env` file is in the project root.
- **No documents loaded**: Confirm your PDFs are in `data/raw/` and that the loader used in `DocumentLoader` supports their format.
- **Model or import errors**: Reinstall dependencies with `pip install -r requirements.txt` and ensure your Python version is compatible.
