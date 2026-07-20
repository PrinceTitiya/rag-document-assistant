# src/api.py

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import RAGPipeline

app = FastAPI(title="RAG Document Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
UPLOAD_VECTORSTORE_ROOT = "data/uploads_vectorstore"

# Wiped once at process startup (nothing has an open handle to it yet at
# this point) so old uploads don't accumulate across restarts. Individual
# per-upload subdirectories below are intentionally NOT cleaned up within a
# running process — see the comment in upload() for why.
shutil.rmtree(UPLOAD_VECTORSTORE_ROOT, ignore_errors=True)

# Built once at startup from data/raw: loads embeddings, vectorstore, and the
# LLM client, which is too slow to redo per-request. This is always the
# fallback pipeline whenever no PDF has been uploaded (or after /reset).
default_pipeline = RAGPipeline()

# The pipeline actually used by /query. Starts out as the default, and is
# swapped to a freshly-indexed pipeline over the uploaded PDF by /upload.
# Single global on purpose: this is a local, single-user tool with no auth
# or session concept, so "the document currently being discussed" is
# process-wide state, same as the rest of the pipeline.
active_pipeline = default_pipeline
active_source: str | None = None


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int


class StatusResponse(BaseModel):
    active_source: str | None


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = active_pipeline.query_with_sources(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return result


@app.get("/status", response_model=StatusResponse)
def status():
    return {"active_source": active_source}


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)):
    global active_pipeline, active_source

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # The previous upload's source file is safe to delete outright (nothing
    # keeps it open past RAGPipeline construction). Its vectorstore
    # directory is NOT deleted here: the previous RAGPipeline's Chroma
    # client may still be alive in memory (e.g. mid-request, or simply not
    # yet garbage collected), and deleting/recreating a Chroma store at a
    # path an existing client still references corrupts it ("attempt to
    # write a readonly database"). So each upload gets its own unique
    # vectorstore directory instead of reusing one path.
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    dest_path = Path(UPLOAD_DIR) / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    persist_dir = f"{UPLOAD_VECTORSTORE_ROOT}/{uuid.uuid4().hex}"

    try:
        new_pipeline = RAGPipeline(
            data_dir=UPLOAD_DIR,
            persist_dir=persist_dir,
        )
    except ValueError as e:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        shutil.rmtree(persist_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Couldn't index this PDF: {e}",
        )

    active_pipeline = new_pipeline
    active_source = file.filename

    return {
        "filename": file.filename,
        "chunks_indexed": len(new_pipeline.retriever.chunks),
    }


@app.post("/reset", response_model=StatusResponse)
def reset():
    global active_pipeline, active_source

    active_pipeline = default_pipeline
    active_source = None

    return {"active_source": active_source}
