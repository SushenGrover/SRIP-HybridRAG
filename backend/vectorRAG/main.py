

import hashlib
import json
import logging
import os
import shutil
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint

from .document_processor import DocumentProcessor
from .llm_provider import get_provider, LLMProvider
from .rag_chain import run_rag_chain
from .vector_store import VectorStore, SearchResult, INDICES_DIR


# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# FastAPI app

app = FastAPI(
    title="VectorRAG API",
    description="Financial document Q&A powered by Vector RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend dev server
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Shared state

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
DOCS_META_PATH = os.path.join(os.path.dirname(__file__), "documents.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

processor = DocumentProcessor()

# Cache: doc_id -> VectorStore (lazy-loaded)
_stores: dict[str, VectorStore] = {}

# LLM provider (initialised lazily to avoid import-time failures)
_provider: LLMProvider | None = None

def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


def _load_docs_meta() -> dict:
    """Load the documents metadata file."""
    if os.path.exists(DOCS_META_PATH):
        with open(DOCS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_docs_meta(meta: dict):
    """Persist documents metadata."""
    with open(DOCS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _get_store(doc_id: str) -> VectorStore:
    """Get or load a VectorStore for the given document."""
    if doc_id in _stores:
        return _stores[doc_id]
    store = VectorStore()
    if store.load(doc_id):
        _stores[doc_id] = store
        return store
    raise HTTPException(status_code=404, detail=f"No index found for document '{doc_id}'")


# Request / Response models

class QueryRequest(BaseModel):
    document_id: str
    query: str
    top_k: conint(gt=0) = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list
    num_sources: int
    query: str
    document_id: str
    time_taken_ms: int


class RetrieveResponse(BaseModel):
    sources: list
    num_sources: int
    query: str
    document_id: str
    time_taken_ms: int



# Endpoints

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a financial PDF document and build its vector index."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Generate a stable doc_id from file content hash
    file_hash = hashlib.md5(content).hexdigest()[:12]
    doc_id = f"{file_hash}"

    # Check if already indexed
    docs_meta = _load_docs_meta()
    if doc_id in docs_meta:
        return {
            "status": "already_indexed",
            "document_id": doc_id,
            "message": f"Document '{file.filename}' is already indexed.",
            **docs_meta[doc_id],
        }

    # Save uploaded file
    upload_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(upload_path, "wb") as f:
        f.write(content)

    try:
        # Process: extract text + chunk
        chunks = processor.process(upload_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF")

        # Build vector index
        provider = _get_provider()
        store = VectorStore()
        num_vectors = store.build_index(chunks, provider, doc_id)
        store.save(doc_id)
        _stores[doc_id] = store

        # Get PDF metadata
        pdf_info = DocumentProcessor.get_pdf_info(upload_path)

        # Save document metadata
        doc_meta = {
            "filename": file.filename,
            "page_count": pdf_info["page_count"],
            "file_size_kb": pdf_info["file_size_kb"],
            "num_chunks": len(chunks),
            "num_vectors": num_vectors,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        docs_meta[doc_id] = doc_meta
        _save_docs_meta(docs_meta)

        logger.info("Document uploaded and indexed: %s (id=%s, %d chunks)", file.filename, doc_id, len(chunks))

        return {
            "status": "indexed",
            "document_id": doc_id,
            "message": f"Document '{file.filename}' processed successfully.",
            **doc_meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on failure
        if os.path.exists(upload_path):
            os.remove(upload_path)
        logger.exception("Failed to process document")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    """Query an uploaded document using basic Vector RAG (simple similarity search)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    docs_meta = _load_docs_meta()
    if req.document_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{req.document_id}' not found")

    provider = _get_provider()
    store = _get_store(req.document_id)

    t0 = time.time()

    # Simple retrieval — basic vector similarity search
    results = store.search(req.query, provider, top_k=req.top_k)

    # Generate answer
    doc_name = docs_meta[req.document_id].get("filename", req.document_id)
    rag_result = run_rag_chain(
        query=req.query,
        results=results,
        provider=provider,
        document_name=doc_name,
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    return QueryResponse(
        answer=rag_result["answer"],
        sources=rag_result["sources"],
        num_sources=rag_result["num_sources"],
        query=req.query,
        document_id=req.document_id,
        time_taken_ms=elapsed_ms,
    )


@app.post("/api/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(req: QueryRequest):
    """Retrieve relevant chunks without invoking generation."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    docs_meta = _load_docs_meta()
    if req.document_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{req.document_id}' not found")

    provider = _get_provider()
    store = _get_store(req.document_id)

    t0 = time.time()
    results = store.search(req.query, provider, top_k=req.top_k)
    elapsed_ms = int((time.time() - t0) * 1000)

    return RetrieveResponse(
        sources=results,
        num_sources=len(results),
        query=req.query,
        document_id=req.document_id,
        time_taken_ms=elapsed_ms,
    )


@app.get("/api/documents")
async def list_documents():
    """List all uploaded and indexed documents."""
    docs_meta = _load_docs_meta()
    documents = []
    for doc_id, meta in docs_meta.items():
        documents.append({"document_id": doc_id, **meta})
    return {"documents": documents}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its index."""
    docs_meta = _load_docs_meta()
    if doc_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Remove index
    VectorStore.delete(doc_id)
    if doc_id in _stores:
        del _stores[doc_id]

    # Remove uploaded file
    upload_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(upload_path):
        os.remove(upload_path)

    # Remove from metadata
    filename = docs_meta[doc_id].get("filename", doc_id)
    del docs_meta[doc_id]
    _save_docs_meta(docs_meta)

    return {"status": "deleted", "message": f"Document '{filename}' deleted successfully."}


@app.get("/")
async def health():
    return {"status": "ok", "service": "VectorRAG API", "version": "1.0.0"}


"""
Vector RAG — FastAPI Application
==================================
REST API for uploading financial PDFs, building vector indices,
and querying them using the RAG pipeline.

Endpoints:
  POST   /api/upload          Upload a PDF and build its vector index
  POST   /api/query           Query an uploaded document
  GET    /api/documents       List all uploaded documents
  DELETE /api/documents/{id}  Delete a document and its index
"""