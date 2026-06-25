"""
GraphRAG — FastAPI Application
=================================
REST API for uploading financial PDFs, extracting knowledge graph triplets,
storing them in Neo4j, and querying via the GraphRAG pipeline.

Runs on port 8001 alongside the VectorRAG server (port 8000).

Endpoints:
  POST   /api/graph/upload              Upload a PDF → extract triplets → store in Neo4j
  POST   /api/graph/query               Query a document using the GraphRAG pipeline
  GET    /api/graph/visualize/{doc_id}   Get graph data for frontend visualisation
  GET    /api/graph/documents            List documents with knowledge graphs
  DELETE /api/graph/documents/{doc_id}   Delete a document's knowledge graph
  GET    /                               Health check
"""

import hashlib
import json
import logging
import os
import time

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from .config import UPLOAD_DIR, DOCS_META_PATH
from .triplet_extractor import extract_triplets
from .graph_store import (
    store_triplets,
    query_graph,
    get_document_graph,
    delete_document_graph,
    list_graph_documents,
    close_driver,
)
from .rag_chain import extract_query_entities, run_graph_rag_chain

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────
app = FastAPI(
    title="GraphRAG API",
    description="Financial document Q&A powered by Knowledge Graph RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Text splitter (matches vectorRAG settings) ─────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Document metadata helpers ──────────────────────────
def _load_docs_meta() -> dict:
    if os.path.exists(DOCS_META_PATH):
        with open(DOCS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_docs_meta(meta: dict):
    with open(DOCS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _count_neo4j_triplets(doc_id: str) -> int:
    """Query Neo4j for the total number of relationships for a document."""
    from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Entity {doc_id: $doc_id})-[r:RELATION {doc_id: $doc_id}]->(o:Entity {doc_id: $doc_id})
            RETURN count(r) AS total
            """,
            doc_id=doc_id,
        ).single()
    driver.close()
    return result["total"] if result else 0


# ── PDF processing ─────────────────────────────────────
def _extract_and_chunk(pdf_path: str) -> list[dict]:
    """Extract text from PDF and split into chunks with metadata."""
    doc = fitz.open(pdf_path)
    chunks = []
    global_idx = 0

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if not text.strip():
                continue

            splits = _splitter.split_text(text)
            for local_idx, chunk_text in enumerate(splits):
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        "source_file": os.path.basename(pdf_path),
                        "page_number": page_num + 1,
                        "chunk_index": global_idx,
                        "page_chunk_index": local_idx,
                    },
                })
                global_idx += 1
    finally:
        doc.close()

    return chunks


# ── Request / Response models ──────────────────────────
class GraphQueryRequest(BaseModel):
    document_id: str
    query: str


class GraphQueryResponse(BaseModel):
    answer: str
    triplets_used: list
    num_triplets: int
    entities_searched: list
    query: str
    document_id: str
    time_taken_ms: int


class GraphRetrieveResponse(BaseModel):
    triplets_used: list
    num_triplets: int
    entities_searched: list
    query: str
    document_id: str
    time_taken_ms: int


# ── Endpoints ──────────────────────────────────────────

@app.post("/api/graph/upload")
async def upload_and_build_graph(file: UploadFile = File(...)):
    """Upload a PDF, extract triplets with GPT-4o, store in Neo4j."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Stable doc_id from content hash (matches vectorRAG)
    file_hash = hashlib.md5(content).hexdigest()[:12]
    doc_id = file_hash

    # Check if already processed
    docs_meta = _load_docs_meta()
    if doc_id in docs_meta:
        # Check if extraction was COMPLETE or partial
        # by querying Neo4j for how many chunks actually have triplets
        existing_meta = docs_meta[doc_id]
        expected_chunks = existing_meta.get("num_chunks", 0)

        try:
            from .triplet_extractor import _get_completed_chunks
            completed_chunks = _get_completed_chunks(doc_id)
            completed_count = len(completed_chunks)
        except Exception:
            completed_count = expected_chunks  # assume complete if can't check

        if completed_count >= expected_chunks and expected_chunks > 0:
            # Fully processed — no need to re-run
            return {
                "status": "already_indexed",
                "document_id": doc_id,
                "message": f"Knowledge graph for '{file.filename}' already exists ({completed_count}/{expected_chunks} chunks processed).",
                **existing_meta,
            }
        else:
            # Partially processed — continue below to resume extraction
            logger.warning(
                "Document '%s' partially processed (%d/%d chunks). Resuming...",
                file.filename, completed_count, expected_chunks,
            )

    # Save file
    upload_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(upload_path, "wb") as f:
        f.write(content)

    try:
        t0 = time.time()

        # Step 1: Extract text & chunk
        chunks = _extract_and_chunk(upload_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        logger.info("Extracted %d chunks from '%s'", len(chunks), file.filename)

        # Step 2: Extract triplets using GPT-4o
        triplets = extract_triplets(chunks, doc_id=doc_id)

        if not triplets:
            logger.warning("No triplets extracted from '%s'", file.filename)

        # Step 3: Store in Neo4j
        stored_count = store_triplets(doc_id, triplets)

        elapsed = time.time() - t0

        # Get TOTAL triplet count from Neo4j (old + new combined)
        try:
            total_triplets_in_neo4j = _count_neo4j_triplets(doc_id)
        except Exception:
            # Fallback: add new to any previously recorded count
            prev_count = docs_meta.get(doc_id, {}).get("num_triplets", 0)
            total_triplets_in_neo4j = prev_count + len(triplets)

        # Save metadata
        doc_meta = {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "num_triplets": total_triplets_in_neo4j,
            "stored_in_neo4j": total_triplets_in_neo4j,
            "processing_time_s": round(elapsed, 1),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        docs_meta[doc_id] = doc_meta
        _save_docs_meta(docs_meta)

        logger.info(
            "GraphRAG upload complete: '%s' → %d chunks → %d triplets → %d stored  (%.1fs)",
            file.filename, len(chunks), len(triplets), stored_count, elapsed,
        )

        return {
            "status": "indexed",
            "document_id": doc_id,
            "message": f"Knowledge graph built for '{file.filename}'.",
            **doc_meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        logger.exception("Failed to process document for GraphRAG")
        raise HTTPException(status_code=500, detail=f"GraphRAG processing failed: {str(e)}")


@app.post("/api/graph/query", response_model=GraphQueryResponse)
async def query_document_graph(req: GraphQueryRequest):
    """Query a document using the GraphRAG pipeline with advanced retrieval."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    docs_meta = _load_docs_meta()
    if req.document_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{req.document_id}' not found in GraphRAG")

    t0 = time.time()

    # Step 1: Extract entities from the query
    entities = extract_query_entities(req.query)

    # Step 1b: Expand entities using advanced retrieval (decomposition + fusion)
    try:
        from backend.shared.groq_provider import GroqLLM
        from backend.shared.retrieval_engine import AdvancedRetriever
        retriever = AdvancedRetriever(groq_llm=GroqLLM())

        # Get expanded queries for broader entity coverage
        expanded_queries = retriever.expand_query_for_graph(req.query)
        # Extract entities from each expanded query
        all_entities = list(entities)
        for eq in expanded_queries:
            if eq != req.query:
                extra = extract_query_entities(eq)
                all_entities.extend(e for e in extra if e not in all_entities)
        entities = all_entities[:12]  # Cap at 12 entities
        logger.info("Advanced entity expansion: %d entities", len(entities))
    except Exception as e:
        logger.warning("Advanced entity expansion not available: %s", e)

    # Step 2: Query the knowledge graph
    graph_triplets = query_graph(req.document_id, entities, max_hops=2)

    # Step 2b: Rerank triplets using cross-encoder
    try:
        from backend.shared.retrieval_engine import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        graph_triplets = reranker.rerank_triplets(req.query, graph_triplets, top_k=30)
        logger.info("Cross-encoder reranked %d triplets", len(graph_triplets))
    except Exception as e:
        logger.warning("Cross-encoder reranking not available: %s", e)

    # Step 3: Generate answer from graph context
    doc_name = docs_meta[req.document_id].get("filename", req.document_id)
    result = run_graph_rag_chain(
        query=req.query,
        graph_triplets=graph_triplets,
        document_name=doc_name,
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    return GraphQueryResponse(
        answer=result["answer"],
        triplets_used=result["triplets_used"],
        num_triplets=result["num_triplets"],
        entities_searched=entities,
        query=req.query,
        document_id=req.document_id,
        time_taken_ms=elapsed_ms,
    )



@app.post("/api/graph/retrieve", response_model=GraphRetrieveResponse)
async def retrieve_graph_triplets(req: GraphQueryRequest):
    """Retrieve graph triplets without invoking generation."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    docs_meta = _load_docs_meta()
    if req.document_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{req.document_id}' not found in GraphRAG")

    t0 = time.time()
    entities = extract_query_entities(req.query)
    graph_triplets = query_graph(req.document_id, entities, max_hops=2)
    elapsed_ms = int((time.time() - t0) * 1000)

    return GraphRetrieveResponse(
        triplets_used=graph_triplets,
        num_triplets=len(graph_triplets),
        entities_searched=entities,
        query=req.query,
        document_id=req.document_id,
        time_taken_ms=elapsed_ms,
    )


@app.get("/api/graph/visualize/{doc_id}")
async def visualize_graph(doc_id: str):
    """Return the full knowledge graph for a document (vis.js format)."""
    docs_meta = _load_docs_meta()
    if doc_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    graph_data = get_document_graph(doc_id)
    return {
        "document_id": doc_id,
        "filename": docs_meta[doc_id].get("filename", doc_id),
        **graph_data,
    }


@app.get("/api/graph/documents")
async def list_documents():
    """List all documents with knowledge graphs."""
    docs_meta = _load_docs_meta()
    documents = []
    for doc_id, meta in docs_meta.items():
        documents.append({"document_id": doc_id, **meta})
    return {"documents": documents}


@app.delete("/api/graph/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document's knowledge graph and metadata."""
    docs_meta = _load_docs_meta()
    if doc_id not in docs_meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Delete from Neo4j
    deleted = delete_document_graph(doc_id)

    # Remove uploaded file
    upload_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(upload_path):
        os.remove(upload_path)

    # Remove metadata
    filename = docs_meta[doc_id].get("filename", doc_id)
    del docs_meta[doc_id]
    _save_docs_meta(docs_meta)

    return {
        "status": "deleted",
        "message": f"Knowledge graph for '{filename}' deleted ({deleted} nodes removed).",
    }


@app.get("/")
async def health():
    return {"status": "ok", "service": "GraphRAG API", "version": "1.0.0"}


@app.on_event("shutdown")
async def shutdown():
    close_driver()
