"""
HybridRAG API
============
Accepts VectorRAG and GraphRAG answers and produces a short, user-friendly
final response using OpenAI.
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_chain import compose_hybrid_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HybridRAG API",
    description="Combine VectorRAG and GraphRAG answers into a final response",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HybridQueryRequest(BaseModel):
    query: str
    vector_answer: str | None = None
    graph_answer: str | None = None


class HybridQueryResponse(BaseModel):
    answer: str
    query: str
    time_taken_ms: int


@app.post("/api/hybrid/compose", response_model=HybridQueryResponse)
async def compose_answer(req: HybridQueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t0 = time.time()

    try:
        answer = compose_hybrid_answer(
            query=req.query,
            vector_answer=req.vector_answer or "",
            graph_answer=req.graph_answer or "",
        )
    except Exception as exc:
        logger.exception("HybridRAG composition failed")
        raise HTTPException(status_code=500, detail=f"HybridRAG failed: {exc}")

    elapsed_ms = int((time.time() - t0) * 1000)

    return HybridQueryResponse(
        answer=answer,
        query=req.query,
        time_taken_ms=elapsed_ms,
    )


@app.get("/")
async def health_check():
    return {"status": "ok"}
