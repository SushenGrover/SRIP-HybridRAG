"""
Vector Store (FAISS)
=====================
Manages FAISS indices for storing and retrieving document chunk embeddings.

Features:
  - Build index from document chunks
  - Similarity search with score
  - Persist / load indices to disk
"""

from openai import embeddings
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np

from .document_processor import Chunk
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Directory to store persisted indices
INDICES_DIR = os.path.join(os.path.dirname(__file__), "indices")
os.makedirs(INDICES_DIR, exist_ok=True)


@dataclass
class SearchResult:
    """A single search result with chunk text, metadata, and similarity score."""
    text: str
    metadata: dict
    score: float  # L2 distance (lower = more similar)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "score": round(float(self.score), 4),
        }


class VectorStore:
    """FAISS-backed vector store for document chunks."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunks: List[dict] = []  # parallel list: [{text, metadata}, ...]
        self.doc_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------
    def build_index(self, chunks: List[Chunk], provider: LLMProvider, doc_id: str) -> int:
        """
        Embed all chunks and build the FAISS index.
        Returns the number of vectors indexed.
        """
        self.doc_id = doc_id
        texts = [c.text for c in chunks]
        self.chunks = [c.to_dict() for c in chunks]

        logger.info("Embedding %d chunks (dim=%d) ...", len(texts), provider.embedding_dim)
        embeddings = provider.embed(texts)
        
        faiss.normalize_L2(embeddings)
        
        actual_dim = embeddings.shape[1]
        
        logger.info(
            "Creating FAISS index with actual embedding dimension=%d",
            actual_dim
        )
        
        self.index = faiss.IndexFlatIP(actual_dim)
        
        self.index.add(embeddings)

        return self.index.ntotal

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, provider: LLMProvider, top_k: int = 5) -> List[SearchResult]:
        """
        Embed the query and return top-k most similar chunks.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = provider.embed_query(query)  # (1, D)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                    score=float(score),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, doc_id: str | None = None):
        """Save the FAISS index + chunk data to disk."""
        doc_id = doc_id or self.doc_id
        if not doc_id or self.index is None:
            return

        doc_dir = os.path.join(INDICES_DIR, doc_id)
        os.makedirs(doc_dir, exist_ok=True)

        faiss.write_index(self.index, os.path.join(doc_dir, "index.faiss"))
        with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        logger.info("Index saved for doc_id=%s", doc_id)

    def load(self, doc_id: str) -> bool:
        """Load a previously saved index. Returns True if successful."""
        doc_dir = os.path.join(INDICES_DIR, doc_id)
        index_path = os.path.join(doc_dir, "index.faiss")
        chunks_path = os.path.join(doc_dir, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return False

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.doc_id = doc_id
        logger.info("Index loaded for doc_id=%s  (%d vectors)", doc_id, self.index.ntotal)
        return True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    @staticmethod
    def delete(doc_id: str):
        """Remove a persisted index."""
        import shutil
        doc_dir = os.path.join(INDICES_DIR, doc_id)
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir)
            logger.info("Deleted index for doc_id=%s", doc_id)
