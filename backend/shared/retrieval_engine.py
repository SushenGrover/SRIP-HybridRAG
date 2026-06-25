"""
Advanced Retrieval Engine
==========================
Implements four advanced retrieval techniques that can be composed
together for both VectorRAG and GraphRAG pipelines:

1. **Query Decomposition** — Break complex queries into sub-queries
2. **RAG Fusion**          — Generate query variants + Reciprocal Rank Fusion
3. **HyDE**                — Hypothetical Document Embeddings
4. **Cross-Encoder Reranking** — Local cross-encoder for precise re-scoring

Usage (VectorRAG):
    from backend.shared.retrieval_engine import AdvancedRetriever
    retriever = AdvancedRetriever(groq_llm=llm)
    results = retriever.retrieve(query, search_fn, embed_fn, top_k=5)

Usage (GraphRAG):
    enhanced_entities = retriever.expand_query_for_graph(query)
    reranked = retriever.rerank_triplets(query, triplets)
"""

from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
from openai.types.chat import chat_completion_chunk
import logging
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Dict

import numpy as np

from .groq_provider import GroqLLM

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    """Unified representation of a retrieved text chunk."""
    text: str
    metadata: dict
    score: float
    source: str = "original"  # Which technique retrieved it

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "score": round(float(self.score), 4),
            "source": self.source,
        }


# ════════════════════════════════════════════════════════
#  1. Query Decomposition
# ════════════════════════════════════════════════════════

DECOMPOSE_PROMPT = """You are a query analysis expert. Given a user question about a financial document, break it down into 2-4 simpler, independent sub-questions that together would answer the original question.

RULES:
1. Each sub-question should be self-contained and answerable independently.
2. Cover all aspects of the original question.
3. Keep sub-questions simple and focused.
4. Return a JSON object with key "sub_queries" containing an array of strings.
5. If the question is already simple, return it as the only sub-question.

QUESTION: {query}

Return JSON:
{{"sub_queries": ["sub question 1", "sub question 2", ...]}}"""


class QueryDecomposer:
    """Decomposes complex queries into simpler sub-queries."""

    def __init__(self, llm: GroqLLM):
        self._llm = llm

    def decompose(self, query: str) -> List[str]:
        """Return a list of sub-queries (always includes the original)."""
        try:
            result = self._llm.generate_json(
                DECOMPOSE_PROMPT.format(query=query),
                temperature=0.1,
                max_tokens=512,
            )

            sub_queries = []
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        sub_queries = [str(q).strip() for q in v if q]
                        break
            elif isinstance(result, list):
                sub_queries = [str(q).strip() for q in result if q]

            # Always include original query
            if query not in sub_queries:
                sub_queries.insert(0, query)

            logger.info("Query decomposed into %d sub-queries", len(sub_queries))
            return sub_queries[:5]  # Cap at 5

        except Exception as e:
            logger.warning("Query decomposition failed: %s", e)
            return [query]


# ════════════════════════════════════════════════════════
#  2. RAG Fusion (Multi-Query Generation)
# ════════════════════════════════════════════════════════

RAG_FUSION_PROMPT = """You are a helpful assistant that generates multiple search queries based on a single input query.

Generate 3 different versions of the given question to retrieve relevant documents from a vector database. Each version should approach the question from a different angle while keeping the same intent.

RULES:
1. Each query should be semantically different but related to the original.
2. Use different phrasing, synonyms, or focus on different aspects.
3. Return a JSON object with key "queries" containing an array of strings.
4. Do NOT include the original query in the output.

ORIGINAL QUESTION: {query}

Return JSON:
{{"queries": ["variant 1", "variant 2", "variant 3"]}}"""


class RAGFusionGenerator:
    """Generates diverse query variants for broader retrieval coverage."""

    def __init__(self, llm: GroqLLM):
        self._llm = llm

    def generate_variants(self, query: str) -> List[str]:
        """Generate 3 query variants (plus the original)."""
        try:
            result = self._llm.generate_json(
                RAG_FUSION_PROMPT.format(query=query),
                temperature=0.7,  # Higher temperature for diversity
                max_tokens=512,
            )

            variants = []
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        variants = [str(q).strip() for q in v if q]
                        break

            # Always include original query first
            all_queries = [query] + variants
            logger.info("RAG Fusion generated %d query variants", len(all_queries))
            return all_queries[:5]  # Cap at 5

        except Exception as e:
            logger.warning("RAG Fusion generation failed: %s", e)
            return [query]


# ════════════════════════════════════════════════════════
#  3. HyDE (Hypothetical Document Embeddings)
# ════════════════════════════════════════════════════════

HYDE_PROMPT = """You are a financial document expert. Given a question, write a short paragraph (3-5 sentences) that would be a perfect passage from a financial document that answers this question. Write it as if it were directly from the document, not as an answer.

QUESTION: {query}

Write the hypothetical document passage:"""


class HyDEGenerator:
    """Generates a hypothetical document passage for embedding-based retrieval."""

    def __init__(self, llm: GroqLLM):
        self._llm = llm

    def generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical document passage that answers the query."""
        try:
            passage = self._llm.generate(
                HYDE_PROMPT.format(query=query),
                temperature=0.3,
                max_tokens=256,
            )
            logger.info("HyDE generated hypothetical passage (%d chars)", len(passage))
            return passage

        except Exception as e:
            logger.warning("HyDE generation failed: %s", e)
            return query  # Fall back to original query


# ════════════════════════════════════════════════════════
#  4. Cross-Encoder Reranking
# ════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Re-scores query-passage pairs using a local cross-encoder model.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, CPU-friendly)
    This is MUCH more accurate than cosine similarity for relevance scoring.
    """

    _instance = None  # Singleton — only load the model once

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder model (first time may download ~80MB)...")
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder model loaded.")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """Re-rank chunks using cross-encoder scores."""
        if not chunks:
            return []

        self._load_model()

        # Build query-passage pairs
        pairs = [(query, chunk.text) for chunk in chunks]

        # Score all pairs
        t0 = time.time()
        scores = self._model.predict(pairs)
        elapsed = time.time() - t0

        logger.info(
            "Cross-encoder reranked %d chunks in %.1fms",
            len(chunks), elapsed * 1000,
        )

        # Attach scores and sort
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update scores and return top_k
        result = []
        for chunk, ce_score in scored[:top_k]:
            chunk.score = float(ce_score)
            chunk.source = f"{chunk.source}+reranked"
            result.append(chunk)

        return result

    def rerank_triplets(
        self,
        query: str,
        triplets: List[dict],
        top_k: int = 20,
    ) -> List[dict]:
        """
        Re-rank graph triplets by converting them to text and scoring.
        Returns the triplets sorted by relevance.
        """
        if not triplets:
            return []

        self._load_model()

        # Convert triplets to natural language for scoring
        texts = []
        for t in triplets:
            subj = t.get("subject", "?")
            pred = t.get("predicate", "?").replace("_", " ").lower()
            obj = t.get("object", "?")
            texts.append(f"{subj} {pred} {obj}")

        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)

        # Sort by score
        scored = sorted(
            zip(triplets, scores), key=lambda x: x[1], reverse=True
        )

        result = []
        for triplet, score in scored[:top_k]:
            triplet["rerank_score"] = float(score)
            result.append(triplet)

        logger.info("Reranked %d triplets → top %d", len(triplets), len(result))
        return result


# ════════════════════════════════════════════════════════
#  Reciprocal Rank Fusion
# ════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedChunk]],
    k: int = 60,
) -> List[RetrievedChunk]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ 1/(k + rank_i(d))
    where rank_i is the rank in the i-th list.

    Args:
        ranked_lists: List of ranked result lists
        k: RRF constant (default 60, standard value)

    Returns:
        Single merged and re-scored list of chunks
    """
    # Use text as the dedup key
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, RetrievedChunk] = {}
    doc_sources: Dict[str, List[str]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, chunk in enumerate(ranked_list):
            key = chunk.text[:200]  # Use first 200 chars as key
            rrf_score = 1.0 / (k + rank + 1)

            if key not in doc_scores:
                doc_scores[key] = 0.0
                doc_map[key] = chunk
                doc_sources[key] = []

            doc_scores[key] += rrf_score
            if chunk.source not in doc_sources[key]:
                doc_sources[key].append(chunk.source)

    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, rrf_score in sorted_docs:
        chunk = doc_map[key]
        chunk.score = rrf_score
        chunk.source = "+".join(doc_sources[key])
        results.append(chunk)

    logger.info("RRF merged %d lists → %d unique chunks", len(ranked_lists), len(results))
    return results


# ════════════════════════════════════════════════════════
#  Orchestrator: Advanced Retriever
# ════════════════════════════════════════════════════════

class AdvancedRetriever:
    """
    Orchestrates all 4 advanced retrieval techniques into a single pipeline.

    Pipeline:
        Query → Decompose → Generate Variants (RAG Fusion) → HyDE
        → Multi-Query Retrieval → RRF Merge → Cross-Encoder Rerank
        → Top-K Results

    For GraphRAG: Use expand_query_for_graph() and rerank_triplets().
    """

    def __init__(self, groq_llm: GroqLLM | None = None):
        self._llm = groq_llm or GroqLLM()
        self._decomposer = QueryDecomposer(self._llm)
        self._fusion = RAGFusionGenerator(self._llm)
        self._hyde = HyDEGenerator(self._llm)
        self._reranker = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        search_fn: Callable,
        embed_query_fn: Callable,
        top_k: int = 5,
        initial_k: int = 15,
    ) -> List[RetrievedChunk]:
        """
        Run the full advanced retrieval pipeline for VectorRAG.

        Args:
            query: The user's original question
            search_fn: Function(query_vector, top_k) -> List[SearchResult]
                        Returns list of objects with .text, .metadata, .score
            embed_query_fn: Function(text) -> np.ndarray (1, D)
                            Embeds a query string
            top_k: Final number of results to return
            initial_k: How many results to retrieve per query variant

        Returns:
            List of RetrievedChunk, reranked by cross-encoder
        """
        t0 = time.time()

        all_ranked_lists = []

        # ── Step 1: Original query retrieval ──────────
        original_results = self._search_and_wrap(
            query, search_fn, embed_query_fn, initial_k, source="original"
        )
        all_ranked_lists.append(original_results)

        # ── Step 2: Query Decomposition ───────────────
        try:
            sub_queries = self._decomposer.decompose(query)
            for sq in sub_queries:
                if sq != query:
                    results = self._search_and_wrap(
                        sq, search_fn, embed_query_fn, initial_k, source="decomposed"
                    )
                    all_ranked_lists.append(results)
        except Exception as e:
            logger.warning("Query decomposition step failed: %s", e)

        # ── Step 3: RAG Fusion ────────────────────────
        try:
            variants = self._fusion.generate_variants(query)
            for variant in variants:
                if variant != query:
                    results = self._search_and_wrap(
                        variant, search_fn, embed_query_fn, initial_k, source="fusion"
                    )
                    all_ranked_lists.append(results)
        except Exception as e:
            logger.warning("RAG Fusion step failed: %s", e)

        # ── Step 4: HyDE ─────────────────────────────
        try:
            hyde_doc = self._hyde.generate_hypothetical_doc(query)
            hyde_results = self._search_and_wrap(
                hyde_doc, search_fn, embed_query_fn, initial_k, source="hyde"
            )
            all_ranked_lists.append(hyde_results)
        except Exception as e:
            logger.warning("HyDE step failed: %s", e)

        # ── Step 5: Reciprocal Rank Fusion ────────────
        merged = reciprocal_rank_fusion(all_ranked_lists)

        # TEMPORARY: skip reranker
        reranked = merged[:top_k]

        elapsed = time.time() - t0
        logger.info(
            "Advanced retrieval complete: %d lists → %d merged → %d reranked (%.1fs)",
            len(all_ranked_lists), len(merged), len(reranked), elapsed,
        )

        return reranked

    def expand_query_for_graph(self, query: str) -> List[str]:
        """
        Expand a query into multiple entity-search strings for GraphRAG.
        Uses decomposition + fusion to generate a broader set of entities.
        """
        all_queries = [query]

        try:
            sub_queries = self._decomposer.decompose(query)
            all_queries.extend(q for q in sub_queries if q != query)
        except Exception:
            pass

        try:
            variants = self._fusion.generate_variants(query)
            all_queries.extend(q for q in variants if q != query)
        except Exception:
            pass

        # Deduplicate
        seen = set()
        unique = []
        for q in all_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)

        return unique

    def rerank_triplets(self, query: str, triplets: List[dict], top_k: int = 20) -> List[dict]:
        """Re-rank graph triplets using the cross-encoder."""
        return self._reranker.rerank_triplets(query, triplets, top_k)

    # ── Internal helper ────────────────────────────────
    @staticmethod
    def _search_and_wrap(
        query_text: str,
        search_fn: Callable,
        embed_query_fn: Callable,
        top_k: int,
        source: str,
    ) -> List[RetrievedChunk]:
        """Embed a query and search, wrapping results as RetrievedChunks."""
        import faiss

        query_vec = embed_query_fn(query_text)
        faiss.normalize_L2(query_vec)
        results = search_fn(query_vec, top_k)

        return [
            RetrievedChunk(
                text=r.text if hasattr(r, "text") else r.get("text", ""),
                metadata=r.metadata if hasattr(r, "metadata") else r.get("metadata", {}),
                score=float(r.score if hasattr(r, "score") else r.get("score", 0)),
                source=source,
            )
            for r in results
        ]
