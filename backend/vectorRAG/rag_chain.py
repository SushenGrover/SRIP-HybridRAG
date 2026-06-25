"""
RAG Chain
==========
Constructs the retrieval-augmented generation prompt and generates
grounded answers from retrieved document chunks.

Key design decisions:
  - The prompt explicitly instructs the LLM to refuse answering if
    the information is not in the provided context (anti-hallucination).
  - Source citations are included with page references.
  - Financial figures must be quoted exactly as they appear.
"""

import logging
from typing import List

from .vector_store import SearchResult
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a **Financial Document Analysis Assistant** built for extracting precise information from corporate financial documents such as earnings call transcripts, annual reports, and investor presentations.

Your responses must be **strictly grounded** in the provided document context. You are an expert at reading dense financial language and extracting key insights."""

RAG_PROMPT_TEMPLATE = """
DOCUMENT: "{document_name}"

RETRIEVED CONTEXT:
{context_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS:
1. Answer the question using ONLY the information found in the RETRIEVED CONTEXT above.
2. If the answer is NOT present in the context, respond EXACTLY with:
   "⚠️ The requested information was not found in the provided document sections. The document may not contain this specific information, or it may be in sections not retrieved by the search."
3. Cite your sources by referencing [Page X] where the information was found.
4. For financial figures (revenue, profit, percentages, ratios), quote them EXACTLY as they appear in the context. Do NOT round, estimate, or modify numbers.
5. Be precise and concise. Provide a well-structured answer.
6. If the context contains partial information, clearly state what was found and what is missing.
7. Do NOT use any knowledge outside the provided context. Do NOT speculate or infer beyond what is explicitly stated.

ANSWER:
"""


# ---------------------------------------------------------------------------
# Build context block from search results
# ---------------------------------------------------------------------------
def _build_context_block(results: List[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block with page refs."""
    if not results:
        return "(No relevant passages were retrieved.)"

    parts = []
    for i, r in enumerate(results, 1):
        page = r.metadata.get("page_number", "?")
        source = r.metadata.get("source_file", "unknown")
        similarity = f"{r.score:.2f}"
        header = f"[Passage {i}]  —  Source: {source}, Page {page}  (relevance: {similarity})"
        parts.append(f"{header}\n{r.text}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAG chain execution
# ---------------------------------------------------------------------------
def run_rag_chain(
    query: str,
    results: List[SearchResult],
    provider: LLMProvider,
    document_name: str = "uploaded document",
) -> dict:
    """
    Build the prompt from retrieved results and generate an answer.

    Returns:
        {
            "answer": str,
            "sources": list[dict],   # the chunks used
            "num_sources": int,
        }
    """
    context_block = _build_context_block(results)

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + RAG_PROMPT_TEMPLATE.format(
            document_name=document_name,
            context_block=context_block,
            query=query,
        )
    )

    logger.info("Generating answer for query: '%s'  (%d source chunks)", query, len(results))

    answer = provider.generate(prompt)

    return {
        "answer": answer.strip(),
        "sources": [r.to_dict() for r in results],
        "num_sources": len(results),
    }
