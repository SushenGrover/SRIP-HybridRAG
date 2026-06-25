"""
GraphRAG Chain
===============
Two‑stage pipeline:
  1. **Entity extraction** (Llama 3.3 70B via Groq) — pull key entities
     from the user query so we know what to look up in the knowledge graph.
  2. **Answer generation** (Llama 3.3 70B via Groq) — synthesize an answer
     from the graph-derived context, with Gemini as a fallback.
"""

import json
import logging
from typing import List, Optional

from .config import (
    GROQ_API_KEY, OPENAI_API_KEY,
    GEMINI_API_KEY, GEMINI_MODEL,
)

logger = logging.getLogger(__name__)

# ── Lazy singletons ────────────────────────────────────
_llm = None
_gemini_model = None


def _get_llm():
    """Get the primary LLM. Prefers Groq (free), falls back to OpenAI."""
    global _llm
    if _llm is not None:
        return _llm

    if GROQ_API_KEY:
        from backend.shared.groq_provider import GroqLLM
        _llm = GroqLLM()
        logger.info("GraphRAG chain using Groq (Llama 3.3 70B)")
        return _llm

    if OPENAI_API_KEY:
        from openai import OpenAI
        from .config import OPENAI_MODEL

        class _OpenAILLM:
            def __init__(self):
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                self._model = OPENAI_MODEL
                self.model_name = OPENAI_MODEL

            def generate(self, prompt, temperature=0.2, max_tokens=1024, **kw):
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()

            def generate_json(self, prompt, temperature=0.0, max_tokens=256, **kw):
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content.strip())

        _llm = _OpenAILLM()
        logger.info("GraphRAG chain using OpenAI (%s)", OPENAI_MODEL)
        return _llm

    raise ValueError("No LLM API key found. Set GROQ_API_KEY or OPEN_API_KEY in .env")


def _get_gemini():
    """Gemini fallback for answer generation."""
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model


# ── 1. Entity extraction from query ────────────────────
ENTITY_EXTRACTION_PROMPT = """You are a Named Entity Recognition expert for financial documents.

Given the following user question, extract the KEY ENTITIES that should be looked up in a knowledge graph.
Focus on: company names, person names, financial metrics, dates/periods, products, locations.

RULES:
1. Return a JSON object with key "entities" containing an array of strings.
2. Normalize names (e.g., "Apple Inc." → "Apple")
3. Include both specific and general forms (e.g., ["Apple", "revenue", "Q3 2024"])
4. Return 2-8 entities — the most important ones for answering the question.
5. Return ONLY valid JSON — no markdown fences, no commentary.

QUESTION: {query}

Return JSON:
{{"entities": ["entity1", "entity2", ...]}}
"""


def extract_query_entities(query: str) -> List[str]:
    """Use LLM to extract key entities from the user's question."""
    llm = _get_llm()

    try:
        result = llm.generate_json(
            ENTITY_EXTRACTION_PROMPT.format(query=query),
            temperature=0.0,
            max_tokens=256,
        )

        # Handle both array and {"entities": [...]} formats
        if isinstance(result, list):
            entities = [str(e).strip() for e in result if e]
        elif isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    entities = [str(e).strip() for e in v if e]
                    break
            else:
                entities = []
        else:
            entities = []

        logger.info("Extracted query entities: %s", entities)
        return entities

    except Exception as e:
        logger.error("Entity extraction failed: %s", e)
        # Fallback: split query into significant words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                       "which", "who", "when", "where", "of", "in", "for", "to",
                       "and", "or", "on", "at", "by", "with", "from", "about",
                       "does", "do", "did", "has", "have", "had", "be", "been",
                       "this", "that", "it", "its", "their", "they", "them"}
        words = [w for w in query.split() if w.lower() not in stop_words and len(w) > 2]
        return words[:6]


# ── 2. Answer generation from graph context ────────────
GRAPH_RAG_PROMPT = """You are a **Financial Document Analysis Assistant** powered by a Knowledge Graph.

You have been given a set of knowledge graph triplets extracted from a financial document.
Each triplet represents a factual relationship: (Subject) —[Relationship]→ (Object).

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS:
1. Answer using ONLY the knowledge graph triplets provided above.
2. If the answer is NOT found in the graph context, respond with:
   "⚠️ The requested information was not found in the knowledge graph. The graph may not contain this specific relationship."
3. Reference the entities and relationships you used to form the answer.
4. For financial figures, quote them EXACTLY as they appear in the graph.
5. Be precise, concise, and well-structured.
6. Do NOT use any knowledge outside the provided graph context.

ANSWER:
"""


def _build_graph_context(triplets: List[dict]) -> str:
    """Format graph triplets into a readable context block."""
    if not triplets:
        return "(No relevant graph triplets were found.)"

    def _format_props(props: dict) -> str:
        if not isinstance(props, dict) or not props:
            return ""
        # Keep it compact: up to 3 properties
        items = []
        for k, v in list(props.items())[:3]:
            items.append(f"{k}={v}")
        return " {" + ", ".join(items) + "}"

    lines = []
    for i, t in enumerate(triplets, 1):
        subj = t.get("subject", "?")
        pred = t.get("predicate", "?").replace("_", " ").title()
        obj = t.get("object", "?")
        subj_type = t.get("subject_type", "")
        obj_type = t.get("object_type", "")
        page = t.get("source_page", "?")
        subj_props = _format_props(t.get("subject_props", {}))
        obj_props = _format_props(t.get("object_props", {}))
        rel_props = _format_props(t.get("relation_props", {}))

        line = (
            f"[{i}] ({subj} [{subj_type}]){subj_props} "
            f"—[{pred}]{rel_props}→ ({obj} [{obj_type}]){obj_props}  [Page {page}]"
        )
        lines.append(line)

    return "\n".join(lines)


def run_graph_rag_chain(
    query: str,
    graph_triplets: List[dict],
    document_name: str = "uploaded document",
) -> dict:
    """
    Generate an answer from graph-derived context using Llama via Groq,
    with Gemini as a fallback.

    Returns:
        {
            "answer": str,
            "triplets_used": list[dict],
            "num_triplets": int,
            "entities_searched": list[str],  (filled by caller)
        }
    """
    graph_context = _build_graph_context(graph_triplets)

    prompt = GRAPH_RAG_PROMPT.format(
        graph_context=graph_context,
        query=query,
    )

    logger.info(
        "Generating GraphRAG answer for: '%s'  (%d graph triplets)",
        query, len(graph_triplets),
    )

    try:
        llm = _get_llm()
        answer = llm.generate(prompt, temperature=0.2, max_tokens=1024)
    except Exception as e:
        logger.error("Primary LLM answer generation failed: %s", e)
        # Fallback: try Gemini
        try:
            model = _get_gemini()
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e2:
            logger.error("Fallback Gemini also failed: %s", e2)
            answer = "⚠️ Answer generation failed. Please try again."

    return {
        "answer": answer,
        "triplets_used": graph_triplets,
        "num_triplets": len(graph_triplets),
    }
