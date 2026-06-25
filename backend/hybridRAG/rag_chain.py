"""
HybridRAG Chain
==============
Combines VectorRAG and GraphRAG answers into a short, user-friendly response.
"""

import logging

from .config import GROQ_API_KEY, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if GROQ_API_KEY:
        from backend.shared.groq_provider import GroqLLM
        _llm = GroqLLM()
        logger.info("HybridRAG chain using Groq (Llama 3.3 70B)")
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

        _llm = _OpenAILLM()
        logger.info("HybridRAG chain using OpenAI (%s)", OPENAI_MODEL)
        return _llm

    raise ValueError("No LLM API key found. Set GROQ_API_KEY or OPEN_API_KEY in .env")


def _is_empty_or_unhelpful(text: str) -> bool:
    if not text:
        return True
    lowered = text.strip().lower()
    if not lowered:
        return True
    return "requested information was not found in the knowledge graph" in lowered


PROMPT_TEMPLATE = """You are a helpful assistant. Your job is to produce a short, clear answer for an end user.

You are given a user question and two draft answers:
- VectorRAG answer (may be incomplete)
- GraphRAG answer (may be incomplete)

Rules:
1) Write a single short paragraph in simple English.
2) Do NOT mention vectors, graphs, triplets, sources, pages, or internal methods.
3) If both answers agree, use that wording.
4) If they disagree, prefer the answer that is more direct and factual.
5) If both are empty or unhelpful, say you cannot find the answer in the document.

User question:
{query}

VectorRAG answer:
{vector_answer}

GraphRAG answer:
{graph_answer}

Final answer:
"""


def compose_hybrid_answer(query: str, vector_answer: str, graph_answer: str) -> str:
    vector_answer = (vector_answer or "").strip()
    graph_answer = (graph_answer or "").strip()

    if _is_empty_or_unhelpful(vector_answer) and _is_empty_or_unhelpful(graph_answer):
        return "I could not find this answer in the document."

    prompt = PROMPT_TEMPLATE.format(
        query=query.strip(),
        vector_answer=vector_answer or "(empty)",
        graph_answer=graph_answer or "(empty)",
    )

    llm = _get_llm()
    return llm.generate(prompt, temperature=0.2, max_tokens=200)
