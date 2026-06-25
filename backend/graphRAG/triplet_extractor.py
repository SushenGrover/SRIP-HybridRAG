"""
Triplet Extractor
==================
Uses Groq-hosted Llama 3.3 70B (free tier) to extract
(subject, predicate, object) triplets from document text chunks.

Each triplet carries source metadata (page number, chunk index) so it can
be traced back to the original document passage.

Features:
    - Chunk-level resume: skips chunks that already have triplets in Neo4j
    - API key rotation: cycles through GROQ_API_KEYS on rate limits
    - Falls back to OpenAI GPT-4o if Groq is unavailable
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

from .config import GROQ_API_KEY, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ── LLM client (lazy singleton) ────────────────────────
_llm = None


def _get_llm():
    """Get the LLM client. Prefers Groq (free), falls back to OpenAI."""
    global _llm
    if _llm is not None:
        return _llm

    if GROQ_API_KEY:
        from backend.shared.groq_provider import GroqLLM
        _llm = GroqLLM()
        logger.info("Triplet extractor using Groq (Llama 3.3 70B)")
        return _llm

    if OPENAI_API_KEY:
        from openai import OpenAI
        from .config import OPENAI_MODEL
        # Wrap OpenAI in a compatible interface
        _llm = _OpenAIWrapper(OPENAI_API_KEY, OPENAI_MODEL)
        logger.info("Triplet extractor using OpenAI (%s)", OPENAI_MODEL)
        return _llm

    raise ValueError("No LLM API key found. Set GROQ_API_KEY or OPEN_API_KEY in .env")


class _OpenAIWrapper:
    """Minimal wrapper to give OpenAI the same interface as GroqLLM."""

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_json(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4096, **kw) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    @property
    def model_name(self) -> str:
        return self._model


# ── Data structures ─────────────────────────────────────
@dataclass
class Triplet:
    """A single (subject, predicate, object) knowledge‑graph triple."""
    subject: str
    predicate: str
    object: str
    subject_type: str = "ENTITY"
    subject_props: Dict[str, Any] = field(default_factory=dict)
    object_type: str = "ENTITY"
    object_props: Dict[str, Any] = field(default_factory=dict)
    relation_props: Dict[str, Any] = field(default_factory=dict)
    source_page: int = 0
    source_chunk: int = 0
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "subject_props": self.subject_props,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type,
            "object_props": self.object_props,
            "relation_props": self.relation_props,
            "source_page": self.source_page,
            "source_chunk": self.source_chunk,
            "confidence": self.confidence,
        }


# ── Extraction prompt ──────────────────────────────────
EXTRACTION_PROMPT = """You are a knowledge-graph extraction expert specializing in financial documents.

Given the following text chunk from a financial document, extract ALL meaningful entity-relationship triplets.

Entity and relationship types are FREE-FORM. Use whatever types are most accurate.
Also attach useful properties to entities/relations (e.g., value, unit, currency,
time_period, source_sentence, ratio, percent, year, quarter). Do not invent facts.

RULES:
1. Extract EVERY factual relationship — be comprehensive
2. Normalize entity names (e.g., "Apple Inc." → "Apple", "Q3 2024" → "Q3 2024")
3. Each triplet must be factual and grounded in the text
4. Assign a confidence score (0.0–1.0) based on how explicit the relationship is
5. Return ONLY valid JSON — no commentary, no markdown fences

TEXT CHUNK:
\"\"\"
{text}
\"\"\"

Return a JSON object with a single key "triplets" containing an array of objects:
{{
  "triplets": [
    {{
      "subject": "entity name",
            "subject_type": "ANY_TYPE",
            "subject_props": {{"key": "value"}},
            "predicate": "RELATIONSHIP_TYPE",
      "object": "entity name",
            "object_type": "ANY_TYPE",
            "object_props": {{"key": "value"}},
            "relation_props": {{"key": "value"}},
            "confidence": 0.95
    }}
  ]
}}

If no meaningful triplets can be extracted, return: {{"triplets": []}}"""

# How many characters per chunk to send
MAX_CHUNK_CHARS = 3000
# Rate-limit: sleep between calls (Groq free: ~30 RPM)
INTER_CALL_DELAY = 2.5  # seconds


# ── Helper: check which chunks already have triplets ───
def _get_completed_chunks(doc_id: str) -> Set[int]:
    """
    Query Neo4j to find which chunk indices already have triplets stored.
    Returns a set of chunk indices that can be skipped.
    """
    try:
        from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        with driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {doc_id: $doc_id})-[r:RELATION {doc_id: $doc_id}]->(o:Entity {doc_id: $doc_id})
                RETURN DISTINCT r.source_chunk AS chunk_idx
                """,
                doc_id=doc_id,
            ).data()

        driver.close()
        chunks = {rec["chunk_idx"] for rec in result if rec.get("chunk_idx") is not None}
        logger.info("Found %d chunks already processed in Neo4j for doc %s", len(chunks), doc_id)
        return chunks

    except Exception as e:
        logger.warning("Could not query Neo4j for completed chunks: %s", e)
        return set()


# ── Core extraction ─────────────────────────────────────
def _extract_from_chunk(text: str, page: int, chunk_idx: int) -> List[Triplet]:
    """Call LLM to extract triplets from a single chunk."""
    llm = _get_llm()

    prompt = EXTRACTION_PROMPT.format(text=text[:MAX_CHUNK_CHARS])

    try:
        parsed = llm.generate_json(prompt, temperature=0.1, max_tokens=4096)

        # Handle both array and {"triplets": [...]} formats
        if isinstance(parsed, dict):
            # Look for any array value in the dict
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                parsed = []
        if not isinstance(parsed, list):
            parsed = []

        triplets = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                t = Triplet(
                    subject=str(item.get("subject", "")).strip(),
                    subject_type=str(item.get("subject_type", "ENTITY")).strip().upper(),
                    predicate=str(item.get("predicate", "")).strip().upper().replace(" ", "_"),
                    object=str(item.get("object", "")).strip(),
                    object_type=str(item.get("object_type", "ENTITY")).strip().upper(),
                    source_page=page,
                    source_chunk=chunk_idx,
                    confidence=float(item.get("confidence", 0.8)),
                )
                # Skip empty triplets
                if t.subject and t.predicate and t.object:
                    triplets.append(t)
            except (ValueError, TypeError):
                continue

        return triplets

    except json.JSONDecodeError as e:
        logger.warning("JSON parse error for chunk %d (page %d): %s", chunk_idx, page, e)
        return []
    except Exception as e:
        logger.error("Triplet extraction failed for chunk %d: %s", chunk_idx, e)
        return []


def extract_triplets(
    chunks: List[dict],
    progress_callback=None,
    doc_id: str = "",
) -> List[Triplet]:
    """
    Extract triplets from all document chunks.

    Features:
        - Chunk-level resume: queries Neo4j to skip chunks that already
          have triplets stored (so you never re-process chunks 1–50 when
          only 51–72 need processing).
        - API key rotation handled by GroqLLM automatically.

    Args:
        chunks: List of {"text": str, "metadata": {"page_number": int, "chunk_index": int}}
        progress_callback: Optional callable(current, total) for progress updates
        doc_id: Document ID, used to check Neo4j for already-processed chunks

    Returns:
        List of Triplet objects (only newly extracted ones)
    """
    all_triplets: List[Triplet] = []
    total = len(chunks)

    # ── Check which chunks are already done ──
    completed_chunks: Set[int] = set()
    if doc_id:
        completed_chunks = _get_completed_chunks(doc_id)
        if completed_chunks:
            logger.info(
                "Resuming: %d/%d chunks already processed, %d remaining",
                len(completed_chunks), total, total - len(completed_chunks),
            )

    llm = _get_llm()
    logger.info(
        "Starting triplet extraction from %d chunks using %s (skipping %d already done)",
        total, llm.model_name, len(completed_chunks),
    )

    skipped = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        page = meta.get("page_number", 0)
        chunk_idx = meta.get("chunk_index", i)

        if not text.strip():
            continue

        # ── SKIP if this chunk is already processed ──
        if chunk_idx in completed_chunks:
            skipped += 1
            logger.info(
                "Chunk %d/%d (page %d): SKIPPED (already in Neo4j)",
                i + 1, total, page,
            )
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        triplets = _extract_from_chunk(text, page, chunk_idx)
        all_triplets.extend(triplets)

        if progress_callback:
            progress_callback(i + 1, total)

        logger.info(
            "Chunk %d/%d (page %d): extracted %d triplets",
            i + 1, total, page, len(triplets),
        )

        # Rate limiting for Groq free tier (~30 RPM = 1 every 2s)
        if i < total - 1:
            time.sleep(INTER_CALL_DELAY)

    # Deduplicate triplets (same subject-predicate-object)
    seen = set()
    unique = []
    for t in all_triplets:
        key = (t.subject.lower(), t.predicate, t.object.lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    logger.info(
        "Extraction complete: %d total → %d unique triplets (skipped %d chunks)",
        len(all_triplets), len(unique), skipped,
    )
    return unique
