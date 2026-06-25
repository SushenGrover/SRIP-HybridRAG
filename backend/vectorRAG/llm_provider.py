"""
LLM Provider Abstraction Layer
===============================
Allows easy switching between LLM providers (Gemini, OpenAI, local models)
by changing a single environment variable: LLM_PROVIDER.

Supports:
    - "openai" (default if OPENAI_API_KEY is set): OpenAI embeddings + GPT-4o
    - "gemini": Google Gemini free tier
    - "local": sentence-transformers embeddings + OpenAI-compatible generation
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Abstract base for embedding + generation providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) float32 numpy array of embeddings."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return (1, D) float32 numpy array for a query embedding."""
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the generated text response."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


# ---------------------------------------------------------------------------
# Gemini provider (free‑tier friendly)
# ---------------------------------------------------------------------------
class GeminiProvider(LLMProvider):
    """
    Uses Google Gemini API.
    - Embeddings: gemini-embedding-001  (3072-dim)
    - Generation: gemini-2.5-flash
    """

    EMBED_MODEL = "models/gemini-embedding-001"
    GEN_MODEL = "gemini-2.5-flash"
    _DIM = 3072

    def __init__(self):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
        import google.generativeai as genai

        api_key = os.getenv("VECTOR_RAG_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set VECTOR_RAG_API_KEY or GEMINI_API_KEY in .env"
            )
        genai.configure(api_key=api_key)
        self._genai = genai
        self._gen_model = genai.GenerativeModel(self.GEN_MODEL)
        logger.info("GeminiProvider initialised  (embed=%s, gen=%s)", self.EMBED_MODEL, self.GEN_MODEL)

    # -- embeddings --
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Handles Gemini's batch limit internally."""
        all_embeddings = []
        batch_size = 100  # Gemini supports up to 100 per call
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._genai.embed_content(
                model=self.EMBED_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            all_embeddings.extend(result["embedding"])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query (uses retrieval_query task type)."""
        result = self._genai.embed_content(
            model=self.EMBED_MODEL,
            content=text,
            task_type="retrieval_query",
        )
        return np.array(result["embedding"], dtype=np.float32).reshape(1, -1)

    # -- generation --
    def generate(self, prompt: str) -> str:
        response = self._gen_model.generate_content(prompt)
        return response.text

    @property
    def embedding_dim(self) -> int:
        return self._DIM


# ---------------------------------------------------------------------------
# Local / fallback provider
# ---------------------------------------------------------------------------
class LocalProvider(LLMProvider):
    """
    Fallback provider using:
    - sentence-transformers for embeddings (runs locally, no API key)
    - OpenAI-compatible API for generation (set OPENAI_API_KEY or OPEN_API_KEY + OPENAI_BASE_URL)
    """

    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
    _DIM = 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s ...", self.EMBED_MODEL_NAME)
        self._embed_model = SentenceTransformer(self.EMBED_MODEL_NAME)

        # Optional: OpenAI-compatible generation
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self._gen_model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self._openai_client = None
        if api_key:
            import openai
            self._openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
            logger.info("OpenAI-compatible generation ready (model=%s)", self._gen_model_name)
        else:
            logger.warning(
                "No OPENAI_API_KEY or OPEN_API_KEY set — generation will use a simple extractive fallback"
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_model.encode([text], convert_to_numpy=True).astype(np.float32)

    def generate(self, prompt: str) -> str:
        if self._openai_client:
            resp = self._openai_client.chat.completions.create(
                model=self._gen_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        # Ultra-basic fallback: return the context as-is
        return "⚠️ No generation model configured. Please set OPENAI_API_KEY or OPEN_API_KEY."

    @property
    def embedding_dim(self) -> int:
        return self._DIM


# ---------------------------------------------------------------------------
# OpenAI provider (preferred when OPENAI_API_KEY is set)
# ---------------------------------------------------------------------------
class OpenAIProvider(LLMProvider):
    """OpenAI embeddings + GPT-4o generation."""

    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY or OPEN_API_KEY is not set in .env")

        base_url = os.getenv("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self._gen_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self._dim = int(os.getenv("OPENAI_EMBED_DIM", "1536"))

        logger.info(
            "OpenAIProvider initialised  (embed=%s, gen=%s)",
            self._embed_model,
            self._gen_model,
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(
                model=self._embed_model,
                input=batch,
            )
            embeddings.extend([d.embedding for d in resp.data])
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self._embed_model,
            input=[text],
        )
        return np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._gen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    @property
    def embedding_dim(self) -> int:
        return self._dim




# ---------------------------------------------------------------------------
# Groq provider (free Llama 3.3 70B + Gemini embeddings)
# ---------------------------------------------------------------------------
class GroqProvider(LLMProvider):
    """
    Uses Groq API for generation (Llama 3.3 70B, free tier) and
    Gemini for embeddings (free, high-quality 768-dim vectors).

    This is the recommended zero-cost provider.
    """

    EMBED_MODEL = "models/gemini-embedding-2"
    _DIM = 768

    def __init__(self):
        import requests
        self._requests = requests
        
        embed_key = os.getenv("VECTOR_RAG_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GRAPH_RAG_API_KEY")
        if not embed_key:
            raise ValueError("API key needed for embeddings.")
        
        self._api_key = embed_key
        self._url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={self._api_key}"

        # Generation via Groq
        from backend.shared.groq_provider import GroqLLM
        self._groq = GroqLLM()

        logger.info(
            "GroqProvider initialised (embed=Gemini REST %s, gen=Groq %s)",
            self.EMBED_MODEL, self._groq.model_name,
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for text in texts:
            resp = self._requests.post(self._url, json={"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}})
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API error: {resp.text}")
            all_embeddings.append(resp.json()["embedding"]["values"])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        resp = self._requests.post(self._url, json={"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}})
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error: {resp.text}")
        return np.array(resp.json()["embedding"]["values"], dtype=np.float32).reshape(1, -1)

    def generate(self, prompt: str) -> str:
        return self._groq.generate(prompt)

    @property
    def embedding_dim(self) -> int:
        return self._DIM


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "local": LocalProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """
    Instantiate and return the requested provider.
    Reads LLM_PROVIDER env var if `name` is None.

    Priority: LLM_PROVIDER env var > GROQ_API_KEY > OPENAI_API_KEY > Gemini
    """
    if name is None:
        env_name = os.getenv("LLM_PROVIDER")
        if env_name:
            name = env_name
        elif os.getenv("GROQ_API_KEY"):
            name = "groq"
        elif os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY"):
            name = "openai"
        else:
            name = "gemini"

    name = name.lower().strip()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{name}'. Choose from: {list(_PROVIDERS.keys())}")
    return cls()

