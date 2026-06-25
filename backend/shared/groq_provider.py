"""
Groq LLM Provider
==================
OpenAI-compatible client for Groq-hosted Llama 3.3 70B.

Groq free tier: ~30 RPM, ~6000 TPM.  The client includes automatic
rate-limit handling with exponential backoff AND key rotation.

When GROQ_API_KEYS is set (comma-separated), the provider automatically
rotates to the next key when the current one hits a rate limit or is
exhausted, so long-running jobs (like triplet extraction) never stop.

Usage:
    from backend.shared.groq_provider import GroqLLM
    llm = GroqLLM()
    answer = llm.generate("Explain RAG in one sentence.")
    entities = llm.generate_json("Extract entities from: ...", schema_hint="entities")
"""

import json
import logging
import os
import time
from typing import Optional, List

from dotenv import load_dotenv

# Load .env from project root
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_env_path, override=True)

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────
# Load all available Groq API keys
_raw_keys = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEY_POOL: List[str] = [
    k.strip() for k in _raw_keys.split(",") if k.strip()
]

# Fallback to single GROQ_API_KEY if pool is empty
if not GROQ_API_KEY_POOL:
    single_key = os.getenv("GROQ_API_KEY", "")
    if single_key:
        GROQ_API_KEY_POOL = [single_key]

GROQ_API_KEY = GROQ_API_KEY_POOL[0] if GROQ_API_KEY_POOL else ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Rate-limit settings
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds


class GroqLLM:
    """
    Lightweight wrapper around the Groq API using the OpenAI client.

    Features:
        - API key pool: rotates through GROQ_API_KEYS when one is exhausted
        - Automatic retry with exponential backoff on rate limits
        - JSON-mode generation for structured outputs
        - Configurable model and temperature
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        from openai import OpenAI

        self._model = model or GROQ_MODEL

        # Build key pool
        if api_key:
            self._key_pool = [api_key]
        else:
            self._key_pool = list(GROQ_API_KEY_POOL)

        if not self._key_pool:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
            )

        self._current_key_idx = 0
        self._api_key = self._key_pool[0]
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=GROQ_BASE_URL,
        )

        logger.info(
            "GroqLLM initialised (model=%s, key_pool_size=%d)",
            self._model, len(self._key_pool),
        )

    def _rotate_key(self) -> bool:
        """
        Switch to the next API key in the pool.
        Returns True if rotation succeeded, False if all keys exhausted.
        """
        from openai import OpenAI

        next_idx = self._current_key_idx + 1
        if next_idx >= len(self._key_pool):
            logger.error("All %d Groq API keys exhausted!", len(self._key_pool))
            return False

        self._current_key_idx = next_idx
        self._api_key = self._key_pool[next_idx]
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=GROQ_BASE_URL,
        )

        logger.warning(
            "Rotated to Groq API key %d/%d (prefix: %s...)",
            next_idx + 1, len(self._key_pool), self._api_key[:15],
        )
        return True

    # ── Core generation ────────────────────────────────
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a text completion with retry on rate limits."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._call(messages, temperature=temperature, max_tokens=max_tokens)

    # ── JSON-mode generation ───────────────────────────
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
    ) -> dict | list:
        """
        Generate a JSON response. Uses json_object response format.
        Returns the parsed Python object (dict or list).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        raw = self._call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            import re
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            logger.warning("Failed to parse JSON from Groq response: %s", raw[:200])
            return {}

    # ── Internal call with retry + key rotation ───────
    def _call(
        self,
        messages: list,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> str:
        """Make an API call with exponential backoff + key rotation on rate limits."""
        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = (
                    "rate_limit" in error_msg
                    or "429" in error_msg
                    or "too many requests" in error_msg
                    or "rate limit" in error_msg
                    or "resource_exhausted" in error_msg
                    or "quota" in error_msg
                )

                if is_rate_limit:
                    # Try rotating to next key first
                    if self._rotate_key():
                        logger.info(
                            "Retrying with new key (attempt %d/%d)...",
                            attempt + 1, MAX_RETRIES,
                        )
                        time.sleep(1)  # Brief pause before retrying with new key
                        continue

                    # No more keys — do backoff on current key
                    if attempt < MAX_RETRIES - 1:
                        delay = BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "All keys rate-limited (attempt %d/%d), retrying in %.1fs...",
                            attempt + 1, MAX_RETRIES, delay,
                        )
                        time.sleep(delay)
                        # Reset to first key for next attempt (it may have recovered)
                        self._current_key_idx = 0
                        from openai import OpenAI
                        self._api_key = self._key_pool[0]
                        self._client = OpenAI(
                            api_key=self._api_key,
                            base_url=GROQ_BASE_URL,
                        )
                        continue

                logger.error("Groq API error: %s", e)
                raise

    @property
    def model_name(self) -> str:
        return self._model
