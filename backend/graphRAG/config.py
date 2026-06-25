"""
GraphRAG Configuration
=======================
Loads all environment variables from the project root .env file.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_env_path, override=True)

# ── Groq (Llama 3.3 70B — free tier) ────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Gemini (answer generation fallback — free tier) ──────
GEMINI_API_KEY = os.getenv("GRAPH_RAG_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Legacy OpenAI (kept for backward compat, not used) ───
OPENAI_API_KEY = os.getenv("OPEN_API_KEY", "")
OPENAI_MODEL = "gpt-4o"

# ── Neo4j ────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_INSTANCE_PASSWORD", "")

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DOCS_META_PATH = os.path.join(BASE_DIR, "documents.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
