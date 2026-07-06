
<h1 align="center">HybridRAG — Advanced Hybrid Retrieval-Augmented Generation Pipeline</h1>

<p align="center">
  <em>Combining Vector Databases &amp; Knowledge Graphs for Superior Document Intelligence</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Neo4j-4581C3?logo=neo4j&logoColor=white" alt="Neo4j"/>
  <img src="https://img.shields.io/badge/FAISS-Facebook-blue" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Llama_3.3_70B-Groq-orange" alt="Groq"/>
  <img src="https://img.shields.io/badge/Gemini-Google-4285F4?logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Services](#running-the-services)
- [API Reference](#-api-reference)
- [Advanced Retrieval Techniques](#-advanced-retrieval-techniques)
- [Evaluation (RAGAS)](#-evaluation-ragas)
- [Frontend](#-frontend)
- [Research Context](#-research-context)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## Overview

**HybridRAG** is a production-grade Retrieval-Augmented Generation system that overcomes the limitations of traditional vector-only RAG by fusing two complementary retrieval paradigms:

| Paradigm | Engine | Strength |
|---|---|---|
| **VectorRAG** | FAISS (cosine similarity) | Semantic meaning, colloquial phrasing |
| **GraphRAG** | Neo4j (2-hop traversal) | Multi-hop reasoning, structured relationships |

A final **HybridRAG** orchestrator queries both engines concurrently, fuses the results, and synthesises a single, grounded answer — achieving **higher faithfulness, context recall, and answer relevancy** than either engine alone.

> **Headline Result:** The open-source Llama 3.3 70B pipeline with advanced retrieval techniques (HyDE, RAG Fusion, Cross-Encoder Reranking) **matched or exceeded** the performance of a GPT-4o baseline on all RAGAS metrics, at **zero inference cost**.

---

## Key Features

- **Dual Retrieval** — Semantic vector search (FAISS) + structural knowledge graph traversal (Neo4j) running concurrently.
- **Advanced Retrieval Engine** — Pluggable pipeline of 4 state-of-the-art techniques:
  - 🔀 **Query Decomposition** — Breaks complex multi-clause queries into simpler sub-queries.
  - 🔄 **RAG Fusion** — Generates diverse query variants + Reciprocal Rank Fusion.
  - 💡 **HyDE** — Hypothetical Document Embeddings for vague/colloquial queries.
  - 🎯 **Cross-Encoder Reranking** — Local `ms-marco-MiniLM-L-6-v2` model for precision re-scoring.
- **Zero-Cost Inference** — Primary generation via Groq-hosted Llama 3.3 70B (free tier), Gemini 2.5 Flash as fallback.
- **API Key Rotation** — Automatic key pool cycling with exponential backoff for uninterrupted long-running jobs.
- **Chunk-Level Resume** — Triplet extraction queries Neo4j to skip already-processed chunks, enabling safe restarts.
- **Interactive Frontend** — Web UI for document upload, querying across all 3 pipelines, and live knowledge graph visualisation.
- **RAGAS Evaluation** — Rigorous benchmarking with Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/JS/CSS)                   │
│                    Upload PDFs · Query · Visualise KG           │
└────────────┬─────────────────┬──────────────────┬───────────────┘
             │                 │                  │
        ┌────▼────┐       ┌────▼────┐       ┌─────▼─────┐
        │ Vector  │       │  Graph  │       │  Hybrid   │
        │  RAG    │       │   RAG   │       │   RAG     │
        │ :8000   │       │  :8001  │       │  :8002    │
        └────┬────┘       └────┬────┘       └─────┬─────┘
             │                 │                  ▼
      ┌──────▼──────┐   ┌──────▼──────┐    Combines both
      │    FAISS    │   │   Neo4j     │    answers via LLM
      │ Vector Index│   │ Knowledge   │
      │ (cosine sim)│   │   Graph     │
      └──────┬──────┘   └──────┬──────┘
             │                 │
      ┌──────▼─────────────────▼───────┐
      │     Shared Retrieval Engine    │
      │  Decompose → Fuse → HyDE →     │
      │  RRF Merge → Cross-Encoder     │
      └────────────────────────────────┘
             │
      ┌──────▼──────┐
      │  Groq LLM   │  ← Llama 3.3 70B (free)
      │  + Gemini   │  ← Fallback
      └─────────────┘
```

### Data Flow

1. **Upload** → PDF parsed with PyMuPDF → chunked via `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
2. **VectorRAG** → Chunks embedded (Gemini Embedding-2) → stored in FAISS `IndexFlatIP`
3. **GraphRAG** → Chunks sent to LLM → (Subject, Predicate, Object) triplets extracted → stored in Neo4j
4. **Query** → Both engines queried concurrently → results fused + reranked → answer generated

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM (Primary)** | Llama 3.3 70B via [Groq](https://groq.com) | Generation, entity extraction, triplet extraction |
| **LLM (Fallback)** | Gemini 2.5 Flash | Fallback generation |
| **Embeddings** | Gemini Embedding-2 (768-dim) | Vector embeddings for FAISS |
| **Vector Store** | FAISS (`IndexFlatIP`) | Semantic nearest-neighbor retrieval |
| **Graph Store** | Neo4j | Knowledge graph storage + Cypher traversal |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder relevance scoring |
| **Backend** | FastAPI + Uvicorn | REST API serving |
| **PDF Parsing** | PyMuPDF (fitz) | Text extraction with OCR fallback |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` | Overlap-aware chunking |
| **Frontend** | Vanilla HTML/CSS/JS | Interactive web interface |
| **Evaluation** | RAGAS Framework | Faithfulness, Relevancy, Precision, Recall |

---

## Project Structure

```
SRIP/
├── .env                          # API keys & database credentials (git-ignored)
├── main_fig.png                  # Architecture diagram
├── projectReport.md              # Full research report
│
├── backend/
│   ├── vectorRAG/                # ── VectorRAG Service (port 8000) ──
│   │   ├── main.py               #   FastAPI app: upload, query, retrieve
│   │   ├── document_processor.py #   PDF → text → chunks pipeline
│   │   ├── vector_store.py       #   FAISS index management
│   │   ├── rag_chain.py          #   Prompt construction + generation
│   │   ├── llm_provider.py       #   Multi-provider abstraction (Groq/Gemini/OpenAI/Local)
│   │   ├── requirements.txt      #   Python dependencies
│   │   ├── uploads/              #   Stored PDF uploads
│   │   └── indices/              #   Persisted FAISS indices
│   │
│   ├── graphRAG/                 # ── GraphRAG Service (port 8001) ──
│   │   ├── main.py               #   FastAPI app: upload, query, visualize
│   │   ├── triplet_extractor.py  #   LLM-powered (S, P, O) extraction
│   │   ├── graph_store.py        #   Neo4j driver, CRUD, subgraph queries
│   │   ├── rag_chain.py          #   Entity extraction + graph-grounded generation
│   │   ├── config.py             #   Environment config loader
│   │   └── requirements.txt      #   Python dependencies
│   │
│   ├── hybridRAG/                # ── HybridRAG Service (port 8002) ──
│   │   ├── main.py               #   FastAPI app: compose hybrid answers
│   │   ├── rag_chain.py          #   Vector+Graph answer fusion logic
│   │   ├── config.py             #   Environment config loader
│   │   └── requirements.txt      #   Python dependencies
│   │
│   └── shared/                   # ── Shared Utilities ──
│       ├── groq_provider.py      #   Groq API client with key rotation + backoff
│       └── retrieval_engine.py   #   Advanced retrieval pipeline (4 techniques)
│
├── frontend/
│   ├── index.html                # Main UI entry point
│   ├── app.js                    # Frontend logic (API calls, KG rendering)
│   └── style.css                 # Styling
│
├── scripts/
│   ├── batch_eval.py             # Batch RAGAS evaluation runner
│   └── refresh_graph_hybrid.py   # Refresh graph and hybrid answers
│
├── analysis1/                    # Analysis 1: GPT-4o baseline (1K words, 18 Qs)
│   ├── charts/                   #   Generated comparison charts
│   └── consolidate_and_visualize.py
│
├── analysis2/                    # Analysis 2: Groq/Llama-3 advanced (1K words, 18 Qs)
│
├── analysis3/                    # Analysis 3: Stress test (10K words, 50 Qs)
│
├── docs/                         # Test PDF documents
└── viz/                          # Knowledge graph visualisations
```

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.10+ | 3.11 or 3.12 recommended |
| **Neo4j** | 5.x | Community Edition works fine. [Install Neo4j](https://neo4j.com/download/) |
| **Node.js** | (optional) | Only if you want to use a dev server for the frontend |

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/SRIP-HybridRAG.git
   cd SRIP-HybridRAG
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r backend/vectorRAG/requirements.txt
   pip install -r backend/graphRAG/requirements.txt
   pip install -r backend/hybridRAG/requirements.txt
   ```

4. **Start Neo4j:**
   Ensure your Neo4j instance is running locally on `bolt://127.0.0.1:7687`.

### Environment Variables

Create a `.env` file in the project root with the following keys:

```env
# ── Groq (Llama 3.3 70B — free tier, primary LLM) ──
GROQ_API_KEYS=<comma-separated-groq-api-keys>
GROQ_MODEL=llama-3.3-70b-versatile

# ── Gemini (embeddings + fallback generation) ──
VECTOR_RAG_API_KEY=<your-gemini-api-key>
GRAPH_RAG_API_KEY=<your-gemini-api-key>

# ── OpenAI (optional legacy fallback) ──
OPEN_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_EMBED_DIM=1536

# ── Neo4j ──
NEO4J_CONNECTION_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_INSTANCE_PASSWORD=<your-neo4j-password>
```

> **Free API Keys:**
> - Groq: [console.groq.com](https://console.groq.com) (30 RPM free tier)
> - Gemini: [aistudio.google.com](https://aistudio.google.com) (free tier)

### Running the Services

Start each microservice in a separate terminal:

```bash
# Terminal 1 — VectorRAG (port 8000)
uvicorn backend.vectorRAG.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — GraphRAG (port 8001)
uvicorn backend.graphRAG.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3 — HybridRAG (port 8002)
uvicorn backend.hybridRAG.main:app --host 0.0.0.0 --port 8002 --reload
```

Then open `frontend/index.html` in your browser (or serve it via a local HTTP server).

---

## API Reference

### VectorRAG — `localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a PDF and build its FAISS vector index |
| `POST` | `/api/query` | Query a document (advanced retrieval + generation) |
| `POST` | `/api/retrieve` | Retrieve relevant chunks without generation |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document and its index |

### GraphRAG — `localhost:8001`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/graph/upload` | Upload PDF → extract triplets → store in Neo4j |
| `POST` | `/api/graph/query` | Query via knowledge graph traversal + generation |
| `POST` | `/api/graph/retrieve` | Retrieve graph triplets without generation |
| `GET` | `/api/graph/visualize/{doc_id}` | Get full knowledge graph (vis.js format) |
| `GET` | `/api/graph/documents` | List documents with knowledge graphs |
| `DELETE` | `/api/graph/documents/{doc_id}` | Delete a document's knowledge graph |

### HybridRAG — `localhost:8002`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/hybrid/compose` | Combine VectorRAG + GraphRAG answers into a final response |

<details>
<summary><strong>Example: Query a Document</strong></summary>

```bash
# Upload a PDF to VectorRAG
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"

# Query it
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "4ec93fc12bff",
    "query": "What is the grace period for premium payment?",
    "top_k": 5
  }'
```

</details>

---

## Advanced Retrieval Techniques

The `backend/shared/retrieval_engine.py` module implements a composable pipeline of four techniques:

```
User Query
    │
    ├──→ [1] Query Decomposition ──→ sub-queries
    │
    ├──→ [2] RAG Fusion ───────────→ query variants
    │
    ├──→ [3] HyDE ─────────────────→ hypothetical document
    │
    ▼
Multi-Query Retrieval (FAISS / Neo4j)
    │
    ▼
[4] Reciprocal Rank Fusion (RRF)
    │
    ▼
[5] Cross-Encoder Reranking
    │
    ▼
Top-K Results → LLM Generation
```

| Technique | Purpose | Impact |
|---|---|---|
| **Query Decomposition** | Handles multi-clause questions (e.g., *"How does X affect Y and what is the timeline?"*) | ↑ Context Recall |
| **RAG Fusion** | Generates semantically diverse reformulations of the query | ↑ Context Coverage |
| **HyDE** | Creates a hypothetical answer passage, then embeds *that* for retrieval | ↑ Precision for vague queries |
| **Cross-Encoder** | Precisely re-scores every query–passage pair using a fine-tuned bi-encoder | ↑ Context Precision |

---

## Evaluation (RAGAS)

The pipeline was rigorously evaluated using the [RAGAS](https://docs.ragas.io/) framework across 3 analyses:

### Analysis 1 — Baseline (GPT-4o, 1K words, 18 questions)
- High Faithfulness (~0.92) due to GPT-4o's strong reasoning
- Moderate Context Recall (~0.75) — standard FAISS missed interconnected clauses

### Analysis 2 — Advanced Pipeline (Groq/Llama-3, 1K words, 18 questions)
- Significant **Context Precision improvement** (~0.88) with HyDE + Cross-Encoder
- **Matched or exceeded** GPT-4o baseline in Answer Relevancy — proving retrieval > model size

### Analysis 3 — Stress Test (Groq/Llama-3, 10K words, 50 questions)
- **GraphRAG outperformed VectorRAG** in Multi-Clause Retrieval
- **VectorRAG outperformed GraphRAG** for colloquial Realistic User Queries
- **HybridRAG achieved the highest overall scores** across all categories
- Perfect hallucination resistance on out-of-scope questions

### RAGAS Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer rely *strictly* on the provided context? |
| **Answer Relevancy** | Does the answer directly address the question? |
| **Context Precision** | Are the most relevant chunks ranked highest? |
| **Context Recall** | Did retrieval fetch *all* necessary information? |

---

## Frontend

The web interface provides:

- ** PDF Upload** — Drag-and-drop document upload to both VectorRAG and GraphRAG
- ** Multi-Pipeline Query** — Run the same question against Vector, Graph, and Hybrid pipelines side-by-side
- ** Knowledge Graph Visualisation** — Interactive vis.js rendering of the extracted knowledge graph
- ** Response Comparison** — View answer provenance (sources, triplets, timing) for each pipeline

Open `frontend/index.html` directly in your browser or serve with:

```bash
python -m http.server 3000 --directory frontend
```

---

## Research Context

This project was developed as part of a **Summer Research Internship** under the guidance of **Dr. Prof. Janaki Meena**.

The core research question was:

> *Can intelligent retrieval pipelines (HyDE, RAG Fusion, Reranking) combined with Knowledge Graph integration offset the need for expensive proprietary LLMs, while maintaining or improving answer quality?*

**Answer: Yes.** The Llama 3.3 70B + Advanced Retrieval pipeline matched or exceeded GPT-4o on all RAGAS metrics at zero inference cost.

For the full research report, see [`projectReport.md`](projectReport.md).

### Reference Paper

The architectural design draws inspiration from the HybridRAG approach detailed in:

> *HybridRAG: Integrating Knowledge Graphs and Vector Retrieval for Efficient Information Extraction* — [Paper.pdf](Paper.pdf)

---

## Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Dr. Prof. Janaki Meena** — Research guidance and mentorship
- [Groq](https://groq.com) — Free-tier ultra-fast Llama inference
- [Google Gemini](https://ai.google.dev) — Free-tier embeddings and fallback generation
- [Neo4j](https://neo4j.com) — Graph database for knowledge graph storage
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [RAGAS](https://docs.ragas.io/) — Evaluation framework for RAG pipelines
- [LangChain](https://python.langchain.com) — Text splitting utilities
- [Sentence Transformers](https://www.sbert.net/) — Cross-encoder reranking model

---

<p align="center">
  <sub>Built with ❤️ during a Summer Research Internship • 2026</sub>
</p>
