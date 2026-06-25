# HybridRAG (SRIP)

Financial document Q&A system with a Vector RAG pipeline, a Graph RAG pipeline, and a frontend UI for uploads, queries, and graph visualization.

## What is implemented

### Vector RAG (backend/vectorRAG)

- FastAPI service on port 8000 with endpoints for upload, query, list, and delete.
- PDF text extraction and chunking with metadata.
- Vector index creation using FAISS and embeddings.
- Answer generation from retrieved chunks with source citations.
- OCR fallback for scanned PDFs (requires Tesseract installed in PATH).

#### Vector RAG algorithm (detailed)

1. **PDF ingestion and text extraction**
   - Each PDF is read page-by-page and text is extracted.
   - If a PDF is scanned or image-only, OCR is attempted to recover text.

2. **Chunking with overlap**
   - Each page is split into overlapping chunks (chunk size and overlap are fixed).
   - Every chunk gets metadata: source file, page number, and chunk index.

3. **Embedding + index build**
   - The embedding model turns each chunk into a dense vector.
   - Vectors are L2-normalized and stored in a FAISS index (inner-product for cosine similarity).
   - The index and chunk metadata are persisted to disk by document ID.

4. **Retrieval**
   - The user query is embedded and compared against the index.
   - Top-k most similar chunks are returned with similarity scores.

5. **Answer generation**
   - A grounded prompt is constructed from the retrieved chunks.
   - The LLM generates a concise answer and must cite the chunk sources by page.
   - If the answer is missing in the context, the model must refuse.

### Graph RAG (backend/graphRAG)

- FastAPI service on port 8001 with endpoints for upload, query, list, delete, and visualize.
- Text chunking and LLM-based triplet extraction.
- Free-form entity/relationship types and properties stored in Neo4j.
- Graph traversal based on query entities to build context.
- Knowledge graph visualization payload for the frontend.

#### Graph RAG algorithm (detailed)

1. **PDF ingestion and text extraction**
   - The PDF is read and split into text chunks with page metadata.

2. **Triplet extraction (LLM)**
   - Each chunk is sent to an LLM to extract knowledge triplets.
   - Triplets are free-form: entity types and relation types are not restricted.
   - The model can attach extra properties (e.g., value, unit, currency, time_period,
     source_sentence) for denser graph facts.
   - Each triplet also includes source page, chunk index, and confidence.

3. **Graph storage (Neo4j)**
   - Entities are stored as nodes with type and properties.
   - Relations are stored as edges with type, properties, and provenance fields.
   - All nodes/edges include a document ID to keep documents isolated.

4. **Graph retrieval by query**
   - An LLM first extracts key entities from the user question.
   - Graph traversal expands from matched entities up to a fixed hop count.
   - Retrieved triplets become the graph context for answering.

5. **Answer generation**
   - A prompt is built from the graph triplets (including properties).
   - The LLM answers only from the graph context, otherwise refuses.

### Frontend (frontend)

- Single-page UI for file upload, querying, and results display.
- Runs Vector RAG and Graph RAG queries in parallel.
- Knowledge Graph Explorer with vis.js rendering.
- Basic markdown rendering for bold text in answers.

## How to run (local)

### Vector RAG

```
python -m uvicorn backend.vectorRAG.main:app --host 127.0.0.1 --port 8000 --reload
```

### Graph RAG

```
python -m uvicorn backend.graphRAG.main:app --host 127.0.0.1 --port 8001 --reload
```

### Frontend

Open frontend/index.html with a local server (VS Code Live Server or similar).

## Notes

- OCR fallback for scanned PDFs requires Tesseract to be installed and in PATH.
- The Vector RAG embedding provider may hit free-tier rate limits for large uploads.

## Planned integration (short note)

- Combine Vector RAG retrieval results with Graph RAG triplets into a unified context.
- Add a hybrid answer generator that cross-checks vector snippets against graph relations.
- Implement a unified upload pipeline that builds both indices and links them by document ID.
- Add retrieval weighting (vector similarity + graph relevance) for more grounded answers.
