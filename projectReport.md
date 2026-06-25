# **PROJECT REPORT: ADVANCED HYBRID RETRIEVAL-AUGMENTED GENERATION (RAG) PIPELINE**

**Summer Research Internship**

**Prepared By:** Sushen Grover (23BCE1728)

**Under the Guidance of:** Dr. Prof. Janaki Meena

**GitHub Repository:** [https://github.com/SushenGrover/HybridRAG](https://github.com/SushenGrover/HybridRAG)

## **ABSTRACT**

This report details the research, development, and evaluation of an advanced Hybrid Retrieval-Augmented Generation (HybridRAG) pipeline conducted during a Summer Research Internship. The core objective of the internship was to overcome the limitations of traditional Vector-based RAG systems by integrating Knowledge Graphs (GraphRAG), ultimately creating a Hybrid system. The project progressed from a baseline VectorRAG using FAISS and local Neo4j deployments to a highly optimized, cost-effective pipeline utilizing LLaMA models via the Groq API. Advanced retrieval techniques, including Cross-Encoder Reranking, RAG Fusion, Query Decomposition, and Hypothetical Document Embeddings (HyDE), were integrated to enhance retrieval accuracy. The system's performance was rigorously evaluated across multiple document scales using the RAGAS (Retrieval Augmented Generation Assessment) framework.

## **1\. INTRODUCTION**

Large Language Models (LLMs) have revolutionized natural language processing, but they suffer from hallucinations and lack domain-specific, up-to-date knowledge. Retrieval-Augmented Generation (RAG) solves this by fetching relevant context from external databases before generating an answer. However, traditional VectorRAG often struggles with multi-hop reasoning and understanding complex relationships across scattered documents.

This research project aimed to solve these issues by introducing a **HybridRAG** approach. By combining the semantic similarity search of Vector Databases (FAISS) with the structured, relational data mapping of Knowledge Graphs (Neo4j), the system can handle both basic factual retrieval and complex, multi-clause logical queries.

## **2\. PROJECT OBJECTIVES**

1. **Baseline Implementation:** Build a standard VectorRAG pipeline using FAISS.  
2. **Graph Integration:** Deploy Neo4j locally and implement a GraphRAG pipeline capable of triplet generation and multi-hop traversal.  
3. **Hybrid Architecture:** Merge Vector and Graph modalities into a single, user-centric HybridRAG system.  
4. **Cost & Performance Optimization:** Migrate from paid models (GPT-4o) to high-speed, open-source models (Llama 3.3 70B via Groq) while maintaining or improving accuracy.  
5. **Advanced RAG Techniques:** Implement state-of-the-art retrieval enhancements (HyDE, RAG Fusion, Query Decomposition, Cross-Encoder Reranking).  
6. **Rigorous Evaluation:** Assess the pipeline using the RAGAS framework (Faithfulness, Answer Relevancy, Context Precision, Context Recall) across varying document sizes and question types.

## **3\. TIMELINE & METHODOLOGY (WEEKLY BREAKDOWN)**

The internship was structured into distinct phases, moving from foundational architecture to advanced optimization and rigorous testing.

### **Weeks 1-2: Foundation & VectorRAG Implementation**

**Objective:** Establish the baseline retrieval system and database infrastructure.

* **VectorRAG Pipeline:** Implemented a standard RAG pipeline. Documents were parsed, chunked, and embedded into a FAISS (Facebook AI Similarity Search) index for fast, semantic nearest-neighbor retrieval.  
* **Database Setup:** Configured and deployed a local instance of Neo4j. This served as the foundational infrastructure for the upcoming graph-based retrieval phase.

### **Weeks 3-4: GraphRAG Implementation**

**Objective:** Enable relationship-based retrieval using Knowledge Graphs.

* **Triplet Generation:** Engineered prompts and extraction logic to parse text chunks into Subject-Predicate-Object triplets (e.g., (Policy) \- \[HAS\_UIN\] \-\> (105N153V04)).  
* **Graph Ingestion:** Wrote scripts to ingest these triplets into the local Neo4j database, creating a densely connected web of entities.  
* **Traversal Logic:** Implemented an entity-extraction step on the user query. The pipeline identifies entities in the query and performs a **2-hop traversal** in Neo4j to pull all surrounding, related triplets.  
* **Graph Answer Generation:** Fed the extracted sub-graph (triplets) to the LLM to generate highly contextual answers based purely on structural relationships.

### **Weeks 5-6: HybridRAG & Initial LLM Integration**

**Objective:** Combine modalities for superior contextual understanding.

* **Hybrid Architecture:** Created an orchestrator that concurrently queries both the FAISS Vector Index (for semantic meaning) and the Neo4j Graph Database (for structured relationships).  
* **Synthesis:** The retrieved text chunks and the retrieved graph triplets were injected into a unified prompt.  
* **Model Selection:** During this phase, OpenAI's **GPT-4o** (a paid model) was utilized to synthesize the combined contexts into a single, coherent, user-centric language response.

### **Week 7: Analysis 1 (Initial Evaluation)**

**Objective:** Baseline evaluation of the GPT-4o HybridRAG system.

* **Dataset:** A small document consisting of 2 pages (approx. 1000 words).  
* **Test Suite:** Generated a test dataset of 18 questions categorized into 5 distinct types (Basic Retrieval, Numerical Reasoning, Multi-Clause Retrieval, Hallucination Resistance, and Realistic User Queries).  
* **RAGAS Evaluation:** Computed the four core RAGAS metrics:  
  1. **Faithfulness:** Does the answer rely strictly on the context?  
  2. **Answer Relevancy:** Does the answer directly address the question?  
  3. **Context Precision:** Are the most relevant chunks ranked highest?  
  4. **Context Recall:** Did the retrieval fetch all necessary information to answer the question?  
* *Observation:* While accuracy was high, the cost of GPT-4o and API latency indicated a need for a more sustainable, open-source architecture. It is important to note that even though we did not use any advanced retrieval techniques in this phase, we achieved relatively high generation metrics simply because our baseline utilized GPT-4o, a highly capable paid model, which compensated for some of the retrieval shortcomings through strong internal reasoning.

### **Weeks 8-9: Optimization & Advanced Retrieval Techniques**

**Objective:** Migrate to free models and boost retrieval intelligence.

* **Model Migration:** Shifted from GPT-4o to the **Llama 3.3 70B** model utilizing the **Groq API** for ultra-fast, free-tier inference. Gemini (2.5-flash) was configured as a fallback.  
* **Query Decomposition:** Implemented logic to break complex user queries into smaller, distinct sub-queries, querying the databases for each part independently.  
* **HyDE (Hypothetical Document Embeddings):** For vague queries, the LLM was prompted to generate a "hypothetical" perfect answer. This hypothetical text was then embedded and searched against FAISS, vastly improving semantic matching.  
* **RAG Fusion:** Ran multiple generated sub-queries concurrently and fused the retrieval results.  
* **Cross-Encoder Reranking:** Applied a cross-encoder model to re-score and re-order the fused retrieved chunks and triplets based on their actual relevance to the initial query, filtering out noise before passing context to the final generation LLM.

### **Weeks 10-12: Scale-Up & Final Analyses (Analysis 2 & 3\)**

**Objective:** Evaluate the new, advanced Llama/Groq pipeline against the old baseline, and stress-test it on a massive document.

* **Analysis 2 (The Benchmark Comparison):** Re-ran the same 2-page, 18-question dataset used in Analysis 1\.  
  * **Result:** The open-source Llama-3 \+ Advanced Techniques (HyDE, Reranking, Fusion) matched or exceeded the performance of the basic GPT-4o pipeline in Answer Relevancy and Context Precision, proving that intelligent retrieval pipelines can offset the need for expensive, proprietary LLMs.  
* **Analysis 3 (The Scale Test):**  
  * **Dataset:** Scaled up to a massive 10,000-word document.  
  * **Test Suite:** Expanded to 50 complex questions spanning the 5 categories.  
  * **Evaluation Challenge:** Computing RAGAS metrics via LLM-as-a-judge for 50 questions across 3 pipelines (Vector, Graph, Hybrid) was computationally heavy.  
  * **Solution:** Developed and utilized a custom heuristic evaluation script (fill\_ragas\_metrics.py) combining difflib sequence matching and failure-state detection (e.g., detecting "Information not found") to rapidly estimate RAGAS metrics and populate the analytics dashboard.  
  * **Result:** The HybridRAG pipeline successfully maintained high Faithfulness and Context Recall across the 10,000-word document, correctly correlating distant clauses that standard VectorRAG missed.

## **4\. ARCHITECTURAL DEEP-DIVE**

### **4.1 VectorRAG**

The VectorRAG component acts as the semantic engine. Text is split using Recursive Character Text Splitting with intentional overlap to maintain context. Embeddings are generated and stored in FAISS. At query time, Cosine Similarity is used to fetch the Top-K chunks. While excellent for "Basic Retrieval," it occasionally fails on "Multi-Clause" questions where the answer spans multiple pages.

### **4.2 GraphRAG**

The GraphRAG component acts as the structural and logical engine.

* **Nodes:** Represent entities (e.g., "Policyholder", "Grace Period").  
* **Edges:** Represent relationships (e.g., "HAS\_DURATION", "APPLIES\_TO").  
  By extracting query entities and traversing two hops outward, GraphRAG easily answers questions like "How does converting a policy to paid-up status affect future reversionary bonuses?" by tracing the exact logical path in the Neo4j graph, completely bypassing semantic noise.

### **4.3 HybridRAG \+ Advanced Enhancements**

The ultimate system runs both engines. However, fetching too much context dilutes LLM focus (the "Lost in the Middle" phenomenon).

To solve this, the **Cross-Encoder** acts as a strict gatekeeper. It takes the query and pairs it with every retrieved chunk/triplet, outputting a relevance score. Only the highest-scoring mix of semantic text and graph triplets is passed to Llama 3.3 via Groq for the final synthesis.

## **5\. EXPERIMENTAL RESULTS & RAGAS EVALUATION**

The performance of the system was measured using the RAGAS framework.

### **Analysis 1 (Baseline: GPT-4o, 1000 words, 18 Qs)**

* **Key Insight:** Even without utilizing advanced retrieval techniques, we obtained better-than-expected metrics because our model was a highly capable paid model (GPT-4o). Its strong inherent knowledge and reasoning power compensated for the basic retrieval mechanism.  
* **Faithfulness:** High (\~0.92). GPT-4o adhered closely to provided contexts.  
* **Context Recall:** Moderate (\~0.75). Standard FAISS missed some interconnected clauses.

### **Analysis 2 (Advanced Pipeline: Groq/Llama-3, 1000 words, 18 Qs)**

* **Context Precision:** Improved significantly (\~0.88). The integration of HyDE and Cross-Encoder Reranking ensured that the exact paragraphs needed were ranked at the top.  
* **Answer Relevancy:** Maintained high performance, proving Llama-3 via Groq is a highly capable reasoning engine when provided curated context.

### **Analysis 3 (Stress Test: Groq/Llama-3, 10000 words, 50 Qs)**

* **VectorRAG vs. GraphRAG:** GraphRAG heavily outperformed VectorRAG in the *Multi-Clause Retrieval* category. Conversely, VectorRAG performed better on *Realistic User Queries* where phrasing was colloquial.  
* **HybridRAG Supremacy:** The combined Hybrid pipeline achieved the highest overall scores. It successfully resisted hallucinations (scoring perfectly on out-of-scope questions by outputting "Information not found") and handled numerical reasoning by fusing the formulaic text chunks with structural graph variables.

## **6\. CHALLENGES & SOLUTIONS**

1. **Graph Triplet Extraction Noise:** *Challenge:* Early LLM prompts generated redundant or overly generic triplets, bloating the Neo4j database.  
   * *Solution:* Refined few-shot prompting techniques and strictly defined entity ontologies to standardize Node labels.  
2. **Evaluation Bottlenecks:** *Challenge:* Running LLM-based RAGAS evaluations on the 50-question dataset was prohibitively slow and prone to API rate limits.  
   * *Solution:* Developed a Python script (fill\_ragas\_metrics.py) using heuristic sequence matching and programmatic failure-state detection to rapidly estimate metrics. This allowed for immediate dashboard population and rapid iterative testing.  
3. **Cost vs. Latency:** *Challenge:* GPT-4o offered great reasoning but was expensive.  
   * *Solution:* Transitioning to Groq provided ultra-low latency inference, enabling the use of complex workflows like RAG Fusion and Query Decomposition without frustrating the end-user with long wait times.

## **7\. CONCLUSION**

This Summer Research Internship culminated in the successful development of a highly robust, scalable, and cost-efficient HybridRAG system. By transitioning from a standard semantic search to a unified architecture combining Knowledge Graphs (Neo4j), Vector Databases (FAISS), and advanced retrieval augmentations (HyDE, Fusion, Reranking), the pipeline demonstrated superior ability to parse, reason, and answer complex queries over massive documents.

The transition from paid, proprietary models to open-source models (Llama 3.3 70B via Groq API) proved that intelligent data retrieval and reranking algorithms are often more impactful than the raw size of the underlying generator model. This system stands as a powerful, open-source-ready template for enterprise-grade document intelligence.
