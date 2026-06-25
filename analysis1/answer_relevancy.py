"""
Answer Relevancy Calculator (Local — No API keys needed)
=========================================================
Uses sentence-transformers to compute cosine similarity between
each question and its answer.  This is a direct semantic-similarity
proxy for RAGAS answer_relevancy, but runs entirely on your machine.

Usage:  Run this script from the analysis/ folder, or paste the cells
        into your Jupyter notebook.
"""

import pandas as pd
import numpy as np

# ── 1. Install sentence-transformers if needed ──────────────────────────
# (Run this once; comment out after first run)
# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── 2. Load data ────────────────────────────────────────────────────────
df = pd.read_csv("answers_updated.csv", encoding="latin1")

# ── 3. Load a lightweight local embedding model ────────────────────────
# all-MiniLM-L6-v2 is ~80 MB, runs on CPU, and gives strong results
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 4. Compute answer relevancy per RAG variant ────────────────────────
def compute_answer_relevancy(questions: list[str], answers: list[str]) -> np.ndarray:
    """
    Embeds each question and its corresponding answer, then returns
    per-row cosine similarity scores in [0, 1].
    """
    q_embeddings = model.encode(questions, show_progress_bar=False)
    a_embeddings = model.encode(answers, show_progress_bar=False)

    # Row-wise cosine similarity
    scores = np.array([
        cosine_similarity([q], [a])[0][0]
        for q, a in zip(q_embeddings, a_embeddings)
    ])
    return scores


questions = df["question"].tolist()

rag_variants = {
    "vectorRAG":  "vectorRAG_answer",
    "graphRAG":   "graphRAG_answer",
    "hybridRAG":  "hybridRAG_answer",
}

results = {}

for name, col in rag_variants.items():
    answers = df[col].tolist()
    scores = compute_answer_relevancy(questions, answers)
    results[name] = scores

    print(f"\n{'='*50}")
    print(f"  {name}  —  Answer Relevancy")
    print(f"{'='*50}")
    for i, (q, a, s) in enumerate(zip(questions, answers, scores)):
        print(f"  Q{i:>2d}  {s:.4f}  │  {q[:60]}...")
    print(f"  {'─'*46}")
    print(f"  MEAN:  {scores.mean():.4f}")
    print(f"  MIN :  {scores.min():.4f}")
    print(f"  MAX :  {scores.max():.4f}")

# ── 5. Build a summary comparison table ────────────────────────────────
summary_df = pd.DataFrame({
    "question": questions,
    "vectorRAG_relevancy":  results["vectorRAG"],
    "graphRAG_relevancy":   results["graphRAG"],
    "hybridRAG_relevancy":  results["hybridRAG"],
})

summary_df.loc["MEAN"] = summary_df.select_dtypes(include="number").mean()
summary_df.at["MEAN", "question"] = "── OVERALL MEAN ──"

print("\n\n")
print(summary_df.to_string(index=True, float_format="%.4f"))

# ── 6. Save results ────────────────────────────────────────────────────
summary_df.to_csv("answer_relevancy_results.csv", index=False)
print("\n✅  Saved to answer_relevancy_results.csv")


# ── 7. Patch the existing RAGAS result CSVs with the new scores ────────
for name, csv_name in [("vectorRAG", "vector_ragas_results.csv"),
                        ("graphRAG",  "graph_ragas_results.csv")]:
    try:
        ragas_df = pd.read_csv(csv_name)
        ragas_df["answer_relevancy"] = results[name]
        ragas_df.to_csv(csv_name, index=False)
        print(f"✅  Patched answer_relevancy into {csv_name}")
    except FileNotFoundError:
        print(f"⚠️  {csv_name} not found, skipping patch")
