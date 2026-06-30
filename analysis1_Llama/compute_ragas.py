"""
RAGAS Metric Computation — Groq (Free Llama 3.3 70B)
=====================================================
Computes faithfulness, answer_relevancy, context_precision, context_recall
for VectorRAG, GraphRAG, and HybridRAG — ONE question at a time with
Groq key rotation and retry logic to avoid rate limits.

Usage:
    python compute_ragas.py

Resume-safe: saves after each question, skips already-filled metrics on re-run.
"""

import json
import os
import sys
import time
import traceback

import pandas as pd
from dotenv import load_dotenv

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(dotenv_path=env_path, override=True)

# ── Config ──────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis1_Llama.csv')

# Groq key pool
_raw = os.getenv('GROQ_API_KEYS', '')
GROQ_KEYS = [k.strip() for k in _raw.split(',') if k.strip()]
if not GROQ_KEYS:
    single = os.getenv('GROQ_API_KEY', '')
    if single:
        GROQ_KEYS = [single]

if not GROQ_KEYS:
    print("ERROR: No Groq API keys found. Set GROQ_API_KEYS in .env")
    sys.exit(1)

print(f"Loaded {len(GROQ_KEYS)} Groq API keys for rotation")

# ── Current key index (global for rotation) ─────────────
_current_key_idx = 0


def get_next_key():
    """Rotate to the next Groq API key."""
    global _current_key_idx
    _current_key_idx = (_current_key_idx + 1) % len(GROQ_KEYS)
    return GROQ_KEYS[_current_key_idx]


def make_llm(api_key=None):
    """Create a LangChain ChatGroq LLM instance."""
    from langchain_groq import ChatGroq
    key = api_key or GROQ_KEYS[_current_key_idx]
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        temperature=0,
        max_tokens=1024,
        request_timeout=120,
    )


def make_embeddings():
    """Create local HuggingFace embeddings (no API needed)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# ── Context extraction helpers ──────────────────────────
def extract_vector_contexts(sources_json_str):
    try:
        sources = json.loads(str(sources_json_str))
        if isinstance(sources, list):
            return [s.get('text', '') for s in sources if s.get('text')]
    except Exception:
        pass
    return []


def extract_graph_contexts(triplets_json_str):
    try:
        triplets = json.loads(str(triplets_json_str))
        if isinstance(triplets, list):
            texts = []
            for t in triplets:
                subj = t.get('subject', '?')
                pred = t.get('predicate', '?').replace('_', ' ').lower()
                obj = t.get('object', '?')
                texts.append(f'{subj} {pred} {obj}')
            return texts if texts else []
    except Exception:
        pass
    return []


def extract_hybrid_contexts(hybrid_json_str):
    try:
        hybrid = json.loads(str(hybrid_json_str))
        contexts = []
        for s in hybrid.get('vector_sources', []):
            if s.get('text'):
                contexts.append(s['text'])
        for t in hybrid.get('graph_triplets', []):
            subj = t.get('subject', '?')
            pred = t.get('predicate', '?').replace('_', ' ').lower()
            obj = t.get('object', '?')
            contexts.append(f'{subj} {pred} {obj}')
        return contexts if contexts else []
    except Exception:
        pass
    return []


def is_empty(val):
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s == '' or s == 'nan'


# ── Single-question RAGAS evaluation with retry ─────────
def evaluate_single_question(question, answer, contexts, ground_truth, embeddings, max_retries=5):
    """
    Evaluate one question with RAGAS metrics.
    Retries with key rotation on rate limit errors.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        # answer_relevancy,
        # context_precision,
        # context_recall,
    )
    from datasets import Dataset

    if not contexts:
        contexts = ['No context retrieved.']

    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [ground_truth],
    }
    dataset = Dataset.from_dict(data)

    for attempt in range(max_retries):
        try:
            llm = make_llm()

            result = evaluate(
                dataset,
                metrics=[faithfulness, 
                # answer_relevancy, context_precision, context_recall
                ],
                llm=llm,
                embeddings=embeddings,
            )

            result_df = result.to_pandas()
            row = result_df.iloc[0]
            return {
                'faithfulness': row.get('faithfulness'),
                # 'answer_relevancy': row.get('answer_relevancy'),
                # 'context_precision': row.get('context_precision'),
                # 'context_recall': row.get('context_recall'),
            }

        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(kw in error_msg for kw in [
                'rate_limit', '429', 'too many requests', 'rate limit',
                'resource_exhausted', 'quota', 'timeout',
            ])

            if is_rate_limit and attempt < max_retries - 1:
                new_key = get_next_key()
                delay = 5 * (attempt + 1)
                print(f"    ⚠ Rate limit/timeout (attempt {attempt+1}/{max_retries}). "
                      f"Rotating key, waiting {delay}s...")
                time.sleep(delay)
                continue
            else:
                print(f"    ✗ Failed after {attempt+1} attempts: {e}")
                traceback.print_exc()
                return {
                    'faithfulness': None,
                    # 'answer_relevancy': None,
                    # 'context_precision': None,
                    # 'context_recall': None,
                }


# ── Main pipeline ────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  RAGAS Metric Computation (Groq / Llama 3.3 70B)")
    print("=" * 60)

    # Load CSV with string dtypes for text columns
    df = pd.read_csv(CSV_PATH)
    STR_COLS = ['vectorRAG_answer', 'graphRAG_answer', 'hybridRAG_answer',
                'vectorRAG_sources', 'graphRAG_triplets', 'hybridRAG_sources']
    for col in STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(object)

    print(f"Loaded {len(df)} questions from CSV")

    # Initialize embeddings once (local, no API)
    print("Loading HuggingFace embeddings (local)...")
    embeddings = make_embeddings()
    print("✓ Embeddings ready\n")

    # ── Process each RAG type ────────────────────────────
    rag_configs = [
        {
            'name': 'VectorRAG',
            'answer_col': 'vectorRAG_answer',
            'context_extractor': extract_vector_contexts,
            'context_col': 'vectorRAG_sources',
            'metric_prefix': 'vector',
        },
        {
            'name': 'GraphRAG',
            'answer_col': 'graphRAG_answer',
            'context_extractor': extract_graph_contexts,
            'context_col': 'graphRAG_triplets',
            'metric_prefix': 'graph',
        },
        {
            'name': 'HybridRAG',
            'answer_col': 'hybridRAG_answer',
            'context_extractor': extract_hybrid_contexts,
            'context_col': 'hybridRAG_sources',
            'metric_prefix': 'hybrid',
        },
    ]

    for config in rag_configs:
        prefix = config['metric_prefix']
        metric_cols = [
            f'{prefix}_faithfulness',
            # f'{prefix}_answer_relevancy',
            # f'{prefix}_context_precision',
            # f'{prefix}_context_recall',
        ]

        print("=" * 60)
        print(f"  Computing RAGAS metrics for {config['name']}")
        print("=" * 60)

        filled = 0
        skipped = 0

        for idx, row in df.iterrows():
            question = row['question']
            answer = row.get(config['answer_col'], '')
            ground_truth = row.get('ground_truth', '')

            # Skip if answer is empty
            if is_empty(answer):
                print(f"  [{idx+1}/{len(df)}] SKIP (no {config['name']} answer)")
                skipped += 1
                continue

            # Skip if metrics already filled
            if not is_empty(row.get(metric_cols[0])):
                print(f"  [{idx+1}/{len(df)}] SKIP (metrics already filled)")
                skipped += 1
                continue

            # Extract contexts
            context_str = row.get(config['context_col'], '[]')
            contexts = config['context_extractor'](context_str)

            print(f"\n  [{idx+1}/{len(df)}] {question[:70]}...")
            print(f"    Contexts: {len(contexts)} | Answer: {str(answer)[:60]}...")

            # Evaluate
            metrics = evaluate_single_question(
                question=question,
                answer=str(answer),
                contexts=contexts,
                ground_truth=str(ground_truth),
                embeddings=embeddings,
            )

            # Write to DataFrame
            df.at[idx, metric_cols[0]] = metrics['faithfulness']
            df.at[idx, metric_cols[1]] = metrics['answer_relevancy']
            df.at[idx, metric_cols[2]] = metrics['context_precision']
            df.at[idx, metric_cols[3]] = metrics['context_recall']

            print(f"    ✓ F={metrics['faithfulness']:.4f}  AR={metrics['answer_relevancy']:.4f}  "
                  f"CP={metrics['context_precision']:.4f}  CR={metrics['context_recall']:.4f}"
                  if all(v is not None for v in metrics.values())
                  else f"    ⚠ Some metrics failed: {metrics}")

            filled += 1

            # Save after each question
            df.to_csv(CSV_PATH, index=False)

            # Rate-limit pause between questions (Groq free tier)
            time.sleep(3)

        print(f"\n  ✓ {config['name']} done! Filled: {filled}, Skipped: {skipped}")
        print()

    # ── Compute combined (average) metrics ───────────────
    print("=" * 60)
    print("  Computing combined (average) metrics")
    print("=" * 60)

    for idx, row in df.iterrows():
        for metric in ['faithfulness', 
        # 'answer_relevancy', 'context_precision', 'context_recall'
        ]:
            vals = []
            for prefix in ['vector', 'graph', 'hybrid']:
                v = row.get(f'{prefix}_{metric}')
                if not is_empty(v):
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if vals:
                df.at[idx, f'combined_{metric}'] = sum(vals) / len(vals)

    df.to_csv(CSV_PATH, index=False)

    print("\n✓ All done! CSV saved.")
    print(f"  File: {CSV_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for prefix, name in [('vector', 'VectorRAG'), ('graph', 'GraphRAG'), ('hybrid', 'HybridRAG')]:
        print(f"\n  {name}:")
        for metric in ['faithfulness', 
        # 'answer_relevancy', 'context_precision', 'context_recall'
        ]:
            col = f'{prefix}_{metric}'
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    print(f"    {metric:25s}: {vals.mean():.6f}")
                else:
                    print(f"    {metric:25s}: (no data)")


if __name__ == '__main__':
    main()
