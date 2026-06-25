"""
RAG Evaluation — Consolidation & Visualization
================================================
Merges all scattered result CSVs into:
  1. ragas_complete_results.csv   – one row per question, all metrics
  2. ragas_summary.csv            – overall means per RAG variant
  3. Six publication-quality charts in analysis/charts/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\grove\Desktop\SRIP\analysis")
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)

# ── 1. Load source files ──────────────────────────────────────────────
base_df       = pd.read_csv(BASE / "answers_updated.csv", encoding="latin1")
vector_ragas  = pd.read_csv(BASE / "vector_ragas_results.csv")
graph_ragas   = pd.read_csv(BASE / "graph_ragas_results.csv")
hybrid_ragas  = pd.read_csv(BASE / "hybrid_ragas_results.csv")
relevancy     = pd.read_csv(BASE / "answer_relevancy_results.csv")

# drop the "OVERALL MEAN" row if present in relevancy
relevancy = relevancy[relevancy["question"] != "── OVERALL MEAN ──"].reset_index(drop=True)
if "OVERALL MEAN" in relevancy["question"].values:
    relevancy = relevancy[relevancy["question"] != "OVERALL MEAN"].reset_index(drop=True)

# ── 2. Build unified dataframe ────────────────────────────────────────
n = len(base_df)

unified = pd.DataFrame()
unified["question"]          = base_df["question"]
unified["question_type"]     = base_df["question_type"]
unified["ground_truth"]      = base_df["ground_truth"]

# Answers
unified["vectorRAG_answer"]  = base_df["vectorRAG_answer"]
unified["graphRAG_answer"]   = base_df["graphRAG_answer"]
unified["hybridRAG_answer"]  = base_df["hybridRAG_answer"]

# Sources
unified["vectorRAG_sources"]   = base_df["vectorRAG_sources"]
unified["graphRAG_triplets"]   = base_df["graphRAG_triplets"]
unified["hybridRAG_sources"]   = base_df["hybridRAG_sources"]

# ── Metrics ────────────────────────────────────────────────────────────
# Faithfulness
unified["vector_faithfulness"] = vector_ragas["faithfulness"].values[:n]
unified["graph_faithfulness"]  = graph_ragas["faithfulness"].values[:n]
unified["hybrid_faithfulness"] = hybrid_ragas["faithfulness"].values[:n]

# Answer Relevancy (from local sentence-transformer computation)
rel = relevancy.head(n)
unified["vector_answer_relevancy"] = rel["vectorRAG_relevancy"].values
unified["graph_answer_relevancy"]  = rel["graphRAG_relevancy"].values
unified["hybrid_answer_relevancy"] = rel["hybridRAG_relevancy"].values

# Context Precision
unified["vector_context_precision"] = vector_ragas["context_precision"].values[:n]
unified["graph_context_precision"]  = graph_ragas["context_precision"].values[:n]
unified["hybrid_context_precision"] = hybrid_ragas["context_precision"].values[:n]

# Context Recall
unified["vector_context_recall"] = vector_ragas["context_recall"].values[:n]
unified["graph_context_recall"]  = graph_ragas["context_recall"].values[:n]
unified["hybrid_context_recall"] = hybrid_ragas["context_recall"].values[:n]

# ── Combined (avg across variants) per question ───────────────────────
for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    cols = [f"vector_{metric}", f"graph_{metric}", f"hybrid_{metric}"]
    unified[f"combined_{metric}"] = unified[cols].mean(axis=1)

# ── 3. Save unified CSV ──────────────────────────────────────────────
unified.to_csv(BASE / "ragas_complete_results.csv", index=False)
print(f"✅  Saved ragas_complete_results.csv  ({len(unified)} rows × {len(unified.columns)} cols)")

# ── 4. Summary table ─────────────────────────────────────────────────
metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
variants = ["vector", "graph", "hybrid"]

summary_rows = []
for m in metrics:
    row = {"Metric": m.replace("_", " ").title()}
    for v in variants:
        col = f"{v}_{m}"
        row[f"{v.title()}RAG"] = unified[col].mean()
    row["Combined (Avg)"] = np.mean([row[f"{v.title()}RAG"] for v in variants])
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(BASE / "ragas_summary.csv", index=False)
print("✅  Saved ragas_summary.csv")
print("\n" + summary.to_string(index=False, float_format="%.4f"))

# ── 5. Visualizations ────────────────────────────────────────────────

# Color palette
COLORS = {
    "VectorRAG":  "#5B8DEE",   # blue
    "GraphRAG":   "#FF6B6B",   # coral
    "HybridRAG":  "#2ECC71",   # green (best)
}
BG_COLOR = "#FAFAFA"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": "#FFFFFF",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ────────────────────────────────────────────────────────────────────
# Chart 1: Overall comparison bar chart
# ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics))
width = 0.25

for i, v in enumerate(["Vector", "Graph", "Hybrid"]):
    vals = [summary.loc[summary["Metric"] == m.replace("_", " ").title(), f"{v}RAG"].values[0]
            for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=f"{v}RAG",
                  color=COLORS[f"{v}RAG"], edgecolor="white", linewidth=0.5)
    # value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Overall RAGAS Metrics Comparison", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(CHARTS / "01_overall_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 01_overall_comparison.png")

# ────────────────────────────────────────────────────────────────────
# Chart 2: Radar / Spider chart
# ────────────────────────────────────────────────────────────────────
labels = [m.replace("_", " ").title() for m in metrics]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_facecolor(BG_COLOR)

for v in ["Vector", "Graph", "Hybrid"]:
    values = [summary.loc[summary["Metric"] == m.replace("_", " ").title(), f"{v}RAG"].values[0]
              for m in metrics]
    values += values[:1]
    lw = 3 if v == "Hybrid" else 1.8
    ax.plot(angles, values, "o-", linewidth=lw, label=f"{v}RAG", color=COLORS[f"{v}RAG"])
    ax.fill(angles, values, alpha=0.08 if v != "Hybrid" else 0.15, color=COLORS[f"{v}RAG"])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_title("RAG Variant Performance Profile", fontsize=14, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.9)
fig.tight_layout()
fig.savefig(CHARTS / "02_radar_chart.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 02_radar_chart.png")

# ────────────────────────────────────────────────────────────────────
# Chart 3: Per-question faithfulness comparison
# ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
q_labels = [f"Q{i+1}" for i in range(n)]
x = np.arange(n)
w = 0.25

for i, (v, c) in enumerate([("vector", COLORS["VectorRAG"]),
                              ("graph",  COLORS["GraphRAG"]),
                              ("hybrid", COLORS["HybridRAG"])]):
    ax.bar(x + i * w, unified[f"{v}_faithfulness"], w,
           label=f"{v.title()}RAG", color=c, edgecolor="white", linewidth=0.5)

ax.set_ylabel("Faithfulness Score", fontweight="bold")
ax.set_title("Faithfulness per Question", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x + w)
ax.set_xticklabels(q_labels, rotation=0, fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(CHARTS / "03_faithfulness_per_question.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 03_faithfulness_per_question.png")

# ────────────────────────────────────────────────────────────────────
# Chart 4: Per-question answer relevancy comparison
# ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

for v, c, ls in [("vector", COLORS["VectorRAG"], "--"),
                  ("graph",  COLORS["GraphRAG"],  ":"),
                  ("hybrid", COLORS["HybridRAG"], "-")]:
    lw = 2.5 if v == "hybrid" else 1.5
    ax.plot(q_labels, unified[f"{v}_answer_relevancy"], ls,
            color=c, linewidth=lw, marker="o", markersize=5, label=f"{v.title()}RAG")

ax.set_ylabel("Answer Relevancy Score", fontweight="bold")
ax.set_title("Answer Relevancy per Question", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, 1.05)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(CHARTS / "04_answer_relevancy_per_question.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 04_answer_relevancy_per_question.png")

# ────────────────────────────────────────────────────────────────────
# Chart 5: Heatmap of all metrics
# ────────────────────────────────────────────────────────────────────
heatmap_data = []
for v in variants:
    row = []
    for m in metrics:
        row.append(unified[f"{v}_{m}"].mean())
    heatmap_data.append(row)

hm = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(hm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(variants)))
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11, fontweight="bold")
ax.set_yticklabels([f"{v.title()}RAG" for v in variants], fontsize=12, fontweight="bold")

# annotate each cell
for i in range(len(variants)):
    for j in range(len(metrics)):
        color = "white" if hm[i, j] < 0.4 else "black"
        ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Score", fontweight="bold")
ax.set_title("Metric Heatmap — All RAG Variants", fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig(CHARTS / "05_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 05_heatmap.png")

# ────────────────────────────────────────────────────────────────────
# Chart 6: HybridRAG Improvement % over Vector & Graph
# ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

metric_labels = [m.replace("_", " ").title() for m in metrics]
hybrid_vals = [summary.loc[summary["Metric"] == ml, "HybridRAG"].values[0] for ml in metric_labels]
vector_vals = [summary.loc[summary["Metric"] == ml, "VectorRAG"].values[0] for ml in metric_labels]
graph_vals  = [summary.loc[summary["Metric"] == ml, "GraphRAG"].values[0]  for ml in metric_labels]

# % improvement of Hybrid over Vector and Graph
def pct_improvement(hybrid, other):
    if other == 0:
        return 100.0  # cap at 100% if base is 0
    return ((hybrid - other) / other) * 100

impr_over_vector = [pct_improvement(h, v) for h, v in zip(hybrid_vals, vector_vals)]
impr_over_graph  = [pct_improvement(h, g) for h, g in zip(hybrid_vals, graph_vals)]

x = np.arange(len(metrics))
w = 0.35

bars1 = ax.bar(x - w/2, impr_over_vector, w, label="vs VectorRAG",
               color=COLORS["VectorRAG"], alpha=0.8, edgecolor="white")
bars2 = ax.bar(x + w/2, impr_over_graph, w, label="vs GraphRAG",
               color=COLORS["GraphRAG"], alpha=0.8, edgecolor="white")

# value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        sign = "+" if height >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, height + (1 if height >= 0 else -3),
                f"{sign}{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("Improvement (%)", fontweight="bold")
ax.set_title("HybridRAG Improvement Over Other Variants", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.legend(framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(CHARTS / "06_hybrid_improvement.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 06_hybrid_improvement.png")

# ────────────────────────────────────────────────────────────────────
# Chart 7: Win count - how often each variant is best per question
# ────────────────────────────────────────────────────────────────────
wins = {"VectorRAG": 0, "GraphRAG": 0, "HybridRAG": 0}

for _, row in unified.iterrows():
    for m in metrics:
        scores = {
            "VectorRAG": row[f"vector_{m}"],
            "GraphRAG":  row[f"graph_{m}"],
            "HybridRAG": row[f"hybrid_{m}"],
        }
        # handle NaN
        valid = {k: v for k, v in scores.items() if pd.notna(v)}
        if valid:
            winner = max(valid, key=valid.get)
            wins[winner] += 1

total_contests = sum(wins.values())

fig, ax = plt.subplots(figsize=(7, 7))
labels_pie = list(wins.keys())
sizes = list(wins.values())
colors_pie = [COLORS[k] for k in labels_pie]
explode = [0.05 if k != "HybridRAG" else 0.1 for k in labels_pie]

wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=labels_pie, colors=colors_pie,
    autopct=lambda pct: f"{pct:.1f}%\n({int(pct * total_contests / 100)})",
    shadow=True, startangle=90, textprops={"fontsize": 12, "fontweight": "bold"}
)

for t in autotexts:
    t.set_fontsize(10)

ax.set_title(f"Best Score Wins ({total_contests} total metric-question pairs)",
             fontsize=14, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(CHARTS / "07_win_count_pie.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("📊  Saved 07_win_count_pie.png")

# ── Done ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"\n  📁  Unified CSV:   {BASE / 'ragas_complete_results.csv'}")
print(f"  📁  Summary CSV:   {BASE / 'ragas_summary.csv'}")
print(f"  📁  Charts:        {CHARTS}")
print(f"       01_overall_comparison.png")
print(f"       02_radar_chart.png")
print(f"       03_faithfulness_per_question.png")
print(f"       04_answer_relevancy_per_question.png")
print(f"       05_heatmap.png")
print(f"       06_hybrid_improvement.png")
print(f"       07_win_count_pie.png")
