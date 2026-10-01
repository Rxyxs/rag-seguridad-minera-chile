"""Figures for the README: run with `python -m src.make_figures`.

Builds the real pipeline and the real classifier, with the same parameters and
seed the modules use on their own, so the numbers drawn here are the numbers
`src/rag/evaluation.py` and `src/nlp/severity_classifier.py` produce. Nothing
is illustrative, and no external LLM is involved.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.nlp.severity_classifier import (
    SEVERITY_LEVELS,
    embed_texts,
    evaluate_classifier,
    get_embedder,
    load_incidents,
    train_severity_classifier,
)
from src.rag.evaluation import (
    QUERY_DATASET,
    evaluate_faithfulness,
    evaluate_retrieval,
    summarize,
)
from src.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

INK = "#2B2B2B"
GRID = "#D9D9D9"
BEFORE = "#8FA8B8"
AFTER = "#B5553D"
OK = "#4C7A3E"
WARN = "#8A5A2C"
LEVEL_COLOR = {"LEVE": "#6E8CA0", "GRAVE": "#B58900", "FATAL": "#A33F2B"}


def _style(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    if title:
        ax.set_title(title, fontsize=11.5, color=INK, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    return ax


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote outputs/figures/{name}")


# --------------------------------------------------------------------------
# 1. What re-ranking actually changes: the rank of the right article
# --------------------------------------------------------------------------
def figure_reranking(pipeline):
    print("1/4 reranking_rank_shift ...")
    # Stage 1 alone returns the wide candidate pool; truncate it to k so the
    # two conditions are compared over the same number of final results.
    before = evaluate_retrieval(lambda q: pipeline.retriever.invoke(q)[: pipeline.k])
    after = evaluate_retrieval(pipeline.retrieve)
    s_before, s_after = summarize(before), summarize(after)

    def ranks(df):
        rr = df["reciprocal_rank"].to_numpy()
        return np.where(rr > 0, np.rint(1 / np.where(rr > 0, rr, 1)), 0).astype(int)

    r_before, r_after = ranks(before), ranks(after)
    n = len(r_before)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    positions = [1, 2, 3, 4, 0]
    labels = ["rank 1", "rank 2", "rank 3", "rank 4", "not in top-4"]
    cb = [int((r_before == p).sum()) for p in positions]
    ca = [int((r_after == p).sum()) for p in positions]
    x = np.arange(len(positions))
    ax1.bar(x - 0.19, cb, width=0.38, color=BEFORE, label="Hybrid retrieval only",
            edgecolor="white", linewidth=1.1)
    ax1.bar(x + 0.19, ca, width=0.38, color=AFTER, label="After Cross-Encoder re-ranking",
            edgecolor="white", linewidth=1.1)
    for i, (b, a) in enumerate(zip(cb, ca)):
        if b:
            ax1.text(i - 0.19, b + 0.4, str(b), ha="center", fontsize=9, color=INK)
        if a:
            ax1.text(i + 0.19, a + 0.4, str(a), ha="center", fontsize=9, color=AFTER,
                     fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    _style(ax1, ylabel=f"Queries (of {n})")
    ax1.set_title(f"Where the relevant article lands\n{cb[0]} -> {ca[0]} queries get it at rank 1",
                  fontsize=11.5, color=INK, pad=12)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0, max(max(cb), max(ca)) * 1.18)

    names = ["MRR", "NDCG@k", "Context Relevance@k"]
    vb = [s_before[k] for k in names]
    va = [s_after[k] for k in names]
    x2 = np.arange(len(names))
    ax2.bar(x2 - 0.19, vb, width=0.38, color=BEFORE, edgecolor="white", linewidth=1.1)
    ax2.bar(x2 + 0.19, va, width=0.38, color=AFTER, edgecolor="white", linewidth=1.1)
    for i, (b, a) in enumerate(zip(vb, va)):
        ax2.text(i - 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.8, color=INK)
        ax2.text(i + 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.8,
                 color=AFTER, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["MRR", "NDCG@4", "Context\nRelevance@4"], fontsize=9)
    ax2.set_ylim(0, 1.12)
    _style(ax2, ylabel="Score")
    ax2.set_title("Context Relevance cannot move; the rank metrics can",
                  fontsize=11.5, color=INK, pad=12)
    ax2.annotate("capped at 1/4 by\nconstruction: one\nrelevant article\nper query",
                 xy=(2.19, va[2]), xytext=(2.3, 0.55), fontsize=8.2, color="#777777",
                 ha="center", arrowprops=dict(arrowstyle="->", color="#999999", linewidth=0.8))

    fig.text(0.5, -0.045,
             f"{n} hand-labelled queries, one per indexed DS 132 article. Both conditions return "
             f"{pipeline.k} results; only the ordering differs.\n"
             "Context Relevance@4 is precision@4, and with exactly one relevant article per query it "
             "is pinned at 0.250 whenever that article is\nanywhere in the top 4 — which it already "
             "was. That it does not move is a property of the eval set, not a failure of re-ranking.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "reranking_rank_shift.png")
    return s_before, s_after, cb, ca


# --------------------------------------------------------------------------
# 2. The severity classifier, and the class it cannot separate
# --------------------------------------------------------------------------
def figure_classifier():
    print("2/4 severity_classifier ...")
    incidents = load_incidents()
    texts = [r["narrative_text"] for r in incidents]
    labels = [r["severity"] for r in incidents]
    counts = Counter(labels)

    embedder = get_embedder()
    X = embed_texts(texts, embedder)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = train_severity_classifier(X_tr, y_tr)
    m = evaluate_classifier(model, X_te, y_te)
    cm = np.array(m["confusion_matrix"])
    report = m["classification_report"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cm_pct, cmap="YlGnBu", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0%}", ha="center", va="center",
                     fontsize=10, color="white" if cm_pct[i, j] > 0.55 else INK)
    ax1.set_xticks(range(3)), ax1.set_yticks(range(3))
    ax1.set_xticklabels(SEVERITY_LEVELS), ax1.set_yticklabels(SEVERITY_LEVELS)
    ax1.set_xlabel("Predicted", fontsize=10, color=INK)
    ax1.set_ylabel("Actual", fontsize=10, color=INK)
    ax1.set_title(f"Confusion matrix — {len(y_te)} held-out incidents\n"
                  f"accuracy {m['accuracy']:.3f}, F1-macro {m['f1_macro']:.3f}",
                  fontsize=11, color=INK, pad=12)
    ax1.tick_params(colors=INK, labelsize=9)
    for s in ax1.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)

    x = np.arange(len(SEVERITY_LEVELS))
    f1s = [report[l]["f1-score"] for l in SEVERITY_LEVELS]
    prec = [report[l]["precision"] for l in SEVERITY_LEVELS]
    rec = [report[l]["recall"] for l in SEVERITY_LEVELS]
    ax2.bar(x - 0.26, prec, width=0.25, color="#A8B8C4", label="Precision",
            edgecolor="white", linewidth=1)
    ax2.bar(x, rec, width=0.25, color="#7E9AAC", label="Recall", edgecolor="white", linewidth=1)
    ax2.bar(x + 0.26, f1s, width=0.25, color=[LEVEL_COLOR[l] for l in SEVERITY_LEVELS],
            label="F1", edgecolor="white", linewidth=1)
    for i, v in enumerate(f1s):
        ax2.text(i + 0.26, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.6, color=INK)
    for i, l in enumerate(SEVERITY_LEVELS):
        ax2.text(i, -0.14, f"{counts[l]} in corpus", ha="center", fontsize=8,
                 color="#777777", transform=ax2.get_xaxis_transform())
    ax2.set_xticks(x)
    ax2.set_xticklabels(SEVERITY_LEVELS)
    ax2.set_ylim(0, 1.1)
    _style(ax2, ylabel="Score")
    ax2.set_title("Per class — the macro average hides the spread",
                  fontsize=11, color=INK, pad=12)
    ax2.legend(frameon=False, fontsize=8.6, ncol=3, loc="upper center")

    # Severity is ordered, so the two kinds of mistake are not equivalent:
    # calling a FATAL incident LEVE is the one that matters in a triage tool.
    under = int(sum(cm[i, j] for i in range(3) for j in range(3) if j < i))
    over = int(sum(cm[i, j] for i in range(3) for j in range(3) if j > i))
    fatal_as_leve = int(cm[SEVERITY_LEVELS.index("FATAL"), SEVERITY_LEVELS.index("LEVE")])
    total = int(cm.sum())

    worst = SEVERITY_LEVELS[int(np.argmin(f1s))]
    best = SEVERITY_LEVELS[int(np.argmax(f1s))]
    fig.text(0.5, -0.155,
             f"{len(incidents)} synthetic incidents, 75/25 stratified split, embeddings + logistic "
             f"regression with balanced class weights.\n"
             f"F1 runs from {min(f1s):.2f} on {worst} to {max(f1s):.2f} on {best}. The macro average "
             f"of {m['f1_macro']:.3f} is the mean of three quite different numbers.\n"
             f"Severity is an ordered label, so the errors are not interchangeable: "
             f"{under}/{total} incidents are under-triaged (predicted milder than they are)\n"
             f"against {over}/{total} over-triaged, and {fatal_as_leve} FATAL incidents are read as "
             "LEVE. For a triage tool that asymmetry matters more than the headline accuracy —\n"
             "over-triage wastes a review, under-triage misses the incident that needed one.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "severity_classifier.png")
    return m, counts, f1s, under, over, fatal_as_leve


# --------------------------------------------------------------------------
# 3. Corpus composition
# --------------------------------------------------------------------------
def figure_corpus(pipeline):
    print("3/4 corpus_composition ...")
    incidents = load_incidents()
    sev = Counter(r["severity"] for r in incidents)
    types = Counter(r.get("incident_type", "?") for r in incidents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    lv = [sev[l] for l in SEVERITY_LEVELS]
    ax1.bar(SEVERITY_LEVELS, lv, color=[LEVEL_COLOR[l] for l in SEVERITY_LEVELS],
            width=0.6, edgecolor="white", linewidth=1.2)
    for i, v in enumerate(lv):
        ax1.text(i, v + 1, f"{v}\n{v / sum(lv):.0%}", ha="center", fontsize=9, color=INK)
    _style(ax1, ylabel="Incidents")
    ax1.set_ylim(0, max(lv) * 1.26)
    ax1.set_title(f"Severity — {sum(lv)} synthetic incidents\n"
                  f"{max(lv) / min(lv):.1f}x imbalance between the extremes",
                  fontsize=11, color=INK, pad=12)

    top = types.most_common()
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    ax2.barh(range(len(names)), vals, color="#8FA8B8", height=0.68,
             edgecolor="white", linewidth=1)
    for i, v in enumerate(vals):
        ax2.text(v + 0.3, i, str(v), va="center", fontsize=8.6, color=INK)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8.6)
    ax2.set_xlim(0, max(vals) * 1.18)
    _style(ax2, xlabel="Incidents", grid_axis="x")
    ax2.set_title(f"{len(types)} incident types, in Chilean mining vocabulary",
                  fontsize=11, color=INK, pad=12)

    fig.text(0.5, -0.07,
             f"Generated by src/nlp/dataset_generator.py. The regulation side is separate: "
             f"{len(pipeline.documents)} DS 132 articles indexed by the RAG pipeline,\n"
             "a curated excerpt rather than the full decree — see the regulatory disclaimer in the "
             "README before reading any of this as legal guidance.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "corpus_composition.png")
    return sev, types


# --------------------------------------------------------------------------
# 4. Citation faithfulness, per query
# --------------------------------------------------------------------------
def figure_faithfulness(pipeline):
    print("4/4 citation_faithfulness ...")
    df = evaluate_faithfulness(pipeline)
    s = summarize(df)
    vals = df["citation_faithfulness"].to_numpy(dtype=float)
    finite = vals[~np.isnan(vals)]
    order = np.argsort(finite)[::-1]
    sorted_vals = finite[order]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    colors = [OK if v >= 0.999 else WARN for v in sorted_vals]
    ax.bar(range(len(sorted_vals)), sorted_vals, color=colors, width=0.78,
           edgecolor="white", linewidth=0.8)
    ax.axhline(float(np.mean(finite)), color=INK, linestyle="--", linewidth=1.2)
    ax.text(len(sorted_vals) - 0.4, float(np.mean(finite)) + 0.012,
            f"mean {np.mean(finite):.3f}", ha="right", fontsize=9, color=INK)

    below = int((sorted_vals < 0.999).sum())
    _style(ax, xlabel="Query (sorted by score)", ylabel="Citation faithfulness")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Every cited article is one that was retrieved, on "
                 f"{len(sorted_vals) - below} of {len(sorted_vals)} queries",
                 fontsize=11.5, color=INK, pad=12)
    ax.set_xticks([])

    backend = df["backend"].iloc[0]
    fig.text(0.5, -0.115,
             f"Backend: {backend}. The metric is the share of article citations in the answer that "
             "appear in the retrieved set —\nit catches a fabricated citation, not a wrong reading of "
             f"a real one. No LLM judge is involved, which is deliberate:\nthe whole pipeline runs "
             "without an external LLM, and an evaluation needing one would undercut that.\n"
             f"The {below} queries below 1.0 share a single verified cause: Article 76's own text "
             "cross-references article 13,\nwhich sits outside the curated excerpt and is never "
             "retrieved. The regex cannot tell that in-text reference\napart from a citation the "
             "pipeline is making — see the honest note in the README.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "citation_faithfulness.png")
    return s, below, len(sorted_vals)


if __name__ == "__main__":
    print(f"Writing figures to {FIG_DIR}\n")
    print("Building the RAG pipeline (chunking + hybrid retriever + re-ranker)...")
    pipeline = RAGPipeline()
    print(f"  indexed articles: {len(pipeline.documents)}  backend: {pipeline.backend_name}\n")

    s_before, s_after, cb, ca = figure_reranking(pipeline)
    metrics, counts, f1s, under, over, fatal_as_leve = figure_classifier()
    sev, types = figure_corpus(pipeline)
    faith, below, n_faith = figure_faithfulness(pipeline)

    print("\nNumbers annotated on the figures:")
    print(f"  MRR            : {s_before['MRR']:.3f} -> {s_after['MRR']:.3f}")
    print(f"  NDCG@4         : {s_before['NDCG@k']:.3f} -> {s_after['NDCG@k']:.3f}")
    print(f"  CtxRelevance@4 : {s_before['Context Relevance@k']:.3f} -> "
          f"{s_after['Context Relevance@k']:.3f}")
    print(f"  rank-1 queries : {cb[0]} -> {ca[0]} (of {len(QUERY_DATASET)})")
    print(f"  classifier     : acc {metrics['accuracy']:.3f}, F1-macro {metrics['f1_macro']:.3f}")
    print("  per-class F1   : " + ", ".join(
        f"{l} {v:.3f}" for l, v in zip(SEVERITY_LEVELS, f1s)))
    print(f"  triage errors  : {under} under-triaged, {over} over-triaged, "
          f"{fatal_as_leve} FATAL read as LEVE")
    print(f"  corpus         : {sum(counts.values())} incidents, {len(types)} types, "
          f"{dict(sev)}")
    print(f"  faithfulness   : {faith['Citation Faithfulness']:.3f} "
          f"({below}/{n_faith} below 1.0)")
