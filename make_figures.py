"""Redraw paper Figs. 4, 6 and 8 from experiment results.

Run evaluate_cce.py, racer_code/racer/experiments/exp1_finqa_retrieval.py and
exp4_generation_hallucination.py first. Writes PNGs to figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CCE_PATH = ROOT / "results" / "cce_results.json"
FINQA_PATH = ROOT / "racer_code" / "racer" / "results" / "exp1_finqa_test.json"
GEN_PATH = ROOT / "racer_code" / "racer" / "results" / "exp4_anthropic.json"
OUT_DIR = ROOT / "figures"

PURPLE, GREEN, TEAL, ORANGE, GREY = "#5349b7", "#1d9e75", "#2196a8", "#d95b33", "#8a8a8a"
LIGHT = {PURPLE: "#9b93e0", GREEN: "#5bbf98", TEAL: "#63b4c0", ORANGE: "#e8a088"}

# (key in exp1 results, label, colour)
RETRIEVERS = [("random", "Random", GREY), ("tfidf", "TF-IDF (RACER)", PURPLE),
              ("bm25", "BM25", TEAL), ("hybrid_rrf", "Hybrid RRF", GREEN)]

# Classifier results as reported in the paper (Tables V-VI); not recomputed here.
D2_RF = {"Train": 90.49, "Test": 91.18, "CV": 86.24, "CV_SD": 2.80}
D3_RF = {"Train": 89.33, "Test": 83.60, "CV": 82.70, "CV_SD": 2.32}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linestyle": "--", "grid.alpha": 0.5, "axes.axisbelow": True,
})


def label_bars(ax, bars, fmt="{:.1f}%", size=9):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=size, fontweight="bold")


def rf_bars(ax, res, x0, color_set):
    xs = [x0, x0 + 1, x0 + 2]
    bars = ax.bar(xs, [res["Train"], res["Test"], res["CV"]], width=0.6, color=color_set)
    ax.errorbar(xs[2], res["CV"], yerr=res["CV_SD"], color="#222", capsize=8, lw=1.5)
    label_bars(ax, bars)
    return xs


def retriever_bars(ax, summary, metric, scale=100, fmt="{:.1f}"):
    """Bars with 95% bootstrap CIs for each retriever on one metric."""
    for i, (key, label, color) in enumerate(RETRIEVERS):
        m = summary[key][metric]
        ax.bar(i, scale * m["mean"], width=0.6, color=color, label=label)
        ax.errorbar(i, scale * m["mean"], yerr=[[scale * (m["mean"] - m["lo"])], [scale * (m["hi"] - m["mean"])]],
                    color="#222", capsize=6, lw=1.2)
        ax.annotate(fmt.format(scale * m["mean"]), (i, scale * m["hi"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(RETRIEVERS)), [label.replace(" (", "\n(") for _, label, _ in RETRIEVERS])


def fig4(finqa):
    closed, opened = finqa["modes"]["closed"]["summary"], finqa["modes"]["open"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    fig.suptitle(f"RACER — FinQA Supporting-Fact Retrieval (Test Set, N = {finqa['n_questions']:,})",
                 fontweight="bold", fontsize=16)
    panels = [(closed, "recall@5", "(a) Recall@5 — Closed Pool\n(facts of the question's own filing)", "Recall@5 (%)"),
              (opened, "recall@5", f"(b) Recall@5 — Open Corpus\n(all {finqa['n_facts']:,} test-split facts)", "Recall@5 (%)"),
              (opened, "mrr", "(c) MRR — Open Corpus", "MRR (×100)")]
    for ax, (summary, metric, title, ylabel) in zip(axes, panels):
        retriever_bars(ax, summary, metric)
        ax.set_ylim(0, 100)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "fig4_finqa.png", dpi=200)


def fig6(cce):
    t5 = cce["theta_5"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
    fig.suptitle("RACER — Mutual Funds India: Financial Product Suitability Evaluation", fontweight="bold", fontsize=16)

    ax = axes[0]
    xs = rf_bars(ax, D2_RF, 0, [PURPLE, GREEN, TEAL])
    ax.set_xticks(xs, ["Train", "Test", "CV (5-fold)"])
    ax.set_ylim(70, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("(a) Supporting RF Classifier Performance", fontweight="bold")

    ax = axes[1]
    tiers = ["LOW", "MEDIUM", "HIGH"]
    colors = [GREEN, PURPLE, ORANGE]
    comp = [100 * t5["per_tier"][t]["compliance"] for t in tiers]
    rej = [100 - c for c in comp]
    xs = range(len(tiers))
    b1 = ax.bar([x - 0.18 for x in xs], comp, width=0.34, color=colors)
    b2 = ax.bar([x + 0.18 for x in xs], rej, width=0.34, color=[LIGHT[c] for c in colors])
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.set_xticks(list(xs), [f"{t}\nRisk (n={t5['per_tier'][t]['n']})" for t in tiers])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(b) CCE Compliance by Risk Group", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#555"), plt.Rectangle((0, 0), 1, 1, color="#bbb")]
    ax.legend(handles, ["Compliant % (dark)", "Rejected % (light)"], loc="upper left")

    ax = axes[2]
    vals = [0.0, 100.0, 100 * t5["compliance"]]
    bars = ax.bar(range(3), vals, width=0.55, color=[GREY, TEAL, PURPLE])
    label_bars(ax, bars)
    ax.set_xticks(range(3), ["Keyword\nMatching", "Standard\nRAG", "RACER\n(Proposed)"])
    ax.set_ylim(0, 120)
    ax.set_ylabel("Compliance (%)")
    ax.set_title("(c) 3-Model Comparison", fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "fig6_mutualfunds.png", dpi=200)


def fig8(cce, finqa, gen):
    t5 = cce["theta_5"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4))
    fig.suptitle("RACER Framework — Combined Results Across All Datasets", fontweight="bold", fontsize=16)

    ax = axes[0]
    xs2 = rf_bars(ax, D2_RF, 0, [PURPLE, GREEN, TEAL])
    xs3 = rf_bars(ax, D3_RF, 3, [LIGHT[PURPLE], LIGHT[GREEN], LIGHT[TEAL]])
    ax.axvline(2.5, color="#bbb", ls="--")
    ax.text(1, 97, "Dataset 2", ha="center", style="italic", color=PURPLE)
    ax.text(4, 97, "Dataset 3", ha="center", style="italic", color=PURPLE)
    ax.set_xticks(xs2 + xs3, ["D2\nTrain", "D2\nTest", "D2\nCV", "D3\nTrain", "D3\nTest", "D3\nCV"])
    ax.set_ylim(70, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) RF Classifier Accuracy\nDatasets 2 & 3", fontweight="bold")

    ax = axes[1]
    retriever_bars(ax, finqa["modes"]["open"]["summary"], "recall@5")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall@5 (%)")
    ax.set_title("(b) FinQA Retrieval, Open Corpus\nDataset 1 (Test Set)", fontweight="bold")

    ax = axes[2]
    variants = list(gen["variants"].values())
    vals = [100 * (1 - t5["compliance"]), 100 * variants[0]["has_hallucination"]["mean"],
            100 * variants[1]["has_hallucination"]["mean"]]
    bars = ax.bar(range(3), vals, width=0.55, color=[ORANGE, LIGHT[ORANGE], PURPLE])
    label_bars(ax, bars)
    ax.set_xticks(range(3), [f"CCE rejection\n(D2, n={t5['n_evaluated']})",
                             f"Hallucinated\nno validator (n={gen['n']})", f"Hallucinated\nwith validator (n={gen['n']})"])
    ax.set_ylim(0, 40)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(c) Compliance Filtering and Validation\nDataset 2", fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "fig8_combined.png", dpi=200)


def main() -> None:
    cce = json.loads(CCE_PATH.read_text())
    finqa = json.loads(FINQA_PATH.read_text())
    gen = json.loads(GEN_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True)
    fig4(finqa)
    fig6(cce)
    fig8(cce, finqa, gen)
    print(f"Saved figures to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
