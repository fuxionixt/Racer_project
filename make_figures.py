"""Redraw paper Figs. 6 and 8 from results/cce_results.json and the reported classifier results.

Run evaluate_cce.py first. Writes PNGs to figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "cce_results.json"
OUT_DIR = ROOT / "figures"

PURPLE, GREEN, TEAL, ORANGE, GREY = "#5349b7", "#1d9e75", "#2196a8", "#d95b33", "#8a8a8a"
LIGHT = {PURPLE: "#9b93e0", GREEN: "#5bbf98", TEAL: "#63b4c0", ORANGE: "#e8a088"}

# Classifier results as reported in the paper (Tables IV-VI); not recomputed here.
D2_RF = {"Train": 90.49, "Test": 91.18, "CV": 86.24, "CV_SD": 2.80}
D3_RF = {"Train": 89.33, "Test": 83.60, "CV": 82.70, "CV_SD": 2.32}
D1_TEST = {  # Keyword Matching, Standard RAG, RACER
    "Recall@K\n(%)": (0.0, 1.0, 6.0),
    "Compliance\n(%)": (0.0, 100.0, 100.0),
    "MRR\n(×100)": (0.0, 1.0, 3.1),
    "Exact\nMatch (%)": (1.0, 1.0, 2.0),
}
D1_DEV_COMPLIANCE = (95.0, 5.0)
D1_TEST_COMPLIANCE = 100.0

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


def fig6(cce):
    t5 = cce["theta_5"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
    fig.suptitle("RACER — Mutual Funds India: Financial Product Suitability Evaluation", fontweight="bold", fontsize=16)

    ax = axes[0]
    xs = rf_bars(ax, D2_RF, 0, [PURPLE, GREEN, TEAL])
    ax.set_xticks(xs, ["Train", "Test", "CV (5-fold)"])
    ax.set_ylim(70, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("(a) RF Classifier Performance", fontweight="bold")

    ax = axes[1]
    tiers = ["LOW", "MEDIUM", "HIGH"]
    colors = [GREEN, PURPLE, ORANGE]
    comp = [100 * t5["per_tier"][t]["compliance"] for t in tiers]
    rej = [100 - c for c in comp]
    xs = range(len(tiers))
    b1 = ax.bar([x - 0.18 for x in xs], comp, width=0.34, color=colors, label="Compliant %")
    b2 = ax.bar([x + 0.18 for x in xs], rej, width=0.34, color=[LIGHT[c] for c in colors], label="Rejected %")
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


def fig8(cce):
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
    comp = [D1_DEV_COMPLIANCE[0], D1_TEST_COMPLIANCE, 100 * t5["compliance"]]
    rej = [D1_DEV_COMPLIANCE[1], None, 100 * t5["rejection_rate"]]
    b1 = ax.bar([x - 0.18 for x in range(3)], comp, width=0.34, color=[PURPLE, GREEN, TEAL], label="Compliant %")
    label_bars(ax, b1)
    rx = [x + 0.18 for x, r in zip(range(3), rej) if r is not None]
    b2 = ax.bar(rx, [r for r in rej if r is not None], width=0.34, color=LIGHT[ORANGE], label="Rejected %")
    label_bars(ax, b2)
    ax.set_xticks(range(3), ["D1\nDev", "D1\nTest", f"D2\nFunds (n={t5['n_evaluated']})"])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(b) CCE Compliance\nDatasets 1 & 2", fontweight="bold")
    ax.legend(loc="upper right")

    ax = axes[2]
    names = ["Keyword Matching", "Standard RAG", "RACER (Proposed)"]
    for i, (name, color) in enumerate(zip(names, [GREY, TEAL, PURPLE])):
        vals = [v[i] for v in D1_TEST.values()]
        bars = ax.bar([x + (i - 1) * 0.26 for x in range(len(D1_TEST))], vals, width=0.26, color=color, label=name)
        label_bars(ax, bars, fmt="{:.1f}", size=8)
    ax.set_xticks(range(len(D1_TEST)), list(D1_TEST))
    ax.set_ylim(0, 120)
    ax.set_ylabel("Score (%)")
    ax.set_title("(c) RACER vs Baselines\nDataset 1 (Test Set, N = 100)", fontweight="bold")
    ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "fig8_combined.png", dpi=200)


def main() -> None:
    cce = json.loads(RESULTS_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True)
    fig6(cce)
    fig8(cce)
    print(f"Saved figures to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
