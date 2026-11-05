"""Score the filled annotation sheets (see scripts/make_annotation_kits.py).

  python scripts/score_annotations.py --cce annotation/filled/cce_*.csv --clauses annotation/filled/clauses_*.csv

CCE external validation: each annotator's agreement with the CCE, Cohen's kappa between the CCE and the
annotators' majority label, and inter-annotator agreement (pairwise Cohen's kappa, Fleiss' kappa).
Clause relevance: inter-annotator quadratic-weighted kappa on the 0/1/2 grades; the consensus grade (rounded mean)
then scores each retriever's top 3 for the 192 tier x query cases by P@3, MRR and nDCG@3.
Writes results/annotation_scores.json.
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer import config  # noqa: E402

LABEL, GRADE = "label (Suitable/Unsuitable)", "relevance (0/1/2)"


def fleiss(labels: np.ndarray) -> float:
    """labels: items x raters, categorical."""
    cats = np.unique(labels)
    counts = np.stack([(labels == c).sum(1) for c in cats], 1)
    n = labels.shape[1]
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else 1.0


def pairwise(frame, weights=None):
    return {f"{a}~{b}": float(cohen_kappa_score(frame[a], frame[b], weights=weights))
            for a, b in itertools.combinations(frame.columns, 2)}


def score_cce(paths):
    key = pd.read_csv(config.ROOT / "annotation" / "key" / "cce_key.csv").set_index("pair_id")
    lab = pd.DataFrame({Path(p).stem: pd.read_csv(p).set_index("pair_id")[LABEL].str.strip().str.capitalize()
                        for p in paths}).loc[key.index]
    if lab.isna().any().any():
        sys.exit(f"unlabelled pairs: {lab[lab.isna().any(axis=1)].index.tolist()[:10]}")
    majority = lab.mode(axis=1)[0]
    out = {"n_pairs": len(lab), "annotators": list(lab.columns),
           "agreement_with_cce": {a: float((lab[a] == key.cce).mean()) for a in lab},
           "majority_vs_cce_accuracy": float((majority == key.cce).mean()),
           "majority_vs_cce_kappa": float(cohen_kappa_score(majority, key.cce))}
    if lab.shape[1] > 1:
        out["inter_annotator_kappa"] = pairwise(lab)
        out["fleiss_kappa"] = fleiss(lab.to_numpy())
    disagree = key.assign(majority=majority)[majority != key.cce]
    out["disagreements_by_failed_check"] = disagree.failed.fillna("(none: CCE passed)").value_counts().to_dict()
    return out


def score_clauses(paths):
    sys.path.insert(0, str(config.ROOT / "experiments"))
    from common import EXTENDED_QUERIES, split  # noqa: E402
    from racer.data import load_funds, load_regulatory  # noqa: E402
    from racer.pipeline import Racer  # noqa: E402

    g = pd.DataFrame({Path(p).stem: pd.read_csv(p).set_index("passage_id")[GRADE] for p in paths})
    if g.isna().any().any():
        sys.exit(f"ungraded passages: {g[g.isna().any(axis=1)].index.tolist()[:10]}")
    g = g.astype(int)
    consensus = g.mean(axis=1).round().astype(int)
    out = {"n_passages": len(g), "annotators": list(g.columns),
           "relevant_passages": int((consensus >= 1).sum()), "highly_relevant": int((consensus == 2).sum())}
    if g.shape[1] > 1:
        out["inter_annotator_weighted_kappa"] = pairwise(g, weights="quadratic")
    funds = load_funds(config.DATA_DIR / "funds" / "funds.csv"); ftr, fte = split(funds, funds["tier"])
    clauses = load_regulatory(config.DATA_DIR / "regulatory" / "sebi_full.jsonl")
    ideal = sorted(consensus.values, reverse=True)[:3]
    idcg = sum(gain / np.log2(r + 2) for r, gain in enumerate(ideal)) or 1.0
    for name in ("bm25", "tfidf", "dense"):
        rac = Racer(ftr, fte, clauses, clause_retriever=name)
        p3, rr, nd = [], [], []
        for t in config.TIER_NAMES:
            for q in EXTENDED_QUERIES:
                grades = [int(consensus.get(c["id"], 0)) for c in rac.recommend(t, q).clauses]
                p3.append(np.mean([x >= 1 for x in grades]))
                rr.append(next((1 / (r + 1) for r, x in enumerate(grades) if x >= 1), 0.0))
                nd.append(sum(x / np.log2(r + 2) for r, x in enumerate(grades)) / idcg)
        out[name] = {"P@3": float(np.mean(p3)), "MRR": float(np.mean(rr)), "nDCG@3": float(np.mean(nd))}
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--cce", nargs="*", default=[]); ap.add_argument("--clauses", nargs="*", default=[])
a = ap.parse_args()
res = {}
if a.cce: res["cce_external_validation"] = score_cce(a.cce)
if a.clauses: res["clause_relevance"] = score_clauses(a.clauses)
print(json.dumps(res, indent=2))
(config.RESULTS_DIR / "annotation_scores.json").write_text(json.dumps(res, indent=2))
