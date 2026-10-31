"""Ranking metrics and statistics (bootstrap CIs, paired tests)."""
from __future__ import annotations

import numpy as np

from .config import BOOTSTRAP_REPS, CI_LEVEL, SEED


def rank_of_gold(scores: np.ndarray, gold_idx: np.ndarray) -> np.ndarray:
    """1-based rank of each gold item when `scores` is sorted descending (ties broken by index)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank[gold_idx]


def per_query_metrics(gold_ranks: np.ndarray, ks=(3, 5, 10)) -> dict:
    """Metrics for ONE query given the ranks of all its gold items."""
    gr = np.sort(gold_ranks)
    out = {"mrr": 1.0 / gr[0]}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(gr <= k))          # fraction of gold facts retrieved
        out[f"hit_all@{k}"] = float(np.all(gr <= k))          # every gold fact retrieved
    dcg = np.sum(1.0 / np.log2(gr[gr <= 10] + 1))
    idcg = np.sum(1.0 / np.log2(np.arange(1, min(len(gr), 10) + 1) + 1))
    out["ndcg@10"] = float(dcg / idcg)
    return out


def bootstrap_ci(values, reps: int = BOOTSTRAP_REPS, level: float = CI_LEVEL, seed: int = SEED):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_bootstrap(a, b, reps: int = BOOTSTRAP_REPS, level: float = CI_LEVEL, seed: int = SEED):
    """Paired bootstrap of mean(a - b). Returns dict(diff, lo, hi, p_two_sided)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(axis=1)
    al = (1 - level) / 2
    # two-sided p: how often the resampled mean crosses zero
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, al)),
            "hi": float(np.quantile(means, 1 - al)), "p": float(min(max(p, 1.0 / reps), 1.0))}


def mean_ci_over_runs(values, level: float = CI_LEVEL):
    """Normal-approx (t) CI over independent repeated runs/seeds. Returns (mean, lo, hi)."""
    from scipy import stats

    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    se = v.std(ddof=1) / np.sqrt(len(v))
    h = float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * se)
    return m, m - h, m + h
