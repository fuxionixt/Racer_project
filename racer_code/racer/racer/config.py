"""Central configuration. Every tunable value lives here so results are reproducible."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42

# ---- retrieval -------------------------------------------------------------
TFIDF_MAX_FEATURES = 15_000
TFIDF_NGRAM = (1, 2)
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
CLAUSE_RETRIEVER = "dense"  # regulatory-clause index; products stay on BM25

# ---- statistics ------------------------------------------------------------
BOOTSTRAP_REPS = 2000
CI_LEVEL = 0.95

# ---- risk profiling (policy) ----------------------------------------------
# phi(u) is a *risk-capacity* score: HIGHER phi => HIGHER capacity to bear risk.
POLICY_WEIGHTS = {
    "income": 0.30,
    "credit_score": 0.25,
    "savings_ratio": 0.25,
    "loan_default_risk": 0.20,
}
# Features where a larger raw value LOWERS capacity (reverse-coded before weighting).
POLICY_REVERSED = {"loan_default_risk"}
TIER_NAMES = ("LOW", "MEDIUM", "HIGH")  # risk-tolerance tier of the *user*

RF_PARAMS = dict(n_estimators=100, max_depth=5, min_samples_leaf=5,
                 class_weight="balanced", random_state=SEED, n_jobs=1)

# ---- compliance constraint engine -----------------------------------------
# Suitability bounds per user tier. "q" values are quantiles of the TRAINING
# catalogue; the others are absolute. These are *illustrative, author-chosen*
# bounds -- they are NOT quoted from MiFID II or SEBI text.
RULE_BOUNDS = {
    #            max_risk_level  max_expense_q  min_rating  categories             max_sd_q  min_sharpe_q
    "LOW":    dict(risk=2, exp_q=0.35, rating=3, cats=("Debt",),            sd_q=0.30, sharpe_q=0.50),
    "MEDIUM": dict(risk=4, exp_q=0.60, rating=2, cats=("Debt", "Hybrid"),   sd_q=0.60, sharpe_q=0.25),
    "HIGH":   dict(risk=6, exp_q=0.85, rating=1, cats=None,                 sd_q=0.90, sharpe_q=0.10),
}
CCE_THETA = 5          # minimum number of the M=6 checks that must pass
CCE_HARD_CHECKS = ("risk_level",)  # checks that must ALWAYS pass (suitability floor)
PROFILE_BOOST = 1.15   # multiplicative boost when product tier == user tier
TOP_K = 5
CANDIDATE_K = 20       # retrieved before compliance filtering
