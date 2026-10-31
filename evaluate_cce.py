"""Evaluate the Compliance Constraint Engine (CCE) on the Mutual Funds India test split.

Each test fund is checked against the suitability rules (paper, Table II) for its own
risk tier. Quantile bounds come from the training split only. Funds missing any checked
attribute are excluded before filtering and reported separately.

Run clean_data.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CLEAN_PATH = ROOT / "data" / "processed" / "mutual_funds_clean.csv"
OUT_PATH = ROOT / "results" / "cce_results.json"

SEED = 42
CHECKED = ["risk_level", "expense_ratio", "rating", "category", "sd", "sharpe"]

# Table II: absolute bounds, or quantiles of the training split (q).
RULES = {
    "LOW": dict(max_risk=2, max_expense_q=0.35, min_rating=3, categories={"Debt"}, max_sd_q=0.30, min_sharpe_q=0.50),
    "MEDIUM": dict(max_risk=4, max_expense_q=0.60, min_rating=2, categories={"Debt", "Hybrid"}, max_sd_q=0.60, min_sharpe_q=0.25),
    "HIGH": dict(max_risk=6, max_expense_q=0.85, min_rating=1, categories=None, max_sd_q=0.90, min_sharpe_q=0.10),
}


def risk_tier(risk_level: pd.Series) -> pd.Series:
    """<=2 LOW, 3-4 MEDIUM, >=5 HIGH."""
    return pd.Series(np.where(risk_level <= 2, "LOW", np.where(risk_level <= 4, "MEDIUM", "HIGH")), index=risk_level.index)


def count_passed(fund: pd.Series, tier: str, train: pd.DataFrame) -> int:
    r = RULES[tier]
    checks = [
        fund["risk_level"] <= r["max_risk"],
        fund["expense_ratio"] <= train["expense_ratio"].quantile(r["max_expense_q"]),
        fund["rating"] >= r["min_rating"],
        r["categories"] is None or fund["category"] in r["categories"],
        fund["sd"] <= train["sd"].quantile(r["max_sd_q"]),
        fund["sharpe"] >= train["sharpe"].quantile(r["min_sharpe_q"]),
    ]
    return sum(bool(c) for c in checks)


def evaluate(test: pd.DataFrame, train: pd.DataFrame, theta: int) -> dict:
    passed = test.apply(lambda f: count_passed(f, f["tier"], train), axis=1)
    compliant = passed >= theta
    per_tier = {
        t: {"n": int((test["tier"] == t).sum()), "compliance": float(compliant[test["tier"] == t].mean())}
        for t in RULES
    }
    return {
        "theta": theta,
        "n_evaluated": len(test),
        "n_rejected": int((~compliant).sum()),
        "compliance": float(compliant.mean()),
        "rejection_rate": float(1 - compliant.mean()),
        "per_tier": per_tier,
    }


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)
    df["tier"] = risk_tier(df["risk_level"])
    train, test = train_test_split(df, test_size=0.25, stratify=df["tier"], random_state=SEED)

    complete = test[CHECKED].notna().all(axis=1)
    results = {
        "n_train": len(train),
        "n_test": len(test),
        "n_excluded_missing": int((~complete).sum()),
        "theta_5": evaluate(test[complete], train, theta=5),
        "theta_4": evaluate(test[complete], train, theta=4),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
