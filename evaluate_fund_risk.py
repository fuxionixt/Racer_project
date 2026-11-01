"""Leakage check for a fund-risk classifier on the Mutual Funds India test split.

A Random Forest predicts each fund's risk tier (LOW/MEDIUM/HIGH, from risk_level) from
the 16 candidate predictors, then again with risk_level itself added as a feature. The
gap shows how far a naive pipeline that keeps the target-derived feature overstates
accuracy. Same split as evaluate_cce.py; missing values are filled with training medians.

Run clean_data.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluate_cce import CLEAN_PATH, SEED, risk_tier

OUT_PATH = Path(__file__).resolve().parent / "results" / "fund_risk_results.json"
N_BOOT = 2000

NUMERIC = ["min_sip", "min_lumpsum", "expense_ratio", "fund_size_cr", "fund_age_yr", "sortino", "alpha", "sd",
           "beta", "sharpe", "rating", "returns_1yr", "returns_3yr", "returns_5yr"]
CATEGORICAL = ["category", "sub_category"]


def model(numeric: list[str]):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                class_weight="balanced", random_state=SEED)
    return make_pipeline(prep, rf)


def bootstrap_ci(correct: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = correct[rng.integers(0, len(correct), (N_BOOT, len(correct)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(numeric, train, test) -> dict:
    X_train, X_test = train[numeric + CATEGORICAL], test[numeric + CATEGORICAL]
    m = model(numeric)
    cv = cross_val_score(m, X_train, train["tier"], cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    m.fit(X_train, train["tier"])
    pred = m.predict(X_test)
    return {
        "n_features": len(numeric) + len(CATEGORICAL),
        "train_accuracy": accuracy_score(train["tier"], m.predict(X_train)),
        "test_accuracy": accuracy_score(test["tier"], pred),
        "test_accuracy_ci": bootstrap_ci((pred == test["tier"].to_numpy()).astype(float)),
        "cv_mean": cv.mean(),
        "cv_sd": cv.std(),
        "f1_macro": f1_score(test["tier"], pred, average="macro"),
    }


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)
    df["tier"] = risk_tier(df["risk_level"])
    train, test = train_test_split(df, test_size=0.25, stratify=df["tier"], random_state=SEED)
    results = {
        "n_train": len(train),
        "n_test": len(test),
        "without_risk_level": evaluate(NUMERIC, train, test),
        "with_risk_level": evaluate(NUMERIC + ["risk_level"], train, test),
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    for name in ("without_risk_level", "with_risk_level"):
        r = results[name]
        print(f"{name:20s} train {r['train_accuracy']:.4f}  test {r['test_accuracy']:.4f} "
              f"[{r['test_accuracy_ci'][0]:.3f}, {r['test_accuracy_ci'][1]:.3f}]  "
              f"cv {r['cv_mean']:.4f}±{r['cv_sd']:.4f}  f1 {r['f1_macro']:.4f}")
    print(f"Saved to {OUT_PATH.relative_to(OUT_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
