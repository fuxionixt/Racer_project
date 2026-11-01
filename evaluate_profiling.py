"""Evaluate the User Risk Profiling Module on the User Financial Profiles dataset (Dataset 3).

Labels follow the paper, eqs. (3)-(5): a weighted sum of min-max normalised income,
credit score, savings ratio and reverse-coded loan default risk, cut at the 33rd and
67th percentiles. The Random Forest (and the baselines) are then trained on the six raw
features to reproduce those labels.

Download financial_dataset.csv to data/raw/profiles/ first (see README).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "profiles" / "financial_dataset.csv"
OUT_PATH = ROOT / "results" / "profiling_results.json"

SEED = 42
N_BOOT = 2000
FEATURES = ["Income", "Credit Score", "Spending Score", "Transaction Count", "Savings Ratio", "Loan Default Risk"]
# Eq. (4): income, credit score, savings ratio, loan default risk (reverse-coded).
WEIGHTS = {"Income": 0.30, "Credit Score": 0.25, "Savings Ratio": 0.25, "Loan Default Risk": 0.20}
CLASSES = ["LOW", "MEDIUM", "HIGH"]


def composite_score(df: pd.DataFrame) -> pd.Series:
    norm = (df - df.min()) / (df.max() - df.min())
    norm["Loan Default Risk"] = 1 - norm["Loan Default Risk"]
    return sum(w * norm[k] for k, w in WEIGHTS.items())


def risk_labels(phi: pd.Series) -> pd.Series:
    """Eq. (5): tiers are risk capacity, so the financially weakest tercile of phi is LOW."""
    t33, t67 = phi.quantile([1 / 3, 2 / 3])
    return pd.Series(np.where(phi <= t33, "LOW", np.where(phi <= t67, "MEDIUM", "HIGH")), index=phi.index)


def bootstrap_ci(correct: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = correct[rng.integers(0, len(correct), (N_BOOT, len(correct)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    cv = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_accuracy_ci": bootstrap_ci((pred == y_test.to_numpy()).astype(float)),
        "cv_mean": cv.mean(),
        "cv_sd": cv.std(),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "f1_weighted": f1_score(y_test, pred, average="weighted"),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro"),
        "confusion": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)[FEATURES]
    y = risk_labels(composite_score(df))
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, stratify=y, random_state=SEED)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                class_weight="balanced", random_state=SEED)
    models = {
        "random_forest": rf,
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "random_stratified": DummyClassifier(strategy="stratified", random_state=SEED),
    }
    results = {
        "n": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_distribution": y.value_counts(normalize=True).round(4).to_dict(),
        "confusion_labels": CLASSES,
        "models": {name: evaluate(m, X_train, X_test, y_train, y_test) for name, m in models.items()},
    }
    results["rf_feature_importance"] = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    for name, r in results["models"].items():
        print(f"{name:20s} train {r['train_accuracy']:.4f}  test {r['test_accuracy']:.4f}  "
              f"[{r['test_accuracy_ci'][0]:.3f}, {r['test_accuracy_ci'][1]:.3f}]  cv {r['cv_mean']:.4f}±{r['cv_sd']:.4f}  f1 {r['f1_macro']:.4f}")
    print("labels", results["label_distribution"])
    print("importance", results["rf_feature_importance"])
    print("rf confusion", results["models"]["random_forest"]["confusion"])
    print(f"Saved to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
