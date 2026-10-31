"""Dataset loaders.

* FinQA (Chen et al., EMNLP 2021): real data, used for fact-level retrieval evaluation.
* Mutual-funds and user-profile CSVs: must be supplied by the user (Kaggle); loaders map
  common column aliases and fail with a clear message instead of guessing.
* Synthetic generators exist ONLY for smoke tests; anything produced from them is flagged
  and must never be reported as a result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- FinQA
def _rm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def row_to_text(header, row) -> str:
    """Verbalise one table row (FinQA's own convention)."""
    res = ""
    if header and header[0]:
        res += header[0] + " "
    for h, c in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + h + " is " + c + " ; "
    return _rm_space(res)


def finqa_facts(ex: dict) -> list[str]:
    """All candidate facts of one FinQA example: text sentences, then table-row sentences."""
    text = list(ex.get("pre_text", [])) + list(ex.get("post_text", []))
    table = ex.get("table", [])
    header = table[0] if table else []
    rows = [row_to_text(header, r) for r in table]  # index i == table_i
    return text, rows


def finqa_gold(ex: dict, n_text: int, n_rows: int) -> list[int]:
    """Gold indices into the concatenation [text facts] + [table-row facts]."""
    out = []
    for key in ex.get("qa", {}).get("gold_inds", {}):
        kind, idx = key.split("_")
        idx = int(idx)
        if kind == "text" and idx < n_text:
            out.append(idx)
        elif kind == "table" and idx < n_rows:
            out.append(n_text + idx)
    return sorted(set(out))


def load_finqa(split: str, data_dir: Path | None = None):
    """Return a list of dicts: id, question, facts, gold (indices into facts)."""
    path = (data_dir or config.DATA_DIR / "finqa") / f"{split}.json"
    raw = json.loads(Path(path).read_text())
    out = []
    for ex in raw:
        text, rows = finqa_facts(ex)
        gold = finqa_gold(ex, len(text), len(rows))
        if not gold:
            continue  # no supporting-fact annotation: cannot be scored
        out.append({"id": ex["id"], "question": ex["qa"]["question"],
                    "facts": text + rows, "gold": gold,
                    "n_text": len(text), "n_rows": len(rows)})
    return out


# --------------------------------------------------------------------------- CSV loaders
FUND_ALIASES = {
    "scheme_name": ["scheme_name", "fund_name", "name", "scheme"],
    "category": ["category", "fund_category", "type"],
    "risk_level": ["risk_level", "risk", "riskometer", "risk_level_numeric"],
    "rating": ["rating", "star_rating", "morningstar_rating"],
    "expense_ratio": ["expense_ratio", "expense", "expense_ratio_pct"],
    "sd": ["sd", "std_dev", "standard_deviation", "volatility"],
    "sharpe": ["sharpe", "sharpe_ratio"],
    "alpha": ["alpha"],
    "beta": ["beta"],
    "return_1yr": ["returns_1yr", "return_1yr", "1yr_return", "returns_1y"],
    "return_3yr": ["returns_3yr", "return_3yr", "3yr_return", "returns_3y"],
    "return_5yr": ["returns_5yr", "return_5yr", "5yr_return", "returns_5y"],
}
PROFILE_ALIASES = {
    "income": ["income", "annual_income", "monthly_income"],
    "credit_score": ["credit_score", "creditscore", "credit"],
    "spending_score": ["spending_score", "spendingscore"],
    "transaction_count": ["transaction_count", "num_transactions", "transactions"],
    "savings_ratio": ["savings_ratio", "savings", "saving_ratio"],
    "loan_default_risk": ["loan_default_risk", "default_risk", "loan_default"],
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def _map_columns(df: pd.DataFrame, aliases: dict, required: list[str], what: str) -> pd.DataFrame:
    norm = {_norm(c): c for c in df.columns}
    rename, missing = {}, []
    for canon, opts in aliases.items():
        hit = next((norm[o] for o in opts if o in norm), None)
        if hit is not None:
            rename[hit] = canon
        elif canon in required:
            missing.append(canon)
    if missing:
        raise ValueError(f"{what}: could not find columns for {missing}. "
                         f"Found columns: {list(df.columns)}. Add aliases in racer/data.py.")
    return df.rename(columns=rename)[[c for c in aliases if c in rename.values()]]


def load_funds(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _map_columns(df, FUND_ALIASES, ["scheme_name", "category", "risk_level"], "funds CSV")
    df["risk_level"] = pd.to_numeric(df["risk_level"], errors="coerce")
    for c in df.columns:
        if c not in ("scheme_name", "category"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["risk_level"]).reset_index(drop=True)
    df["tier"] = fund_tier(df["risk_level"])
    return df


def fund_tier(risk_level) -> pd.Series:
    """<=2 LOW, 3-4 MEDIUM, >=5 HIGH (as in the paper's D2 description)."""
    r = pd.Series(risk_level)
    return pd.Series(np.where(r <= 2, "LOW", np.where(r <= 4, "MEDIUM", "HIGH")), index=r.index)


def load_profiles(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = list(config.POLICY_WEIGHTS)
    df = _map_columns(df, PROFILE_ALIASES, req, "profiles CSV")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def load_regulatory(path: Path | str) -> list[dict]:
    """Regulatory corpus: a JSONL file with {"id","source","text"} per clause.

    Build it from text YOU obtained (EUR-Lex / ESMA / SEBI) with scripts/split_clauses.py.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append({"id": d["id"], "source": d.get("source", ""), "text": d["text"]})
    return out


# --------------------------------------------------------------------------- synthetic (SMOKE ONLY)
SYNTHETIC = True  # marker; anything derived from the functions below is NOT a result


def synth_funds(n=300, seed=config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["Debt", "Hybrid", "Equity"], size=n, p=[0.3, 0.25, 0.45])
    base = {"Debt": 1.5, "Hybrid": 3.5, "Equity": 5.0}
    risk = np.clip(np.round([base[c] + rng.normal(0, 0.8) for c in cat]), 1, 6)
    df = pd.DataFrame({
        "scheme_name": [f"Synthetic Fund {i}" for i in range(n)],
        "category": cat, "risk_level": risk,
        "rating": rng.integers(1, 6, n).astype(float),
        "expense_ratio": np.abs(rng.normal(1.0 + 0.2 * risk, 0.3)),
        "sd": np.abs(rng.normal(2 + 2 * risk, 1.0)),
        "sharpe": rng.normal(0.8, 0.4, n),
    })
    df["tier"] = fund_tier(df["risk_level"])
    return df


def synth_profiles(n=500, seed=config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "income": rng.lognormal(10.5, 0.5, n),
        "credit_score": rng.normal(650, 80, n).clip(300, 850),
        "spending_score": rng.uniform(0, 100, n),
        "transaction_count": rng.poisson(40, n).astype(float),
        "savings_ratio": rng.beta(2, 5, n),
        "loan_default_risk": rng.uniform(0, 1, n),
    })
