"""Build the two human-annotation kits for the paper's external-validation steps.

  python scripts/make_annotation_kits.py        # writes annotation/ (sheets) and annotation/key/ (do NOT share)

1. cce_expert_sheet.csv   100 fund-client pairs for annotators with a finance background to label Suitable /
                          Unsuitable, blind to the CCE. Pairs come from the 166 test funds with complete checked
                          attributes x 3 client tiers, stratified by tier and balanced between CCE pass and fail.
2. clause_relevance_sheet.csv  the 322 passages of the full SEBI corpus, each graded once 0/1/2 for relevance to
                          judging whether a fund recommendation is suitable (every investor query asks for one).
Score filled sheets with scripts/score_annotations.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer import config  # noqa: E402
from racer.data import load_funds, load_regulatory  # noqa: E402
from racer.rules import CHECKS, RuleTable  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

OUT = config.ROOT / "annotation"
KEY = OUT / "key"
RISKOMETER = {1: "Low", 2: "Low to Moderate", 3: "Moderate", 4: "Moderately High", 5: "High", 6: "Very High"}
CAPACITY = {"LOW": "LOW: weakest third of clients by income, savings and credit; cannot bear losses",
            "MEDIUM": "MEDIUM: middle third; can bear moderate fluctuations",
            "HIGH": "HIGH: strongest third; can bear high volatility and drawdowns"}
N_PAIRS = 100

funds = load_funds(config.DATA_DIR / "funds" / "funds.csv")
train, test = train_test_split(funds, test_size=0.25, stratify=funds["tier"], random_state=config.SEED)
rules = RuleTable().fit(train)
cols = ["risk_level", "expense_ratio", "rating", "category", "sd", "sharpe"]
test = test.dropna(subset=cols).reset_index(drop=True)  # the 166 funds evaluated in the paper

rows = []
for i, f in test.iterrows():
    for tier in config.TIER_NAMES:
        ok, checks = rules.passes(f, tier)
        rows.append({"fund": i, "tier": tier, "cce": "Suitable" if ok else "Unsuitable",
                     "failed": ";".join(c for c in CHECKS if not checks[c])})
pairs = pd.DataFrame(rows)

rng = np.random.default_rng(config.SEED)
per_tier = {t: n for t, n in zip(config.TIER_NAMES, (34, 33, 33))}
picked = []
for tier, n in per_tier.items():
    pool = pairs[pairs.tier == tier]
    k_pass = min(n // 2, (pool.cce == "Suitable").sum())
    k_fail = n - k_pass if (pool.cce == "Unsuitable").sum() >= n - k_pass else (pool.cce == "Unsuitable").sum()
    k_pass = n - k_fail
    for dec, k in (("Suitable", k_pass), ("Unsuitable", k_fail)):
        sub = pool[pool.cce == dec]
        picked.append(sub.iloc[rng.choice(len(sub), size=k, replace=False)])
sample = pd.concat(picked).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
sample["pair_id"] = [f"P{i + 1:03d}" for i in range(len(sample))]

sheet = pd.DataFrame({
    "pair_id": sample.pair_id,
    "client_risk_capacity": sample.tier.map(CAPACITY),
    "fund_name": sample.fund.map(test.scheme_name),
    "category": sample.fund.map(test.category),
    "riskometer_level": sample.fund.map(lambda i: f"{int(test.risk_level[i])} ({RISKOMETER[int(test.risk_level[i])]})"),
    "expense_ratio_pct": sample.fund.map(test.expense_ratio),
    "rating_0_5": sample.fund.map(test.rating),
    "std_dev_pct": sample.fund.map(test.sd),
    "sharpe": sample.fund.map(test.sharpe),
    "return_3yr_pct": sample.fund.map(test.return_3yr),
    "label (Suitable/Unsuitable)": "", "confidence (1-3)": "", "comment": "",
})
OUT.mkdir(exist_ok=True); KEY.mkdir(exist_ok=True)
sheet.to_csv(OUT / "cce_expert_sheet.csv", index=False)
sample[["pair_id", "tier", "cce", "failed"]].assign(fund_name=sheet.fund_name).to_csv(KEY / "cce_key.csv", index=False)

clauses = load_regulatory(config.DATA_DIR / "regulatory" / "sebi_full.jsonl")
pd.DataFrame({"passage_id": [c["id"] for c in clauses], "source": [c["source"] for c in clauses],
              "text": [c["text"] for c in clauses], "relevance (0/1/2)": "", "comment": ""}
             ).to_csv(OUT / "clause_relevance_sheet.csv", index=False)

med = train[cols[1:3] + ["sd", "sharpe"]].median().round(2).to_dict()
print(f"pairs: {len(sheet)} | by tier {sample.tier.value_counts().to_dict()} | CCE {sample.cce.value_counts().to_dict()}")
print(f"passages: {len(clauses)} | training-catalogue medians for the instructions: {med}")
