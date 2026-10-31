"""Compliance Constraint Engine: M=6 product-suitability checks, threshold theta, hard checks.

The numeric bounds are ILLUSTRATIVE and author-chosen (see config.RULE_BOUNDS); they are not
transcriptions of any regulation.  Missing product attributes FAIL their check (conservative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CHECKS = ("risk_level", "expense_ratio", "rating", "category", "volatility", "sharpe")


class RuleTable:
    def __init__(self, bounds=None, theta=config.CCE_THETA, hard=config.CCE_HARD_CHECKS):
        self.bounds, self.theta, self.hard = bounds or config.RULE_BOUNDS, theta, hard

    def fit(self, catalogue: pd.DataFrame):
        """Resolve quantile bounds on the TRAIN catalogue only."""
        self.cut = {}
        for tier, b in self.bounds.items():
            self.cut[tier] = {
                "expense": catalogue["expense_ratio"].quantile(b["exp_q"]) if "expense_ratio" in catalogue else np.nan,
                "sd": catalogue["sd"].quantile(b["sd_q"]) if "sd" in catalogue else np.nan,
                "sharpe": catalogue["sharpe"].quantile(b["sharpe_q"]) if "sharpe" in catalogue else np.nan,
            }
        return self

    def check(self, p: pd.Series, tier: str) -> dict:
        b, c = self.bounds[tier], self.cut[tier]

        def ok(cond):  # NaN comparisons are False -> fail
            return bool(cond)

        return {
            "risk_level": ok(p.get("risk_level", np.nan) <= b["risk"]),
            "expense_ratio": ok(p.get("expense_ratio", np.nan) <= c["expense"]),
            "rating": ok(p.get("rating", np.nan) >= b["rating"]),
            "category": ok(b["cats"] is None or p.get("category") in b["cats"]),
            "volatility": ok(p.get("sd", np.nan) <= c["sd"]),
            "sharpe": ok(p.get("sharpe", np.nan) >= c["sharpe"]),
        }

    def passes(self, p: pd.Series, tier: str) -> tuple[bool, dict]:
        r = self.check(p, tier)
        good = sum(r.values()) >= self.theta and all(r[h] for h in self.hard)
        return good, r

    def filter(self, products: pd.DataFrame, tier: str) -> pd.DataFrame:
        keep = [self.passes(row, tier)[0] for _, row in products.iterrows()]
        return products[np.array(keep, dtype=bool)]
