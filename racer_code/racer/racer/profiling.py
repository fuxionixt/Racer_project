"""User risk profiling: policy score phi (transparent formula) + Random-Forest imitation.

IMPORTANT (honest framing): the tier labels come from the policy formula, so a classifier
trained on them measures *how well it reproduces the policy*, not real-world risk
tolerance.  `fidelity` is therefore reported separately from any claim about users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config

POLICY_COLS = list(config.POLICY_WEIGHTS)


class PolicyScorer:
    """phi = sum_k w_k * x~_k  with x~ = min-max on TRAIN (clipped to [0,1]); reversed features flipped."""

    def fit(self, df: pd.DataFrame):
        self.lo = df[POLICY_COLS].min()
        self.hi = df[POLICY_COLS].max()
        self.t33, self.t67 = np.quantile(self.phi(df), [1 / 3, 2 / 3])
        return self

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = (self.hi - self.lo).replace(0, 1.0)
        x = ((df[POLICY_COLS] - self.lo) / rng).clip(0, 1)
        for c in config.POLICY_REVERSED:
            x[c] = 1.0 - x[c]
        return x

    def phi(self, df: pd.DataFrame) -> np.ndarray:
        x = self.normalise(df)
        w = np.array([config.POLICY_WEIGHTS[c] for c in POLICY_COLS])
        return (x.values * w).sum(axis=1)  # NaN if any policy feature missing

    def tier(self, df: pd.DataFrame) -> np.ndarray:
        p = self.phi(df)
        out = np.where(p <= self.t33, "LOW", np.where(p <= self.t67, "MEDIUM", "HIGH")).astype(object)
        out[np.isnan(p)] = None
        return out


class RFProfiler:
    """Random Forest trained to imitate the policy tiers from (possibly more) features."""

    def __init__(self, features=None, params=None):
        self.features = features
        self.params = params or config.RF_PARAMS

    def fit(self, df: pd.DataFrame, tiers):
        self.features = self.features or list(df.columns)
        self.med = df[self.features].median()
        self.rf = RandomForestClassifier(**self.params).fit(self._x(df), np.asarray(tiers, dtype=str))
        return self

    def _x(self, df):
        return df[self.features].fillna(self.med).values

    def predict(self, df):
        return self.rf.predict(self._x(df))


class RiskProfiler:
    """Hybrid: RF prediction when available, policy tier as fallback (e.g. RF unavailable)."""

    def __init__(self, mode: str = "policy", rf_features=None):
        assert mode in ("policy", "rf")
        self.mode, self.rf_features = mode, rf_features

    def fit(self, train: pd.DataFrame):
        self.policy = PolicyScorer().fit(train)
        if self.mode == "rf":
            self.rf = RFProfiler(self.rf_features).fit(train, self.policy.tier(train))
        return self

    def predict(self, df: pd.DataFrame):
        return self.rf.predict(df) if self.mode == "rf" else self.policy.tier(df)
