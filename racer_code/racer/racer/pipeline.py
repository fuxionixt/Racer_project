"""End-to-end RACER pipeline with ablation switches."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .generator import ExtractiveGenerator, build_prompt
from .retrieval import BM25Retriever, TfidfRetriever
from .rules import RuleTable
from .validator import validate


def fund_doc(r: pd.Series) -> str:
    return (f"{r['scheme_name']} {r.get('category','')} fund risk level {r.get('risk_level','')} "
            f"rating {r.get('rating','')} expense ratio {r.get('expense_ratio','')}").strip()


@dataclass
class Options:
    use_profile_boost: bool = True
    use_cce: bool = True
    use_validator: bool = True
    max_regen: int = 2
    top_k: int = config.TOP_K
    candidate_k: int = config.CANDIDATE_K


@dataclass
class Result:
    tier: str
    products: list
    clauses: list
    text: str
    report: object
    attempts: int
    trace: list = field(default_factory=list)


class Racer:
    def __init__(self, funds_train: pd.DataFrame, funds_all: pd.DataFrame, clauses: list[dict] | None = None,
                 generator=None, retriever: str = "bm25", options: Options | None = None):
        self.o = options or Options()
        self.funds = funds_all.reset_index(drop=True)
        self.rules = RuleTable().fit(funds_train)
        docs = [fund_doc(r) for _, r in self.funds.iterrows()]
        R = BM25Retriever if retriever == "bm25" else TfidfRetriever
        self.fr = R().fit([fund_doc(r) for _, r in funds_train.iterrows()]).index(docs)
        self.clauses = clauses or []
        if self.clauses:
            self.cr = R().fit([c["text"] for c in self.clauses]).index([c["text"] for c in self.clauses])
        self.gen = generator or ExtractiveGenerator()
        self.names = list(self.funds["scheme_name"])

    def retrieve_products(self, query: str, tier: str):
        s = self.fr.scores(query)
        if self.o.use_profile_boost:
            same = (self.funds["tier"].values == tier)
            s = s * np.where(same, config.PROFILE_BOOST, 1.0)
        order = np.argsort(-s, kind="stable")[: self.o.candidate_k]
        cand = self.funds.iloc[order]
        if self.o.use_cce:
            cand = self.rules.filter(cand, tier)
        return cand.head(self.o.top_k)

    def recommend(self, tier: str, query: str) -> Result:
        prods = self.retrieve_products(query, tier)
        plist = [{"name": r["scheme_name"], **{k: (None if pd.isna(r.get(k)) else r.get(k))
                  for k in ("category", "risk_level", "expense_ratio", "rating", "sharpe")}}
                 for _, r in prods.iterrows()]
        clauses = []
        if self.clauses:
            s = self.cr.scores(query + " " + tier.lower() + " risk suitability")
            clauses = [self.clauses[i] for i in np.argsort(-s, kind="stable")[:3]]
        ctx = {"tier": tier, "products": plist, "clauses": clauses}
        feedback, trace = "", []
        for attempt in range(1, self.o.max_regen + 2):
            text = self.gen.generate(build_prompt(tier, plist, clauses, feedback), ctx)
            rep = validate(text, ctx, self.names, self.rules, self.funds)
            trace.append((attempt, rep.ok, rep.feedback))
            if rep.ok or not self.o.use_validator:
                break
            feedback = rep.feedback
        return Result(tier, plist, clauses, text, rep, attempt, trace)
