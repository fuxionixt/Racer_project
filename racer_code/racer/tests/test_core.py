import numpy as np, pandas as pd, pytest
from racer import config
from racer.metrics import rank_of_gold, per_query_metrics, bootstrap_ci, paired_bootstrap
from racer.retrieval import TfidfRetriever, BM25Retriever, RRFHybrid
from racer.profiling import PolicyScorer, RiskProfiler
from racer.rules import RuleTable
from racer.validator import validate
from racer.generator import ScriptedBackend, build_prompt
from racer.pipeline import Racer, Options
from racer.data import synth_funds, synth_profiles, row_to_text, finqa_gold


def test_rank_and_metrics():
    s = np.array([0.1, 0.9, 0.5, 0.3])
    assert list(rank_of_gold(s, np.array([1, 2]))) == [1, 2]
    m = per_query_metrics(np.array([1, 2]), ks=(1, 3))
    assert m["recall@1"] == 0.5 and m["hit_all@3"] == 1.0 and m["mrr"] == 1.0 and abs(m["ndcg@10"] - 1) < 1e-9
    m = per_query_metrics(np.array([4, 20]), ks=(3,))
    assert m["recall@3"] == 0 and m["mrr"] == 0.25


def test_bootstrap():
    m, lo, hi = bootstrap_ci(np.r_[np.ones(50), np.zeros(50)])
    assert lo < 0.5 < hi
    r = paired_bootstrap(np.ones(100), np.zeros(100))
    assert r["lo"] > 0 and r["p"] < 0.01


DOCS = ["the expense ratio of the fund is low", "revenue grew by ten percent in 2019", "net income fell sharply",
        "the board approved a dividend", "debt fund with low risk", "equity growth fund high volatility"]


@pytest.mark.parametrize("R", [TfidfRetriever, BM25Retriever])
def test_retrievers_rank_matching_doc_first(R):
    r = R().fit(DOCS).index(DOCS)
    assert int(np.argmax(r.scores("revenue grew in 2019"))) == 1
    sub = r.scores("revenue grew in 2019", rows=np.array([0, 1, 2]))
    assert int(np.argmax(sub)) == 1 and len(sub) == 3


def test_rrf():
    a, b = TfidfRetriever().fit(DOCS).index(DOCS), BM25Retriever().fit(DOCS).index(DOCS)
    assert int(np.argmax(RRFHybrid([a, b]).scores("dividend board"))) == 3


def test_finqa_helpers():
    assert "the revenue of 2019 is 5" in row_to_text(["", "2019"], ["revenue", "5"])
    ex = {"qa": {"gold_inds": {"text_1": "x", "table_0": "y", "table_99": "z"}}}
    assert finqa_gold(ex, 3, 2) == [1, 3]   # table_0 -> 3+0 ; table_99 out of range dropped


def test_policy_direction_and_tiers():
    df = synth_profiles(600)
    ps = PolicyScorer().fit(df)
    hi = df.iloc[[0]].copy(); lo = df.iloc[[0]].copy()
    hi[["income", "credit_score", "savings_ratio"]] = df[["income", "credit_score", "savings_ratio"]].max().values
    hi["loan_default_risk"] = df["loan_default_risk"].min()
    lo[["income", "credit_score", "savings_ratio"]] = df[["income", "credit_score", "savings_ratio"]].min().values
    lo["loan_default_risk"] = df["loan_default_risk"].max()
    assert ps.phi(hi)[0] > ps.phi(lo)[0]                    # higher default risk => LOWER capacity
    assert abs(ps.phi(hi)[0] - 1.0) < 1e-9 and abs(ps.phi(lo)[0]) < 1e-9
    t = pd.Series(ps.tier(df)).value_counts()
    assert set(t.index) == {"LOW", "MEDIUM", "HIGH"} and t.min() > 150   # terciles


def test_policy_missing_is_none():
    df = synth_profiles(100); ps = PolicyScorer().fit(df)
    d2 = df.iloc[:2].copy(); d2.loc[d2.index[0], "income"] = np.nan
    assert ps.tier(d2)[0] is None and ps.tier(d2)[1] is not None


def test_rf_profiler_no_leakage_of_fit_stats():
    df = synth_profiles(600); tr, te = df.iloc[:450], df.iloc[450:]
    p = RiskProfiler("rf").fit(tr)
    acc = np.mean(p.predict(te) == p.policy.tier(te))
    assert acc > 0.6


def _funds():
    return synth_funds(300)


def test_cce_hard_risk_check_and_theta():
    f = _funds(); rt = RuleTable().fit(f)
    low = f[f.risk_level >= 5].iloc[0]
    ok, r = rt.passes(low, "LOW"); assert not ok and not r["risk_level"]
    kept = rt.filter(f, "LOW"); assert (kept.risk_level <= 2).all()
    nan = low.copy(); nan["risk_level"] = np.nan
    assert not rt.passes(nan, "HIGH")[0]                      # missing hard attribute fails


def _ctx():
    return {"tier": "LOW", "products": [{"name": "Alpha Debt Fund", "risk_level": 2.0, "expense_ratio": 0.5}],
            "clauses": [{"text": "Firms must assess suitability at least annually."}]}


def test_validator_accepts_grounded():
    r = validate("Alpha Debt Fund [P1] has risk level 2 and expense ratio 0.5. See [R1].", _ctx(), ["Alpha Debt Fund"])
    assert r.ok and r.metrics["numeric_grounding"] == 1.0


def test_validator_catches_each_hallucination_type():
    ctx = _ctx(); names = ["Alpha Debt Fund", "Beta Equity Fund"]
    assert "retrieved set" in validate("Buy Beta Equity Fund [P1].", ctx, names).feedback
    assert "unknown product" in validate("Try the Gamma Super Growth Fund [P1].", ctx, names).feedback
    assert "invalid citations" in validate("Alpha Debt Fund [P3] per [R7].", ctx, names).feedback
    r = validate("Alpha Debt Fund [P1] returned 18.5% last year.", ctx, names)
    assert not r.ok and r.metrics["numeric_grounding"] < 1 and r.metrics["has_hallucination"]


def test_validator_ignores_list_markers_and_grounds_clause_sources():
    ctx = _ctx(); ctx["clauses"][0]["source"] = "SEBI IA Regulations, 2013, Reg. 17(d)"
    text = "1. Alpha Debt Fund [P1] has risk level 2.\n2) Per [R1] (Reg. 17(d)), suitability applies."
    r = validate(text, ctx, ["Alpha Debt Fund"])
    assert r.ok and r.metrics["numeric_grounding"] == 1.0
    assert not validate("1. Alpha Debt Fund [P1] costs 3 more.", ctx, ["Alpha Debt Fund"]).ok  # inline numbers still checked
    table = "| Rank | Product |\n|---|---|\n| 1 | Alpha Debt Fund [P1] (risk 2) |"
    assert validate(table, ctx, ["Alpha Debt Fund"]).ok


def test_validator_matches_names_across_dash_and_ampersand_spacing():
    ctx = {"tier": "LOW", "products": [{"name": "Quant Tax Plan- Direct Growth"}, {"name": "PGIM India Banking&PSU Debt Fund"}],
           "clauses": []}
    names = [p["name"] for p in ctx["products"]] + ["Other Fund"]
    r = validate("Quant Tax Plan – Direct Growth [P1] and PGIM India Banking & PSU Debt Fund [P2].", ctx, names)
    assert r.ok and r.metrics["product_grounding"] == 1.0


def test_validator_keeps_negative_numbers_intact():
    ctx = {"tier": "LOW", "products": [{"name": "Tata Overnight Fund", "sharpe": -0.3}], "clauses": []}
    assert validate("Tata Overnight Fund [P1] has a negative Sharpe (-0.3).", ctx, ["Tata Overnight Fund"]).ok
    assert not validate("Tata Overnight Fund [P1] has a Sharpe of 0.3.", ctx, ["Tata Overnight Fund"]).ok


def test_regeneration_loop_repairs_hallucination():
    f = _funds(); f.loc[0, ["scheme_name", "risk_level", "category", "expense_ratio", "rating", "sd", "sharpe"]] = [
        "Alpha Debt Fund", 1.0, "Debt", 0.1, 5.0, 0.1, 5.0]
    good = "Alpha Debt Fund [P1] is suitable."
    gen = ScriptedBackend(["Buy Ghost Magic Fund [P1] for 40% returns.", good])
    rac = Racer(f, f, generator=gen, options=Options(max_regen=2, use_cce=False))
    res = rac.recommend("LOW", "alpha debt fund")
    assert res.products[0]["name"] == "Alpha Debt Fund"
    assert res.attempts == 2 and res.report.ok and res.text == good
    # without the validator the bad text is returned as-is
    gen2 = ScriptedBackend(["Buy Ghost Magic Fund [P1] for 40% returns."])
    r2 = Racer(f, f, generator=gen2, options=Options(use_validator=False, use_cce=False)).recommend("LOW", "alpha")
    assert r2.attempts == 1 and "Ghost" in r2.text


def test_pipeline_cce_never_recommends_over_risk():
    f = _funds()
    rac = Racer(f.iloc[:200], f, options=Options())
    for tier, cap in (("LOW", 2), ("MEDIUM", 4)):
        res = rac.recommend(tier, "growth income fund")
        assert all(p["risk_level"] <= cap for p in res.products)


def test_prompt_contains_context_and_feedback():
    p = build_prompt("LOW", _ctx()["products"], _ctx()["clauses"], feedback="bad number")
    assert "[P1] Alpha Debt Fund" in p and "[R1]" in p and "bad number" in p
