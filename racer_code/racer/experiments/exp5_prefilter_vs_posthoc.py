"""Exp 5 -- where the rules are applied, and how deep the filtered candidate pool is.
For every user tier x investor query, compares the products delivered by
  no CCE:               top-5 retrieved products, unfiltered;
  post-hoc filter @K:   top-K retrieved products, non-compliant ones removed, then top-5 (K = 5, 10);
  pre-filter (RACER):   top-20 retrieved, CCE applied, then top-5 (identical to post-hoc @20);
  pre-filter, no boost: as RACER but without the profile tier boost (ablation of Eq. 4).
No LLM is involved: this isolates the placement and depth of the rule filter."""
from common import *
from racer.metrics import bootstrap_ci, paired_bootstrap
from racer.pipeline import Racer, Options

a = parser(__doc__).parse_args()
funds = get_funds(a); ftr, fte = split(funds, funds["tier"])
rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
nocce = Racer(ftr, fte, options=Options(use_cce=False, top_k=config.CANDIDATE_K))
pre = Racer(ftr, fte, options=Options(use_cce=True))
noboost = Racer(ftr, fte, options=Options(use_cce=True, use_profile_boost=False))
rules = nocce.rules
cells = [(t, q) for t in config.TIER_NAMES for q in EXTENDED_QUERIES]
METRICS = ("delivered", "empty", "violation", "overshoot", "tier_match")

def outcome(p, t):
    ok = [rules.passes(r, t)[0] for _, r in p.iterrows()]
    frac = lambda xs: float(np.mean(xs)) if len(p) else 0.0
    return {"delivered": float(len(p)), "empty": float(len(p) == 0), "violation": frac([not x for x in ok]),
            "overshoot": frac([rank[x] > rank[t] for x in p["tier"]]), "tier_match": frac([x == t for x in p["tier"]])}

def posthoc(top, t, k):
    return rules.filter(top.head(k), t).head(config.TOP_K)

VARIANTS = ("no CCE", "post-hoc filter @5", "post-hoc filter @10", "pre-filter (RACER)", "pre-filter, no tier boost")
rows = {v: {k: [] for k in METRICS} for v in VARIANTS}
for t, q in cells:
    top = nocce.retrieve_products(q, t)  # boosted top-20, unfiltered
    pre_p = pre.retrieve_products(q, t)
    assert list(pre_p.index) == list(posthoc(top, t, config.CANDIDATE_K).index)  # pre-filter == post-hoc @20
    outs = (top.head(config.TOP_K), posthoc(top, t, 5), posthoc(top, t, 10), pre_p, noboost.retrieve_products(q, t))
    for name, p in zip(VARIANTS, outs):
        for k, v in outcome(p, t).items(): rows[name][k].append(v)

res = {name: {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v))) for k, v in r.items()} for name, r in rows.items()}
for name in res: print(f"{name:26s}", {k: round(res[name][k]["mean"], 3) for k in res[name]})
paired = {k: paired_bootstrap(rows["pre-filter (RACER)"][k], rows["post-hoc filter @5"][k]) for k in ("delivered", "empty")}
boost = {k: paired_bootstrap(rows["pre-filter (RACER)"][k], rows["pre-filter, no tier boost"][k])
         for k in ("delivered", "empty", "tier_match")}
print("pre - post@5:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in paired.items()})
print("boost - no boost:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in boost.items()})
save(a, "exp5_prefilter_vs_posthoc.json", {"n_cases": len(cells), "n_funds_test": len(fte), "variants": res,
                                           "paired_pre_minus_post": paired, "paired_boost_minus_no_boost": boost})
