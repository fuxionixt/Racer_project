"""Exp 2 -- product retrieval + suitability on the funds catalogue (needs Mutual Funds India CSV).
Reports OBJECTIVE outcomes from the official risk level; rule conformance is shown separately and
is satisfied by construction when the CCE is on."""
from common import *
from racer.metrics import bootstrap_ci, paired_bootstrap
from racer.pipeline import Racer, Options
from racer.profiling import PolicyScorer
from racer.rules import RuleTable

a = parser(__doc__).parse_args()
funds, prof = get_funds(a), get_profiles(a)
ftr, fte = split(funds, funds["tier"])
ptr, pte = split(prof.dropna(subset=list(config.POLICY_WEIGHTS)), None)
ps = PolicyScorer().fit(ptr); users = ps.tier(pte)
rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
rt = RuleTable().fit(ftr)

VARIANTS = {"baseline (no boost, no CCE)": Options(False, False, False),
            "+ profile boost": Options(True, False, False),
            "+ CCE only": Options(False, True, False),
            "RACER (boost + CCE)": Options(True, True, False)}
res, per_user = {}, {}
for name, opt in VARIANTS.items():
    rac = Racer(ftr, fte, generator=None, options=opt)
    rows = {k: [] for k in ("overshoot", "tier_match", "sharpe", "covered", "rule_conform")}
    for t in users:
        vals = {k: [] for k in rows}
        for q in QUERIES:
            p = rac.retrieve_products(q, t)
            vals["covered"].append(float(len(p) > 0))
            if len(p):
                vals["overshoot"].append(np.mean([rank[x] > rank[t] for x in p["tier"]]))
                vals["tier_match"].append(np.mean([x == t for x in p["tier"]]))
                vals["sharpe"].append(p["sharpe"].mean() if "sharpe" in p else np.nan)
                vals["rule_conform"].append(np.mean([rt.passes(r, t)[0] for _, r in p.iterrows()]))
        for k in rows: rows[k].append(np.nanmean(vals[k]) if vals[k] else np.nan)
    per_user[name] = rows
    res[name] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(np.array(v)[~np.isnan(v)]))) for k, v in rows.items()}
    print(f"{name:30s}", {k: round(res[name][k]["mean"], 3) for k in rows})
base = per_user["baseline (no boost, no CCE)"]; full = per_user["RACER (boost + CCE)"]
ok = ~(np.isnan(base["overshoot"]) | np.isnan(full["overshoot"]))
paired = {k: paired_bootstrap(np.array(full[k])[ok], np.array(base[k])[ok]) for k in ("overshoot", "tier_match")}
save(a, "exp2_funds.json", {"n_users": len(users), "n_funds_test": len(fte), "n_queries": len(QUERIES),
                            "variants": res, "paired_racer_vs_baseline": paired})
