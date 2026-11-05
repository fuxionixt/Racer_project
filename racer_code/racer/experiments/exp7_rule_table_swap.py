"""Exp 7 -- the CCE is agnostic to its rule table: swap in a MiFID-flavoured table on the same catalogue.
The SEBI table's mandatory check c1 uses the SEBI riskometer level. The swapped table replaces it with the UCITS
synthetic risk and reward indicator (SRRI, classes 1-7 by annualized volatility; ESMA/CESR bands 0.5/2/5/10/15/25%),
computed from the catalogue's standard deviation, with LOW <= SRRI 3, MEDIUM <= 5, HIGH <= 7 (an illustrative
mapping). Checks c2-c6, theta and the engine code are unchanged. Reports compliance on the 166 complete test funds,
decision agreement with the SEBI table over all fund x tier pairs, and filtering coverage over the 192 cases."""
from common import *
from racer.metrics import bootstrap_ci
from racer.pipeline import Racer, Options
from racer.rules import RuleTable

SRRI_EDGES = (0.5, 2.0, 5.0, 10.0, 15.0, 25.0)
SRRI_MAX = {"LOW": 3, "MEDIUM": 5, "HIGH": 7}


def srri(sd):
    return 1 + sum(sd >= e for e in SRRI_EDGES)


class SRRIRuleTable(RuleTable):
    """Same checks and engine; c1 compares the fund's SRRI class instead of its SEBI riskometer level."""

    def check(self, p, tier):
        r = super().check(p, tier)
        sd = p.get("sd", np.nan)
        r["risk_level"] = bool(sd == sd and srri(sd) <= SRRI_MAX[tier])  # NaN fails, as for every check
        return r


a = parser(__doc__).parse_args()
funds = get_funds(a); ftr, fte = split(funds, funds["tier"])
tables = {"SEBI (riskometer)": RuleTable().fit(ftr), "MiFID-flavoured (SRRI)": SRRIRuleTable().fit(ftr)}
cols = ["risk_level", "expense_ratio", "rating", "category", "sd", "sharpe"]
complete = fte.dropna(subset=cols)

res = {}
for name, rt in tables.items():
    per_tier = {t: float(np.mean([rt.passes(r, t)[0] for _, r in complete[complete.tier == t].iterrows()]))
                for t in config.TIER_NAMES}
    overall = float(np.mean([rt.passes(r, r["tier"])[0] for _, r in complete.iterrows()]))
    rac = Racer(ftr, fte, options=Options()); rac.rules = rt  # swap the table into the deployed pipeline
    delivered, empty = [], []
    for t in config.TIER_NAMES:
        for q in EXTENDED_QUERIES:
            p = rac.retrieve_products(q, t); delivered.append(float(len(p))); empty.append(float(len(p) == 0))
    res[name] = {"compliance_own_tier": overall, "compliance_by_tier": per_tier,
                 "delivered": dict(zip(("mean", "lo", "hi"), bootstrap_ci(delivered))),
                 "empty": dict(zip(("mean", "lo", "hi"), bootstrap_ci(empty)))}
    print(f"{name:24s} compliance {overall:.3f} {per_tier} delivered {np.mean(delivered):.2f} empty {np.mean(empty):.3f}")

sebi, srri_t = tables.values()
pairs = [(sebi.passes(r, t)[0], srri_t.passes(r, t)[0]) for _, r in complete.iterrows() for t in config.TIER_NAMES]
agree = float(np.mean([x == y for x, y in pairs]))
srri_dist = complete["sd"].map(srri).value_counts().sort_index().to_dict()
print(f"decision agreement over {len(pairs)} fund x tier pairs: {agree:.3f}; SRRI classes of test funds: {srri_dist}")
save(a, "exp7_rule_table_swap.json", {"n_funds": len(complete), "n_pairs": len(pairs), "tables": res,
                                      "decision_agreement": agree, "srri_distribution": srri_dist})
