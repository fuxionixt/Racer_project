"""Exp 6 -- regulatory-clause retrieval over the SEBI IA Regulations corpus (D4): BM25 vs TF-IDF vs dense.
There are no human relevance labels for these clauses, so for every tier x investor query we record
which clauses reach the top 3 and report how often a suitability provision (Reg. 17) is among them,
how often Reg. 17(a) is, and how often the record-keeping clause (Reg. 19(1)) takes a slot."""
from collections import Counter
from common import *
from racer.metrics import bootstrap_ci, paired_bootstrap
from racer.pipeline import Racer

a = parser(__doc__).parse_args()
if not Path(a.regulatory).exists(): sys.exit(f"Missing {a.regulatory}: build the SEBI clause corpus first (see README).")
funds = get_funds(a); ftr, fte = split(funds, funds["tier"])
clauses = load_regulatory(a.regulatory)
cells = [(t, q) for t in config.TIER_NAMES for q in EXTENDED_QUERIES]

rows, counts = {}, {}
for name in ("bm25", "tfidf", "dense"):
    rac = Racer(ftr, fte, clauses, clause_retriever=name)
    r = {"reg17_in_top3": [], "reg17a_in_top3": [], "reg19_in_top3": []}
    c = Counter()
    for t, q in cells:
        ids = [x["id"] for x in rac.recommend(t, q).clauses]
        c.update(ids)
        r["reg17_in_top3"].append(float(any(i.startswith("SEBI-IA-17") for i in ids)))
        r["reg17a_in_top3"].append(float("SEBI-IA-17(a)" in ids))
        r["reg19_in_top3"].append(float("SEBI-IA-19(1)" in ids))
    rows[name], counts[name] = r, dict(c.most_common())

res = {n: {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v))) for k, v in r.items()} for n, r in rows.items()}
for n in res: print(f"{n:6s}", {k: round(res[n][k]["mean"], 3) for k in res[n]}, "distinct:", len(counts[n]))
paired = {k: paired_bootstrap(rows["dense"][k], rows["bm25"][k]) for k in rows["bm25"]}
print("dense - bm25:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in paired.items()})
save(a, "exp6_clause_retrieval.json", {"n_cases": len(cells), "n_clauses": len(clauses), "variants": res,
                                       "clause_counts": counts, "paired_dense_minus_bm25": paired})
