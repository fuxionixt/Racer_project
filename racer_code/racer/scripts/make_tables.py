"""Turn results/*.json into LaTeX table bodies (paper/tables/*.tex). Only real result files are used."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent
out = R / "paper" / "tables"; out.mkdir(parents=True, exist_ok=True)
def f(s, k):
    x = s[k]; h = max(x["hi"] - x["mean"], x["mean"] - x["lo"])
    return f"{x['mean']:.3f}{{\\scriptsize$\\pm${h:.3f}}}"
p = R / "results" / "exp1_finqa_test.json"
if p.exists():
    d = json.load(open(p))
    names = {"random": "Random", "tfidf": "TF-IDF", "bm25": "BM25", "hybrid_rrf": "TF-IDF+BM25 (RRF)"}
    for mode in d["modes"]:
        lines = []
        for m in ("random", "tfidf", "bm25", "hybrid_rrf"):
            s = d["modes"][mode]["summary"][m]
            lines.append(f"{names[m]} & {f(s,'recall@5')} & {f(s,'hit_all@5')} & {f(s,'mrr')} & {f(s,'ndcg@10')} \\\\")
        (out / f"exp1_{mode}.tex").write_text("\n".join(lines) + "\n")
    print("wrote exp1 tables")
for name in ("exp2_funds", "exp3_profiling"):
    if (R / "results" / f"{name}.json").exists(): print(f"{name}: results found -> add a table writer or fill by hand")
    else: print(f"{name}: NOT RUN (data missing) -- paper must say so")

sys.path.insert(0, str(R))
from racer import config
rows = []
for t, b in config.RULE_BOUNDS.items():
    cats = "any" if b["cats"] is None else "/".join(b["cats"])
    rows.append(f"{t} & {b['risk']} & P{int(b['exp_q']*100)} & {b['rating']} & {cats} & P{int(b['sd_q']*100)} & P{int(b['sharpe_q']*100)} \\\\")
(out / "bounds.tex").write_text("\n".join(rows) + "\n"); print("wrote bounds")
