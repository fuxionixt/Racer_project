"""Exp 1 -- supporting-fact retrieval on FinQA (REAL data, independent gold labels).

Closed-pool: rank the facts of the question's own filing.  Open-corpus: rank every fact
in the split.  Retrievers are fitted (IDF / length statistics only) on the TRAIN split.
Usage: python experiments/exp1_finqa_retrieval.py [--split test] [--limit N]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from racer import config
from racer.data import load_finqa
from racer.metrics import per_query_metrics, rank_of_gold, bootstrap_ci, paired_bootstrap
from racer.retrieval import TfidfRetriever, BM25Retriever, RRFHybrid

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--open_corpus", type=int, default=1)
args = ap.parse_args()

t0 = time.time()
train = load_finqa("train")
ev = load_finqa(args.split)
if args.limit:
    ev = ev[: args.limit]
print(f"train={len(train)} {args.split}={len(ev)}")

fit_texts = [f for ex in train for f in ex["facts"]] + [ex["question"] for ex in train]
tfidf = TfidfRetriever().fit(fit_texts)
bm25 = BM25Retriever().fit(fit_texts)

all_facts = [f for ex in ev for f in ex["facts"]]
offsets = np.cumsum([0] + [len(ex["facts"]) for ex in ev])
tfidf.index(all_facts); bm25.index(all_facts)
methods = {"tfidf": tfidf, "bm25": bm25, "hybrid_rrf": RRFHybrid([tfidf, bm25])}
print(f"indexed {len(all_facts)} facts in {time.time()-t0:.1f}s")

KS = (3, 5, 10)
METRICS = ["mrr", "ndcg@10"] + [f"{m}@{k}" for k in KS for m in ("recall", "hit_all")]
rng = np.random.default_rng(config.SEED)

def run(mode):
    per = {m: {k: [] for k in METRICS} for m in list(methods) + ["random"]}
    for qi, ex in enumerate(ev):
        lo, hi = offsets[qi], offsets[qi + 1]
        rows = np.arange(lo, hi) if mode == "closed" else None
        gold = np.array(ex["gold"]) + (0 if mode == "closed" else lo)
        for name, r in methods.items():
            gr = rank_of_gold(r.scores(ex["question"], rows), gold)
            for k, v in per_query_metrics(gr, KS).items():
                per[name][k].append(v)
        n = (hi - lo) if mode == "closed" else len(all_facts)
        if mode == "closed":
            gr = rank_of_gold(rng.random(n), np.array(ex["gold"]))
        else:  # random rank of gold in a huge pool: sample ranks uniformly without replacement
            gr = np.sort(rng.choice(n, size=len(gold), replace=False) + 1)
        for k, v in per_query_metrics(gr, KS).items():
            per["random"][k].append(v)
        if (qi + 1) % 200 == 0:
            print(f"  [{mode}] {qi+1}/{len(ev)} {time.time()-t0:.0f}s", flush=True)
    return per

out = {"split": args.split, "n_questions": len(ev), "n_facts": len(all_facts),
       "mean_pool_size": float(np.mean([len(e["facts"]) for e in ev])),
       "mean_gold": float(np.mean([len(e["gold"]) for e in ev])), "modes": {}}
for mode in (["closed"] + (["open"] if args.open_corpus else [])):
    per = run(mode)
    res = {"summary": {}, "paired_vs_tfidf": {}}
    for name, d in per.items():
        res["summary"][name] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v))) for k, v in d.items()}
    for name in ("bm25", "hybrid_rrf"):
        res["paired_vs_tfidf"][name] = {k: paired_bootstrap(per[name][k], per["tfidf"][k])
                                        for k in ("recall@5", "hit_all@5", "mrr", "ndcg@10")}
    out["modes"][mode] = res
    print(f"\n== {mode} ==")
    for name in per:
        s = res["summary"][name]
        print(f"{name:11s} " + "  ".join(f"{k}={s[k]['mean']:.3f}[{s[k]['lo']:.3f},{s[k]['hi']:.3f}]" for k in ("recall@5", "hit_all@5", "mrr", "ndcg@10")))

d = config.RESULTS_DIR; d.mkdir(exist_ok=True)
suffix = "" if not args.limit else f"_limit{args.limit}"
(d / f"exp1_finqa_{args.split}{suffix}.json").write_text(json.dumps(out, indent=2))
print(f"done in {time.time()-t0:.0f}s")
