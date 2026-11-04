"""Exp 4 -- hallucination with vs without the post-generation validator (needs an LLM backend).
  python exp4_generation_hallucination.py --backend openai --model gpt-4o-mini   (OPENAI_API_KEY)
  python exp4_generation_hallucination.py --backend anthropic --model <id>       (ANTHROPIC_API_KEY)
  python exp4_generation_hallucination.py --backend extractive                   (sanity: grounded by construction)"""
import os
from common import *
from racer.generator import CachedBackend, ExtractiveGenerator, OpenAICompatBackend, AnthropicBackend
from racer.metrics import bootstrap_ci, paired_bootstrap
from racer.pipeline import Racer, Options
from racer.validator import validate

ap = parser(__doc__)
ap.add_argument("--backend", default="extractive"); ap.add_argument("--model", default="")
ap.add_argument("--base_url", default="https://api.openai.com/v1"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--queries", choices=("base", "extended"), default="base", help="8 base queries or 64 extended ones")
ap.add_argument("--max_tokens", type=int, default=2048, help="output limit; 600 truncated most answers")
ap.add_argument("--clause_retriever", choices=("bm25", "tfidf", "dense"), default=config.CLAUSE_RETRIEVER)
a = ap.parse_args()
if a.backend == "openai": gen = OpenAICompatBackend(a.base_url, os.environ["OPENAI_API_KEY"], a.model)
elif a.backend == "anthropic": gen = AnthropicBackend(os.environ["ANTHROPIC_API_KEY"], a.model, max_tokens=a.max_tokens)
else: gen = ExtractiveGenerator()
funds = get_funds(a); ftr, fte = split(funds, funds["tier"])
clauses = [] if a.smoke or not Path(a.regulatory).exists() else load_regulatory(a.regulatory)
cells = [(t, q) for t in config.TIER_NAMES for q in (EXTENDED_QUERIES if a.queries == "extended" else QUERIES)][: a.n]
tag = f"exp4_{a.backend}{'_' + a.model if a.model else ''}_{a.clause_retriever}{'_mt' + str(a.max_tokens) if a.max_tokens != 600 else ''}{'_extended' if a.queries == 'extended' else ''}"
res, per = {}, {}
for name, opt in {"no validator": Options(use_validator=False), "with validator (<=2 regenerations)": Options(use_validator=True)}.items():
    g = gen if a.backend == "extractive" else CachedBackend(gen, out_dir(a.smoke) / "cache" / f"{tag}.jsonl", name)
    rac = Racer(ftr, fte, clauses, generator=g, clause_retriever=a.clause_retriever, options=opt)
    rows = {"has_hallucination": [], "product_grounding": [], "citation_validity": [], "numeric_grounding": [], "attempts": [], "final_valid": [], "truncated": []}
    for t, q in cells:
        r = rac.recommend(t, q)
        m = validate(r.text, {"tier": t, "products": r.products, "clauses": r.clauses}, rac.names, rac.rules, rac.funds).metrics
        for k in rows: rows[k].append(float(r.attempts if k == "attempts" else r.report.ok if k == "final_valid"
                                            else getattr(g, "last_stop", None) == "max_tokens" if k == "truncated" else m[k]))
    per[name] = rows
    res[name] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v))) for k, v in rows.items()}
    print(f"{name:38s}", {k: round(res[name][k]["mean"], 3) for k in rows})
paired = paired_bootstrap(per["with validator (<=2 regenerations)"]["has_hallucination"], per["no validator"]["has_hallucination"])
save(a, f"{tag}.json", {"backend": gen.name, "clause_retriever": a.clause_retriever, "max_tokens": a.max_tokens, "n": len(cells), "variants": res, "paired_validator_effect_on_hallucination": paired})
