"""Exp 4 -- hallucination with vs without the post-generation validator (needs an LLM backend).
  python exp4_generation_hallucination.py --backend openai --model gpt-4o-mini   (OPENAI_API_KEY)
  python exp4_generation_hallucination.py --backend anthropic --model <id>       (ANTHROPIC_API_KEY)
  python exp4_generation_hallucination.py --backend extractive                   (sanity: grounded by construction)"""
import os
from common import *
from racer.generator import ExtractiveGenerator, OpenAICompatBackend, AnthropicBackend
from racer.metrics import bootstrap_ci, paired_bootstrap
from racer.pipeline import Racer, Options
from racer.validator import validate

ap = parser(__doc__)
ap.add_argument("--backend", default="extractive"); ap.add_argument("--model", default="")
ap.add_argument("--base_url", default="https://api.openai.com/v1"); ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
if a.backend == "openai": gen = OpenAICompatBackend(a.base_url, os.environ["OPENAI_API_KEY"], a.model)
elif a.backend == "anthropic": gen = AnthropicBackend(os.environ["ANTHROPIC_API_KEY"], a.model)
else: gen = ExtractiveGenerator()
funds = get_funds(a); ftr, fte = split(funds, funds["tier"])
clauses = [] if a.smoke or not Path(a.regulatory).exists() else load_regulatory(a.regulatory)
cells = [(t, q) for t in config.TIER_NAMES for q in QUERIES][: a.n]
res, per = {}, {}
for name, opt in {"no validator": Options(use_validator=False), "with validator (<=2 regenerations)": Options(use_validator=True)}.items():
    rac = Racer(ftr, fte, clauses, generator=gen, options=opt)
    rows = {"has_hallucination": [], "product_grounding": [], "citation_validity": [], "numeric_grounding": [], "attempts": []}
    for t, q in cells:
        r = rac.recommend(t, q)
        m = validate(r.text, {"tier": t, "products": r.products, "clauses": r.clauses}, rac.names, rac.rules, rac.funds).metrics
        for k in rows: rows[k].append(float(r.attempts if k == "attempts" else m[k]))
    per[name] = rows
    res[name] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v))) for k, v in rows.items()}
    print(f"{name:38s}", {k: round(res[name][k]["mean"], 3) for k in rows})
paired = paired_bootstrap(per["with validator (<=2 regenerations)"]["has_hallucination"], per["no validator"]["has_hallucination"])
save(a, f"exp4_{a.backend}.json", {"backend": gen.name, "n": len(cells), "variants": res, "paired_validator_effect_on_hallucination": paired})
