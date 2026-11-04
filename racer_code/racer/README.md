# RACER — regulatory-aware RAG for financial advisory (rebuilt, tested, honestly evaluated)

```
pip install -r requirements.txt
python -m pytest -q tests                      # 21 tests
```

## What is already run (real data)
`experiments/exp1_finqa_retrieval.py` — FinQA fact retrieval (TF-IDF / BM25 / RRF), closed-pool + open-corpus.
Needs `data/finqa/{train,test}.json` from https://github.com/czyssrs/FinQA (folder `dataset/`).
Results: `results/exp1_finqa_test.json`; tables: `python scripts/make_tables.py` -> `paper/tables/`.

## What YOU still have to run (data / API access I did not have)
1. **Funds** — download "Mutual Funds India" from Kaggle, save as `data/funds/funds.csv`.
2. **Profiles** — download "User Financial Profiles" from Kaggle, save as `data/profiles/profiles.csv`.
   Column names are matched through aliases in `racer/data.py`; if one is not found you get a clear error listing the CSV's columns.
3. `cd experiments && python exp2_funds_suitability.py && python exp3_profiling.py`
4. **Regulatory text** (optional but needed for citations): save MiFID II / ESMA / SEBI text you obtained as .txt and run
   `python scripts/split_clauses.py file.txt --source "MiFID II" --append`  -> `data/regulatory/clauses.jsonl`
5. **LLM experiment**: `export OPENAI_API_KEY=...; python exp4_generation_hallucination.py --backend openai --model <name>`
   (or `--backend anthropic`). `--backend extractive` is a no-LLM sanity check.
6. `python scripts/make_tables.py`, then fill the `\pending{...}` blocks in `paper/main.tex` with the printed numbers and `cd paper && pdflatex main && pdflatex main`.

`--smoke` runs any experiment on SYNTHETIC data (writes to `results/smoke/`); never report those numbers.

## Layout
`racer/` retrieval, metrics (bootstrap CIs, paired tests), profiling (policy + RF), rules (CCE), generator (extractive / scripted / OpenAI-compatible / Anthropic via urllib), validator (hallucination metrics), pipeline (ablation switches, regeneration loop).
