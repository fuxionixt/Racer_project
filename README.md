# Racer

Code and data-cleaning pipeline for an analysis of Indian mutual funds (814 schemes).

## Data

The raw dataset is not included in this repository. Download it from Kaggle:

- **Mutual Funds India – Detailed**: https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed

Save `comprehensive_mutual_funds_data.csv` to `data/raw/`. The results were produced with Kaggle version 2 (SHA-256 `2672b241c0715e4e099d81771b09da2108f5c2415693c0df2982cfc9ec155b5e`). To download it with `kagglehub`:

```bash
mkdir -p data/raw
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('ravibarnawal/mutual-funds-india-detailed/versions/2'); shutil.copy(p + '/comprehensive_mutual_funds_data.csv', 'data/raw/')"
```

The user-profiling experiment (Dataset 3) uses the synthetic **Financial Risk Dataset** (CC0): https://www.kaggle.com/datasets/berkereryilmaz/financial-risk-dataset. Save `financial_dataset.csv` to `data/raw/profiles/` (version 1, SHA-256 `32335ac88fe4bd76161b459b8beb303b36556922c5b841d785dc3b004dbdb1a9`):

```bash
mkdir -p data/raw/profiles
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('berkereryilmaz/financial-risk-dataset/versions/1'); shutil.copy(p + '/financial_dataset.csv', 'data/raw/profiles/')"
```

## Experiments

```bash
python evaluate_cce.py         # suitability filtering on Mutual Funds India -> results/cce_results.json
python evaluate_fund_risk.py   # risk_level leakage check on Mutual Funds India -> results/fund_risk_results.json
python evaluate_profiling.py   # user risk profiling on Financial Risk Dataset -> results/profiling_results.json
python make_figures.py         # paper Fig. 1 (framework) and Figs. 4, 6, 7 and 8 -> figures/
```

`make_figures.py` also reads the FinQA and generation results in `racer_code/racer/results/`.

The pipeline experiments live in `racer_code/racer/experiments/` (FinQA files go in `racer_code/racer/data/finqa/`, the fund CSV in `racer_code/racer/data/funds/funds.csv`):

```bash
cd racer_code/racer/experiments
python exp1_finqa_retrieval.py                 # FinQA supporting-fact retrieval
python exp5_prefilter_vs_posthoc.py            # filtering depth (5/10/20) and tier-boost ablation (192 cases, no LLM)
python exp6_clause_retrieval.py                # BM25 vs TF-IDF vs dense retrieval over the SEBI clauses
python exp4_generation_hallucination.py --backend anthropic --model claude-sonnet-5-5 --queries extended \
    --n 192 --clause_retriever dense --max_tokens 2048
                                               # generation with/without validator (needs ANTHROPIC_API_KEY)
```

The regulatory index uses `racer_code/racer/data/regulatory/clauses.jsonl`: 16 clauses copied verbatim from the
SEBI (Investment Advisers) Regulations, 2013, as amended up to 25 Nov 2025 (sebi.gov.in). Dense retrieval downloads
`BAAI/bge-small-en-v1.5` on first use. Model responses are cached in `racer_code/racer/results/cache/`, so rerunning
exp4 re-scores the saved answers without new API calls.

Results behind the paper's generation table: `exp4_anthropic_claude-sonnet-5-5_dense_mt2048_extended.json`.
The other `exp4_*` files are earlier configurations (600-token limit, BM25 clauses, Claude Haiku 5.5) kept for the record;
most of their 600-token answers were truncated.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the cleaned dataset

```bash
python clean_data.py
```

This writes `data/processed/mutual_funds_clean.csv`.

## Cleaning steps

`clean_data.py` applies the following, leaving the raw file unchanged:

| Step | Columns | Rationale |
|---|---|---|
| `"-"` read as missing | `sortino`, `alpha`, `sd`, `beta`, `sharpe` | Placeholder for unavailable risk metrics |
| `0` set to missing | `rating` | 0 means the fund is unrated |
| `0` set to missing | `min_sip`, `min_lumpsum`, `expense_ratio` | Not offered or not reported |
| `fund_id` added | — | Scheme names are truncated, so distinct variants share a name |
| `is_closed_ended` added | — | Flags FTP/FMP funds that do not accept new investment |
| Left missing | `returns_3yr`, `returns_5yr` | The fund is younger than the return period |
