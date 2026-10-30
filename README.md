# Racer

Code and data-cleaning pipeline for an analysis of Indian mutual funds (814 schemes).

## Data

The raw dataset is not included in this repository. Download it from Kaggle:

- **Mutual Funds India – Detailed**: https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed

Save `comprehensive_mutual_funds_data.csv` to `data/raw/`.

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
