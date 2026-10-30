"""Clean the raw mutual funds dataset and save it to data/processed/."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "comprehensive_mutual_funds_data.csv"
OUT_PATH = ROOT / "data" / "processed" / "mutual_funds_clean.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    # The source uses "-" for missing risk metrics (sortino, alpha, sd, beta, sharpe).
    return pd.read_csv(path, na_values="-")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Scheme names are truncated, so different variants share a name; give each row an id.
    df.insert(0, "fund_id", range(len(df)))

    # Closed-ended fixed term / fixed maturity plans can't take new money.
    df["is_closed_ended"] = df["scheme_name"].str.contains(r"\b(?:FTP|FMP)\b", regex=True)

    # 0 means "not rated" / "not offered" / "not reported", not a real value.
    for col in ["rating", "min_sip", "min_lumpsum", "expense_ratio"]:
        df[col] = df[col].replace(0, np.nan)

    for col in ["scheme_name", "fund_manager", "amc_name", "category", "sub_category"]:
        df[col] = df[col].str.strip()

    # returns_3yr / returns_5yr stay NaN: the fund is younger than the period.
    return df


def report(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    print(f"Closed-ended funds: {df['is_closed_ended'].sum()}")
    missing = df.isna().sum()
    print("Missing values:")
    print(missing[missing > 0].to_string())


def main() -> None:
    df = clean(load_raw())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    report(df)
    print(f"Saved to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
