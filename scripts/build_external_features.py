"""Build leakage-safe external features from raw FRED CSVs.

Produces `data/processed/external_features_daily.parquet` with daily
indexed external features aligned for as-of joins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running scripts directly
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copper_market_intelligence.features.external import (  # noqa: E402
    expand_monthly_as_of,
    inventory_features,
    usd_features,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "fred"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_usd():
    path = RAW_DIR / "DTWEXBGS.csv"
    df = pd.read_csv(path)
    # FRED CSVs sometimes use 'observation_date' as the date column
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    series = df.iloc[:, 0]
    series.index.name = "date"
    usd = usd_features(series, prefix="usd")
    return usd


def build_china_imports(daily_index):
    path = RAW_DIR / "VALIMPCNM052N.csv"
    df = pd.read_csv(path)
    # detect date column name (DATE or observation_date)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    value_col_name = df.columns[1]
    df = df.rename(columns={date_col: "obs_date", value_col_name: "value"})
    out = expand_monthly_as_of(
        df,
        daily_index,
        obs_date_col="obs_date",
        value_col="value",
        publication_lag_months=1,
    )
    # compute mom/yoy
    out = out.rename(columns={"value": "china_imports"})
    out["china_imports_mom"] = out["china_imports"].pct_change(1)
    out["china_imports_yoy"] = out["china_imports"].pct_change(12)
    return out


def main():
    # build USD features
    usd = build_usd()
    daily_index = usd.index

    china = build_china_imports(daily_index)

    # Inventory: placeholder (not fetched) — keep empty for now
    inventory = None

    # combine
    merged = usd.join(china, how="outer")
    if inventory is not None:
        inv = inventory_features(inventory, prefix="lme")
        merged = merged.join(inv, how="outer")

    out_path = OUT_DIR / "external_features_daily.parquet"
    merged.to_parquet(out_path)
    print(f"wrote external features to {out_path}")


if __name__ == "__main__":
    main()
