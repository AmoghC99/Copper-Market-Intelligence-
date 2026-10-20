from pathlib import Path
from typing import List

import pandas as pd


def _flatten_columns(cols) -> List[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [p for p in c if p and str(p).strip()]
            out.append("_".join(parts) if parts else "")
        else:
            out.append(str(c))
    return out


def normalize_yahoo(raw_path: Path, return_stats: bool = False):
    """Normalize a Yahoo raw CSV (or yfinance-produced CSV) to canonical daily schema.

    Returns a DataFrame with columns:
    timestamp, open, high, low, close, volume, source, instrument
    """
    df = pd.read_csv(raw_path)
    # flatten columns if multi-index-like
    df.columns = _flatten_columns(df.columns)

    # common column names
    # ensure Date column exists
    date_cols = [c for c in df.columns if c.lower() == "date"]
    if not date_cols:
        # try first column
        date_col = df.columns[0]
    else:
        date_col = date_cols[0]

    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # stats: rows_in before any removals
    rows_in = int(len(df))

    # dedupe exact duplicates
    df = df.drop_duplicates()

    # numeric conversions
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # map canonical names
    canonical = pd.DataFrame()
    canonical["timestamp"] = df["timestamp"]
    # prefer Close over Adj Close
    if "Open" in df.columns:
        canonical["open"] = df["Open"]
    else:
        canonical["open"] = pd.NA
    if "High" in df.columns:
        canonical["high"] = df["High"]
    else:
        canonical["high"] = pd.NA
    if "Low" in df.columns:
        canonical["low"] = df["Low"]
    else:
        canonical["low"] = pd.NA
    if "Close" in df.columns:
        canonical["close"] = df["Close"]
    elif "Adj Close" in df.columns:
        canonical["close"] = df["Adj Close"]
    else:
        canonical["close"] = pd.NA

    if "Volume" in df.columns:
        canonical["volume"] = df["Volume"]
    else:
        canonical["volume"] = pd.NA

    canonical["source"] = "Yahoo Finance"
    canonical["instrument"] = "COMEX copper futures (HG=F)"

    # keep original rows order and metadata
    canonical = canonical.loc[:, [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "instrument",
    ]]

    # Drop fully-blank rows: timestamp is null AND all numeric market fields null
    market_cols = ["open", "high", "low", "close", "volume"]
    is_ts_null = canonical["timestamp"].isna()
    are_markets_all_null = canonical[market_cols].isna().all(axis=1)
    to_remove = is_ts_null & are_markets_all_null
    rows_removed = int(to_remove.sum())

    # Per requirement: DO NOT drop partially populated rows here
    if rows_removed > 0:
        canonical = canonical.loc[~to_remove].reset_index(drop=True)

    if return_stats:
        stats = {
            "rows_in": rows_in,
            "rows_removed": rows_removed,
            "rows_out": int(len(canonical)),
        }
        return canonical, stats

    return canonical


def normalize_fred(raw_path: Path) -> pd.DataFrame:
    """Normalize FRED CSV for PCOPPUSDM into monthly canonical schema.

    Returns dataframe with columns: timestamp, copper_benchmark_usd_t, source, frequency
    """
    df = pd.read_csv(raw_path)
    # find date and value columns
    date_col = None
    val_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
        if c.lower() in ("value", "pcoppusdm") or "value" in c.lower():
            val_col = c

    if date_col is None:
        date_col = df.columns[0]
    if val_col is None and len(df.columns) > 1:
        val_col = df.columns[1]

    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["copper_benchmark_usd_t"] = pd.to_numeric(df[val_col], errors="coerce")
    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"]
    out["copper_benchmark_usd_t"] = df["copper_benchmark_usd_t"]
    out["source"] = "FRED"
    out["frequency"] = "monthly"
    # units can be filled if available elsewhere; leave as metadata
    return out
