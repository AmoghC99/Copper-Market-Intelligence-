"""ETL runner: normalize raw FRED and Yahoo data into processed canonical datasets.

Idempotent: writing to data/processed/* with predictable filenames.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from copper_market_intelligence.config import config
from copper_market_intelligence.etl import validation
from copper_market_intelligence.etl.processing import normalize_fred, normalize_yahoo

RAW_YAHOO = config.RAW_DIR / "yahoo"
RAW_YAHOO_LIVE = RAW_YAHOO / "live"
RAW_FRED = config.RAW_DIR / "fred"
PROC_DIR = config.PROCESSED_DIR
PROC_DIR.mkdir(parents=True, exist_ok=True)
RAW_YAHOO_LIVE.mkdir(parents=True, exist_ok=True)


def write_parquet_or_csv(df, path: Path):
    try:
        df.to_parquet(path)
        return str(path)
    except Exception:
        # fall back to CSV
        csvp = path.with_suffix(".csv")
        df.to_csv(csvp, index=False)
        return str(csvp)


def run():
    report = {}

    # Yahoo: only process files under data/raw/yahoo/live/ (avoid fixtures)
    # If live directory is empty, as a fallback do not process fixtures.
    yahoo_files = sorted(
        RAW_YAHOO_LIVE.glob("yahoo_HG=F_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not yahoo_files:
        # report presence of raw yahoo files but do not process fixtures located in parent dir
        parent_live_candidates = sorted(
            RAW_YAHOO.glob("yahoo_HG=F_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if parent_live_candidates:
            report["yahoo"] = {
                "ok": False,
                "reason": (
                    "live directory empty; fixture files present in data/raw/yahoo/. "
                    "Move live files to data/raw/yahoo/live/ to be processed."
                ),
                "candidates": [str(p) for p in parent_live_candidates[:5]],
            }
    else:
        fy = yahoo_files[0]
        dfy, stats = normalize_yahoo(fy, return_stats=True)
        outp = PROC_DIR / "market_daily.parquet"
        written = write_parquet_or_csv(dfy, outp)
        # validations
        v = validation.check_date_ordering(dfy)
        dup = validation.check_duplicates(dfy)
        num_nulls = validation.check_numeric_fields(dfy, ["open", "high", "low", "close", "volume"])
        ohlc_issues = validation.check_ohlc_relationships(dfy)
        neg_vol = validation.check_negative_volume(dfy)
        report["yahoo"] = {
            "raw": str(fy),
            "processed": written,
            "rows_in": stats.get("rows_in"),
            "rows_removed": stats.get("rows_removed"),
            "rows_out": stats.get("rows_out"),
            "earliest": str(dfy["timestamp"].min()),
            "latest": str(dfy["timestamp"].max()),
            "validation": {
                "date": v,
                "duplicates": dup,
                "numeric_nulls": num_nulls,
                "ohlc_issues": ohlc_issues,
                "negative_volume": neg_vol,
            },
        }

    # FRED: pick latest from raw/fred/
    fred_files = sorted(RAW_FRED.glob("fred_PCOPPUSDM_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fred_files:
        ff = fred_files[0]
        dff = normalize_fred(ff)
        outf = PROC_DIR / "fred_monthly.parquet"
        written = write_parquet_or_csv(dff, outf)
        # FRED validations
        f_nulls = validation.check_date_ordering(dff)
        f_dups = validation.check_duplicates(dff)
        f_nums = validation.check_numeric_fields(dff, ["copper_benchmark_usd_t"])
        f_neg = {"negative_values": int((dff["copper_benchmark_usd_t"] < 0).sum())}
        # monthly consistency: ensure day is start of month for most rows
        months_ok = None
        try:
            months = dff["timestamp"].dt.day
            months_ok = int((months == 1).sum()) if len(months) > 0 else 0
        except Exception:
            months_ok = 0

        report["fred"] = {
            "raw": str(ff),
            "processed": written,
            "rows": len(dff),
            "earliest": str(dff["timestamp"].min()),
            "latest": str(dff["timestamp"].max()),
            "validation": {
                "date": f_nulls,
                "duplicates": f_dups,
                "numeric_nulls": f_nums,
                "negative_values": f_neg,
                "monthly_day_1_counts": months_ok,
            },
        }

    out = config.DATA_DIR / "processed" / "etl_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # attach run metadata
    report["run_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(report, indent=2, default=str))
    print("ETL report written to:", out)


if __name__ == "__main__":
    run()
