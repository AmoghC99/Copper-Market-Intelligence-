"""Fetch specified FRED series CSVs and write manifests.

Usage (from repo root):
    python scripts/fetch_fred.py

The script attempts to download public CSVs from FRED's graph endpoint
(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIESID`) which does
not require an API key for public series. Files are saved to
`data/raw/fred/{SERIESID}.csv` and a manifest CSV is written to
`data/raw/fred/manifest.csv`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "fred"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SERIES = [
    {
        "series_id": "DTWEXBGS",
        "note": "Nominal Broad U.S. Dollar Index",
    },
    {
        "series_id": "VALIMPCNM052N",
        "note": "China imports of goods (monthly, USD millions)",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_series(series_id: str) -> dict:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    out_path = RAW_DIR / f"{series_id}.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

    # read to infer dates
    df = pd.read_csv(out_path)
    if "DATE" in df.columns:
        dates = pd.to_datetime(df["DATE"])
        earliest = str(dates.min().date())
        latest = str(dates.max().date())
    else:
        earliest = ""
        latest = ""

    record = {
        "series_id": series_id,
        "source": "FRED",
        "url": url,
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
        "checksum": sha256_file(out_path),
        "earliest_date": earliest,
        "latest_date": latest,
        "frequency": "daily" if series_id == "DTWEXBGS" else "monthly",
        "units": "unknown",
    }
    return record


def main():
    manifest = []
    for s in SERIES:
        sid = s["series_id"]
        try:
            rec = fetch_series(sid)
            rec["note"] = s.get("note", "")
            manifest.append(rec)
            print(f"fetched {sid}: {rec['earliest_date']} -> {rec['latest_date']}")
        except Exception as e:
            print(f"failed to fetch {sid}: {e}", file=sys.stderr)

    mf_path = RAW_DIR / "manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {mf_path}")


if __name__ == "__main__":
    main()
