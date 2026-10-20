"""Live ingestion runner for Stage 2 (FRED, USGS, LME).

This script attempts to download public data for the requested sources,
saves raw files under `data/raw/<source>/`, writes per-file manifests and
updates the global manifest at `data/raw/manifest.json`.

Run locally; CI must not depend on this.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from copper_market_intelligence.config import config
from copper_market_intelligence.data import manifest as manifest_mod
from copper_market_intelligence.data.base import BaseIngestor, IngestResult


class TempIngestor(BaseIngestor):
    def __init__(self, source_name: str):
        super().__init__(source_name=source_name)
        # use per-source subdirectory under data/raw
        self.raw_dir = config.RAW_DIR / source_name
        self.raw_dir.mkdir(parents=True, exist_ok=True)


def sha256_of_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def save_global_manifest_entry(key: str, entry: dict) -> None:
    man = manifest_mod.load_manifest()
    man[key] = entry
    manifest_mod.save_manifest(man)


def ingest_fred_public(series_id: str) -> dict:
    src = "fred"
    ing = TempIngestor(src)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Fetching FRED CSV: {url}")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return {"ok": False, "reason": f"HTTP {resp.status_code}"}
    filename = f"fred_{series_id}_{int(datetime.now(timezone.utc).timestamp())}.csv"
    raw_path = ing._save_raw(resp.content, filename)
    checksum = ing._checksum(raw_path)

    # try parsing CSV
    try:
        df = pd.read_csv(raw_path)
        # attempt to find date column and parse
        date_col = None
        for c in df.columns:
            if re.search(r"date", c, flags=re.I):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # ensure numeric for other columns
        for c in df.columns:
            if c != date_col:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # frequency inference
        idx = pd.DatetimeIndex(df[date_col].dropna())
        freq = pd.infer_freq(idx) if len(idx) > 2 else None
        # note: earliest/latest are available if needed; avoid unused assignments
        # earliest = idx.min() if len(idx) > 0 else None
        # latest = idx.max() if len(idx) > 0 else None
    except Exception as e:
        return {"ok": False, "reason": f"parse_error: {e}"}

    result = IngestResult(
        df=df,
        raw_path=raw_path,
        checksum=checksum,
        retrieved_at=datetime.now(timezone.utc),
        source_url=url,
        dataset=series_id,
    )
    ing._write_manifest(result)

    # try extract units from series page (best-effort)
    units = None
    try:
        page = requests.get(f"https://fred.stlouisfed.org/series/{series_id}", timeout=20)
        m = re.search(r"Units:</span>\s*<span[^>]*>([^<]+)</span>", page.text)
        if m:
            units = m.group(1).strip()
    except Exception:
        units = None

    entry = {
        "source": src,
        "series_id": series_id,
        "retrieved_at": result.retrieved_at.isoformat(),
        "source_url": url,
        "checksum": checksum,
        "frequency": freq,
        "units": units,
        "raw_filename": raw_path.name,
        "rows": len(df),
    }
    save_global_manifest_entry(raw_path.name, entry)

    return {"ok": True, "result": result, "meta": entry}


def try_find_usgs_csv() -> str | None:
    # Search the USGS NMIC commodity page for CSV links mentioning copper (best-effort)
    base = "https://www.usgs.gov/centers/nmic/commodity-statistics-and-information"
    try:
        r = requests.get(base, timeout=20)
        if r.status_code != 200:
            return None
        for m in re.finditer(r"href=[\'\"](?P<link>[^\"'>]+\.csv)[\'\"]", r.text, flags=re.I):
            link = m.group("link")
            if "copper" in link.lower() or "copper" in r.text.lower():
                if link.startswith("/"):
                    return "https://www.usgs.gov" + link
                if link.startswith("http"):
                    return link
        return None
    except Exception:
        return None


def ingest_usgs_minimal() -> dict:
    src = "usgs"
    found = try_find_usgs_csv()
    if not found:
        return {"ok": False, "reason": "no_public_csv_found_on_usgs_nmic"}
    ing = TempIngestor(src)
    resp = requests.get(found, timeout=30)
    if resp.status_code != 200:
        return {"ok": False, "reason": f"HTTP {resp.status_code}"}
    filename = f"usgs_copper_{int(datetime.now(timezone.utc).timestamp())}.csv"
    raw_path = ing._save_raw(resp.content, filename)
    checksum = ing._checksum(raw_path)
    try:
        df = pd.read_csv(raw_path)
        # parse dates where present
        for c in df.columns:
            if re.search(r"date", c, flags=re.I):
                df[c] = pd.to_datetime(df[c], errors="coerce")
    except Exception as e:
        return {"ok": False, "reason": f"parse_error: {e}"}

    result = IngestResult(
        df=df,
        raw_path=raw_path,
        checksum=checksum,
        retrieved_at=datetime.now(timezone.utc),
        source_url=found,
        dataset="usgs_copper",
    )
    ing._write_manifest(result)
    entry = {
        "source": src,
        "retrieved_at": result.retrieved_at.isoformat(),
        "source_url": found,
        "checksum": checksum,
        "rows": len(df),
        "raw_filename": raw_path.name,
    }
    save_global_manifest_entry(raw_path.name, entry)
    return {"ok": True, "result": result, "meta": entry}


def check_lme_public():
    # Best-effort check for public LME CSVs; LME historically restricts data
    base = "https://www.lme.com/en-GB/"
    try:
        r = requests.get(base, timeout=20)
        if r.status_code != 200:
            return {"ok": False, "reason": f"HTTP {r.status_code}"}
        # look for any CSV links or clear indicators of free download
        if ".csv" in r.text.lower():
            return {"ok": False, "reason": "csv_link_found_but_manual_inspect_needed"}
        return {"ok": False, "reason": "no_public_csv_found; likely paywalled"}
    except Exception as e:
        return {"ok": False, "reason": f"error: {e}"}


def main():
    report = {"fred": None, "usgs": None, "lme": None}

    # FRED
    print("=== FRED ingestion ===")
    fred_res = ingest_fred_public("PCOPPUSDM")
    report["fred"] = fred_res
    if fred_res.get("ok"):
        r = fred_res["result"]
        print(f"Saved: {r.raw_path} ({r.raw_path.stat().st_size} bytes)")
        print(r.df.head().to_string(index=False))
        # earliest/latest
        date_col = r.df.columns[0]
        try:
            idx = pd.DatetimeIndex(r.df[date_col])
            print("Earliest:", idx.min())
            print("Latest:", idx.max())
        except Exception:
            pass

    # USGS
    print("\n=== USGS ingestion ===")
    usgs_res = ingest_usgs_minimal()
    report["usgs"] = usgs_res
    if usgs_res.get("ok"):
        r = usgs_res["result"]
        print(f"Saved: {r.raw_path} ({r.raw_path.stat().st_size} bytes)")
        print(r.df.head().to_string(index=False))

    # LME
    print("\n=== LME public check ===")
    lme_res = check_lme_public()
    report["lme"] = lme_res
    print(lme_res)

    # write a short run report
    out = Path(config.DATA_DIR) / "raw" / "live_ingest_report.json"
    out.write_text(json.dumps(report, default=str, indent=2))
    print(f"Run report: {out}")


if __name__ == "__main__":
    main()
