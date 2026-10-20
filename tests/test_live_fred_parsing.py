from pathlib import Path

import pandas as pd
import pytest

RAW_DIR = Path("data/raw/fred")


@pytest.mark.skipif(not RAW_DIR.exists(), reason="No live FRED raw files present")
def test_live_fred_parsing():
    # pick the most recent fred csv if present
    files = sorted(
        RAW_DIR.glob("fred_PCOPPUSDM_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        pytest.skip("No fred CSV files found")
    p = files[0]
    df = pd.read_csv(p)
    # basic checks
    assert df.shape[0] > 0
    # date parse
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    assert date_col is not None
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    assert parsed.notna().any()

