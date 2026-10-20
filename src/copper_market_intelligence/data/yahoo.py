"""Simple Yahoo Finance CSV downloader for commodity tickers (e.g.
HG=F for copper futures).

This adapter uses the public Yahoo CSV download endpoint which does not
require an API key.
"""
import json
from datetime import datetime, timezone
from urllib.parse import urlencode

import pandas as pd
import requests

from ..config import config
from .base import BaseIngestor, IngestResult

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None


class YahooIngestor(BaseIngestor):
    def __init__(self, ticker: str = "HG=F"):
        super().__init__(source_name="yahoo")
        self.ticker = ticker
        # prefer per-source subdirectory under data/raw
        self.raw_dir = config.RAW_DIR / "yahoo"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _build_url(
        self, period1: int = 0, period2: int = 9999999999, interval: str = "1d"
    ) -> str:
        params = {
            "period1": str(period1),
            "period2": str(period2),
            "interval": interval,
            "events": "history",
        }
        return (
            f"https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}?{urlencode(params)}"
        )

    def ingest(self, period1: int = 0, period2: int = 9999999999) -> IngestResult:
        # prefer yfinance when available (more robust for programmatic access)
        if yf is not None:
            # use yfinance to download historical OHLCV
            hist = yf.download(
                self.ticker, period="max", interval="1d", progress=False
            )
            if hist is None or hist.empty:
                raise RuntimeError(
                    "yfinance returned no data for ticker: %s" % self.ticker
                )
            # reset index to have Date column
            df = hist.reset_index()
            # yfinance often returns a DataFrame with MultiIndex columns
            # like ('Close', 'HG=F'). Flatten to single-level columns
            if hasattr(df.columns, "nlevels") and getattr(df.columns, "nlevels") > 1:
                new_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        # prefer the first element (e.g., 'Close')
                        new_cols.append(col[0] if col[0] else col[1])
                    else:
                        new_cols.append(col)
                df.columns = new_cols
            # write raw CSV for reproducibility
            filename = (
                f"yahoo_{self.ticker}_{int(datetime.now(timezone.utc).timestamp())}.csv"
            )
            raw_path = self._save_raw(df.to_csv(index=False).encode("utf-8"), filename)
            checksum = self._checksum(raw_path)
            url = f"yfinance://{self.ticker}"
        else:
            url = self._build_url(period1=period1, period2=period2)
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to download {self.ticker} from Yahoo: {resp.status_code}"
                )

            filename = (
                f"yahoo_{self.ticker}_{int(datetime.now(timezone.utc).timestamp())}.csv"
            )
            raw_path = self._save_raw(resp.content, filename)
            checksum = self._checksum(raw_path)
            df = pd.read_csv(raw_path, parse_dates=["Date"])  # may raise

        result = IngestResult(
            df=df,
            raw_path=raw_path,
            checksum=checksum,
            retrieved_at=datetime.now(timezone.utc),
            source_url=url,
            dataset=self.ticker,
        )
        # build extended manifest metadata for Yahoo
        manifest = {
            "source": "yahoo",
            "ticker": self.ticker,
            "source_url": url,
            "retrieved_at": result.retrieved_at.isoformat(),
            "checksum": checksum,
            "raw_path": str(raw_path.as_posix()),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        # earliest/latest and frequency
        if "Date" in df.columns:
            try:
                idx = pd.to_datetime(df["Date"], errors="coerce")
                manifest["earliest_date"] = str(idx.min())
                manifest["latest_date"] = str(idx.max())
                manifest["frequency"] = pd.infer_freq(idx.dropna())
            except Exception:
                pass
        # currency and instrument notes
        manifest["currency"] = "USD"
        manifest["instrument"] = "COMEX futures (HG=F)"
        manifest["notes"] = (
            "Daily historical OHLC+Volume from Yahoo Finance (COMEX HG=F). "
            "Treat as convenience public market data; not LME cash price."
        )

        # write per-file manifest next to raw
        manifest_path = raw_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        # also update global manifest
        try:
            from . import manifest as manifest_mod

            gm = manifest_mod.load_manifest()
            gm[raw_path.name] = manifest
            manifest_mod.save_manifest(gm)
        except Exception:
            # non-fatal
            pass

        return result
