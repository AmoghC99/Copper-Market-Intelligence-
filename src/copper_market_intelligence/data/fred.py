"""FRED adapter using the FRED API when `FRED_API_KEY` is provided via env var.

If no API key is present the adapter will raise an informative error and not
attempt unauthenticated scraping.
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from .base import BaseIngestor, IngestResult


class FredIngestor(BaseIngestor):
    def __init__(self, series_id: str):
        super().__init__(source_name="fred")
        self.series_id = series_id
        self.api_key = os.environ.get("FRED_API_KEY")

    def ingest(
        self,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
    ) -> IngestResult:
        if not self.api_key:
            raise RuntimeError(
                "FRED API requires an API key. Set environment variable"
                " FRED_API_KEY to use this adapter."
            )
        params = {
            "series_id": self.series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        url = "https://api.stlouisfed.org/fred/series/observations"
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"FRED request failed: {resp.status_code} {resp.text}")
        content = resp.content
        filename = f"fred_{self.series_id}_{int(datetime.utcnow().timestamp())}.json"
        raw_path = self._save_raw(content, filename)
        checksum = self._checksum(raw_path)
        data = resp.json()
        observations = data.get("observations", [])
        df = pd.DataFrame(observations)
        # parse date and numeric values
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        result = IngestResult(
            df=df,
            raw_path=raw_path,
            checksum=checksum,
            retrieved_at=datetime.utcnow(),
            source_url=resp.url,
            dataset=self.series_id,
        )
        self._write_manifest(result)
        return result
