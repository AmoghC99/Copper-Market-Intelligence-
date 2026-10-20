"""LME adapter: attempts to download public LME datasets if a direct CSV URL is known.

Note: LME often requires paid subscriptions for historical datasets. This
adapter will accept a direct URL to a CSV or other file and fetch it, but
does not bypass paywalls.
"""
from datetime import datetime

import pandas as pd
import requests

from .base import BaseIngestor, IngestResult


class LMEIngestor(BaseIngestor):
    def __init__(self, url: str, dataset_name: str):
        super().__init__(source_name="lme")
        self.url = url
        self.dataset_name = dataset_name

    def ingest(self) -> IngestResult:
        resp = requests.get(self.url, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"LME download failed: {resp.status_code}")

        filename = (
            f"lme_{self.dataset_name}_{int(datetime.utcnow().timestamp())}.csv"
        )
        raw_path = self._save_raw(resp.content, filename)
        checksum = self._checksum(raw_path)

        # try read as csv, but allow caller to handle other formats
        try:
            df = pd.read_csv(raw_path)
        except Exception:
            df = pd.DataFrame({"raw": [str(raw_path)]})

        result = IngestResult(
            df=df,
            raw_path=raw_path,
            checksum=checksum,
            retrieved_at=datetime.utcnow(),
            source_url=self.url,
            dataset=self.dataset_name,
        )
        self._write_manifest(result)
        return result
