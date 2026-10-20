"""USGS adapter placeholder: downloads a CSV from a provided URL.

USGS publishes many reports; specific dataset URLs should be set in the
registry. This adapter accepts a direct CSV URL and will download it.
"""
from datetime import datetime

import pandas as pd
import requests

from .base import BaseIngestor, IngestResult


class USGSIngestor(BaseIngestor):
    def __init__(self, url: str, dataset_name: str):
        super().__init__(source_name="usgs")
        self.url = url
        self.dataset_name = dataset_name

    def ingest(self) -> IngestResult:
        resp = requests.get(self.url, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"USGS download failed: {resp.status_code}")
        filename = (
            f"usgs_{self.dataset_name}_{int(datetime.utcnow().timestamp())}.csv"
        )
        raw_path = self._save_raw(resp.content, filename)
        checksum = self._checksum(raw_path)
        df = pd.read_csv(raw_path)

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
