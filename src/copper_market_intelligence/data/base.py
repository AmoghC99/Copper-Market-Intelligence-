import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import config


@dataclass
class IngestResult:
    df: pd.DataFrame
    raw_path: Path
    checksum: str
    retrieved_at: datetime
    source_url: str
    dataset: str


class BaseIngestor:
    """Base ingestion helper providing raw save and manifest writing."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.raw_dir = config.RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _save_raw(self, content: bytes, filename: str) -> Path:
        path = self.raw_dir / filename
        path.write_bytes(content)
        return path

    def _checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_manifest(self, result: IngestResult) -> Path:
        manifest = {
            "source": self.source_name,
            "dataset": result.dataset,
            "source_url": result.source_url,
            "retrieved_at": result.retrieved_at.astimezone(timezone.utc).isoformat(),
            "raw_path": str(result.raw_path.as_posix()),
            "checksum": result.checksum,
        }
        manifest_path = result.raw_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    def ingest(self) -> IngestResult:
        raise NotImplementedError()
