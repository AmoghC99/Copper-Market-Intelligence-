"""Data ingestion modules and canonical readers/writers.

Adapters for public data sources are implemented in separate modules.
"""

from .base import BaseIngestor

__all__ = ["BaseIngestor"]
