import json
from typing import Any, Dict

from ..config import config

MANIFEST_PATH = config.BASE_DIR / "data" / "raw" / "manifest.json"


def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(man: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(man, indent=2))
