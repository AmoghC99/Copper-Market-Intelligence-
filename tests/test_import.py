from importlib import import_module
from pathlib import Path


def test_package_importable():
    # ensure package imports and config attributes exist
    mod = import_module("copper_market_intelligence.config")
    assert hasattr(mod, "config")
    cfg = mod.config
    assert isinstance(cfg.BASE_DIR, Path)
    assert cfg.RAW_DIR.name == "raw"
    assert cfg.PROCESSED_DIR.name == "processed"
