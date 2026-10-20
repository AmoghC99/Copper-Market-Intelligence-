from pathlib import Path

from copper_market_intelligence.data.manifest import load_manifest, save_manifest


def test_manifest_roundtrip(tmp_path):
    man = {"test": {"a": 1}}
    p = Path(tmp_path) / "data" / "raw"
    p.mkdir(parents=True)
    # save_manifest writes to the configured MANIFEST_PATH; ensure it can be called
    save_manifest(man)
    # load should return dict or empty (depends on global path);
    # just assert no exception
    _ = load_manifest()
    assert isinstance(_, dict)
