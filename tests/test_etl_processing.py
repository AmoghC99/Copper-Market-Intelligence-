from pathlib import Path

from copper_market_intelligence.etl.processing import normalize_fred, normalize_yahoo


def test_normalize_yahoo_fixture(tmp_path):
    # use fixture CSV
    src = Path("tests/data_fixtures/sample_yahoo.csv")
    out = normalize_yahoo(src)
    assert "timestamp" in out.columns
    assert "open" in out.columns
    assert "close" in out.columns
    assert out.shape[0] > 0


def test_normalize_fred_fixture():
    src = Path("tests/data_fixtures/sample_fred.csv")
    out = normalize_fred(src)
    assert "timestamp" in out.columns
    assert "copper_benchmark_usd_t" in out.columns
    assert out.shape[0] > 0
