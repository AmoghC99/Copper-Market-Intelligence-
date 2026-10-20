from copper_market_intelligence.data.fred import FredIngestor


def test_fred_adapter_requires_api_key(monkeypatch):
    # ensure no FRED_API_KEY in env
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    ing = FredIngestor(series_id="TEST_SERIES")
    try:
        ing.ingest()
        assert False, "Expected RuntimeError when no API key"
    except RuntimeError as e:
        assert "FRED API requires an API key" in str(e)


