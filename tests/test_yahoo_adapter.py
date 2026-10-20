from pathlib import Path

import pandas as pd

from copper_market_intelligence.data.yahoo import YahooIngestor


def test_yahoo_adapter_reads_fixture(tmp_path, monkeypatch):
    # point RAW_DIR to tmp_path to avoid writing in repo
    from copper_market_intelligence import config as config_module
    from copper_market_intelligence.config import Config

    new_cfg = Config(BASE_DIR=tmp_path)
    monkeypatch.setattr(config_module, "config", new_cfg)
    fixture = Path("tests/data_fixtures/sample_yahoo.csv")

    # monkeypatch requests.get to return fixture content
    class DummyResp:
        status_code = 200

        def __init__(self, content):
            self.content = content


    def fake_get(url, timeout=30):
        return DummyResp(fixture.read_bytes())


    import copper_market_intelligence.data.yahoo as yahoo_mod

    yahoo_mod.requests.get = fake_get

    ing = YahooIngestor(ticker="HG=F")
    res = ing.ingest()
    assert isinstance(res.df, pd.DataFrame)
    assert "Date" in res.df.columns
    # check numeric OHLC and Volume parse
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in res.df.columns:
            assert pd.to_numeric(res.df[col], errors="coerce").notna().any()
    assert res.checksum is not None
    assert res.raw_path.exists()
    assert res.raw_path.suffix == ".csv"


def test_yahoo_adapter_quality_checks(tmp_path, monkeypatch):
    from copper_market_intelligence import config as config_module
    from copper_market_intelligence.config import Config

    new_cfg = Config(BASE_DIR=tmp_path)
    monkeypatch.setattr(config_module, "config", new_cfg)
    fixture = Path("tests/data_fixtures/sample_yahoo.csv")

    class DummyResp:
        status_code = 200

        def __init__(self, content):
            self.content = content


    def fake_get(url, timeout=30):
        return DummyResp(fixture.read_bytes())


    import copper_market_intelligence.data.yahoo as yahoo_mod

    yahoo_mod.requests.get = fake_get

    ing = YahooIngestor(ticker="HG=F")
    res = ing.ingest()
    df = res.df
    # dates parse
    assert pd.to_datetime(df["Date"], errors="coerce").notna().any()
    # duplicates
    assert df["Date"].duplicated().sum() == 0
    # metadata manifest exists
    mpath = res.raw_path.with_suffix(".manifest.json")
    assert mpath.exists()


def test_yahoo_adapter_yfinance_parsing(monkeypatch, tmp_path):
    """Simulate yfinance download by monkeypatching yf.download"""
    try:
        from copper_market_intelligence.data.yahoo import yf as yf_mod
    except Exception:
        yf_mod = None

    if yf_mod is None:
        # yfinance optional not installed; simulate module for parser
        class DummyDF:
            pass

        class DummyYF:
            @staticmethod
            def download(ticker, period, interval, progress):
                import pandas as pd

                df = pd.DataFrame({
                    "Date": ["2021-01-01", "2021-01-02"],
                    "Open": [4.0, 4.05],
                    "High": [4.1, 4.2],
                    "Low": [3.95, 4.0],
                    "Close": [4.05, 4.15],
                    "Adj Close": [4.05, 4.15],
                    "Volume": [1000, 1100],
                })
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                return df

        monkeypatch.setitem(__import__("sys").modules, "yfinance", DummyYF())
        # reload module to pick up patched yf
        import importlib

        import copper_market_intelligence.data.yahoo as yahoo_mod

        importlib.reload(yahoo_mod)
        ing = yahoo_mod.YahooIngestor("HG=F")
        res = ing.ingest()
        df = res.df
        assert "Date" in df.columns or df.index.name == "Date"
        assert df.shape[0] == 2
    else:
        # yfinance present; monkeypatch download to return small df
        def fake_download(ticker, period, interval, progress):
            import pandas as pd

            df = pd.DataFrame({
                "Date": ["2021-01-01", "2021-01-02"],
                "Open": [4.0, 4.05],
                "High": [4.1, 4.2],
                "Low": [3.95, 4.0],
                "Close": [4.05, 4.15],
                "Adj Close": [4.05, 4.15],
                "Volume": [1000, 1100],
            })
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            return df

        monkeypatch.setattr(yf_mod, "download", fake_download)
        import importlib

        import copper_market_intelligence.data.yahoo as yahoo_mod

        importlib.reload(yahoo_mod)
        ing = yahoo_mod.YahooIngestor("HG=F")
        res = ing.ingest()
        df = res.df
        assert df.shape[0] == 2
