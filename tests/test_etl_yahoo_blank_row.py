import pandas as pd

from copper_market_intelligence.etl.processing import normalize_yahoo


def test_normalize_yahoo_drops_fully_blank_row(tmp_path):
    # create a CSV with one valid row, one partially populated, and one fully blank
    p = tmp_path / "yahoo_test.csv"
    df = pd.DataFrame(
        [
            {
                "Date": "2020-01-01",
                "Open": 1.0,
                "High": 2.0,
                "Low": 0.5,
                "Close": 1.5,
                "Volume": 100,
            },
            {
                "Date": "2020-01-02",
                "Open": pd.NA,
                "High": pd.NA,
                "Low": pd.NA,
                "Close": pd.NA,
                "Volume": pd.NA,
            },
            {
                "Date": pd.NA,
                "Open": pd.NA,
                "High": pd.NA,
                "Low": pd.NA,
                "Close": pd.NA,
                "Volume": pd.NA,
            },
        ]
    )
    df.to_csv(p, index=False)

    canonical, stats = normalize_yahoo(p, return_stats=True)
    # one fully blank row removed
    assert stats["rows_in"] == 3
    assert stats["rows_removed"] == 1
    assert stats["rows_out"] == 2
    # partially populated row (has timestamp but null markets) should be preserved
    # We expect one row with timestamp 2020-01-02 and NaNs for markets
    assert canonical.shape[0] == 2
    # check that there is at least one row with NaN in market columns
    assert canonical["open"].isna().sum() >= 1
