
import pandas as pd

from copper_market_intelligence.features import external


def make_daily_index(start="2020-01-01", periods=10):
    return pd.date_range(start=start, periods=periods, freq="D")


def test_as_of_join_basic():
    left = pd.DataFrame({"date": pd.to_datetime(["2020-01-05", "2020-01-10"])})
    right = pd.DataFrame({
        "available_date": pd.to_datetime(["2020-01-01", "2020-01-07"]),
        "value": [1.0, 2.0],
    })
    merged = external.as_of_join(
        left,
        right,
        left_on="date",
        right_available_col="available_date",
        right_value_col="value",
    )
    assert merged.loc[0, "value"] == 1.0
    assert merged.loc[1, "value"] == 2.0


def test_as_of_join_future_not_used():
    left = pd.DataFrame({"date": pd.to_datetime(["2020-01-03"])})
    right = pd.DataFrame({
        "available_date": pd.to_datetime(["2020-01-05"]),
        "value": [9.0],
    })
    merged = external.as_of_join(
        left,
        right,
        left_on="date",
        right_available_col="available_date",
        right_value_col="value",
    )
    assert pd.isna(merged.loc[0, "value"])  # future unavailable


def test_as_of_join_exact_boundary():
    left = pd.DataFrame({"date": pd.to_datetime(["2020-01-05"])})
    right = pd.DataFrame({
        "available_date": pd.to_datetime(["2020-01-05"]),
        "value": [7.0],
    })
    merged = external.as_of_join(
        left,
        right,
        left_on="date",
        right_available_col="available_date",
        right_value_col="value",
    )
    assert merged.loc[0, "value"] == 7.0


def test_expand_monthly_as_of_respects_lag():
    monthly = pd.DataFrame({
        "obs_date": ["2020-01-01", "2020-02-01"],
        "value": [100, 110],
    })
    daily_index = pd.date_range("2020-01-15", periods=60, freq="D")
    out_default = external.expand_monthly_as_of(
        monthly, daily_index, publication_lag_months=1
    )
    # With 1-month lag, the Feb observation becomes available on 2020-03-01
    assert (
        pd.isna(out_default.loc["2020-02-15", "value"]) or out_default.loc[
            "2020-02-15", "value"
        ]
        == 100
    )
    out_no_lag = external.expand_monthly_as_of(
        monthly, daily_index, publication_lag_months=0
    )
    assert out_no_lag.loc["2020-02-15", "value"] == 110


def test_inventory_features_none_returns_none():
    assert external.inventory_features(None) is None


def test_inventory_features_computes_changes():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "stock": [10, 11, 12, 12, 13],
        }
    )
    out = external.inventory_features(df)
    assert "lme_pct_1" in out.columns
    assert out.shape[0] == 5


def test_feature_alignment_no_duplicates():
    daily = pd.DataFrame({"date": make_daily_index("2020-01-01", 10)})
    monthly = pd.DataFrame({"obs_date": ["2019-12-01"], "value": [1.0]})
    out = external.expand_monthly_as_of(
        monthly, daily["date"], publication_lag_months=1
    )
    assert len(out.index) == len(daily)
    assert out.index.is_unique
