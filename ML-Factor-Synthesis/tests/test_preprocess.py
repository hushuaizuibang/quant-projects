import numpy as np
import pandas as pd
import pytest

from data.preprocess import preprocess_factors
from factors.neutralize import neutralize


def test_preprocessing_is_cross_sectional_and_standardized():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), list("ABCDE")], names=["date", "symbol"]
    )
    frame = pd.DataFrame({"alpha": [1, 2, 3, 4, 100, 10, 11, 12, 13, 14]}, index=index)
    result = preprocess_factors(frame)
    by_date = result.groupby(level="date")["alpha"]
    assert np.allclose(by_date.mean(), 0, atol=1e-12)
    assert np.allclose(by_date.std(), 1, atol=1e-12)


def test_changing_future_date_does_not_change_past():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), list("ABCDE")], names=["date", "symbol"]
    )
    frame = pd.DataFrame({"alpha": np.arange(10, dtype=float)}, index=index)
    baseline = preprocess_factors(frame)
    frame.loc[(pd.Timestamp("2024-01-02"), "E"), "alpha"] = 1e9
    changed = preprocess_factors(frame)
    pd.testing.assert_series_equal(
        baseline.xs("2024-01-01")["alpha"], changed.xs("2024-01-01")["alpha"]
    )


def test_vectorized_joint_neutralization_matches_ols():
    date = pd.Timestamp("2024-01-31")
    symbols = list("ABCDEF")
    index = pd.MultiIndex.from_product([[date], symbols], names=["date", "symbol"])
    factors = pd.DataFrame({"alpha": [1.0, 2.0, 4.0, 3.0, 7.0, 8.0]}, index=index)
    market_cap = np.exp(np.array([1.0, 1.4, 1.9, 1.2, 1.7, 2.1]))
    metadata = pd.DataFrame(
        {
            "date": date,
            "symbol": symbols,
            "industry": ["X", "X", "X", "Y", "Y", "Y"],
            "market_cap": market_cap,
        }
    )
    result = neutralize(factors, metadata, "both")["alpha"].to_numpy()
    design = np.column_stack(
        [np.ones(6), np.array([0, 0, 0, 1, 1, 1], dtype=float), np.log(market_cap)]
    )
    expected = factors["alpha"].to_numpy() - design @ np.linalg.lstsq(
        design, factors["alpha"].to_numpy(), rcond=None
    )[0]
    assert np.allclose(result, expected)


def test_neutralization_rejects_missing_point_in_time_metadata():
    date = pd.Timestamp("2024-01-31")
    symbols = ["A", "B"]
    index = pd.MultiIndex.from_product([[date], symbols], names=["date", "symbol"])
    factors = pd.DataFrame({"alpha": [1.0, 2.0]}, index=index)
    metadata = pd.DataFrame(
        {"date": date, "symbol": symbols, "industry": "Unknown", "market_cap": np.nan}
    )
    with pytest.raises(ValueError, match="Industry neutralization requires"):
        neutralize(factors, metadata, "industry")
    with pytest.raises(ValueError, match="Size neutralization requires"):
        neutralize(factors, metadata, "size")
