import numpy as np
import pandas as pd
import pytest

from factors.feature_pipeline import build_monthly_samples, time_split, transform_cross_sectional_target
from model.train import walk_forward_split


def test_feature_lags_and_forward_target_are_aligned():
    dates = pd.bdate_range("2023-01-02", "2023-04-28")
    symbols = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    factors = pd.DataFrame({"alpha": np.arange(len(index), dtype=float)}, index=index)
    market = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "symbol": symbols * len(dates),
            "close": np.repeat(np.arange(1, len(dates) + 1, dtype=float), 2),
        }
    )
    samples = build_monthly_samples(factors, market, lookback_days=3, forward_days=2)
    date, symbol = samples.index[0]
    assert samples.loc[(date, symbol), "alpha__lag_00"] == factors.loc[(date, symbol), "alpha"]
    prior = dates[dates.get_loc(date) - 2]
    assert samples.loc[(date, symbol), "alpha__lag_02"] == factors.loc[(prior, symbol), "alpha"]
    position = dates.get_loc(date)
    expected = (position + 3) / (position + 1) - 1
    assert np.isclose(samples.loc[(date, symbol), "target"], expected)
    assert samples.loc[(date, symbol), "_label_date"] == dates[position + 2]


def test_future_factor_mutation_does_not_change_earlier_sample():
    dates = pd.bdate_range("2023-01-02", "2023-04-28")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    factors = pd.DataFrame({"alpha": np.arange(len(index), dtype=float)}, index=index)
    market = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * len(dates), "close": 10.0})
    before = build_monthly_samples(factors, market, 3, 2)
    cutoff = before.index.get_level_values("date")[0]
    factors.loc[factors.index.get_level_values("date") > cutoff, "alpha"] = 999999
    after = build_monthly_samples(factors, market, 3, 2)
    pd.testing.assert_series_equal(before.loc[(cutoff, "A")], after.loc[(cutoff, "A")])


def test_partial_rolling_gaps_do_not_delete_a_sample():
    dates = pd.bdate_range("2023-01-02", "2023-03-31")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    factors = pd.DataFrame({"available": 1.0, "sometimes_missing": np.nan}, index=index)
    factors.loc[(slice(None), "A"), "sometimes_missing"] = 2.0
    market = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * len(dates), "close": 10.0})
    samples = build_monthly_samples(factors, market, 3, 2)
    assert (samples.index.get_level_values("symbol") == "B").any()


def test_time_split_excludes_warmup_history():
    dates = pd.to_datetime(["2017-12-29", "2018-01-31", "2022-12-30", "2023-12-29", "2024-01-31"])
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    frame = pd.DataFrame({"feature": 1.0, "target": 0.01}, index=index)
    train, validation, test = time_split(frame)
    assert train.index.get_level_values("date").min().year == 2018
    assert validation.index.get_level_values("date").year.unique().tolist() == [2023]
    assert test.index.get_level_values("date").year.unique().tolist() == [2024]


def test_walk_forward_split_purges_unobservable_year_end_labels():
    dates = pd.to_datetime(
        ["2018-01-31", "2022-12-30", "2023-10-31", "2023-11-30", "2023-12-29", "2024-01-31"]
    )
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    frame = pd.DataFrame({"feature": 1.0, "target": 0.01}, index=index)
    train, validation, test = walk_forward_split(frame, 2024, forward_days=20)
    assert train.index.get_level_values("date").max().year == 2022
    assert validation.index.get_level_values("date").max() == pd.Timestamp("2023-11-30")
    assert test.index.get_level_values("date").unique().tolist() == [pd.Timestamp("2024-01-31")]


def test_walk_forward_split_requires_prior_validation_year():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2018-01-31", "2024-01-31"]), ["A"]], names=["date", "symbol"]
    )
    frame = pd.DataFrame({"feature": 1.0, "target": 0.01}, index=index)
    with pytest.raises(ValueError, match="Insufficient data"):
        walk_forward_split(frame, 2024)


def test_walk_forward_uses_exact_label_availability_date():
    dates = pd.to_datetime(["2018-01-31", "2023-10-31", "2023-11-30", "2023-12-01", "2024-01-31"])
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    frame = pd.DataFrame({"feature": 1.0, "target": 0.01}, index=index)
    frame["_label_date"] = pd.to_datetime(
        ["2018-03-01", "2023-11-28", "2023-12-29", "2024-01-02", "2024-02-29"]
    )
    _, validation, _ = walk_forward_split(frame, 2024, forward_days=20)
    assert pd.Timestamp("2023-11-30") in validation.index.get_level_values("date")
    assert pd.Timestamp("2023-12-01") not in validation.index.get_level_values("date")


def test_cross_sectional_rank_target_preserves_raw_return_order():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2023-01-31")], ["A", "B", "C"]], names=["date", "symbol"]
    )
    raw = pd.Series([-0.1, 0.0, 0.2], index=index, name="target_return")
    ranked = transform_cross_sectional_target(raw, "rank")
    assert ranked.tolist() == pytest.approx([-1 / 6, 1 / 6, 0.5])
    assert raw.tolist() == [-0.1, 0.0, 0.2]
