import numpy as np
import pandas as pd

from factors.operators import correlation, delay, delta, rank, ts_sum


def panel_series() -> pd.Series:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), ["A", "B"]], names=["date", "symbol"]
    )
    return pd.Series(np.arange(10, dtype=float), index=index)


def test_delay_and_delta_stay_within_symbol():
    values = panel_series()
    shifted = delay(values, 1)
    assert np.isnan(shifted.loc[("2024-01-01", "A")])
    assert shifted.loc[("2024-01-02", "A")] == values.loc[("2024-01-01", "A")]
    assert delta(values, 1).loc[("2024-01-02", "B")] == 2


def test_rank_is_cross_sectional():
    ranked = rank(panel_series())
    assert ranked.loc[("2024-01-03", "A")] == 0.5
    assert ranked.loc[("2024-01-03", "B")] == 1.0


def test_rolling_operators_only_use_past_and_present():
    values = panel_series()
    summed = ts_sum(values, 2)
    assert summed.loc[("2024-01-02", "A")] == 2
    corr = correlation(values, values * 2, 3)
    assert corr.loc[("2024-01-03", "A")] == 1

