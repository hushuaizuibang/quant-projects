import numpy as np
import pandas as pd

from backtest.constraints import cap_turnover
from backtest.metrics import moving_block_bootstrap, performance_metrics
from backtest.portfolio import run_backtest, single_factor_benchmarks


def test_turnover_is_capped():
    previous = pd.Series({"A": 1.0, "B": -1.0})
    target = pd.Series({"A": -1.0, "B": 1.0})
    constrained, used = cap_turnover(previous, target, 0.3)
    assert np.isclose(used, 0.3)
    assert np.isclose(constrained.abs().sum(), 1.4)


def test_backtest_charges_cost_and_respects_constraint():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    index = pd.MultiIndex.from_product([dates, list("ABCD")], names=["date", "symbol"])
    score = pd.Series([4, 3, 2, 1, 1, 2, 3, 4], index=index)
    returns = pd.Series(0.01, index=index)
    result = run_backtest(score, returns, quantile=0.25, commission=0.001, slippage=0, max_turnover=0.3)
    assert (result["turnover"] <= 0.3 + 1e-12).all()
    assert np.allclose(result["cost"], 0.0006)
    assert np.allclose(result["net_return"], result["gross_return"] - result["cost"])


def test_performance_metrics_are_finite_for_variable_returns():
    frame = pd.DataFrame(
        {"net_return": [0.02, -0.01, 0.03], "turnover": [0.2] * 3, "cost": [0.001] * 3}
    )
    metrics = performance_metrics(frame)
    assert metrics["annual_return"] > 0
    assert np.isfinite(metrics["sharpe"])
    assert metrics["max_drawdown"] <= 0


def test_moving_block_bootstrap_is_reproducible():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
    first = moving_block_bootstrap(returns, samples=20, seed=7, block_size=2)
    second = moving_block_bootstrap(returns, samples=20, seed=7, block_size=2)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["annual_return", "information_ratio", "max_drawdown"]


def test_single_factor_orientation_uses_pretest_history_only():
    dates = pd.to_datetime(
        ["2022-01-31", "2022-02-28", "2022-03-31", "2023-01-31", "2024-01-31"]
    )
    symbols = list("ABCDE")
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    signal = np.tile(np.arange(5, dtype=float), len(dates))
    target = signal.copy()
    frame = pd.DataFrame({"alpha__lag_00": signal, "target": target}, index=index)
    scores, report = single_factor_benchmarks(frame, 2024, forward_days=1)
    assert report.loc["alpha", "direction"] == 1
    baseline = scores["alpha"].copy()
    frame.loc[(pd.Timestamp("2024-01-31"), slice(None)), "target"] *= -1
    changed, changed_report = single_factor_benchmarks(frame, 2024, forward_days=1)
    assert changed_report.loc["alpha", "direction"] == 1
    pd.testing.assert_series_equal(baseline, changed["alpha"])
