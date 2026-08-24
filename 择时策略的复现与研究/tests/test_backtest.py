import unittest

import numpy as np
import pandas as pd

from src.factor_timing.backtest import simulate_weight_strategy
from src.factor_timing.comparison import build_strategy_targets


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2024-01-01", periods=8)
        self.returns = pd.DataFrame(
            {"a": np.arange(8) / 100, "b": -np.arange(8) / 200},
            index=self.index,
        )

    def test_target_is_executed_one_day_later(self):
        targets = pd.DataFrame({"a": 1.0, "b": 0.0}, index=self.index)
        result, weights = simulate_weight_strategy(
            self.returns,
            targets,
            transaction_cost=0.0,
            rebalance_freq="D",
        )
        self.assertAlmostEqual(weights.iloc[0]["a"], 0.5)
        self.assertAlmostEqual(weights.iloc[1]["a"], 1.0)
        self.assertAlmostEqual(result.iloc[1]["gross_return"], self.returns.iloc[1]["a"])

    def test_equal_weight_has_no_meta_portfolio_turnover(self):
        targets = pd.DataFrame(0.5, index=self.index, columns=self.returns.columns)
        result, _ = simulate_weight_strategy(
            self.returns,
            targets,
            transaction_cost=0.001,
            slippage=0.001,
            rebalance_freq="W-FRI",
        )
        self.assertAlmostEqual(result["turnover"].sum(), 0.0)
        self.assertAlmostEqual(result["trading_cost"].sum(), 0.0)

    def test_all_comparison_targets_are_long_only_and_fully_invested(self):
        scores = pd.DataFrame(
            np.arange(16).reshape(8, 2),
            index=self.index,
            columns=self.returns.columns,
        )
        targets = build_strategy_targets(self.returns, scores, top_k=1)
        for target in targets.values():
            self.assertTrue((target >= 0).all().all())
            np.testing.assert_allclose(target.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
