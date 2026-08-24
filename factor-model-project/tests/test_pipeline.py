from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.ff_three_factor.backtest import _construct_portfolio
from src.ff_three_factor.config import BacktestConfig
from src.ff_three_factor.factors import preprocess_factor


class FactorProcessingTests(unittest.TestCase):
    def test_neutralization_removes_industry_and_size_exposure(self) -> None:
        rng = np.random.default_rng(7)
        codes = pd.Index([f"S{i:03d}" for i in range(80)])
        industry = pd.Series(np.repeat(["A", "B", "C", "D"], 20), index=codes)
        log_cap = pd.Series(rng.normal(20, 1, len(codes)), index=codes)
        values = 2.5 * log_cap + industry.map({"A": -3, "B": 1, "C": 2, "D": 5})
        values += pd.Series(rng.normal(0, 0.3, len(codes)), index=codes)

        result = preprocess_factor(values, industry, log_cap)

        self.assertAlmostEqual(float(result.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(result.std(ddof=0)), 1.0, places=10)
        self.assertLess(abs(float(result.corr(log_cap))), 1e-10)
        self.assertLess(result.groupby(industry).mean().abs().max(), 1e-10)


class PortfolioConstraintTests(unittest.TestCase):
    def test_optimizer_respects_weight_industry_and_size_limits(self) -> None:
        rng = np.random.default_rng(11)
        codes = pd.Index([f"S{i:03d}" for i in range(60)])
        industry = pd.Series(np.repeat(["A", "B", "C", "D", "E", "F"], 10), index=codes)
        score = pd.Series(rng.normal(size=len(codes)), index=codes)
        size = pd.Series(rng.normal(size=len(codes)), index=codes)
        covariance = pd.DataFrame(np.eye(len(codes)) * 0.03, index=codes, columns=codes)
        previous = pd.Series(1 / len(codes), index=codes)
        config = BacktestConfig(
            max_stock_weight=0.10,
            max_industry_deviation=0.05,
            max_size_exposure=0.15,
        )

        weights, status = _construct_portfolio(
            score, size, industry, covariance, previous, config
        )

        self.assertTrue(status.startswith("optimized"))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertLessEqual(float(weights.max()), config.max_stock_weight + 1e-8)
        active = weights.groupby(industry).sum() - industry.value_counts(normalize=True)
        self.assertLessEqual(float(active.abs().max()), config.max_industry_deviation + 1e-7)
        self.assertLessEqual(abs(float(weights @ size)), config.max_size_exposure + 1e-7)


if __name__ == "__main__":
    unittest.main()
