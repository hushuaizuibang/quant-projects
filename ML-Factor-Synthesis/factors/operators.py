"""Vectorized operators from Appendix A of 101 Formulaic Alphas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def by_symbol(series: pd.Series):
    return series.groupby(level="symbol", group_keys=False)


def delay(x: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).shift(days)


def delta(x: pd.Series, days: int) -> pd.Series:
    return x - delay(x, days)


def ts_sum(x: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).rolling(days, min_periods=days).sum().droplevel(0)


def ts_min(x: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).rolling(days, min_periods=days).min().droplevel(0)


def ts_max(x: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).rolling(days, min_periods=days).max().droplevel(0)


def stddev(x: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).rolling(days, min_periods=days).std(ddof=0).droplevel(0)


def correlation(x: pd.Series, y: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).apply(
        lambda group: group.rolling(days, min_periods=days).corr(y.reindex(group.index))
    )


def covariance(x: pd.Series, y: pd.Series, days: int) -> pd.Series:
    return by_symbol(x).apply(
        lambda group: group.rolling(days, min_periods=days).cov(y.reindex(group.index), ddof=0)
    )


def rank(x: pd.Series) -> pd.Series:
    return x.groupby(level="date").rank(pct=True)


def ts_rank(x: pd.Series, days: int) -> pd.Series:
    def last_rank(values: np.ndarray) -> float:
        return pd.Series(values).rank(pct=True).iloc[-1]

    return by_symbol(x).rolling(days, min_periods=days).apply(last_rank, raw=True).droplevel(0)


def safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    if isinstance(denominator, pd.Series):
        denominator = denominator.mask(denominator.abs() < 1e-12)
    return numerator / denominator
