from __future__ import annotations

import numpy as np
import pandas as pd


def build_timing_scores(factor_returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    aligned_market = market_returns.reindex(factor_returns.index).fillna(0)
    scores = pd.DataFrame(index=factor_returns.index, columns=factor_returns.columns, dtype=float)

    factor_momentum = _zscore_ts(factor_returns.rolling(20).mean())
    factor_reversal = -_zscore_ts(factor_returns.rolling(5).sum())
    factor_vol_penalty = -_zscore_ts(factor_returns.rolling(20).std())
    market_trend = _zscore_series(aligned_market.rolling(60).mean()).reindex(factor_returns.index)
    market_vol = -_zscore_series(aligned_market.rolling(20).std()).reindex(factor_returns.index)
    dispersion = _factor_dispersion_score(factor_returns)

    for factor in factor_returns.columns:
        raw_score = (
            0.35 * factor_momentum[factor]
            + 0.15 * factor_reversal[factor]
            + 0.20 * factor_vol_penalty[factor]
            + 0.15 * market_trend
            + 0.10 * market_vol
            + 0.05 * dispersion
        )
        scores[factor] = raw_score

    return scores.clip(-3, 3).fillna(0)


def timing_weights(scores: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    positive = scores.clip(lower=0)
    fallback = pd.DataFrame(1 / scores.shape[1], index=scores.index, columns=scores.columns)
    weights = positive.div(positive.sum(axis=1).replace(0, np.nan), axis=0).fillna(fallback)
    weights = weights.clip(upper=max_weight)
    weights = weights.div(weights.sum(axis=1), axis=0)
    return weights.fillna(fallback)


def _zscore_ts(frame: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=20).mean()
    std = frame.rolling(window, min_periods=20).std()
    return (frame - mean) / std.replace(0, np.nan)


def _zscore_series(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window, min_periods=20).mean()
    std = series.rolling(window, min_periods=20).std()
    return (series - mean) / std.replace(0, np.nan)


def _factor_dispersion_score(factor_returns: pd.DataFrame) -> pd.Series:
    rolling_corr = factor_returns.rolling(60, min_periods=30).corr()
    values = []
    dates = []
    for date in factor_returns.index:
        try:
            matrix = rolling_corr.loc[date]
        except KeyError:
            values.append(np.nan)
            dates.append(date)
            continue
        upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
        values.append(-upper.stack().mean())
        dates.append(date)
    return _zscore_series(pd.Series(values, index=dates))
