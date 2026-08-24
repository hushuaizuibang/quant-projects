"""Leakage-aware monthly sample construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def month_end_mask(index: pd.MultiIndex) -> np.ndarray:
    dates = pd.Series(index.get_level_values("date"), index=index)
    periods = dates.dt.to_period("M")
    return dates.groupby(periods).transform("max").eq(dates).to_numpy()


def build_monthly_samples(
    factors: pd.DataFrame,
    market: pd.DataFrame,
    lookback_days: int = 60,
    forward_days: int = 20,
    target_mode: str = "raw",
) -> pd.DataFrame:
    """Flatten the previous 60 daily factor observations at each month end.

    Lag 0 is information available on the signal date. The target uses close at
    t+20 and is retained only for research evaluation, never as an input.
    """
    factors = factors.sort_index()
    monthly_index = factors.index[month_end_mask(factors.index)]
    features: list[pd.DataFrame] = []
    grouped = factors.groupby(level="symbol", group_keys=False)
    for lag in range(lookback_days):
        # Slice each lag before concatenation. Building the full daily
        # 20 x 60 matrix first is prohibitively wasteful for a 300-stock panel.
        shifted = grouped.shift(lag).reindex(monthly_index)
        shifted.columns = [f"{name}__lag_{lag:02d}" for name in factors.columns]
        features.append(shifted)
    matrix = pd.concat(features, axis=1)

    prices = market.sort_values(["symbol", "date"]).set_index(["date", "symbol"])["close"]
    target = prices.groupby(level="symbol").shift(-forward_days) / prices - 1
    trading_dates = pd.Series(
        prices.index.get_level_values("date"), index=prices.index, dtype="datetime64[ns]"
    )
    label_date = trading_dates.groupby(level="symbol").shift(-forward_days)
    matrix["target_return"] = target.reindex(matrix.index)
    matrix["target"] = transform_cross_sectional_target(matrix["target_return"], target_mode)
    matrix["_target_mode"] = target_mode
    matrix["_label_date"] = label_date.reindex(matrix.index)
    # Rolling correlations can be undefined in a constant window. Preserve these
    # local gaps: XGBoost handles them natively and the fallback median-imputes.
    feature_columns = [
        column
        for column in matrix
        if column not in {"target", "target_return"} and not column.startswith("_")
    ]
    matrix = matrix.dropna(subset=feature_columns, how="all")
    return matrix.sort_index()


def transform_cross_sectional_target(target: pd.Series, mode: str) -> pd.Series:
    """Create a model label while preserving raw returns for the backtest."""
    if mode == "raw":
        return target.rename("target")
    grouped = target.groupby(level="date")
    if mode == "demean":
        return (target - grouped.transform("mean")).rename("target")
    if mode == "rank":
        # Centered percentile ranks have a stable scale across changing universes.
        return (grouped.rank(method="average", pct=True) - 0.5).rename("target")
    raise ValueError("target_mode must be 'raw', 'demean', or 'rank'")


def time_split(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = samples.index.get_level_values("date")
    train = samples.loc[(dates.year >= 2018) & (dates.year <= 2022)]
    validation = samples.loc[dates.year == 2023]
    test = samples.loc[dates.year >= 2024]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected non-empty 2018-2022 train, 2023 validation, and 2024+ test sets")
    return train, validation, test


def purge_unobservable_labels(samples: pd.DataFrame, cutoff: pd.Timestamp, forward_days: int = 20) -> pd.DataFrame:
    """Keep rows whose complete forward label would have been known by cutoff."""
    if "_label_date" in samples:
        return samples.loc[samples["_label_date"].notna() & (samples["_label_date"] < cutoff)]
    conservative_gap = pd.offsets.BDay(forward_days)
    dates = samples.index.get_level_values("date")
    return samples.loc[dates + conservative_gap < cutoff]
