"""Portfolio turnover constraints."""

from __future__ import annotations

import pandas as pd


def turnover(previous: pd.Series, target: pd.Series) -> float:
    universe = previous.index.union(target.index)
    return float(0.5 * (target.reindex(universe, fill_value=0) - previous.reindex(universe, fill_value=0)).abs().sum())


def cap_turnover(previous: pd.Series, target: pd.Series, maximum: float) -> tuple[pd.Series, float]:
    universe = previous.index.union(target.index)
    old = previous.reindex(universe, fill_value=0.0)
    desired = target.reindex(universe, fill_value=0.0)
    requested = turnover(old, desired)
    if requested <= maximum or requested == 0:
        return desired, requested
    fraction = maximum / requested
    constrained = old + fraction * (desired - old)
    return constrained, turnover(old, constrained)

