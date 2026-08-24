from __future__ import annotations

import numpy as np
import pandas as pd


def performance_table(returns: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in returns.columns if c != "turnover"]
    rows = {}
    for col in cols:
        series = returns[col].dropna()
        rows[col] = {
            "annual_return": annual_return(series),
            "annual_volatility": series.std() * np.sqrt(252),
            "sharpe": sharpe(series),
            "max_drawdown": max_drawdown(series),
            "win_rate": (series > 0).mean(),
        }
    return pd.DataFrame(rows).T


def annual_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return (1 + series).prod() ** (252 / len(series)) - 1


def sharpe(series: pd.Series) -> float:
    std = series.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return series.mean() / std * np.sqrt(252)


def max_drawdown(series: pd.Series) -> float:
    nav = (1 + series).cumprod()
    drawdown = nav / nav.cummax() - 1
    return drawdown.min()
