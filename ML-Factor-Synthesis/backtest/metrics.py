"""Performance, IC, quantile, and bootstrap statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(result: pd.DataFrame, periods_per_year: int = 12) -> dict[str, float]:
    returns = result["net_return"].dropna()
    if returns.empty:
        return {name: np.nan for name in ("annual_return", "sharpe", "max_drawdown", "information_ratio")}
    wealth = (1 + returns).cumprod()
    years = len(returns) / periods_per_year
    annual = float(wealth.iloc[-1] ** (1 / years) - 1) if wealth.iloc[-1] > 0 else -1.0
    volatility = returns.std(ddof=1)
    sharpe = float(np.sqrt(periods_per_year) * returns.mean() / volatility) if volatility > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1
    return {
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "information_ratio": sharpe,
        "average_turnover": float(result["turnover"].mean()),
        "win_rate": float((returns > 0).mean()),
        "average_return": float(returns.mean()),
        "total_cost": float(result["cost"].sum()),
    }


def monthly_ic(score: pd.Series, target: pd.Series) -> pd.Series:
    joined = pd.concat([score.rename("score"), target.rename("target")], axis=1)
    return joined.groupby(level="date").apply(lambda x: x["score"].corr(x["target"], method="spearman"))


def bootstrap_confidence_interval(
    returns: pd.Series,
    samples: int = 500,
    seed: int = 42,
    periods_per_year: int = 12,
    block_size: int = 3,
) -> dict[str, float]:
    distribution = moving_block_bootstrap(returns, samples, seed, periods_per_year, block_size)
    if distribution.empty:
        return {"annual_return_ci_low": np.nan, "annual_return_ci_high": np.nan, "ir_ci_low": np.nan, "ir_ci_high": np.nan}
    return {
        "annual_return_ci_low": float(distribution["annual_return"].quantile(0.025)),
        "annual_return_ci_high": float(distribution["annual_return"].quantile(0.975)),
        "ir_ci_low": float(distribution["information_ratio"].quantile(0.025)),
        "ir_ci_high": float(distribution["information_ratio"].quantile(0.975)),
    }


def moving_block_bootstrap(
    returns: pd.Series,
    samples: int = 500,
    seed: int = 42,
    periods_per_year: int = 12,
    block_size: int = 3,
) -> pd.DataFrame:
    """Circular moving-block bootstrap, preserving short-horizon dependence."""
    clean = returns.dropna().to_numpy()
    if len(clean) < 2:
        return pd.DataFrame(columns=["annual_return", "information_ratio", "max_drawdown"])
    if block_size < 1:
        raise ValueError("block_size must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    blocks_needed = int(np.ceil(len(clean) / block_size))
    offsets = np.arange(block_size)
    for sample_id in range(samples):
        starts = rng.integers(0, len(clean), size=blocks_needed)
        indices = ((starts[:, None] + offsets) % len(clean)).ravel()[: len(clean)]
        draw = clean[indices]
        terminal = np.prod(1 + draw)
        annualized = terminal ** (periods_per_year / len(draw)) - 1 if terminal > 0 else -1.0
        std = draw.std(ddof=1)
        wealth = np.cumprod(1 + draw)
        max_drawdown = np.min(wealth / np.maximum.accumulate(wealth) - 1)
        rows.append(
            {
                "sample": sample_id,
                "annual_return": annualized,
                "information_ratio": np.sqrt(periods_per_year) * draw.mean() / std if std else np.nan,
                "max_drawdown": max_drawdown,
            }
        )
    return pd.DataFrame(rows).set_index("sample")


def quantile_returns(score: pd.Series, target: pd.Series, groups: int = 5) -> pd.DataFrame:
    joined = pd.concat([score.rename("score"), target.rename("target")], axis=1).dropna()

    def assign(day: pd.DataFrame) -> pd.DataFrame:
        ranked = day["score"].rank(method="first")
        day = day.copy()
        day["quantile"] = pd.qcut(ranked, groups, labels=False, duplicates="drop") + 1
        return day

    assigned = joined.groupby(level="date", group_keys=False).apply(assign)
    return assigned.groupby([assigned.index.get_level_values("date"), "quantile"])["target"].mean().unstack()
