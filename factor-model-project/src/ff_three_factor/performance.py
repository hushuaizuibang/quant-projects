from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .backtest import BacktestResult
from .factors import FactorModel


@dataclass
class PerformanceReport:
    overall_metrics: pd.Series
    yearly_metrics: pd.DataFrame
    sample_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame


def build_performance_report(
    result: BacktestResult, out_of_sample_start: str | None = None
) -> PerformanceReport:
    overall = _metrics(result.strategy_returns, result.benchmark_returns)
    yearly = pd.DataFrame(
        {
            year: _metrics(values, result.benchmark_returns.reindex(values.index))
            for year, values in result.strategy_returns.groupby(result.strategy_returns.index.year)
        }
    ).T
    sample_rows = {"Full sample": overall}
    if out_of_sample_start:
        split = pd.Timestamp(out_of_sample_start)
        for label, mask in {
            "In sample": result.strategy_returns.index < split,
            "Out of sample": result.strategy_returns.index >= split,
        }.items():
            values = result.strategy_returns.loc[mask]
            if not values.empty:
                sample_rows[label] = _metrics(values, result.benchmark_returns.reindex(values.index))
    samples = pd.DataFrame(sample_rows).T
    return PerformanceReport(overall, yearly, samples, transaction_cost_sensitivity(result))


def transaction_cost_sensitivity(
    result: BacktestResult,
    cost_rates: tuple[float, ...] = (0.0, 0.0005, 0.0013, 0.0020, 0.0050),
) -> pd.DataFrame:
    rows = {}
    for rate in cost_rates:
        returns = result.gross_returns - rate * result.turnover
        metrics = _metrics(returns, result.benchmark_returns)
        rows[f"{rate * 10000:.0f} bps"] = metrics[
            ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"]
        ]
    return pd.DataFrame(rows).T


def factor_ic_summary(factor_ic: pd.DataFrame) -> pd.DataFrame:
    mean = factor_ic.mean()
    std = factor_ic.std(ddof=1)
    count = factor_ic.count()
    return pd.DataFrame(
        {
            "Mean IC": mean,
            "IC Std": std,
            "ICIR": mean / std.replace(0, np.nan),
            "t-stat": mean / (std / np.sqrt(count)).replace(0, np.nan),
            "Positive IC Ratio": (factor_ic > 0).sum() / count.replace(0, np.nan),
            "Observations": count,
        }
    )


def factor_quantile_returns(model: FactorModel, quantiles: int = 5) -> pd.DataFrame:
    """Return next-month average returns for each factor-signal quantile."""
    records = []
    dates = list(model.monthly_returns.index)
    for signal_date, exposure in model.exposures.items():
        position = dates.index(signal_date)
        if position + 1 >= len(dates):
            continue
        holding_date = dates[position + 1]
        future = model.monthly_returns.loc[holding_date]
        for factor in exposure.columns:
            sample = pd.concat([exposure[factor].rename("signal"), future.rename("return")], axis=1).dropna()
            if sample["signal"].nunique() < quantiles:
                continue
            labels = pd.qcut(sample["signal"], quantiles, labels=False, duplicates="drop") + 1
            for bucket, value in sample.groupby(labels)["return"].mean().items():
                records.append(
                    {
                        "signal_date": signal_date,
                        "holding_date": holding_date,
                        "factor": factor,
                        "quantile": f"Q{int(bucket)}",
                        "return": value,
                    }
                )
    return pd.DataFrame(records)


def _metrics(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    returns = returns.dropna()
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.reindex(returns.index).rename("benchmark")],
        axis=1,
    ).dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (12 / len(returns)) - 1
    volatility = returns.std(ddof=1) * np.sqrt(12)
    sharpe = annualized_return / volatility if volatility else np.nan
    downside = returns.clip(upper=0).std(ddof=1) * np.sqrt(12)
    sortino = annualized_return / downside if downside else np.nan
    nav = (1 + returns).cumprod()

    alpha = beta = information_ratio = np.nan
    if len(aligned) >= 3:
        active = aligned["strategy"] - aligned["benchmark"]
        tracking_error = active.std(ddof=1) * np.sqrt(12)
        information_ratio = active.mean() * 12 / tracking_error if tracking_error else np.nan
        if aligned["benchmark"].std(ddof=1):
            regression = sm.OLS(
                aligned["strategy"], sm.add_constant(aligned["benchmark"])
            ).fit()
            alpha = regression.params["const"] * 12
            beta = regression.params["benchmark"]
    return pd.Series(
        {
            "Cumulative Return": cumulative_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": volatility,
            "Max Drawdown": _max_drawdown(nav),
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Information Ratio": information_ratio,
            "Win Rate": (returns > 0).mean(),
            "Alpha": alpha,
            "Beta": beta,
        }
    )


def _max_drawdown(nav: pd.Series) -> float:
    return (nav / nav.cummax() - 1).min()


def format_metrics(metrics: pd.Series, title: str) -> str:
    percentage_metrics = {
        "Cumulative Return", "Annualized Return", "Annualized Volatility",
        "Max Drawdown", "Win Rate", "Alpha",
    }
    lines = [title, "-" * len(title)]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.2%}" if key in percentage_metrics else f"{key}: {value:.4f}")
    return "\n".join(lines)
