from __future__ import annotations

import numpy as np
import pandas as pd


def market_proxy_returns(prices: pd.DataFrame) -> pd.Series:
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    return close.pct_change().mean(axis=1).dropna().rename("market_proxy")


def run_factor_timing_backtest(
    factor_returns: pd.DataFrame,
    timing_weights: pd.DataFrame,
    market_returns: pd.Series,
    transaction_cost: float,
    rebalance_freq: str,
    slippage: float = 0.0,
    max_rebalance_turnover: float | None = None,
) -> pd.DataFrame:
    result, weights = simulate_weight_strategy(
        factor_returns=factor_returns,
        target_weights=timing_weights,
        transaction_cost=transaction_cost,
        rebalance_freq=rebalance_freq,
        slippage=slippage,
        max_rebalance_turnover=max_rebalance_turnover,
    )
    common_index = result.index
    strategy = result["net_return"]
    turnover = result["turnover"]
    equal_weight = factor_returns.mean(axis=1)
    market = market_returns.reindex(common_index).fillna(0)

    return pd.DataFrame(
        {
            "timing_strategy": strategy,
            "equal_weight_factors": equal_weight,
            "market_proxy": market,
            "turnover": turnover,
        }
    )


def simulate_weight_strategy(
    factor_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost: float,
    rebalance_freq: str,
    slippage: float = 0.0,
    max_rebalance_turnover: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest factor weights with next-day execution and common cost rules.

    Turnover is the L1 change in factor weights.  Targets observed on day t are
    first tradable on day t+1, which prevents the day-t factor return from
    leaking into the position that earns that return.
    """
    common_index = factor_returns.index.intersection(target_weights.index).sort_values()
    returns = factor_returns.loc[common_index].fillna(0.0)
    targets = _normalise_weights(target_weights.loc[common_index], returns.columns)
    rebalance_targets = targets.groupby(pd.Grouper(freq=rebalance_freq)).tail(1)
    desired = rebalance_targets.reindex(common_index).ffill().shift(1)

    initial = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    actual = _apply_turnover_limit(desired, initial, max_rebalance_turnover)
    turnover = actual.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = (actual.iloc[0] - initial).abs().sum()
    turnover = turnover.fillna(0.0)

    gross = (actual * returns).sum(axis=1)
    trading_cost = turnover * (transaction_cost + slippage)
    result = pd.DataFrame(
        {
            "gross_return": gross,
            "net_return": gross - trading_cost,
            "turnover": turnover,
            "trading_cost": trading_cost,
        }
    )
    return result, actual


def _normalise_weights(weights: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    aligned = weights.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
    aligned = aligned.clip(lower=0.0)
    totals = aligned.sum(axis=1).replace(0.0, np.nan)
    normalised = aligned.div(totals, axis=0)
    fallback = pd.DataFrame(1.0 / len(columns), index=aligned.index, columns=columns)
    return normalised.fillna(fallback)


def _apply_turnover_limit(
    desired: pd.DataFrame,
    initial: pd.Series,
    max_turnover: float | None,
) -> pd.DataFrame:
    rows = []
    current = initial.astype(float).copy()
    previous_target = None
    for _, target in desired.iterrows():
        if target.isna().all():
            rows.append(current.copy())
            continue
        target = target.fillna(current)
        is_new_target = previous_target is None or not target.equals(previous_target)
        if is_new_target:
            delta = target - current
            requested = delta.abs().sum()
            if max_turnover is not None and requested > max_turnover:
                delta *= max_turnover / requested
            current = current + delta
            current /= current.sum()
            previous_target = target.copy()
        rows.append(current.copy())
    return pd.DataFrame(rows, index=desired.index, columns=desired.columns)
