from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_factor_timing_backtest
from .metrics import performance_table, sharpe
from .signals import timing_weights


@dataclass(frozen=True)
class TimingCandidate:
    name: str
    signal_weights: dict[str, float]
    max_factor_weight: float


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 504
    validation_days: int = 126
    test_days: int = 126
    step_days: int = 126


def run_walk_forward_validation(
    factor_returns: pd.DataFrame,
    market_returns: pd.Series,
    transaction_cost: float,
    rebalance_freq: str,
    output_dir: Path,
    config: WalkForwardConfig | None = None,
    slippage: float = 0.0,
    max_rebalance_turnover: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or WalkForwardConfig()
    components = build_leak_safe_signal_components(factor_returns, market_returns)
    candidates = default_candidates()
    folds = make_walk_forward_folds(factor_returns.index, config)

    fold_rows = []
    oos_returns = []
    for fold_id, fold in enumerate(folds, start=1):
        selected, diagnostics = select_candidate_for_fold(
            factor_returns=factor_returns,
            market_returns=market_returns,
            components=components,
            candidates=candidates,
            train_dates=fold["train"],
            validation_dates=fold["validation"],
            transaction_cost=transaction_cost,
            rebalance_freq=rebalance_freq,
            slippage=slippage,
            max_rebalance_turnover=max_rebalance_turnover,
        )
        test_result = evaluate_candidate(
            factor_returns=factor_returns,
            market_returns=market_returns,
            components=components,
            candidate=selected,
            dates=fold["test"],
            transaction_cost=transaction_cost,
            rebalance_freq=rebalance_freq,
            slippage=slippage,
            max_rebalance_turnover=max_rebalance_turnover,
        )
        selected_train = diagnostics.loc[selected.name, "train_sharpe"]
        selected_validation = diagnostics.loc[selected.name, "validation_sharpe"]
        selected_test = sharpe(test_result["timing_strategy"])
        fold_rows.append(
            {
                "fold": fold_id,
                "candidate": selected.name,
                "train_start": fold["train"][0],
                "train_end": fold["train"][-1],
                "validation_start": fold["validation"][0],
                "validation_end": fold["validation"][-1],
                "test_start": fold["test"][0],
                "test_end": fold["test"][-1],
                "train_sharpe": selected_train,
                "validation_sharpe": selected_validation,
                "test_sharpe": selected_test,
                "validation_turnover": diagnostics.loc[selected.name, "validation_turnover"],
            }
        )
        oos_returns.append(test_result.assign(fold=fold_id, candidate=selected.name))

    if not oos_returns:
        raise ValueError("Not enough data to build walk-forward folds.")

    wf_returns = pd.concat(oos_returns).sort_index()
    wf_metrics = performance_table(wf_returns[["timing_strategy", "equal_weight_factors", "market_proxy"]])
    fold_table = pd.DataFrame(fold_rows)
    regime_metrics = regime_performance(wf_returns, market_returns)

    output_dir.mkdir(parents=True, exist_ok=True)
    wf_returns.to_csv(output_dir / "walk_forward_returns.csv")
    wf_metrics.to_csv(output_dir / "walk_forward_metrics.csv")
    fold_table.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    regime_metrics.to_csv(output_dir / "walk_forward_regime_metrics.csv")
    write_walk_forward_summary(output_dir, config, wf_metrics, fold_table, regime_metrics)
    return wf_returns, wf_metrics, fold_table, regime_metrics


def build_leak_safe_signal_components(factor_returns: pd.DataFrame, market_returns: pd.Series) -> dict[str, pd.DataFrame | pd.Series]:
    # Day-t observations form a target executed on day t+1 by the backtester.
    lagged_factor_returns = factor_returns
    aligned_market = market_returns.reindex(factor_returns.index).fillna(0)

    factor_momentum = _zscore_ts(lagged_factor_returns.rolling(20, min_periods=10).mean())
    factor_reversal = -_zscore_ts(lagged_factor_returns.rolling(5, min_periods=3).sum())
    factor_vol_penalty = -_zscore_ts(lagged_factor_returns.rolling(20, min_periods=10).std())
    market_trend = _zscore_series(aligned_market.rolling(60, min_periods=20).mean()).reindex(factor_returns.index)
    market_vol = -_zscore_series(aligned_market.rolling(20, min_periods=10).std()).reindex(factor_returns.index)
    dispersion = _factor_dispersion_score(lagged_factor_returns)

    return {
        "factor_momentum": factor_momentum,
        "factor_reversal": factor_reversal,
        "factor_vol_penalty": factor_vol_penalty,
        "market_trend": market_trend,
        "market_vol": market_vol,
        "dispersion": dispersion,
    }


def default_candidates() -> list[TimingCandidate]:
    return [
        TimingCandidate(
            name="balanced",
            signal_weights={
                "factor_momentum": 0.35,
                "factor_reversal": 0.15,
                "factor_vol_penalty": 0.20,
                "market_trend": 0.15,
                "market_vol": 0.10,
                "dispersion": 0.05,
            },
            max_factor_weight=0.35,
        ),
        TimingCandidate(
            name="momentum_defensive",
            signal_weights={
                "factor_momentum": 0.45,
                "factor_reversal": 0.05,
                "factor_vol_penalty": 0.25,
                "market_trend": 0.10,
                "market_vol": 0.10,
                "dispersion": 0.05,
            },
            max_factor_weight=0.30,
        ),
        TimingCandidate(
            name="market_state",
            signal_weights={
                "factor_momentum": 0.20,
                "factor_reversal": 0.10,
                "factor_vol_penalty": 0.15,
                "market_trend": 0.30,
                "market_vol": 0.20,
                "dispersion": 0.05,
            },
            max_factor_weight=0.35,
        ),
        TimingCandidate(
            name="low_turnover",
            signal_weights={
                "factor_momentum": 0.30,
                "factor_reversal": 0.05,
                "factor_vol_penalty": 0.30,
                "market_trend": 0.15,
                "market_vol": 0.15,
                "dispersion": 0.05,
            },
            max_factor_weight=0.25,
        ),
    ]


def make_walk_forward_folds(index: pd.DatetimeIndex, config: WalkForwardConfig) -> list[dict[str, pd.DatetimeIndex]]:
    dates = pd.DatetimeIndex(index).sort_values()
    total = config.train_days + config.validation_days + config.test_days
    folds = []
    start = 0
    while start + total <= len(dates):
        train = dates[start : start + config.train_days]
        validation = dates[start + config.train_days : start + config.train_days + config.validation_days]
        test = dates[start + config.train_days + config.validation_days : start + total]
        folds.append({"train": train, "validation": validation, "test": test})
        start += config.step_days
    return folds


def select_candidate_for_fold(
    factor_returns: pd.DataFrame,
    market_returns: pd.Series,
    components: dict[str, pd.DataFrame | pd.Series],
    candidates: list[TimingCandidate],
    train_dates: pd.DatetimeIndex,
    validation_dates: pd.DatetimeIndex,
    transaction_cost: float,
    rebalance_freq: str,
    slippage: float = 0.0,
    max_rebalance_turnover: float | None = None,
) -> tuple[TimingCandidate, pd.DataFrame]:
    rows = []
    for candidate in candidates:
        train_result = evaluate_candidate(
            factor_returns, market_returns, components, candidate, train_dates, transaction_cost, rebalance_freq,
            slippage, max_rebalance_turnover
        )
        validation_result = evaluate_candidate(
            factor_returns, market_returns, components, candidate, validation_dates, transaction_cost, rebalance_freq,
            slippage, max_rebalance_turnover
        )
        rows.append(
            {
                "candidate": candidate.name,
                "train_sharpe": sharpe(train_result["timing_strategy"]),
                "validation_sharpe": sharpe(validation_result["timing_strategy"]),
                "validation_turnover": validation_result["turnover"].mean(),
            }
        )
    diagnostics = pd.DataFrame(rows).set_index("candidate")
    selected_name = diagnostics.sort_values(["validation_sharpe", "train_sharpe"], ascending=False).index[0]
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    return selected, diagnostics


def evaluate_candidate(
    factor_returns: pd.DataFrame,
    market_returns: pd.Series,
    components: dict[str, pd.DataFrame | pd.Series],
    candidate: TimingCandidate,
    dates: pd.DatetimeIndex,
    transaction_cost: float,
    rebalance_freq: str,
    slippage: float = 0.0,
    max_rebalance_turnover: float | None = None,
) -> pd.DataFrame:
    scores = candidate_scores(components, candidate).reindex(factor_returns.index)
    weights = timing_weights(scores, candidate.max_factor_weight)
    full_result = run_factor_timing_backtest(
        factor_returns=factor_returns,
        timing_weights=weights,
        market_returns=market_returns,
        transaction_cost=transaction_cost,
        rebalance_freq=rebalance_freq,
        slippage=slippage,
        max_rebalance_turnover=max_rebalance_turnover,
    )
    return full_result.loc[full_result.index.intersection(dates)]


def candidate_scores(components: dict[str, pd.DataFrame | pd.Series], candidate: TimingCandidate) -> pd.DataFrame:
    first_frame = next(value for value in components.values() if isinstance(value, pd.DataFrame))
    scores = pd.DataFrame(0.0, index=first_frame.index, columns=first_frame.columns)
    for name, weight in candidate.signal_weights.items():
        component = components[name]
        if isinstance(component, pd.Series):
            expanded = pd.DataFrame(
                np.repeat(component.reindex(scores.index).fillna(0).to_numpy()[:, None], scores.shape[1], axis=1),
                index=scores.index,
                columns=scores.columns,
            )
            scores = scores.add(expanded * weight, fill_value=0)
        else:
            scores = scores.add(component.reindex_like(scores).fillna(0) * weight, fill_value=0)
    return scores.clip(-3, 3).fillna(0)


def regime_performance(wf_returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    market = market_returns.reindex(wf_returns.index).fillna(0)
    trend_60 = market.rolling(60, min_periods=20).sum()
    regimes = pd.Series("sideways", index=wf_returns.index)
    regimes[trend_60 >= trend_60.quantile(0.67)] = "bull"
    regimes[trend_60 <= trend_60.quantile(0.33)] = "bear"

    tables = []
    for regime in ["bull", "sideways", "bear"]:
        subset = wf_returns.loc[regimes == regime, ["timing_strategy", "equal_weight_factors", "market_proxy"]]
        if subset.empty:
            continue
        table = performance_table(subset)
        table.insert(0, "regime", regime)
        tables.append(table)
    return pd.concat(tables) if tables else pd.DataFrame()


def write_walk_forward_summary(
    output_dir: Path,
    config: WalkForwardConfig,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    regime_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Walk-Forward 验证结果",
        "",
        f"- 训练窗口：{config.train_days} 个交易日",
        f"- 验证窗口：{config.validation_days} 个交易日",
        f"- 测试窗口：{config.test_days} 个交易日",
        f"- 滚动步长：{config.step_days} 个交易日",
        f"- 折数：{len(folds)}",
        "- 防泄漏处理：t 日收盘后生成目标权重，回测器在 t+1 日执行；候选规则只用验证窗口选择，最终指标只拼接测试窗口。",
        "",
        "## 样本外指标",
        "",
        _markdown_table(metrics.round(4)),
        "",
        "## 每折选择",
        "",
        _markdown_table(folds),
    ]
    if not regime_metrics.empty:
        lines.extend(["", "## 市场状态分段", "", _markdown_table(regime_metrics.round(4))])
    (output_dir / "walk_forward_summary.md").write_text("\n".join(lines), encoding="utf-8")


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


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = ["index"] + list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for index, row in frame.iterrows():
        values = [str(index)] + [_format_value(value) for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)
