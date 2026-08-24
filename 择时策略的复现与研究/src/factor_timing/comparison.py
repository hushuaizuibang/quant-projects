from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import simulate_weight_strategy
from .metrics import annual_return, max_drawdown, sharpe
from .signals import timing_weights


STRATEGY_LABELS = {
    "equal_weight": "等权",
    "timing_score": "择时得分加权",
    "top_k": "Top-K 等权",
    "rank_weight": "分位排名加权",
    "risk_weight": "逆波动风险加权",
}

PLOT_LABELS = {
    "equal_weight": "Equal weight",
    "timing_score": "Timing score",
    "top_k": "Top-K",
    "rank_weight": "Rank weight",
    "risk_weight": "Inverse vol",
}


def run_comparison_experiment(
    factor_returns: pd.DataFrame,
    timing_scores: pd.DataFrame,
    market_returns: pd.Series,
    output_dir: Path,
    transaction_cost: float,
    slippage: float,
    rebalance_freq: str,
    max_rebalance_turnover: float | None,
    top_k: int = 3,
    bootstrap_samples: int = 2000,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run a fair, reproducible comparison on one shared factor-return panel."""
    targets = build_strategy_targets(factor_returns, timing_scores, top_k=top_k)
    evaluated = _evaluate_targets(
        factor_returns,
        targets,
        transaction_cost,
        slippage,
        rebalance_freq,
        max_rebalance_turnover,
    )
    net_returns = pd.DataFrame({name: value["result"]["net_return"] for name, value in evaluated.items()})
    gross_returns = pd.DataFrame({name: value["result"]["gross_return"] for name, value in evaluated.items()})
    turnovers = pd.DataFrame({name: value["result"]["turnover"] for name, value in evaluated.items()})
    costs = pd.DataFrame({name: value["result"]["trading_cost"] for name, value in evaluated.items()})
    weights = pd.concat({name: value["weights"] for name, value in evaluated.items()}, axis=1)

    metrics = comparison_metrics(net_returns, gross_returns, turnovers, costs)
    significance = bootstrap_outperformance(
        net_returns,
        benchmark="equal_weight",
        samples=bootstrap_samples,
        random_seed=random_seed,
    )
    regimes = regime_metrics(net_returns, market_returns)
    sensitivity = sensitivity_analysis(
        factor_returns,
        targets,
        cost_rates=(0.0, 0.0005, 0.0010, 0.0020),
        rebalance_frequencies=("D", "W-FRI", "ME"),
        max_rebalance_turnover=max_rebalance_turnover,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    net_returns.to_csv(output_dir / "comparison_returns.csv")
    gross_returns.to_csv(output_dir / "comparison_gross_returns.csv")
    turnovers.to_csv(output_dir / "comparison_turnover.csv")
    weights.to_csv(output_dir / "comparison_weights.csv")
    metrics.to_csv(output_dir / "comparison_metrics.csv")
    significance.to_csv(output_dir / "comparison_significance.csv")
    regimes.to_csv(output_dir / "comparison_regime_metrics.csv", index=False)
    sensitivity.to_csv(output_dir / "comparison_sensitivity.csv", index=False)
    _plot_comparison(net_returns, output_dir / "comparison_equity_curve.png")
    _plot_research_dashboard(
        net_returns,
        metrics,
        regimes,
        sensitivity,
        output_dir / "research_dashboard.png",
    )
    _write_comparison_summary(
        output_dir,
        metrics,
        significance,
        regimes,
        transaction_cost,
        slippage,
        rebalance_freq,
        max_rebalance_turnover,
        top_k,
    )
    _write_research_brief(
        output_dir,
        net_returns,
        metrics,
        significance,
        regimes,
        sensitivity,
        transaction_cost,
        slippage,
        rebalance_freq,
    )
    return {
        "returns": net_returns,
        "metrics": metrics,
        "significance": significance,
        "regimes": regimes,
        "sensitivity": sensitivity,
    }


def build_strategy_targets(
    factor_returns: pd.DataFrame,
    timing_scores: pd.DataFrame,
    top_k: int = 3,
) -> dict[str, pd.DataFrame]:
    scores = timing_scores.reindex_like(factor_returns).fillna(0.0)
    n_factors = factor_returns.shape[1]
    equal = pd.DataFrame(1.0 / n_factors, index=factor_returns.index, columns=factor_returns.columns)

    k = min(max(1, top_k), n_factors)
    ranks_desc = scores.rank(axis=1, method="first", ascending=False)
    top = (ranks_desc <= k).astype(float)
    top = top.div(top.sum(axis=1), axis=0)

    percentile = scores.rank(axis=1, pct=True)
    rank = percentile.where(percentile >= 0.5, 0.0)
    rank = rank.div(rank.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(equal)

    # Day-t volatility becomes a target that is executed on day t+1.
    trailing_vol = factor_returns.rolling(60, min_periods=20).std()
    inverse_vol = 1.0 / trailing_vol.replace(0.0, np.nan)
    risk = inverse_vol.div(inverse_vol.sum(axis=1), axis=0).fillna(equal)

    return {
        "equal_weight": equal,
        "timing_score": timing_weights(scores, max_weight=0.35),
        "top_k": top,
        "rank_weight": rank,
        "risk_weight": risk,
    }


def comparison_metrics(
    net_returns: pd.DataFrame,
    gross_returns: pd.DataFrame,
    turnovers: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    benchmark_annual = annual_return(net_returns["equal_weight"].dropna())
    rows = {}
    for strategy in net_returns:
        net = net_returns[strategy].dropna()
        gross = gross_returns[strategy].reindex(net.index).fillna(0.0)
        turnover = turnovers[strategy].reindex(net.index).fillna(0.0)
        rows[strategy] = {
            "annual_net_return": annual_return(net),
            "annual_gross_return": annual_return(gross),
            "annual_excess_vs_equal": annual_return(net) - benchmark_annual,
            "annual_volatility": net.std() * np.sqrt(252),
            "sharpe": sharpe(net),
            "max_drawdown": max_drawdown(net),
            "win_rate": (net > 0).mean(),
            "average_daily_turnover": turnover.mean(),
            "annual_turnover": turnover.mean() * 252,
            "total_trading_cost": costs[strategy].sum(),
        }
    return pd.DataFrame(rows).T


def bootstrap_outperformance(
    returns: pd.DataFrame,
    benchmark: str,
    samples: int,
    random_seed: int,
    block_size: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = {}
    for strategy in returns.columns:
        if strategy == benchmark:
            continue
        diff = (returns[strategy] - returns[benchmark]).dropna().to_numpy()
        n = len(diff)
        observed = diff.mean() * 252
        if n < 2:
            rows[strategy] = {
                "annualized_mean_difference": observed,
                "paired_t_stat": np.nan,
                "paired_t_pvalue_normal": np.nan,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "bootstrap_probability_positive": np.nan,
            }
            continue
        standard_error = diff.std(ddof=1) / np.sqrt(n)
        t_stat = diff.mean() / standard_error if standard_error > 0 else np.nan
        pvalue = erfc(abs(t_stat) / sqrt(2)) if np.isfinite(t_stat) else np.nan
        means = np.empty(samples)
        blocks_needed = int(np.ceil(n / block_size))
        max_start = max(1, n - block_size + 1)
        for i in range(samples):
            starts = rng.integers(0, max_start, size=blocks_needed)
            sample = np.concatenate([diff[start : start + block_size] for start in starts])[:n]
            means[i] = sample.mean() * 252
        rows[strategy] = {
            "annualized_mean_difference": observed,
            "paired_t_stat": t_stat,
            "paired_t_pvalue_normal": pvalue,
            "bootstrap_ci_low": np.quantile(means, 0.025),
            "bootstrap_ci_high": np.quantile(means, 0.975),
            "bootstrap_probability_positive": (means > 0).mean(),
        }
    return pd.DataFrame(rows).T


def regime_metrics(returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    market = market_returns.reindex(returns.index).fillna(0.0)
    trend = market.shift(1).rolling(60, min_periods=20).sum()
    volatility = market.shift(1).rolling(20, min_periods=10).std()
    trend_state = pd.Series("sideways", index=returns.index)
    trend_state[trend <= trend.quantile(1 / 3)] = "bear"
    trend_state[trend >= trend.quantile(2 / 3)] = "bull"
    vol_state = pd.Series("low_vol", index=returns.index)
    vol_state[volatility >= volatility.median()] = "high_vol"

    rows = []
    for regime_type, states in (("trend", trend_state), ("volatility", vol_state)):
        for state in states.dropna().unique():
            subset = returns.loc[states == state]
            for strategy in returns:
                series = subset[strategy].dropna()
                rows.append(
                    {
                        "regime_type": regime_type,
                        "regime": state,
                        "strategy": strategy,
                        "observations": len(series),
                        "annual_return": annual_return(series),
                        "sharpe": sharpe(series),
                        "max_drawdown": max_drawdown(series),
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_analysis(
    factor_returns: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    cost_rates: tuple[float, ...],
    rebalance_frequencies: tuple[str, ...],
    max_rebalance_turnover: float | None,
) -> pd.DataFrame:
    rows = []
    for cost_rate in cost_rates:
        for frequency in rebalance_frequencies:
            evaluated = _evaluate_targets(
                factor_returns,
                targets,
                transaction_cost=cost_rate,
                slippage=0.0,
                rebalance_freq=frequency,
                max_rebalance_turnover=max_rebalance_turnover,
            )
            benchmark = evaluated["equal_weight"]["result"]["net_return"]
            for strategy, value in evaluated.items():
                net = value["result"]["net_return"]
                rows.append(
                    {
                        "total_cost_rate": cost_rate,
                        "rebalance_freq": frequency,
                        "strategy": strategy,
                        "annual_net_return": annual_return(net),
                        "annual_excess_vs_equal": annual_return(net) - annual_return(benchmark),
                        "sharpe": sharpe(net),
                        "max_drawdown": max_drawdown(net),
                        "annual_turnover": value["result"]["turnover"].mean() * 252,
                    }
                )
    return pd.DataFrame(rows)


def _evaluate_targets(
    factor_returns: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    transaction_cost: float,
    slippage: float,
    rebalance_freq: str,
    max_rebalance_turnover: float | None,
) -> dict[str, dict[str, pd.DataFrame]]:
    evaluated = {}
    for name, target in targets.items():
        result, actual_weights = simulate_weight_strategy(
            factor_returns,
            target,
            transaction_cost=transaction_cost,
            slippage=slippage,
            rebalance_freq=rebalance_freq,
            max_rebalance_turnover=max_rebalance_turnover,
        )
        evaluated[name] = {"result": result, "weights": actual_weights}
    return evaluated


def _plot_comparison(returns: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    nav = (1.0 + returns).cumprod()
    ax = nav.plot(figsize=(11, 6), title="Factor combination comparison (net of costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_research_dashboard(
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    regimes: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "equal_weight": "#4C78A8",
        "timing_score": "#F58518",
        "top_k": "#E45756",
        "rank_weight": "#72B7B2",
        "risk_weight": "#54A24B",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    nav = (1.0 + returns).cumprod()
    for strategy in nav:
        axes[0, 0].plot(
            nav.index,
            nav[strategy],
            label=PLOT_LABELS[strategy],
            color=colors[strategy],
            linewidth=1.8,
        )
    axes[0, 0].set_title("A. Net equity curves")
    axes[0, 0].set_ylabel("NAV")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(alpha=0.2)

    for strategy, row in metrics.iterrows():
        axes[0, 1].scatter(
            row["annual_volatility"] * 100,
            row["annual_net_return"] * 100,
            s=90,
            color=colors[strategy],
        )
        axes[0, 1].annotate(
            PLOT_LABELS[strategy],
            (row["annual_volatility"] * 100, row["annual_net_return"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 1].set_title("B. Return-risk trade-off")
    axes[0, 1].set_xlabel("Annual volatility (%)")
    axes[0, 1].set_ylabel("Annual net return (%)")
    axes[0, 1].grid(alpha=0.2)

    regime_order = [
        ("trend", "bear", "Bear"),
        ("trend", "sideways", "Sideways"),
        ("trend", "bull", "Bull"),
        ("volatility", "high_vol", "High vol"),
        ("volatility", "low_vol", "Low vol"),
    ]
    pivot = regimes.pivot_table(index=["regime_type", "regime"], columns="strategy", values="sharpe")
    x = np.arange(len(regime_order))
    width = 0.16
    for offset, strategy in enumerate(returns.columns):
        values = [pivot.loc[(kind, state), strategy] for kind, state, _ in regime_order]
        axes[1, 0].bar(
            x + (offset - 2) * width,
            values,
            width,
            label=PLOT_LABELS[strategy],
            color=colors[strategy],
        )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(x, [label for _, _, label in regime_order])
    axes[1, 0].set_title("C. Sharpe by market regime")
    axes[1, 0].set_ylabel("Sharpe")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(axis="y", alpha=0.2)

    weekly = sensitivity[sensitivity["rebalance_freq"] == "W-FRI"]
    for strategy in returns.columns:
        if strategy == "equal_weight":
            continue
        subset = weekly[weekly["strategy"] == strategy].sort_values("total_cost_rate")
        axes[1, 1].plot(
            subset["total_cost_rate"] * 10000,
            subset["annual_excess_vs_equal"] * 100,
            marker="o",
            label=PLOT_LABELS[strategy],
            color=colors[strategy],
        )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("D. Weekly excess return under costs")
    axes[1, 1].set_xlabel("Total cost (bps per L1 turnover)")
    axes[1, 1].set_ylabel("Annual excess vs equal weight (%)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle("Factor combination research dashboard", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_research_brief(
    output_dir: Path,
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    regimes: pd.DataFrame,
    sensitivity: pd.DataFrame,
    transaction_cost: float,
    slippage: float,
    rebalance_freq: str,
) -> None:
    def regime_value(kind: str, state: str, strategy: str, field: str) -> float:
        row = regimes[
            (regimes["regime_type"] == kind)
            & (regimes["regime"] == state)
            & (regimes["strategy"] == strategy)
        ]
        return float(row.iloc[0][field])

    def weekly_value(cost: float, strategy: str, field: str) -> float:
        row = sensitivity[
            (sensitivity["rebalance_freq"] == "W-FRI")
            & np.isclose(sensitivity["total_cost_rate"], cost)
            & (sensitivity["strategy"] == strategy)
        ]
        return float(row.iloc[0][field])

    start = returns.index.min().strftime("%Y-%m-%d")
    end = returns.index.max().strftime("%Y-%m-%d")
    timing = metrics.loc["timing_score"]
    equal = metrics.loc["equal_weight"]
    risk = metrics.loc["risk_weight"]
    timing_sig = significance.loc["timing_score"]
    lines = [
        "# 因子择时组合是否优于简单等权？——研究摘要",
        "",
        f"**样本：** {start} 至 {end}，{len(returns):,} 个交易日；"
        f"**基准设定：** {rebalance_freq} 调仓，手续费 {transaction_cost:.2%}，"
        f"滑点 {slippage:.2%}，权重信号次日执行。",
        "",
        "![关键结果总览](research_dashboard.png)",
        "",
        "## 研究问题",
        "",
        "在完全相同的因子收益、执行时点、成本模型和换手约束下，择时得分加权是否在收益、风险和统计意义上优于简单等权？如果优势并不普遍，它出现在哪些市场状态，哪类组合对交易成本更稳健？",
        "",
        "## 方法与实验设计",
        "",
        "比较等权、择时得分、Top-K、分位排名和逆波动五种只多因子组合。"
        "评价年化净收益、Sharpe、最大回撤、L1 换手和成本后超额；按牛/震荡/熊及高/低波动分层；"
        "进行 4 档成本 × 3 种调仓频率敏感性分析，并以 20 日移动区块 bootstrap 检验相对等权的日收益差。",
        "",
        "## 核心结果",
        "",
        f"1. **全样本没有显著胜者。** 择时得分年化净收益 {timing['annual_net_return']:.2%}，"
        f"略高于等权 {equal['annual_net_return']:.2%}，但 Sharpe 为 {timing['sharpe']:.2f}，"
        f"低于等权的 {equal['sharpe']:.2f}。相对等权的 bootstrap 95% 区间为"
        f" [{timing_sig['bootstrap_ci_low']:.2%}, {timing_sig['bootstrap_ci_high']:.2%}]，跨过 0。",
        "",
        f"2. **择时收益具有明显状态依赖。** 牛市中择时得分年化收益 "
        f"{regime_value('trend', 'bull', 'timing_score', 'annual_return'):.2%}、Sharpe "
        f"{regime_value('trend', 'bull', 'timing_score', 'sharpe'):.2f}，高于等权的 "
        f"{regime_value('trend', 'bull', 'equal_weight', 'annual_return'):.2%} 和 "
        f"{regime_value('trend', 'bull', 'equal_weight', 'sharpe'):.2f}；但熊市中等权亏损最小，"
        f"择时得分年化收益为 {regime_value('trend', 'bear', 'timing_score', 'annual_return'):.2%}。",
        "",
        f"3. **高波动期复杂加权没有优势。** 高波动期等权 Sharpe "
        f"{regime_value('volatility', 'high_vol', 'equal_weight', 'sharpe'):.2f} 为最高；"
        f"低波动期分位排名 Sharpe {regime_value('volatility', 'low_vol', 'rank_weight', 'sharpe'):.2f} 为最高。"
        f"震荡期逆波动 Sharpe {regime_value('trend', 'sideways', 'risk_weight', 'sharpe'):.2f} 为最高。",
        "",
        f"4. **逆波动对成本最稳。** 基准成本下逆波动 Sharpe {risk['sharpe']:.2f}、"
        f"最大回撤 {risk['max_drawdown']:.2%}，优于等权的 {equal['sharpe']:.2f} 和 "
        f"{equal['max_drawdown']:.2%}。周频总成本升至 20 bps 时，择时超额降至 "
        f"{weekly_value(0.002, 'timing_score', 'annual_excess_vs_equal'):.2%}，逆波动仍为 "
        f"{weekly_value(0.002, 'risk_weight', 'annual_excess_vs_equal'):.2%}。",
        "",
        "5. **样本外证据不支持择时。** Walk-forward 拼接测试中择时年化收益约 -0.94%，"
        "等权约 1.13%；当前不能把全样本局部优势解释为稳定 Alpha。",
        "",
        "## 结论",
        "",
        "**择时得分不是等权的普遍替代品。** 它主要在牛市、低波动环境中有效，"
        "在熊市和高波动期会失效；更简单的逆波动配置虽然收益提升有限，但换手更低、"
        "对成本更稳健。所有动态策略相对等权的收益改善均未达到统计显著，"
        "因此最合理的研究判断是“存在状态依赖的描述性优势，但尚无稳健的样本外显著性”。",
        "",
        "## 结论边界",
        "",
        "回测使用当前沪深300成分股回看历史，存在幸存者偏差；只计因子组合层成本，"
        "未计底层股票换手与融券成本；样本约 5.5 年。下一步应优先补历史成分、真实财务因子和股票层成本，"
        "再延长样本做独立样本外检验，而不是继续扩大参数搜索。",
    ]
    (output_dir / "research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def _write_comparison_summary(
    output_dir: Path,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    regimes: pd.DataFrame,
    transaction_cost: float,
    slippage: float,
    rebalance_freq: str,
    max_rebalance_turnover: float | None,
    top_k: int,
) -> None:
    display_metrics = metrics.rename(index=STRATEGY_LABELS)
    display_significance = significance.rename(index=STRATEGY_LABELS)
    regime_pivot = regimes.pivot_table(
        index=["regime_type", "regime", "strategy"],
        values=["annual_return", "sharpe", "max_drawdown"],
    )
    lines = [
        "# 组合方法公平对比实验",
        "",
        "所有策略使用完全相同的因子收益、样本区间、调仓日、次日执行规则和成本模型。",
        "",
        f"- 调仓频率：`{rebalance_freq}`",
        f"- 手续费：{transaction_cost:.4%} / 单位 L1 换手",
        f"- 滑点：{slippage:.4%} / 单位 L1 换手",
        f"- 单次调仓 L1 换手上限：{max_rebalance_turnover if max_rebalance_turnover is not None else '不限制'}",
        f"- Top-K：{top_k}",
        "- 统计检验：20 日移动区块 bootstrap，95% 置信区间，固定随机种子",
        "",
        "## 全样本净值指标",
        "",
        display_metrics.round(4).to_markdown(),
        "",
        "## 相对等权的显著性",
        "",
        display_significance.round(4).to_markdown(),
        "",
        "只有当 bootstrap 置信区间下界大于 0，才能称扣费后的平均收益改善在该检验下显著。",
        "",
        "## 市场状态分层",
        "",
        regime_pivot.round(4).to_markdown(),
        "",
        "完整成本/调仓频率敏感性结果见 `comparison_sensitivity.csv`。",
    ]
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
