from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult


def plot_nav_curves(result: BacktestResult, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    result.strategy_nav.plot(ax=axes[0], label="Strategy NAV", linewidth=2)
    result.benchmark_nav.plot(
        ax=axes[0], label="CSI 300 Current-Constituent Equal-Weight Proxy", linewidth=2
    )
    axes[0].set(title="Multi-Factor Strategy vs Benchmark Proxy", ylabel="Net Asset Value")
    axes[0].legend()
    drawdown = result.strategy_nav / result.strategy_nav.cummax() - 1
    axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="tab:red")
    axes[1].set(ylabel="Drawdown", xlabel="Date")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_monthly_distribution(returns: pd.Series, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    returns.dropna().plot(kind="hist", bins=20, alpha=0.75, edgecolor="white", ax=axis)
    axis.axvline(returns.mean(), color="tab:red", linestyle="--", label="Mean")
    axis.set(title="Monthly Return Distribution", xlabel="Monthly Return", ylabel="Frequency")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_factor_diagnostics(
    factor_ic: pd.DataFrame, factor_weights: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    factor_ic.rolling(6, min_periods=3).mean().plot(ax=axes[0], linewidth=1.4)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Six-Month Rolling Factor IC", ylabel="Rank IC")
    factor_weights.plot(ax=axes[1], linewidth=1.4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Dynamic Composite-Factor Weights", ylabel="Weight", xlabel="Date")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_portfolio_diagnostics(result: BacktestResult, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    result.industry_exposure.plot(ax=axes[0], linewidth=1)
    axes[0].set(title="Active Industry Weights vs Equal-Weight Universe", ylabel="Active weight")
    result.size_exposure.plot(ax=axes[1], color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Portfolio Size-Factor Exposure", ylabel="Exposure")
    result.turnover.plot(kind="bar", ax=axes[2], color="tab:orange", width=0.8)
    axes[2].set(title="Monthly One-Way Turnover", ylabel="Turnover", xlabel="Date")
    axes[2].set_xticks([])
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
