"""Research plots and result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.metrics import monthly_ic, quantile_returns


def plot_performance(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    colors = {"ML": "#1b4965", "Equal weight": "#267365", "ICIR weight": "#c75146"}
    for name, frame in results.items():
        wealth = (1 + frame["net_return"]).cumprod()
        axis.plot(wealth.index, wealth, label=name, color=colors.get(name), linewidth=2)
    axis.axhline(1, color="#777777", linewidth=0.8)
    axis.set(title="Out-of-sample long-short performance", ylabel="Cumulative wealth")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "ml_synthesis_performance.png", dpi=170)
    plt.close(fig)


def plot_ic_analysis(score: pd.Series, target: pd.Series, output_dir: Path) -> pd.Series:
    ic = monthly_ic(score, target)
    quantiles = quantile_returns(score, target)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    axes[0].bar(ic.index, ic.values, width=18, color=ic.map(lambda x: "#267365" if x >= 0 else "#c75146"))
    axes[0].axhline(ic.mean(), color="#1b4965", label=f"Mean IC {ic.mean():.3f}")
    axes[0].set(title="Monthly out-of-sample Spearman IC", ylabel="IC")
    axes[0].legend(frameon=False)
    (1 + quantiles).cumprod().plot(ax=axes[1], colormap="viridis")
    axes[1].set(title="Score quantile cumulative returns", ylabel="Cumulative wealth", xlabel="")
    axes[1].legend(title="Quantile", ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ic_analysis.png", dpi=170)
    plt.close(fig)
    return ic


def plot_bootstrap(distribution: pd.DataFrame, output_dir: Path) -> None:
    strategies = distribution.index.get_level_values("strategy").unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1b4965", "#267365", "#c75146", "#c49a00"]
    for name, color in zip(strategies, colors):
        values = distribution.xs(name, level="strategy")
        axes[0].hist(values["annual_return"], bins=30, alpha=0.45, label=name, color=color)
        axes[1].hist(values["information_ratio"].dropna(), bins=30, alpha=0.45, label=name, color=color)
    axes[0].set(title="Moving-block bootstrap annual return", xlabel="Annual return", ylabel="Frequency")
    axes[1].set(title="Moving-block bootstrap information ratio", xlabel="Information ratio")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_analysis.png", dpi=170)
    plt.close(fig)


def plot_yearly_performance(yearly: pd.DataFrame, output_dir: Path) -> None:
    pivot = yearly.pivot(index="year", columns="strategy", values="net_return")
    axis = pivot.plot(kind="bar", figsize=(10, 5), color=["#1b4965", "#267365", "#c75146", "#c49a00"])
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set(title="Out-of-sample performance by test year", ylabel="Compounded net return", xlabel="")
    axis.legend(frameon=False)
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "yearly_performance.png", dpi=170)
    plt.close(axis.figure)


def plot_neutralization_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["annual_return", "sharpe", "max_drawdown", "average_turnover"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, axis in zip(metrics, axes.ravel()):
        values = summary[metric]
        colors = np.where(values >= 0, "#267365", "#c75146")
        axis.bar(values.index, values, color=colors)
        axis.axhline(0, color="#666666", linewidth=0.7)
        axis.set_title(metric.replace("_", " ").title())
    fig.suptitle("ML factor neutralization experiment")
    fig.tight_layout()
    fig.savefig(output_dir / "neutralization_comparison.png", dpi=170)
    plt.close(fig)
