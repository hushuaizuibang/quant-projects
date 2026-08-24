"""Signal baselines and monthly long-short simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.constraints import cap_turnover


def equal_weight_score(samples: pd.DataFrame) -> pd.Series:
    current = [column for column in samples if column.endswith("__lag_00")]
    return samples[current].mean(axis=1).rename("equal_weight")


def icir_weighted_score(samples: pd.DataFrame, forward_days: int = 20) -> pd.Series:
    """Build expanding ICIR weights using only labels observable before each signal date."""
    current = [column for column in samples if column.endswith("__lag_00")]
    dates = samples.index.get_level_values("date").unique().sort_values()
    results: list[pd.Series] = []
    monthly_ic: list[pd.Series] = []
    ic_dates: list[pd.Timestamp] = []
    for date in dates:
        day = samples.xs(date, level="date")
        observable = pd.Timestamp(date) - pd.offsets.BDay(forward_days)
        history = pd.DataFrame(monthly_ic, index=ic_dates)
        history = history.loc[history.index <= observable]
        if len(history) >= 6:
            dispersion = history.std(ddof=1).replace(0, np.nan)
            weights = (history.mean() / dispersion).replace([np.inf, -np.inf], np.nan).fillna(0)
            if weights.abs().sum() == 0:
                weights[:] = 1
        else:
            weights = pd.Series(1.0, index=current)
        weights /= weights.abs().sum()
        results.append((day[current] @ weights).rename(date))
        if day["target"].notna().sum() >= 5:
            monthly_ic.append(day[current].corrwith(day["target"], method="spearman"))
            ic_dates.append(pd.Timestamp(date))
    score = pd.concat(results, keys=dates, names=["date", "symbol"])
    return score.rename("icir_weighted")


def single_factor_benchmarks(
    samples: pd.DataFrame, first_test_year: int = 2024, forward_days: int = 20
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Orient and rank single factors using pre-test IC evidence only."""
    current = [column for column in samples if column.endswith("__lag_00")]
    dates = samples.index.get_level_values("date")
    cutoff = pd.Timestamp(first_test_year, 1, 1)
    observable = dates + pd.offsets.BDay(forward_days) < cutoff
    history = samples.loc[observable & samples["target"].notna()]
    monthly = history[current].join(history["target"]).groupby(level="date").apply(
        lambda day: day[current].corrwith(day["target"], method="spearman")
    )
    report = pd.DataFrame({"mean_ic": monthly.mean(), "ic_std": monthly.std(ddof=1)})
    report["icir"] = report["mean_ic"] / report["ic_std"]
    report["direction"] = np.sign(report["mean_ic"]).replace(0, 1)
    report["selection_score"] = report["icir"].abs()
    report.index = report.index.str.removesuffix("__lag_00")
    test = samples.loc[dates.year >= first_test_year]
    scores = {
        column.removesuffix("__lag_00"): (
            test[column] * report.loc[column.removesuffix("__lag_00"), "direction"]
        ).rename(column.removesuffix("__lag_00"))
        for column in current
    }
    return scores, report.sort_values("selection_score", ascending=False)


def quantile_target(scores: pd.Series, quantile: float) -> pd.Series:
    valid = scores.dropna().sort_values()
    count = max(1, int(np.floor(len(valid) * quantile)))
    if len(valid) < count * 2:
        return pd.Series(dtype=float)
    weights = pd.Series(0.0, index=valid.index)
    weights.loc[valid.index[:count]] = -1 / count
    weights.loc[valid.index[-count:]] = 1 / count
    return weights


def run_backtest(
    score: pd.Series,
    forward_returns: pd.Series,
    quantile: float = 0.10,
    commission: float = 0.0003,
    slippage: float = 0.001,
    max_turnover: float = 0.30,
) -> pd.DataFrame:
    if score.index.names != ["date", "symbol"]:
        score.index = score.index.set_names(["date", "symbol"])
    previous = pd.Series(dtype=float)
    records = []
    for date in score.index.get_level_values("date").unique().sort_values():
        day_score = score.xs(date, level="date")
        target = quantile_target(day_score, quantile)
        actual, used_turnover = cap_turnover(previous, target, max_turnover)
        returns = forward_returns.xs(date, level="date").reindex(actual.index).fillna(0)
        gross = float((actual * returns).sum())
        traded_notional = 2 * used_turnover
        cost = (commission + slippage) * traded_notional
        records.append(
            {"date": date, "gross_return": gross, "cost": cost, "net_return": gross - cost, "turnover": used_turnover}
        )
        previous = actual
    return pd.DataFrame(records).set_index("date")
