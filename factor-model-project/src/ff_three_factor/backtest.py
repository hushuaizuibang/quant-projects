from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .config import BacktestConfig
from .factors import FactorModel


@dataclass
class BacktestResult:
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    strategy_nav: pd.Series
    benchmark_nav: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    factor_score_weights: pd.DataFrame
    gross_returns: pd.Series
    transaction_costs: pd.Series
    industry_exposure: pd.DataFrame
    size_exposure: pd.Series
    optimizer_status: pd.Series


def run_backtest(model: FactorModel, config: BacktestConfig) -> BacktestResult:
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    score_weight_frames: dict[pd.Timestamp, dict[str, float]] = {}
    previous = pd.Series(0.0, index=model.monthly_returns.columns)
    records = []

    month_index = model.monthly_returns.index
    for signal_date, exposure in model.exposures.items():
        location = month_index.get_indexer([signal_date])[0]
        if location < 0 or location + 1 >= len(month_index):
            continue
        holding_date = month_index[location + 1]
        score_weights = _resolve_score_weights(model, signal_date, config)
        score = _score_exposures(exposure, score_weights)
        covariance = _trailing_covariance(model.monthly_returns, signal_date, config)
        weights, status = _construct_portfolio(
            score, exposure["size"], model.industry, covariance, previous, config
        )

        turnover = float((weights - previous).abs().sum())
        gross = float((weights * model.monthly_returns.loc[holding_date]).sum(skipna=True))
        cost = config.total_cost_rate * turnover
        records.append(
            {
                "date": holding_date,
                "gross_return": gross,
                "cost": cost,
                "net_return": gross - cost,
                "benchmark": model.benchmark_returns.get(holding_date, np.nan),
                "turnover": turnover,
                "size_exposure": float((weights * exposure["size"]).sum()),
                "status": status,
            }
        )
        weights_by_date[holding_date] = weights
        score_weight_frames[holding_date] = score_weights
        previous = weights

    if not records:
        raise RuntimeError("Backtest produced no holdings. Increase the date range or check factor inputs.")
    summary = pd.DataFrame(records).set_index("date")
    strategy_returns = summary["net_return"].rename("strategy")
    benchmark_returns = summary["benchmark"].rename("benchmark")
    weights = pd.DataFrame(weights_by_date).T.reindex(strategy_returns.index).fillna(0.0)
    industry_exposure = _industry_active_weights(weights, model.industry)
    return BacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        strategy_nav=(1 + strategy_returns).cumprod().rename("strategy_nav"),
        benchmark_nav=(1 + benchmark_returns.fillna(0)).cumprod().rename("benchmark_nav"),
        weights=weights,
        turnover=summary["turnover"],
        factor_score_weights=pd.DataFrame(score_weight_frames).T.reindex(strategy_returns.index),
        gross_returns=summary["gross_return"],
        transaction_costs=summary["cost"],
        industry_exposure=industry_exposure,
        size_exposure=summary["size_exposure"],
        optimizer_status=summary["status"],
    )


def _construct_portfolio(
    score: pd.Series,
    size_signal: pd.Series,
    industry: pd.Series,
    covariance: pd.DataFrame,
    previous: pd.Series,
    config: BacktestConfig,
) -> tuple[pd.Series, str]:
    valid = score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    minimum = int(np.ceil(1 / config.max_stock_weight))
    selected_count = max(minimum, int(np.ceil(len(valid) * config.top_quantile)))
    primary = _make_feasible_candidate_set(
        valid,
        size_signal.reindex(valid.index).fillna(0),
        industry.reindex(valid.index).fillna("Unknown"),
        selected_count,
        config,
    )
    candidate_sets = [primary]
    if len(primary) < len(valid):
        candidate_sets.append(valid.index)
    last_message = "not attempted"
    for candidates in candidate_sets:
        result = _optimize_candidates(
            valid.reindex(candidates),
            size_signal.reindex(candidates).fillna(0),
            industry.reindex(candidates).fillna("Unknown"),
            covariance.reindex(index=candidates, columns=candidates),
            previous.reindex(candidates).fillna(0),
            industry,
            config,
        )
        if result.success:
            weights = pd.Series(0.0, index=score.index)
            weights.loc[candidates] = np.where(result.x < 1e-8, 0.0, result.x)
            weights /= weights.sum()
            return weights, f"optimized ({len(candidates)} candidates)"
        last_message = result.message

    # A deterministic capped equal-weight fallback keeps the backtest running
    # while making optimizer failures visible in the exported status.
    candidates = valid.index[: max(minimum, selected_count)]
    weights = pd.Series(0.0, index=score.index)
    weights.loc[candidates] = 1 / len(candidates)
    return weights, f"fallback: {last_message}"


def _optimize_candidates(
    score: pd.Series,
    size: pd.Series,
    industry: pd.Series,
    covariance: pd.DataFrame,
    previous: pd.Series,
    universe_industry: pd.Series,
    config: BacktestConfig,
):
    n = len(score)
    universe_mix = universe_industry.fillna("Unknown").value_counts(normalize=True)
    candidate_counts = industry.value_counts()
    industry_x0 = np.array(
        [float(universe_mix.get(sector, 0)) / candidate_counts[sector] for sector in industry]
    )
    industry_x0 /= industry_x0.sum()
    cov = covariance.fillna(0).to_numpy(dtype=float)
    cov = (cov + cov.T) / 2 + np.eye(n) * 1e-6
    alpha = _safe_zscore(score).fillna(0).to_numpy() * 0.01
    prev = previous.to_numpy(dtype=float)
    size_vector = size.to_numpy(dtype=float)

    def objective(w: np.ndarray) -> float:
        smooth_turnover = np.sqrt((w - prev) ** 2 + 1e-8).sum()
        return float(
            -alpha @ w
            + config.risk_aversion * (w @ cov @ w)
            + config.turnover_penalty * config.total_cost_rate * smooth_turnover
        )

    def objective_jacobian(w: np.ndarray) -> np.ndarray:
        turnover_gradient = (w - prev) / np.sqrt((w - prev) ** 2 + 1e-8)
        return (
            -alpha
            + 2 * config.risk_aversion * (cov @ w)
            + config.turnover_penalty * config.total_cost_rate * turnover_gradient
        )

    ones = np.ones(n)
    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: ones}
    ]
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for sector in industry.unique():
        mask = (industry == sector).to_numpy(dtype=float)
        baseline = float(universe_mix.get(sector, 0))
        upper = min(1.0, baseline + config.max_industry_deviation)
        lower = max(0.0, baseline - config.max_industry_deviation)
        if upper < 1.0 - 1e-12:
            constraints.append(
                {"type": "ineq", "fun": lambda w, m=mask, u=upper: u - m @ w,
                 "jac": lambda w, m=mask: -m}
            )
            a_ub.append(mask)
            b_ub.append(upper)
        if lower > 1e-12:
            constraints.append(
                {"type": "ineq", "fun": lambda w, m=mask, low=lower: m @ w - low,
                 "jac": lambda w, m=mask: m}
            )
            a_ub.append(-mask)
            b_ub.append(-lower)
    constraints.extend(
        [
            {
                "type": "ineq",
                "fun": lambda w: config.max_size_exposure - size_vector @ w,
                "jac": lambda w: -size_vector,
            },
            {
                "type": "ineq",
                "fun": lambda w: config.max_size_exposure + size_vector @ w,
                "jac": lambda w: size_vector,
            },
        ]
    )
    a_ub.extend([size_vector, -size_vector])
    b_ub.extend([config.max_size_exposure, config.max_size_exposure])
    bounds = [(0.0, config.max_stock_weight)] * n
    feasibility = linprog(
        np.zeros(n),
        A_ub=np.asarray(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=ones.reshape(1, -1),
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not feasibility.success:
        return feasibility
    x0 = 0.85 * feasibility.x + 0.15 * industry_x0
    if abs(size_vector @ x0) > config.max_size_exposure:
        x0 = feasibility.x
    return minimize(
        objective,
        x0,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )


def _make_feasible_candidate_set(
    ranked_score: pd.Series,
    size: pd.Series,
    industry: pd.Series,
    selected_count: int,
    config: BacktestConfig,
) -> pd.Index:
    """Add enough names per sector for industry lower bounds to be feasible."""
    selected = set(ranked_score.index[:selected_count])
    mix = industry.value_counts(normalize=True)
    for sector, baseline in mix.items():
        lower = max(0.0, float(baseline) - config.max_industry_deviation)
        required = int(np.ceil(lower / config.max_stock_weight - 1e-12))
        sector_ranked = ranked_score.index[industry.reindex(ranked_score.index) == sector]
        selected.update(sector_ranked[:required])
    # Composite scores often favor one side of the Size factor. Include both
    # tails so the active-size constraint has a feasible convex combination.
    tail_count = max(5, int(np.ceil(0.05 * len(ranked_score))))
    selected.update(size.nsmallest(tail_count).index)
    selected.update(size.nlargest(tail_count).index)
    # Preserve score order to keep optimization inputs deterministic.
    return ranked_score.index[ranked_score.index.isin(selected)]


def _score_exposures(exposure: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    missing = set(weights).difference(exposure.columns)
    if missing:
        raise KeyError(f"Missing factor exposures: {sorted(missing)}")
    aligned = pd.Series(weights, dtype=float).reindex(exposure.columns).fillna(0)
    return exposure.mul(aligned, axis=1).sum(axis=1).rename("score")


def _resolve_score_weights(
    model: FactorModel, signal_date: pd.Timestamp, config: BacktestConfig
) -> dict[str, float]:
    active = list(model.available_factors)
    if not active:
        raise RuntimeError("No usable factor is available for portfolio scoring.")
    base = pd.Series(config.factor_weights, dtype=float).reindex(active).fillna(0.0)
    if config.scoring_method == "fixed":
        return _normalize_weights(base).to_dict()
    if config.scoring_method != "icir":
        raise ValueError(f"Unsupported scoring method: {config.scoring_method}")

    # IC(signal_date) uses the following month's return, so only strictly older
    # signal rows are observable at the current rebalance.
    history = model.factor_ic.loc[model.factor_ic.index < signal_date].tail(config.icir_lookback_months)
    if len(history.dropna(how="all")) < config.icir_min_periods:
        return _normalize_weights(base).to_dict()
    icir = history.mean() / history.std(ddof=1).replace(0, np.nan)
    icir = icir.reindex(active).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-3, 3)

    # Penalize redundant factors using a regularized IC correlation matrix.
    corr = history.reindex(columns=active).corr().fillna(0).to_numpy()
    corr = corr + np.eye(len(corr)) * 0.25
    try:
        adjusted = pd.Series(np.linalg.solve(corr, icir.to_numpy()), index=active)
    except np.linalg.LinAlgError:
        adjusted = icir
    if adjusted.abs().sum() < 1e-12:
        adjusted = base
    return _normalize_weights(adjusted).to_dict()


def _normalize_weights(weights: pd.Series) -> pd.Series:
    gross = weights.abs().sum()
    return weights / gross if gross else weights


def _trailing_covariance(
    returns: pd.DataFrame, signal_date: pd.Timestamp, config: BacktestConfig
) -> pd.DataFrame:
    window = returns.loc[:signal_date].tail(config.covariance_lookback_months)
    cov = window.cov(min_periods=max(6, config.covariance_lookback_months // 3)) * 12
    diagonal = np.diag(np.diag(cov.fillna(0)))
    # Simple shrinkage makes small-sample covariance matrices stable.
    return pd.DataFrame(
        0.6 * cov.fillna(0).to_numpy() + 0.4 * diagonal,
        index=returns.columns,
        columns=returns.columns,
    )


def _industry_active_weights(weights: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    sectors = industry.reindex(weights.columns).fillna("Unknown")
    portfolio = pd.DataFrame(
        {sector: weights.loc[:, sectors == sector].sum(axis=1) for sector in sectors.unique()}
    )
    benchmark = sectors.value_counts(normalize=True)
    return portfolio.subtract(benchmark, axis=1)


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std and not pd.isna(std) else series * 0
