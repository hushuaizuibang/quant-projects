from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import FACTOR_NAMES, BacktestConfig
from .data import MarketData


@dataclass
class FactorModel:
    monthly_returns: pd.DataFrame
    benchmark_returns: pd.Series
    exposures: dict[pd.Timestamp, pd.DataFrame]
    raw_exposures: dict[pd.Timestamp, pd.DataFrame]
    market_cap: pd.DataFrame
    industry: pd.Series
    factor_ic: pd.DataFrame
    available_factors: tuple[str, ...]

    @property
    def factor_returns(self) -> pd.DataFrame:
        """Compatibility alias: cross-sectional factor-mimicking returns."""
        rows = {}
        for signal_date, exposure in self.exposures.items():
            future = _next_row(self.monthly_returns, signal_date)
            if future is None:
                continue
            rows[future.name] = {
                factor: _factor_mimicking_return(exposure[factor], future)
                for factor in FACTOR_NAMES
            }
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build_factor_model(data: MarketData, config: BacktestConfig) -> FactorModel:
    """Build point-in-time factor signals at each month end.

    Direction is consistent: a larger processed value is always preferred.
    Size therefore uses negative log market cap and low volatility uses negative
    trailing volatility.
    """
    monthly_close = data.close.resample("ME").last()
    monthly_returns = monthly_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_returns = (
        data.benchmark_close.resample("ME").last().pct_change(fill_method=None).reindex(monthly_returns.index)
    )
    market_cap_source = (
        data.market_cap if data.market_cap is not None else data.close * data.outstanding_share
    )
    market_cap = market_cap_source.resample("ME").last().reindex(monthly_close.index)
    navps = _to_monthly_frame(data.net_asset_per_share, monthly_close.index, monthly_close.columns)
    roe = _to_monthly_frame(data.roe, monthly_close.index, monthly_close.columns)
    cashflow = _to_monthly_frame(
        data.operating_cashflow_to_assets, monthly_close.index, monthly_close.columns
    )
    industry = (
        data.industry.reindex(monthly_close.columns).fillna("Unknown")
        if data.industry is not None
        else pd.Series("Unknown", index=monthly_close.columns)
    )

    raw_panels = {
        "size": -np.log(market_cap.where(market_cap > 0)),
        "value": np.log((navps / monthly_close).where((navps / monthly_close) > 0)),
        "momentum": (
            monthly_close.shift(config.momentum_skip_months)
            / monthly_close.shift(config.momentum_lookback_months)
            - 1
        ),
        "quality": (roe + cashflow) / 2,
        "low_volatility": -(
            data.close.pct_change(fill_method=None)
            .rolling(config.volatility_lookback_days, min_periods=config.volatility_lookback_days // 2)
            .std()
            .resample("ME")
            .last()
            .reindex(monthly_close.index)
        ),
    }

    exposures: dict[pd.Timestamp, pd.DataFrame] = {}
    raw_exposures: dict[pd.Timestamp, pd.DataFrame] = {}
    for date in monthly_close.index:
        raw = pd.DataFrame({name: panel.loc[date] for name, panel in raw_panels.items()})
        processed = pd.DataFrame(index=monthly_close.columns)
        log_cap = np.log(market_cap.loc[date].where(market_cap.loc[date] > 0))
        for factor in FACTOR_NAMES:
            controls = None if factor == "size" else log_cap
            processed[factor] = preprocess_factor(
                raw[factor], industry, controls, winsor_mad=config.winsor_mad
            )
        # An unavailable optional fundamental factor is neutral, rather than
        # deleting every stock from the composite signal.
        processed = processed.fillna(0.0)
        if (processed.abs().sum(axis=0) > 0).sum() >= 3:
            exposures[date] = processed
            raw_exposures[date] = raw

    available_factors = tuple(
        factor
        for factor in FACTOR_NAMES
        if any(frame[factor].nunique(dropna=True) > 1 for frame in exposures.values())
    )
    factor_ic = compute_factor_ic(monthly_returns, exposures, config.ic_method)
    return FactorModel(
        monthly_returns=monthly_returns,
        benchmark_returns=benchmark_returns,
        exposures=exposures,
        raw_exposures=raw_exposures,
        market_cap=market_cap,
        industry=industry,
        factor_ic=factor_ic,
        available_factors=available_factors,
    )


def preprocess_factor(
    values: pd.Series,
    industry: pd.Series,
    log_market_cap: pd.Series | None,
    winsor_mad: float = 3.0,
) -> pd.Series:
    """MAD winsorize, neutralize and standardize one cross-section."""
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clipped = _mad_winsorize(values, winsor_mad)
    sample = pd.DataFrame({"factor": clipped, "industry": industry})
    if log_market_cap is not None:
        sample["log_market_cap"] = log_market_cap
    sample = sample.dropna()
    if len(sample) < 5 or sample["factor"].nunique() < 2:
        return pd.Series(np.nan, index=values.index)

    dummies = pd.get_dummies(sample["industry"].astype(str), prefix="industry", drop_first=True, dtype=float)
    x_parts = [dummies]
    if log_market_cap is not None:
        x_parts.insert(0, sample[["log_market_cap"]].astype(float))
    x = sm.add_constant(pd.concat(x_parts, axis=1), has_constant="add")
    if x.shape[1] >= len(sample):
        residual = sample["factor"] - sample.groupby("industry")["factor"].transform("mean")
    else:
        residual = sm.OLS(sample["factor"].astype(float), x).fit().resid
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[residual.index] = _zscore(residual)
    return result


def compute_factor_ic(
    monthly_returns: pd.DataFrame,
    exposures: dict[pd.Timestamp, pd.DataFrame],
    method: str = "spearman",
) -> pd.DataFrame:
    rows = {}
    for signal_date, frame in exposures.items():
        future = _next_row(monthly_returns, signal_date)
        if future is None:
            continue
        row = {}
        for factor in FACTOR_NAMES:
            sample = pd.concat([frame[factor], future], axis=1).dropna()
            row[factor] = (
                sample.iloc[:, 0].corr(sample.iloc[:, 1], method=method)
                if len(sample) >= 5 and sample.iloc[:, 0].nunique() > 1
                else np.nan
            )
        rows[signal_date] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _to_monthly_frame(
    values: pd.Series | pd.DataFrame | None,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    if isinstance(values, pd.Series):
        aligned = pd.to_numeric(values.reindex(columns), errors="coerce")
        return pd.DataFrame(np.tile(aligned.to_numpy(), (len(index), 1)), index=index, columns=columns)
    dated = values.copy()
    dated.index = pd.to_datetime(dated.index)
    return dated.reindex(index.union(dated.index)).sort_index().ffill().reindex(index=index, columns=columns)


def _mad_winsorize(series: pd.Series, threshold: float) -> pd.Series:
    if series.notna().sum() == 0:
        return series
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        return series
    scale = 1.4826 * mad
    return series.clip(median - threshold * scale, median + threshold * scale)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return series * np.nan
    return (series - series.mean()) / std


def _next_row(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    locations = np.flatnonzero(frame.index == date)
    if not len(locations) or locations[0] + 1 >= len(frame):
        return None
    return frame.iloc[locations[0] + 1]


def _factor_mimicking_return(signal: pd.Series, future_return: pd.Series) -> float:
    sample = pd.concat([signal.rename("signal"), future_return.rename("return")], axis=1).dropna()
    if len(sample) < 5 or sample["signal"].nunique() < 2:
        return np.nan
    high = sample["signal"] >= sample["signal"].quantile(0.8)
    low = sample["signal"] <= sample["signal"].quantile(0.2)
    return sample.loc[high, "return"].mean() - sample.loc[low, "return"].mean()
