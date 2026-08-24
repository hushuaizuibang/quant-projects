"""Daily cross-sectional industry and size neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd


VALID_MODES = {"none", "industry", "size", "both"}


def _demean(frame: pd.DataFrame, keys: list[pd.Series]) -> pd.DataFrame:
    return frame - frame.groupby(keys).transform("mean")


def _remove_size(frame: pd.DataFrame, log_size: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """Vectorized OLS residuals after any fixed effects have been removed."""
    centered_size = log_size - log_size.groupby(dates).transform("mean")
    centered_frame = frame - frame.groupby(dates).transform("mean")
    cross_product = centered_frame.mul(centered_size, axis=0)
    numerator = cross_product.groupby(dates).transform("sum")
    denominator = centered_size.pow(2).groupby(dates).transform("sum").replace(0, np.nan)
    beta = numerator.div(denominator, axis=0).fillna(0)
    return centered_frame - beta.mul(centered_size, axis=0)


def neutralize(factors: pd.DataFrame, metadata: pd.DataFrame, mode: str = "none") -> pd.DataFrame:
    """Residualize factors against daily industry fixed effects and/or log size.

    The ``both`` mode uses the Frisch-Waugh-Lovell result: industry-demean the
    factors and size, then remove the common size slope. This is algebraically
    equivalent to OLS on an intercept, industry dummies, and log market cap.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"neutralization must be one of {sorted(VALID_MODES)}")
    if mode == "none":
        return factors.copy()
    meta = metadata.set_index(["date", "symbol"])[["industry", "market_cap"]].reindex(factors.index)
    dates = pd.Series(factors.index.get_level_values("date"), index=factors.index, name="date_key")
    industry = meta["industry"].fillna("Unknown").rename("industry_key")
    log_size = np.log(meta["market_cap"].where(meta["market_cap"] > 0)).rename("log_market_cap")

    valid_industry = meta["industry"].notna() & meta["industry"].ne("Unknown")
    if mode in {"industry", "both"} and (
        industry.nunique() <= 1 or valid_industry.mean() < 0.95
    ):
        raise ValueError(
            "Industry neutralization requires at least 95% point-in-time industry coverage"
        )
    if mode in {"size", "both"} and log_size.notna().mean() < 0.95:
        raise ValueError(
            "Size neutralization requires point-in-time positive market_cap coverage of at least 95%"
        )

    if mode == "industry":
        return _demean(factors, [dates, industry])
    if mode == "size":
        return _remove_size(factors, log_size, dates)

    industry_factors = _demean(factors, [dates, industry])
    industry_size = log_size - log_size.groupby([dates, industry]).transform("mean")
    return _remove_size(industry_factors, industry_size, dates)
