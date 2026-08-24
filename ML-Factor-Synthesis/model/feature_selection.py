"""Training-only ICIR screening and redundant-lag removal."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FeatureConfig

LOGGER = logging.getLogger(__name__)
FEATURE_PATTERN = re.compile(r"^(?P<factor>.+)__lag_(?P<lag>\d+)$")


@dataclass
class FeatureSelectionResult:
    features: list[str]
    factor_summary: pd.DataFrame
    lag_summary: pd.DataFrame


def _monthly_feature_ic(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    joined = frame[features + ["target"]].dropna(subset=["target"])
    return joined.groupby(level="date").apply(
        lambda month: month[features].corrwith(month["target"], method="spearman"),
        include_groups=False,
    )


def select_features_by_icir(
    train: pd.DataFrame,
    config: FeatureConfig,
) -> FeatureSelectionResult:
    """Select factors and lags using only the supplied training period.

    Factor eligibility is based on lag-0 monthly ICIR, annualized by sqrt(12).
    Eligible factors then contribute their lags in descending absolute mean IC,
    one lag per round, until the configured 20-30 dimensional target is met.
    """
    features = [
        column
        for column in train
        if FEATURE_PATTERN.match(column) and column != "target" and not column.startswith("_")
    ]
    if not features:
        raise ValueError("No factor lag features are available for ICIR selection")

    monthly_ic = _monthly_feature_ic(train, features)
    lag_summary = pd.DataFrame(
        {
            "mean_ic": monthly_ic.mean(),
            "ic_std": monthly_ic.std(ddof=1),
            "positive_ic_rate": (monthly_ic > 0).mean(),
            "months": monthly_ic.notna().sum(),
        }
    )
    lag_summary["icir"] = lag_summary["mean_ic"] / lag_summary["ic_std"] * np.sqrt(12)
    lag_summary["abs_mean_ic"] = lag_summary["mean_ic"].abs()
    parsed = lag_summary.index.to_series().str.extract(FEATURE_PATTERN)
    lag_summary["factor"] = parsed["factor"].to_numpy()
    lag_summary["lag"] = parsed["lag"].astype(int).to_numpy()

    lag_zero = lag_summary.loc[lag_summary["lag"] == 0].copy()
    factor_summary = lag_zero.set_index("factor")[
        ["mean_ic", "ic_std", "icir", "positive_ic_rate", "months"]
    ]
    factor_summary["eligible"] = factor_summary["icir"] > config.factor_icir_threshold
    factor_summary = factor_summary.sort_values("icir", ascending=False)
    eligible = factor_summary.index[factor_summary["eligible"]].tolist()

    ranked_lags: dict[str, list[str]] = {}
    for factor in eligible:
        candidates = lag_summary.loc[lag_summary["factor"] == factor].sort_values(
            ["abs_mean_ic", "lag"], ascending=[False, True]
        )
        ranked_lags[factor] = candidates.index[: config.lags_per_factor].tolist()

    selected: list[str] = []
    for rank in range(config.lags_per_factor):
        for factor in eligible:
            candidates = ranked_lags[factor]
            if rank < len(candidates) and len(selected) < config.max_selected_features:
                selected.append(candidates[rank])
        if len(selected) >= config.min_selected_features:
            break

    lag_summary["selected"] = lag_summary.index.isin(selected)
    factor_summary["selected_lags"] = [
        ",".join(
            str(lag_summary.loc[name, "lag"])
            for name in selected
            if name.startswith(f"{factor}__")
        )
        for factor in factor_summary.index
    ]
    if len(selected) < config.min_selected_features:
        LOGGER.warning(
            "ICIR threshold retained only %d features; not backfilling below threshold %.2f",
            len(selected),
            config.factor_icir_threshold,
        )
    LOGGER.info(
        "Training-only ICIR selection retained %d/%d factors and %d/%d lag features",
        len(eligible),
        len(factor_summary),
        len(selected),
        len(features),
    )
    return FeatureSelectionResult(
        selected,
        factor_summary,
        lag_summary.sort_values("abs_mean_ic", ascending=False),
    )
