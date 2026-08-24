"""Cross-sectional preprocessing that never pools information across dates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(frame: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Clip each date/factor at median +/- threshold * 1.4826 * MAD."""
    def clip(group: pd.DataFrame) -> pd.DataFrame:
        median = group.median()
        scale = (group - median).abs().median() * 1.4826
        scale = scale.replace(0, np.nan)
        return group.clip(median - threshold * scale, median + threshold * scale, axis=1)

    return frame.groupby(level="date", group_keys=False).apply(clip)


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (frame - mean) / std


def preprocess_factors(frame: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    return zscore(winsorize_mad(frame, threshold)).replace([np.inf, -np.inf], np.nan)

