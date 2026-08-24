"""Local cache with a parquet-first, CSV fallback policy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_frame(path: Path) -> pd.DataFrame | None:
    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv.gz")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        frame = pd.read_csv(csv, parse_dates=["date"])
        return frame
    return None


def save_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet = path.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet, index=False)
        return parquet
    except (ImportError, ModuleNotFoundError):
        csv = path.with_suffix(".csv.gz")
        frame.to_csv(csv, index=False)
        return csv

