"""Feature importance and optional SHAP output."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.train import TrainedModel

LOGGER = logging.getLogger(__name__)


def _save_importance(
    feature_names: list[str], importance: np.ndarray, output_dir: Path, top_n: int
) -> pd.DataFrame:
    feature_table = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    feature_table.to_csv(output_dir / "feature_importance.csv", index=False)
    factor_table = (
        feature_table.assign(factor=feature_table["feature"].str.split("__", n=1).str[0])
        .groupby("factor", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    factor_table.to_csv(output_dir / "factor_importance.csv", index=False)
    shown = factor_table.head(top_n).sort_values("importance")
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(shown["factor"], shown["importance"], color="#267365")
    axis.set(title="Aggregated Alpha importance", xlabel="Importance across 60 lags")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
    return factor_table


def save_feature_importance(model: TrainedModel, output_dir: Path, top_n: int = 30) -> pd.DataFrame:
    importance = model.original_feature_importances()
    return _save_importance(model.feature_names, importance, output_dir, top_n)


def save_walk_forward_importance(
    models: dict[int, TrainedModel], output_dir: Path, top_n: int = 30
) -> pd.DataFrame:
    if not models:
        raise ValueError("At least one walk-forward model is required")
    rows = []
    for year, model in sorted(models.items()):
        row = pd.Series(
            model.original_feature_importances(), index=model.feature_names, name=year
        )
        rows.append(row)
    yearly = pd.DataFrame(rows).fillna(0.0)
    yearly.to_csv(output_dir / "yearly_feature_importance.csv", index_label="test_year")
    return _save_importance(yearly.columns.tolist(), yearly.mean().to_numpy(), output_dir, top_n)


def save_shap_summary(model: TrainedModel, samples: pd.DataFrame, output_dir: Path) -> bool:
    if model.backend != "xgboost":
        return False
    if model.transformer is not None:
        LOGGER.info("Skipping SHAP because PCA components are not direct factor features")
        return False
    try:
        import shap
    except ImportError:
        LOGGER.info("SHAP is not installed; skipping optional shap_summary.png")
        return False
    sample = samples[model.feature_names].dropna().sample(min(1000, len(samples)), random_state=42)
    values = shap.TreeExplainer(model.estimator).shap_values(sample)
    shap.summary_plot(values, sample, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    return True
