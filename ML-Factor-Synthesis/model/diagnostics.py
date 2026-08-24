"""Diagnostics that connect regression quality to portfolio behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.train import TrainedModel


def build_model_diagnostics(
    score: pd.Series,
    target: pd.Series,
    samples: pd.DataFrame,
    models: dict[int, TrainedModel],
    backtest: pd.DataFrame,
) -> dict[str, object]:
    aligned = pd.concat([score.rename("score"), target.rename("target")], axis=1).dropna()
    monthly = aligned.groupby(level="date")
    monthly_ic = monthly.apply(lambda day: day["score"].corr(day["target"], method="spearman"))
    prediction_std = monthly["score"].std()
    target_std = monthly["target"].std()
    error = aligned["score"] - aligned["target"]
    feature_names = next(iter(models.values())).feature_names
    test_rmse = float(np.sqrt(np.mean(error**2)))
    zero_rmse = float(np.sqrt(np.mean(aligned["target"] ** 2)))
    gross_wealth = float((1 + backtest["gross_return"]).prod())
    net_wealth = float((1 + backtest["net_return"]).prod())
    periods = len(backtest)

    dates = samples.index.get_level_values("date")
    train_counts = {
        year: int(((dates.year >= 2018) & (dates.year <= year - 2) & samples["target"].notna()).sum())
        for year in sorted(models)
    }
    return {
        "feature_count": len(feature_names),
        "test_rows": len(aligned),
        "test_months": int(aligned.index.get_level_values("date").nunique()),
        "symbols_per_month_mean": float(monthly.size().mean()),
        "train_samples_by_test_year": train_counts,
        "feature_to_train_sample_ratio": {
            str(year): len(feature_names) / count if count else None
            for year, count in train_counts.items()
        },
        "best_iteration_by_test_year": {
            str(year): int(getattr(model.estimator, "best_iteration", -1)) for year, model in models.items()
        },
        "prediction_mean": float(aligned["score"].mean()),
        "prediction_std": float(aligned["score"].std()),
        "prediction_monthly_std_mean": float(prediction_std.mean()),
        "target_std": float(aligned["target"].std()),
        "target_monthly_std_mean": float(target_std.mean()),
        "prediction_to_target_monthly_dispersion": float(prediction_std.mean() / target_std.mean()),
        "overall_pearson": float(aligned["score"].corr(aligned["target"])),
        "overall_spearman": float(aligned["score"].corr(aligned["target"], method="spearman")),
        "monthly_ic_mean": float(monthly_ic.mean()),
        "monthly_ic_std": float(monthly_ic.std(ddof=1)),
        "monthly_ic_positive_rate": float((monthly_ic > 0).mean()),
        "test_rmse": test_rmse,
        "zero_prediction_rmse": zero_rmse,
        "rmse_skill_vs_zero": float(1 - test_rmse / zero_rmse),
        "missing_feature_rate": float(samples[feature_names].isna().to_numpy().mean()),
        "annualized_gross_return": float(gross_wealth ** (12 / periods) - 1),
        "annualized_net_return": float(net_wealth ** (12 / periods) - 1),
        "annualized_cost_drag": float(gross_wealth ** (12 / periods) - net_wealth ** (12 / periods)),
        "average_turnover": float(backtest["turnover"].mean()),
    }


def save_model_diagnostics(diagnostics: dict[str, object], output_dir: Path) -> None:
    (output_dir / "model_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
