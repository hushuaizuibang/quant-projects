"""Leakage-aware XGBoost regression/ranking with IC-oriented selection."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import FeatureConfig, ModelConfig
from model.feature_selection import FeatureSelectionResult, select_features_by_icir

LOGGER = logging.getLogger(__name__)


@dataclass
class RidgeFallback:
    alpha: float = 20.0
    coefficients_: np.ndarray | None = None
    center_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeFallback":
        self.center_ = np.nanmedian(x, axis=0)
        clean = np.where(np.isfinite(x), x, self.center_)
        self.intercept_ = float(y.mean())
        centered_y = y - self.intercept_
        self.coefficients_ = (clean.T @ centered_y) / ((clean * clean).sum(axis=0) + self.alpha)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(x), x, self.center_)
        return self.intercept_ + clean @ self.coefficients_

    @property
    def feature_importances_(self) -> np.ndarray:
        values = np.abs(self.coefficients_)
        return values / values.sum() if values.sum() else values


@dataclass
class PCATransformer:
    """Train-only median imputation, standardization, and PCA."""

    median_: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    components_: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, components: int) -> "PCATransformer":
        from sklearn.decomposition import PCA

        median = np.nanmedian(x, axis=0)
        clean = np.where(np.isfinite(x), x, median)
        mean = clean.mean(axis=0)
        scale = clean.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (clean - mean) / scale
        count = min(components, standardized.shape[1], standardized.shape[0])
        pca = PCA(n_components=count, random_state=42).fit(standardized)
        return cls(median, mean, scale, pca.components_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(x), x, self.median_)
        return ((clean - self.mean_) / self.scale_) @ self.components_.T


@dataclass
class TrainedModel:
    estimator: object
    feature_names: list[str]
    backend: str
    validation_rmse: float
    validation_mae: float
    validation_ic: float = np.nan
    transformer: PCATransformer | None = None
    feature_selection: FeatureSelectionResult | None = None
    candidate_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    scaling_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_names].to_numpy()
        if self.transformer is not None:
            values = self.transformer.transform(values)
        return np.asarray(self.estimator.predict(values))

    def original_feature_importances(self) -> np.ndarray:
        importance = np.asarray(self.estimator.feature_importances_, dtype=float)
        if self.transformer is not None:
            importance = np.abs(self.transformer.components_).T @ importance
        return importance / importance.sum() if importance.sum() else importance

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    report: pd.DataFrame
    models: dict[int, TrainedModel]


def walk_forward_split(
    samples: pd.DataFrame, test_year: int, forward_days: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return expanding train, purged prior-year validation, and current-year test."""
    dates = samples.index.get_level_values("date")
    cutoff = pd.Timestamp(test_year, 1, 1)
    if "_label_date" in samples:
        observable = samples["_label_date"].notna() & (samples["_label_date"] < cutoff)
    else:
        observable = dates + pd.offsets.BDay(forward_days) < cutoff
    history = samples.loc[observable & (dates.year >= 2018)]
    history_dates = history.index.get_level_values("date")
    train = history.loc[history_dates.year <= test_year - 2]
    validation = history.loc[history_dates.year == test_year - 1]
    test = samples.loc[dates.year == test_year]
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"Insufficient data for {test_year} walk-forward split")
    return train, validation, test


def walk_forward_train_predict(
    samples: pd.DataFrame,
    config: ModelConfig,
    forward_days: int = 20,
    first_test_year: int = 2024,
    model_dir: Path | None = None,
    feature_config: FeatureConfig | None = None,
) -> WalkForwardResult:
    years = sorted(
        year for year in samples.index.get_level_values("date").year.unique() if year >= first_test_year
    )
    predictions: list[pd.Series] = []
    reports: list[dict[str, float | int | str]] = []
    models: dict[int, TrainedModel] = {}
    for year in years:
        train, validation, test = walk_forward_split(samples, year, forward_days)
        model = train_model(train, validation, config, feature_config)
        score = pd.Series(model.predict(test), index=test.index, name="ml_score")
        evaluated = test["target"].notna()
        error = score.loc[evaluated] - test.loc[evaluated, "target"]
        monthly_ic = (
            pd.concat([score.rename("score"), test["target"]], axis=1)
            .groupby(level="date")
            .apply(lambda day: day["score"].corr(day["target"], method="spearman"))
        )
        reports.append(
            {
                "test_year": year,
                "backend": model.backend,
                "objective": config.objective,
                "selected_features": len(model.feature_names),
                "model_dimensions": (
                    model.transformer.components_.shape[0]
                    if model.transformer is not None
                    else len(model.feature_names)
                ),
                "train_start": str(train.index.get_level_values("date").min().date()),
                "train_end": str(train.index.get_level_values("date").max().date()),
                "validation_start": str(validation.index.get_level_values("date").min().date()),
                "validation_end": str(validation.index.get_level_values("date").max().date()),
                "train_samples": len(train.dropna(subset=["target"])),
                "validation_samples": len(validation.dropna(subset=["target"])),
                "validation_rmse": model.validation_rmse,
                "validation_mae": model.validation_mae,
                "validation_ic": model.validation_ic,
                "test_rmse": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                "test_mae": float(np.mean(np.abs(error))) if len(error) else np.nan,
                "mean_monthly_ic": float(monthly_ic.mean()),
                "monthly_icir": float(monthly_ic.mean() / monthly_ic.std(ddof=1)),
            }
        )
        predictions.append(score)
        models[year] = model
        if model_dir:
            model.save(model_dir / f"model_{year}.pkl")
        LOGGER.info(
            "Walk-forward %d: train=%d validation=%d test=%d features=%d mean IC=%.4f",
            year,
            len(train),
            len(validation),
            len(test),
            len(model.feature_names),
            monthly_ic.mean(),
        )
    combined = pd.concat(predictions).sort_index()
    return WalkForwardResult(combined, pd.DataFrame(reports).set_index("test_year"), models)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    error = prediction - y
    return float(np.sqrt(np.mean(error**2))), float(np.mean(np.abs(error)))


def _mean_monthly_ic(index: pd.MultiIndex, y: np.ndarray, prediction: np.ndarray) -> float:
    frame = pd.DataFrame({"target": y, "prediction": prediction}, index=index)
    values = frame.groupby(level="date").apply(
        lambda day: day["prediction"].corr(day["target"], method="spearman")
    )
    return float(values.mean())


def _is_better_candidate(
    candidate: tuple[float, float, float, object],
    best: tuple[float, float, float, object] | None,
    metric: str,
) -> bool:
    if metric not in {"rmse", "ic"}:
        raise ValueError("selection_metric must be 'rmse' or 'ic'")
    if best is None:
        return True
    if metric == "rmse":
        return candidate[0] < best[0]
    candidate_ic = candidate[2] if np.isfinite(candidate[2]) else -np.inf
    best_ic = best[2] if np.isfinite(best[2]) else -np.inf
    return candidate_ic > best_ic


def _distribution(split: str, y: np.ndarray, prediction: np.ndarray) -> dict[str, float | str | int]:
    prediction_std = float(np.std(prediction, ddof=1)) if len(prediction) > 1 else 0.0
    target_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    return {
        "split": split,
        "samples": len(prediction),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": prediction_std,
        "prediction_min": float(np.min(prediction)),
        "prediction_p05": float(np.quantile(prediction, 0.05)),
        "prediction_p50": float(np.quantile(prediction, 0.50)),
        "prediction_p95": float(np.quantile(prediction, 0.95)),
        "prediction_max": float(np.max(prediction)),
        "target_mean": float(np.mean(y)),
        "target_std": target_std,
        "prediction_to_target_std": prediction_std / target_std if target_std else np.nan,
    }


def _scaling_table(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for split, frame in (("train", train), ("validation", validation)):
        for feature in features:
            values = frame[feature]
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "missing_rate": values.isna().mean(),
                }
            )
    return pd.DataFrame(rows)


def _ranking_labels(index: pd.MultiIndex, target: np.ndarray) -> np.ndarray:
    values = pd.Series(target, index=index)
    return values.groupby(level="date").rank(method="dense").sub(1).to_numpy()


def _group_sizes(index: pd.MultiIndex) -> list[int]:
    return pd.Series(1, index=index).groupby(level="date", sort=False).sum().astype(int).tolist()


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ModelConfig,
    feature_config: FeatureConfig | None = None,
) -> TrainedModel:
    feature_selection = None
    if feature_config is not None and feature_config.select_features:
        feature_selection = select_features_by_icir(train, feature_config)
        features = feature_selection.features
        if not features:
            raise ValueError(
                "No features passed the training-only ICIR threshold; lower the threshold explicitly"
            )
    else:
        features = [
            column
            for column in train
            if column not in {"target", "target_return"} and not column.startswith("_")
        ]

    train_clean = train.dropna(subset=["target"]).sort_index()
    valid_clean = validation.dropna(subset=["target"]).sort_index()
    x_train, y_train = train_clean[features].to_numpy(), train_clean["target"].to_numpy()
    x_valid, y_valid = valid_clean[features].to_numpy(), valid_clean["target"].to_numpy()
    fallback_transformer = (
        PCATransformer.fit(x_train, config.pca_components)
        if config.pca_components is not None
        else None
    )
    x_train_fallback = fallback_transformer.transform(x_train) if fallback_transformer else x_train
    x_valid_fallback = fallback_transformer.transform(x_valid) if fallback_transformer else x_valid

    try:
        from xgboost import XGBRanker, XGBRegressor
    except ImportError:
        LOGGER.warning("xgboost is unavailable; using deterministic ridge fallback")
        estimator = RidgeFallback().fit(x_train_fallback, y_train)
        validation_prediction = estimator.predict(x_valid_fallback)
        train_prediction = estimator.predict(x_train_fallback)
        rmse, mae = _metrics(y_valid, validation_prediction)
        validation_ic = _mean_monthly_ic(valid_clean.index, y_valid, validation_prediction)
        return TrainedModel(
            estimator,
            features,
            "ridge-fallback",
            rmse,
            mae,
            validation_ic,
            transformer=fallback_transformer,
            feature_selection=feature_selection,
            prediction_distribution=pd.DataFrame(
                [
                    _distribution("train", y_train, train_prediction),
                    _distribution("validation", y_valid, validation_prediction),
                ]
            ),
            scaling_diagnostics=_scaling_table(train_clean, valid_clean, features),
        )

    if config.objective not in {"reg:squarederror", "rank:pairwise", "rank:ndcg"}:
        raise ValueError("objective must be reg:squarederror, rank:pairwise, or rank:ndcg")

    best: tuple[float, float, float, object, PCATransformer | None] | None = None
    candidate_rows: list[dict[str, object]] = []
    pca_options: list[int | None] = [None]
    requested_pca = config.pca_components
    if requested_pca is not None:
        pca_options.append(requested_pca)
    elif config.try_pca and len(features) > config.pca_fallback_components:
        pca_options.append(config.pca_fallback_components)
    candidate_id = 0
    for pca_components in pca_options:
        transformer = (
            PCATransformer.fit(x_train, pca_components)
            if pca_components is not None
            else None
        )
        x_train_model = transformer.transform(x_train) if transformer else x_train
        x_valid_model = transformer.transform(x_valid) if transformer else x_valid
        for params in config.parameter_grid:
            candidate_id += 1
            common = {
                "random_state": config.random_state,
                "n_jobs": -1,
                "early_stopping_rounds": config.early_stopping_rounds,
                **params,
            }
            if config.objective.startswith("rank:"):
                # CSI 300 relevance labels exceed the default exponential-gain limit of 31.
                common["ndcg_exp_gain"] = False
            if config.objective.startswith("rank:"):
                estimator = XGBRanker(objective=config.objective, eval_metric="ndcg", **common)
                estimator.fit(
                    x_train_model,
                    _ranking_labels(train_clean.index, y_train),
                    group=_group_sizes(train_clean.index),
                    eval_set=[(x_valid_model, _ranking_labels(valid_clean.index, y_valid))],
                    eval_group=[_group_sizes(valid_clean.index)],
                    verbose=False,
                )
            else:
                estimator = XGBRegressor(
                    objective=config.objective,
                    eval_metric="rmse",
                    **common,
                )
                estimator.fit(
                    x_train_model,
                    y_train,
                    eval_set=[(x_valid_model, y_valid)],
                    verbose=False,
                )
            prediction = estimator.predict(x_valid_model)
            rmse, mae = _metrics(y_valid, prediction)
            validation_ic = _mean_monthly_ic(valid_clean.index, y_valid, prediction)
            candidate_rows.append(
                {
                    "candidate": candidate_id,
                    "objective": config.objective,
                    "pca_components": pca_components,
                    "params": json.dumps(params, sort_keys=True),
                    "validation_rmse": rmse,
                    "validation_mae": mae,
                    "validation_ic": validation_ic,
                    "best_iteration": getattr(estimator, "best_iteration", np.nan),
                }
            )
            LOGGER.info(
                "Validation RMSE %.6f MAE %.6f IC %.4f PCA=%s for %s",
                rmse,
                mae,
                validation_ic,
                pca_components,
                params,
            )
            candidate = (rmse, mae, validation_ic, estimator, transformer)
            if _is_better_candidate(candidate, best, config.selection_metric):
                best = candidate
    assert best is not None
    selected_estimator = best[3]
    selected_transformer = best[4]
    selected_x_train = selected_transformer.transform(x_train) if selected_transformer else x_train
    selected_x_valid = selected_transformer.transform(x_valid) if selected_transformer else x_valid
    train_prediction = selected_estimator.predict(selected_x_train)
    validation_prediction = selected_estimator.predict(selected_x_valid)
    candidate_metrics = pd.DataFrame(candidate_rows)
    if config.selection_metric == "ic":
        selected_index = candidate_metrics["validation_ic"].idxmax()
    else:
        selected_index = candidate_metrics["validation_rmse"].idxmin()
    candidate_metrics["selected"] = False
    candidate_metrics.loc[selected_index, "selected"] = True
    return TrainedModel(
        selected_estimator,
        features,
        "xgboost",
        best[0],
        best[1],
        best[2],
        transformer=selected_transformer,
        feature_selection=feature_selection,
        candidate_metrics=candidate_metrics,
        prediction_distribution=pd.DataFrame(
            [
                _distribution("train", y_train, train_prediction),
                _distribution("validation", y_valid, validation_prediction),
            ]
        ),
        scaling_diagnostics=_scaling_table(train_clean, valid_clean, features),
    )
