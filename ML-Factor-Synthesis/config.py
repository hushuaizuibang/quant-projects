"""Central configuration for the factor research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class DataConfig:
    source: str = "baostock"
    start_date: str = "2018-01-01"
    end_date: str = "2026-07-31"
    synthetic_symbols: int = 50
    seed: int = 42
    cache_dir: Path = ROOT / "data" / "cache"
    constituents_file: Path | None = None
    index_code: str = "000300.SH"


@dataclass(slots=True)
class FeatureConfig:
    lookback_days: int = 60
    forward_days: int = 20
    neutralization: str = "none"
    mad_threshold: float = 3.0
    target_mode: str = "rank"
    select_features: bool = True
    factor_icir_threshold: float = 0.5
    lags_per_factor: int = 3
    min_selected_features: int = 20
    max_selected_features: int = 30


@dataclass(slots=True)
class ModelConfig:
    early_stopping_rounds: int = 50
    random_state: int = 42
    selection_metric: str = "ic"
    objective: str = "rank:pairwise"
    pca_components: int | None = None
    try_pca: bool = True
    pca_fallback_components: int = 10
    parameter_grid: tuple[dict, ...] = field(
        default_factory=lambda: (
            {
                "n_estimators": 350,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 2,
                "reg_lambda": 5.0,
            },
            {
                "n_estimators": 450,
                "max_depth": 3,
                "learning_rate": 0.035,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 3,
                "reg_lambda": 3.0,
            },
        )
    )


@dataclass(slots=True)
class BacktestConfig:
    quantile: float = 0.10
    commission: float = 0.0003
    slippage: float = 0.001
    max_turnover: float = 0.30
    bootstrap_samples: int = 500
    bootstrap_block_size: int = 3
    seed: int = 42


@dataclass(slots=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output_dir: Path = ROOT / "outputs_optimized"
