from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


FACTOR_NAMES = ("size", "value", "momentum", "quality", "low_volatility")


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the point-in-time multi-factor backtest."""

    start_date: str = "20210101"
    end_date: str | None = None
    stock_count: int = 300
    rebalance_rule: str = "monthly"

    # Signal construction
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "size": 0.15,
            "value": 0.25,
            "momentum": 0.25,
            "quality": 0.20,
            "low_volatility": 0.15,
        }
    )
    scoring_method: str = "icir"
    icir_lookback_months: int = 24
    icir_min_periods: int = 12
    ic_method: str = "spearman"
    winsor_mad: float = 3.0
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1
    volatility_lookback_days: int = 60

    # Portfolio construction
    top_quantile: float = 0.20
    covariance_lookback_months: int = 24
    risk_aversion: float = 4.0
    turnover_penalty: float = 0.10
    max_stock_weight: float = 0.10
    max_industry_deviation: float = 0.05
    max_size_exposure: float = 0.15
    optimizer: str = "mean_variance"

    # Costs are charged on one-way turnover: sum(abs(w_t - w_{t-1})).
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0010

    # Kept as a compatibility alias for older command lines.
    lookback_months: int = 12

    data_source: str = "akshare"
    use_sample_data: bool = False
    random_seed: int = 42
    out_of_sample_start: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.top_quantile <= 1:
            raise ValueError("top_quantile must be in (0, 1].")
        if not 0 < self.max_stock_weight <= 1:
            raise ValueError("max_stock_weight must be in (0, 1].")
        unknown = set(self.factor_weights).difference(FACTOR_NAMES)
        if unknown:
            raise ValueError(f"Unknown factors: {sorted(unknown)}")

    @property
    def resolved_end_date(self) -> str:
        return self.end_date or datetime.today().strftime("%Y%m%d")

    @property
    def total_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate
