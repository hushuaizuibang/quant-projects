from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    universe_name: str = "CSI300"
    start: str = "2018-01-01"
    end: str = "2025-12-31"
    rebalance_freq: str = "W-FRI"
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_rebalance_turnover: float | None = 0.50
    min_factor_weight: float = 0.0
    max_factor_weight: float = 0.35
    random_seed: int = 42
