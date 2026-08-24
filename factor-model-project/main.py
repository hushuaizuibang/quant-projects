from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.ff_three_factor.backtest import run_backtest
from src.ff_three_factor.config import BacktestConfig
from src.ff_three_factor.data import load_market_data
from src.ff_three_factor.factors import build_factor_model
from src.ff_three_factor.performance import (
    build_performance_report,
    factor_ic_summary,
    factor_quantile_returns,
    format_metrics,
)
from src.ff_three_factor.visualization import (
    plot_factor_diagnostics,
    plot_monthly_distribution,
    plot_nav_curves,
    plot_portfolio_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSI 300 industry- and size-neutral multi-factor portfolio backtest."
    )
    parser.add_argument("--start-date", default="20210101", help="YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="YYYYMMDD; defaults to today")
    parser.add_argument("--stock-count", type=int, default=300)
    parser.add_argument("--top-quantile", type=float, default=0.20)
    parser.add_argument("--scoring-method", choices=["fixed", "icir"], default="icir")
    parser.add_argument("--icir-lookback-months", type=int, default=24)
    parser.add_argument("--icir-min-periods", type=int, default=12)
    parser.add_argument("--max-stock-weight", type=float, default=0.10)
    parser.add_argument("--max-industry-deviation", type=float, default=0.05)
    parser.add_argument("--max-size-exposure", type=float, default=0.15)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.0010)
    parser.add_argument("--out-of-sample-start", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--data-source",
        choices=["akshare", "yfinance", "sample"],
        default="akshare",
        help="akshare prefers data/akshare/csi300_prices.csv and only downloads when the cache is absent.",
    )
    parser.add_argument("--use-sample-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Also rerun a small parameter grid (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = "sample" if args.use_sample_data else args.data_source
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        stock_count=args.stock_count,
        top_quantile=args.top_quantile,
        scoring_method=args.scoring_method,
        icir_lookback_months=args.icir_lookback_months,
        icir_min_periods=args.icir_min_periods,
        max_stock_weight=args.max_stock_weight,
        max_industry_deviation=args.max_industry_deviation,
        max_size_exposure=args.max_size_exposure,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        out_of_sample_start=args.out_of_sample_start,
        data_source=source,
        use_sample_data=source == "sample",
    )

    market_data = load_market_data(config)
    factor_model = build_factor_model(market_data, config)
    result = run_backtest(factor_model, config)
    report = build_performance_report(result, config.out_of_sample_start)

    output_dir = args.output_dir / f"{source}_{config.scoring_method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_nav_curves(result, output_dir / "strategy_performance.png")
    plot_monthly_distribution(result.strategy_returns, output_dir / "monthly_return_distribution.png")
    plot_factor_diagnostics(
        factor_model.factor_ic,
        result.factor_score_weights,
        output_dir / "factor_diagnostics.png",
    )
    plot_portfolio_diagnostics(result, output_dir / "portfolio_diagnostics.png")

    report.overall_metrics.rename("value").to_csv(output_dir / "overall_metrics.csv")
    report.yearly_metrics.to_csv(output_dir / "yearly_metrics.csv", index_label="year")
    report.sample_metrics.to_csv(output_dir / "sample_metrics.csv", index_label="sample")
    report.cost_sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index_label="cost")
    factor_model.factor_ic.to_csv(output_dir / "factor_ic.csv", index_label="signal_date")
    factor_ic_summary(factor_model.factor_ic).to_csv(output_dir / "factor_ic_summary.csv")
    factor_model.factor_returns.to_csv(output_dir / "factor_mimicking_returns.csv", index_label="date")
    factor_quantile_returns(factor_model).to_csv(output_dir / "factor_quantile_returns.csv", index=False)
    result.factor_score_weights.to_csv(output_dir / "factor_weights.csv", index_label="date")
    result.weights.to_csv(output_dir / "portfolio_weights.csv", index_label="date")
    result.turnover.to_csv(output_dir / "turnover.csv", index_label="date")
    result.industry_exposure.to_csv(output_dir / "industry_active_weights.csv", index_label="date")
    result.size_exposure.to_csv(output_dir / "size_exposure.csv", index_label="date")
    result.optimizer_status.to_csv(output_dir / "optimizer_status.csv", index_label="date")
    run_metadata = dict(market_data.metadata or {})
    run_metadata.update(
        {
            "available_factors": ",".join(factor_model.available_factors),
            "scoring_method": config.scoring_method,
            "commission_rate": config.commission_rate,
            "slippage_rate": config.slippage_rate,
        }
    )
    pd.Series(run_metadata, name="value").to_csv(output_dir / "data_quality.csv", index_label="field")
    pd.concat(
        [
            result.gross_returns.rename("gross_return"),
            result.transaction_costs.rename("transaction_cost"),
            result.strategy_returns.rename("net_return"),
            result.benchmark_returns.rename("benchmark_return"),
        ],
        axis=1,
    ).to_csv(output_dir / "monthly_returns.csv", index_label="date")

    if args.run_sensitivity:
        _parameter_sensitivity(factor_model, config).to_csv(
            output_dir / "parameter_sensitivity.csv", index=False
        )

    print(format_metrics(report.overall_metrics, title="Overall Performance"))
    print("\nFactor IC Summary")
    print(factor_ic_summary(factor_model.factor_ic).round(4).to_string())
    print(f"\nAvailable factors: {', '.join(factor_model.available_factors)}")
    fallback_count = result.optimizer_status.str.startswith("fallback").sum()
    print(f"\nOptimizer fallbacks: {fallback_count}/{len(result.optimizer_status)}")
    print(f"Outputs: {output_dir.resolve()}")


def _parameter_sensitivity(model, config: BacktestConfig) -> pd.DataFrame:
    rows = []
    cases = [
        ("base", config),
        ("top_10pct", replace(config, top_quantile=0.10)),
        ("top_30pct", replace(config, top_quantile=0.30)),
        ("industry_3pct", replace(config, max_industry_deviation=0.03)),
        ("industry_10pct", replace(config, max_industry_deviation=0.10)),
    ]
    for label, variant in cases:
        result = run_backtest(model, variant)
        metrics = build_performance_report(result).overall_metrics
        rows.append({"case": label, **metrics.to_dict(), "Average Turnover": result.turnover.mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
