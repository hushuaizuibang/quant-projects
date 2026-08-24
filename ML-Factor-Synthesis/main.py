"""Command-line entry point for the complete factor synthesis study."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
from pathlib import Path

import pandas as pd

from backtest.metrics import bootstrap_confidence_interval, moving_block_bootstrap, performance_metrics
from backtest.portfolio import (
    equal_weight_score,
    icir_weighted_score,
    run_backtest,
    single_factor_benchmarks,
)
from config import ProjectConfig
from data.downloader import load_market_data
from data.preprocess import preprocess_factors
from factors import calculate_factors
from factors.feature_pipeline import build_monthly_samples, time_split
from factors.neutralize import neutralize
from model.explain import save_feature_importance, save_shap_summary, save_walk_forward_importance
from model.diagnostics import build_model_diagnostics, save_model_diagnostics
from model.train import train_model, walk_forward_train_predict
from reporting import (
    plot_bootstrap,
    plot_ic_analysis,
    plot_neutralization_comparison,
    plot_performance,
    plot_yearly_performance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML synthesis of Alpha101 signals")
    parser.add_argument(
        "--source",
        choices=["synthetic", "akshare", "tushare", "baostock"],
        default="baostock",
    )
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-07-31")
    parser.add_argument("--symbols", type=int, default=50, help="Synthetic universe size")
    parser.add_argument("--neutralization", choices=["none", "industry", "size", "both"], default="none")
    parser.add_argument("--constituents-file", type=Path)
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_optimized"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--shap", action="store_true", help="Create optional SHAP plot when shap is installed")
    parser.add_argument("--static-model", action="store_true", help="Disable annual walk-forward retraining")
    parser.add_argument(
        "--model-selection",
        choices=["rmse", "ic"],
        default="ic",
        help="Select hyperparameters by validation RMSE or mean monthly IC",
    )
    parser.add_argument(
        "--target-mode",
        choices=["raw", "demean", "rank"],
        default="rank",
        help="Model label; raw returns are always retained for backtesting",
    )
    parser.add_argument(
        "--objective",
        choices=["reg:squarederror", "rank:pairwise", "rank:ndcg"],
        default="rank:pairwise",
    )
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--no-pca-candidate", action="store_true")
    parser.add_argument("--icir-threshold", type=float, default=0.5)
    parser.add_argument("--lags-per-factor", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--no-feature-selection", action="store_true")
    parser.add_argument(
        "--neutralization-experiment", action="store_true", help="Run none/industry/size/both ML comparison"
    )
    return parser.parse_args()


def _prepare_samples(
    raw_factors: pd.DataFrame, market: pd.DataFrame, config: ProjectConfig, mode: str
) -> pd.DataFrame:
    adjusted = neutralize(raw_factors, market, mode)
    factors = preprocess_factors(adjusted, config.features.mad_threshold)
    return build_monthly_samples(
        factors,
        market,
        config.features.lookback_days,
        config.features.forward_days,
        config.features.target_mode,
    )


def _backtest_score(score: pd.Series, target: pd.Series, config: ProjectConfig) -> pd.DataFrame:
    return run_backtest(
        score.reindex(target.index),
        target,
        config.backtest.quantile,
        config.backtest.commission,
        config.backtest.slippage,
        config.backtest.max_turnover,
    )


def _run_ml(
    samples: pd.DataFrame,
    config: ProjectConfig,
    model_dir: Path,
    walk_forward: bool,
) -> tuple[pd.Series, object, str, float, float]:
    train, validation, test = time_split(samples)
    all_samples = pd.concat([train, validation, test]).sort_index()
    test_evaluation = test.dropna(subset=["target"])
    if walk_forward:
        walk = walk_forward_train_predict(
            all_samples,
            config.model,
            config.features.forward_days,
            model_dir=model_dir,
            feature_config=config.features,
        )
        score = walk.predictions.reindex(test_evaluation.index).rename("ml_score")
        backend = ",".join(sorted({model.backend for model in walk.models.values()}))
        return (
            score,
            walk,
            backend,
            float(walk.report["validation_rmse"].mean()),
            float(walk.report["validation_mae"].mean()),
        )
    model = train_model(train, validation, config.model, config.features)
    model.save(model_dir / "model.pkl")
    score = pd.Series(model.predict(test_evaluation), index=test_evaluation.index, name="ml_score")
    return score, model, model.backend, model.validation_rmse, model.validation_mae


def _neutralization_experiment(
    raw_factors: pd.DataFrame,
    market: pd.DataFrame,
    config: ProjectConfig,
    primary_result: pd.DataFrame,
    walk_forward: bool,
) -> None:
    output = config.output_dir
    summary_rows, yearly_rows = [], []
    for mode in ("none", "industry", "size", "both"):
        logging.info("Neutralization experiment: %s", mode)
        if mode == config.features.neutralization:
            result = primary_result
        else:
            samples = _prepare_samples(raw_factors, market, config, mode)
            _, _, test = time_split(samples)
            target = test.dropna(subset=["target_return"])["target_return"]
            score, fitted, _, _, _ = _run_ml(
                samples, config, output / "neutralization_models" / mode, walk_forward
            )
            result = _backtest_score(score, target, config)
            if walk_forward:
                fitted.report.to_csv(output / f"walk_forward_{mode}.csv")
        result.to_csv(output / f"neutralization_backtest_{mode}.csv")
        summary_rows.append({"neutralization": mode, **performance_metrics(result)})
        for year, group in result.groupby(result.index.year):
            yearly_rows.append(
                {"neutralization": mode, "year": year, "net_return": (1 + group["net_return"]).prod() - 1}
            )
    summary = pd.DataFrame(summary_rows).set_index("neutralization")
    summary.to_csv(output / "neutralization_summary.csv")
    pd.DataFrame(yearly_rows).to_csv(output / "neutralization_yearly.csv", index=False)
    plot_neutralization_comparison(summary, output)


def run(
    config: ProjectConfig,
    refresh: bool = False,
    with_shap: bool = False,
    walk_forward: bool = True,
    neutralization_experiment: bool = False,
) -> pd.DataFrame:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s market data", config.data.source)
    market = load_market_data(config.data, refresh=refresh)
    logging.info("Calculating 20 paper-defined Alpha101 factors")
    raw_factors = calculate_factors(market)
    logging.info("Building monthly samples with %d daily lags", config.features.lookback_days)
    samples = _prepare_samples(raw_factors, market, config, config.features.neutralization)
    train, validation, test = time_split(samples)
    all_samples = pd.concat([train, validation, test]).sort_index()
    test_evaluation = test.dropna(subset=["target", "target_return"])
    model_target = test_evaluation["target"]
    target = test_evaluation["target_return"]
    logging.info("Samples: train=%d validation=%d test=%d", len(train), len(validation), len(test))

    ml_score, fitted, backend, validation_rmse, validation_mae = _run_ml(
        samples, config, output / "models", walk_forward
    )
    if walk_forward:
        fitted.report.to_csv(output / "walk_forward_report.csv")
        save_walk_forward_importance(fitted.models, output)
        latest_model = fitted.models[max(fitted.models)]
    else:
        latest_model = fitted
        save_feature_importance(latest_model, output)
    logging.info("Backend=%s mean validation RMSE=%.6f MAE=%.6f", backend, validation_rmse, validation_mae)
    _save_training_diagnostics(fitted, output, walk_forward)
    if with_shap:
        latest_year = test.index.get_level_values("date").year.max()
        latest_samples = test.loc[test.index.get_level_values("date").year == latest_year]
        save_shap_summary(latest_model, latest_samples, output)

    equal_score = equal_weight_score(test_evaluation)
    icir_score = icir_weighted_score(all_samples, config.features.forward_days).reindex(test_evaluation.index)

    current_features = [column for column in all_samples if column.endswith("__lag_00")]
    factor_history = all_samples[current_features].rename(columns=lambda name: name.removesuffix("__lag_00"))
    factor_history.corr(method="spearman").to_csv(output / "factor_correlation.csv")
    factor_monthly_ic = factor_history.join(all_samples["target"]).groupby(level="date").apply(
        lambda day: day.drop(columns="target").corrwith(day["target"], method="spearman")
    )
    factor_ic_summary = pd.DataFrame(
        {
            "mean_ic": factor_monthly_ic.mean(),
            "ic_std": factor_monthly_ic.std(ddof=1),
            "positive_ic_rate": (factor_monthly_ic > 0).mean(),
        }
    )
    factor_ic_summary["icir"] = factor_ic_summary["mean_ic"] / factor_ic_summary["ic_std"]
    factor_ic_summary.sort_values("icir", ascending=False).to_csv(output / "factor_ic_summary.csv")

    single_scores, single_selection = single_factor_benchmarks(
        all_samples, 2024, config.features.forward_days
    )
    single_results = {
        name: _backtest_score(score, target, config) for name, score in single_scores.items()
    }
    single_summary = pd.DataFrame(
        [{"factor": name, **performance_metrics(result)} for name, result in single_results.items()]
    ).set_index("factor").join(single_selection)
    single_summary.to_csv(output / "single_factor_summary.csv")
    best_single = single_selection.index[0]

    strategy_scores = {
        "ML": ml_score,
        "Equal weight": equal_score,
        "ICIR weight": icir_score,
        f"Single: {best_single}": single_scores[best_single].reindex(test_evaluation.index),
    }
    results = {name: _backtest_score(score, target, config) for name, score in strategy_scores.items()}
    diagnostic_models = fitted.models if walk_forward else {2024: latest_model}
    save_model_diagnostics(
        build_model_diagnostics(
            ml_score, model_target, all_samples, diagnostic_models, results["ML"]
        ),
        output,
    )

    summary_rows, bootstrap_frames = [], []
    for name, result in results.items():
        filename = name.lower().replace(" ", "_").replace(":", "")
        result.to_csv(output / f"backtest_{filename}.csv")
        row = {"strategy": name, **performance_metrics(result)}
        row.update(
            bootstrap_confidence_interval(
                result["net_return"],
                config.backtest.bootstrap_samples,
                config.backtest.seed,
                block_size=config.backtest.bootstrap_block_size,
            )
        )
        summary_rows.append(row)
        bootstrap_frames.append(
            moving_block_bootstrap(
                result["net_return"],
                config.backtest.bootstrap_samples,
                config.backtest.seed,
                block_size=config.backtest.bootstrap_block_size,
            )
        )
    summary = pd.DataFrame(summary_rows).set_index("strategy")
    summary.to_csv(output / "backtest_summary.csv")
    bootstrap_distribution = pd.concat(
        bootstrap_frames, keys=results.keys(), names=["strategy", "sample"]
    )
    bootstrap_distribution.to_csv(output / "bootstrap_distribution.csv")
    plot_bootstrap(bootstrap_distribution, output)
    plot_performance(results, output)

    yearly_rows = []
    for name, result in results.items():
        for year, group in result.groupby(result.index.year):
            metrics = performance_metrics(group)
            yearly_rows.append(
                {
                    "strategy": name,
                    "year": year,
                    "net_return": (1 + group["net_return"]).prod() - 1,
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(output / "yearly_performance.csv", index=False)
    plot_yearly_performance(yearly, output)

    ic = plot_ic_analysis(ml_score, target, output)
    ic.rename("spearman_ic").to_csv(output / "monthly_ic.csv")
    pd.concat(strategy_scores, axis=1).join(
        pd.concat([model_target.rename("model_target"), target.rename("target_return")], axis=1)
    ).to_csv(output / "test_scores.csv")

    if neutralization_experiment:
        _neutralization_experiment(raw_factors, market, config, results["ML"], walk_forward)

    metadata = {
        "source": config.data.source,
        "index_code": config.data.index_code,
        "membership_source": (
            str(market["_membership_source"].iloc[0])
            if "_membership_source" in market
            else config.data.source
        ),
        "point_in_time_membership": (
            "_membership_source" in market
            and not market["_membership_source"].astype(str).str.contains(
                "current|survivorship", case=False, regex=True
            ).any()
        ),
        "point_in_time_industry_and_market_cap": (
            config.data.source == "tushare"
            and market["market_cap"].gt(0).mean() >= 0.95
            and (market["industry"].notna() & market["industry"].ne("Unknown")).mean() >= 0.95
        ),
        "market_cap_coverage": float(market["market_cap"].gt(0).mean()),
        "industry_coverage": float(
            (market["industry"].notna() & market["industry"].ne("Unknown")).mean()
        ),
        "start_date": config.data.start_date,
        "end_date": config.data.end_date,
        "synthetic_symbols": config.data.synthetic_symbols if config.data.source == "synthetic" else None,
        "neutralization": config.features.neutralization,
        "lookback_days": config.features.lookback_days,
        "forward_days": config.features.forward_days,
        "target_mode": config.features.target_mode,
        "feature_selection": config.features.select_features,
        "factor_icir_threshold": config.features.factor_icir_threshold,
        "lags_per_factor": config.features.lags_per_factor,
        "selected_feature_counts": {
            str(year): len(model.feature_names) for year, model in diagnostic_models.items()
        },
        "training_mode": "annual_walk_forward" if walk_forward else "static_2018_2022",
        "model_backend": backend,
        "model_estimator_class": type(latest_model.estimator).__name__,
        "fallback_used": backend != "xgboost",
        "model_selection_metric": config.model.selection_metric,
        "model_objective": config.model.objective,
        "pca_components": config.model.pca_components,
        "try_pca_candidate": config.model.try_pca,
        "python_version": platform.python_version(),
        "xgboost_version": importlib.metadata.version("xgboost") if backend == "xgboost" else None,
        "scikit_learn_version": importlib.metadata.version("scikit-learn") if backend == "xgboost" else None,
        "validation_rmse_mean": validation_rmse,
        "validation_mae_mean": validation_mae,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples_evaluated": len(test_evaluation),
        "single_factor_selected_pretest": best_single,
        "bootstrap_method": "circular_moving_block",
        "bootstrap_block_months": config.backtest.bootstrap_block_size,
        "neutralization_experiment": neutralization_experiment,
        "price_adjustment": {
            "baostock": "backward_adjusted",
            "akshare": "forward_adjusted",
            "tushare": "raw_price_times_cumulative_adjustment_factor",
            "synthetic": "not_applicable",
        }.get(config.data.source, "unknown"),
        "vwap_method": (
            "typical_price_(high+low+close)/3"
            if config.data.source in {"baostock", "akshare"}
            else "provider_or_generated_vwap"
        ),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("\n%s", summary.to_string(float_format=lambda value: f"{value:.4f}"))
    return summary


def _save_training_diagnostics(fitted: object, output: Path, walk_forward: bool) -> None:
    models = fitted.models if walk_forward else {2024: fitted}
    prediction_frames = []
    scaling_frames = []
    candidate_frames = []
    factor_frames = []
    lag_frames = []
    for year, model in sorted(models.items()):
        prediction_frames.append(model.prediction_distribution.assign(test_year=year))
        scaling_frames.append(model.scaling_diagnostics.assign(test_year=year))
        if not model.candidate_metrics.empty:
            candidate_frames.append(model.candidate_metrics.assign(test_year=year))
        if model.feature_selection is not None:
            factor_frames.append(
                model.feature_selection.factor_summary.reset_index().assign(test_year=year)
            )
            lag_frames.append(
                model.feature_selection.lag_summary.reset_index(names="feature").assign(
                    test_year=year
                )
            )
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output / "prediction_distribution.csv", index=False
        )
    if scaling_frames:
        pd.concat(scaling_frames, ignore_index=True).to_csv(
            output / "feature_scaling_diagnostics.csv", index=False
        )
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            output / "model_candidate_metrics.csv", index=False
        )
    if factor_frames:
        pd.concat(factor_frames, ignore_index=True).to_csv(
            output / "training_factor_icir.csv", index=False
        )
    if lag_frames:
        pd.concat(lag_frames, ignore_index=True).to_csv(
            output / "training_lag_ic.csv", index=False
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = ProjectConfig()
    config.data.source = args.source
    config.data.start_date = args.start_date
    config.data.end_date = args.end_date
    config.data.synthetic_symbols = args.symbols
    config.data.constituents_file = args.constituents_file
    config.data.index_code = args.index_code
    config.features.neutralization = args.neutralization
    config.features.target_mode = args.target_mode
    config.features.select_features = not args.no_feature_selection
    config.features.factor_icir_threshold = args.icir_threshold
    config.features.lags_per_factor = args.lags_per_factor
    config.model.selection_metric = args.model_selection
    config.model.objective = args.objective
    config.model.pca_components = args.pca_components
    config.model.try_pca = not args.no_pca_candidate
    config.output_dir = args.output_dir.resolve()
    run(
        config,
        refresh=args.refresh,
        with_shap=args.shap,
        walk_forward=not args.static_model,
        neutralization_experiment=args.neutralization_experiment,
    )


if __name__ == "__main__":
    main()
