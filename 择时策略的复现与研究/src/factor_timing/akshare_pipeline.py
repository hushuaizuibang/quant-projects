from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import market_proxy_returns, run_factor_timing_backtest
from .config import ResearchConfig
from .comparison import run_comparison_experiment
from .factors import compute_factor_exposures, compute_factor_returns
from .metrics import performance_table
from .signals import build_timing_scores, timing_weights
from .walk_forward import run_walk_forward_validation


def run_akshare_pipeline(project_root: Path, config: ResearchConfig | None = None) -> None:
    config = config or ResearchConfig(start="2021-01-01", end="2026-07-29")
    output_dir = project_root / "outputs" / "akshare_csi300"
    output_dir.mkdir(parents=True, exist_ok=True)

    prices, constituents, failed = load_or_fetch_akshare_csi300(project_root, config)
    market_returns = market_proxy_returns(prices)
    exposures = compute_factor_exposures(prices)
    factor_returns = compute_factor_returns(exposures, config.long_quantile, config.short_quantile)
    scores = build_timing_scores(factor_returns, market_returns)
    weights = timing_weights(scores, config.max_factor_weight)
    strategy_returns = run_factor_timing_backtest(
        factor_returns=factor_returns,
        timing_weights=weights,
        market_returns=market_returns,
        transaction_cost=config.transaction_cost,
        rebalance_freq=config.rebalance_freq,
        slippage=config.slippage,
        max_rebalance_turnover=config.max_rebalance_turnover,
    )
    metrics = performance_table(strategy_returns)

    # Raw inputs already have canonical copies under data/akshare/.  Do not
    # duplicate them in outputs; that directory is reserved for derived results.
    factor_returns.to_csv(output_dir / "factor_returns.csv")
    scores.to_csv(output_dir / "timing_scores.csv")
    weights.to_csv(output_dir / "factor_weights.csv")
    strategy_returns.to_csv(output_dir / "strategy_returns.csv")
    metrics.to_csv(output_dir / "metrics.csv")
    _plot_equity_curve(strategy_returns, output_dir / "equity_curve.png")
    comparison = run_comparison_experiment(
        factor_returns=factor_returns,
        timing_scores=scores,
        market_returns=market_returns,
        output_dir=output_dir,
        transaction_cost=config.transaction_cost,
        slippage=config.slippage,
        rebalance_freq=config.rebalance_freq,
        max_rebalance_turnover=config.max_rebalance_turnover,
        random_seed=config.random_seed,
    )
    wf_returns, wf_metrics, wf_folds, wf_regimes = run_walk_forward_validation(
        factor_returns=factor_returns,
        market_returns=market_returns,
        transaction_cost=config.transaction_cost,
        rebalance_freq=config.rebalance_freq,
        output_dir=output_dir,
        slippage=config.slippage,
        max_rebalance_turnover=config.max_rebalance_turnover,
    )
    _write_summary(output_dir, config, constituents, prices, failed, metrics)

    print("Done. AKShare CSI300 outputs written to:", output_dir)
    print(metrics.round(4).to_string())
    print("\nWalk-forward out-of-sample metrics:")
    print(wf_metrics.round(4).to_string())
    print("\nFair combination comparison (net of costs):")
    print(comparison["metrics"].round(4).to_string())


def load_or_fetch_akshare_csi300(project_root: Path, config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cache_dir = project_root / "data" / "akshare"
    price_path = cache_dir / "csi300_prices.csv"
    constituent_path = cache_dir / "csi300_constituents.csv"
    failed_path = cache_dir / "failed_tickers.csv"
    stock_cache_dir = cache_dir / "stocks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stock_cache_dir.mkdir(parents=True, exist_ok=True)

    if price_path.exists() and constituent_path.exists():
        prices = pd.read_csv(price_path, parse_dates=["date"])
        constituents = pd.read_csv(constituent_path, dtype=str)
        failed = pd.read_csv(failed_path)["ticker"].tolist() if failed_path.exists() else []
        if prices["ticker"].nunique() >= 250:
            return prices, constituents, failed
        print("existing aggregated price cache is incomplete; rebuilding from per-stock cache", flush=True)

    import akshare as ak

    constituents = ak.index_stock_cons_csindex(symbol="000300")
    constituents.to_csv(constituent_path, index=False, encoding="utf-8-sig")

    frames = []
    start = config.start.replace("-", "")
    end = config.end.replace("-", "")

    codes = [str(code).zfill(6) for code in constituents["成分券代码"]]
    cached_codes = []
    missing_codes = []
    for code in codes:
        stock_path = stock_cache_dir / f"{code}.csv"
        if stock_path.exists():
            frames.append(pd.read_csv(stock_path, parse_dates=["date"]))
            cached_codes.append(code)
        else:
            missing_codes.append(code)

    print(f"cached stocks: {len(cached_codes)}, missing stocks: {len(missing_codes)}", flush=True)
    fetched_frames, failed = _fetch_missing_stocks(ak, missing_codes, start, end, stock_cache_dir)
    frames.extend(fetched_frames)

    if not frames:
        raise RuntimeError("AKShare 没有成功获取任何沪深300成分股行情。")

    prices = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    prices.to_csv(price_path, index=False)
    pd.DataFrame({"ticker": failed}).to_csv(failed_path, index=False)
    return prices, constituents, failed


def _fetch_stock_history(ak, code: str, start: str, end: str) -> pd.DataFrame:
    last_error = None
    for _ in range(3):
        try:
            symbol = _ak_market_symbol(code)
            return ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
        except Exception as exc:
            last_error = exc
            time.sleep(0.8)
    try:
        symbol = _ak_market_symbol(code)
        return ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    except Exception:
        raise last_error


def _fetch_missing_stocks(ak, codes: list[str], start: str, end: str, stock_cache_dir: Path) -> tuple[list[pd.DataFrame], list[str]]:
    if not codes:
        return [], []

    frames = []
    failed = []
    total = len(codes)
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_and_cache_one_stock, ak, code, start, end, stock_cache_dir): code
            for code in codes
        }
        for done, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                frame = future.result()
                if frame.empty:
                    failed.append(code)
                    print(f"[{done:03d}/{total}] empty {code}", flush=True)
                else:
                    frames.append(frame)
                    print(f"[{done:03d}/{total}] fetched {code}", flush=True)
            except Exception as exc:
                failed.append(code)
                print(f"[{done:03d}/{total}] failed {code}: {exc}", flush=True)
    return frames, failed


def _fetch_and_cache_one_stock(ak, code: str, start: str, end: str, stock_cache_dir: Path) -> pd.DataFrame:
    stock_path = stock_cache_dir / f"{code}.csv"
    raw = _fetch_stock_history(ak, code, start, end)
    if raw.empty:
        return pd.DataFrame()
    normalized = _normalize_stock_history(raw, code)
    normalized.to_csv(stock_path, index=False)
    time.sleep(0.1)
    return normalized


def _normalize_stock_history(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "日期": "date",
            "股票代码": "ticker",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
    ).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = code
    for col in ["open", "close", "high", "low", "volume", "amount", "turnover", "outstanding_share"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close", "high", "low", "volume", "amount"])
    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]

    if "outstanding_share" in df:
        df["market_cap"] = (df["close"] * df["outstanding_share"]).clip(lower=1)
    else:
        scale_proxy = df["amount"].rolling(60, min_periods=5).median().bfill()
        df["market_cap"] = scale_proxy.clip(lower=1)
    return df[["date", "ticker", "open", "high", "low", "close", "volume", "amount", "market_cap"]]


def _ak_market_symbol(code: str) -> str:
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{code}"


def _plot_equity_curve(strategy_returns: pd.DataFrame, output_path: Path) -> None:
    nav = (1 + strategy_returns.drop(columns=["turnover"], errors="ignore")).cumprod()
    ax = nav.plot(figsize=(10, 5), title="CSI 300 AKShare Factor Timing")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_summary(
    output_dir: Path,
    config: ResearchConfig,
    constituents: pd.DataFrame,
    prices: pd.DataFrame,
    failed: list[str],
    metrics: pd.DataFrame,
) -> None:
    start = prices["date"].min().date()
    end = prices["date"].max().date()
    lines = [
        "# AKShare 沪深300成分股因子择时结果",
        "",
        "- 数据源：AKShare",
        f"- 成分股日期：{constituents['日期'].iloc[0] if '日期' in constituents else 'unknown'}",
        f"- 当前成分股数量：{len(constituents)}",
        f"- 成功获取行情股票数：{prices['ticker'].nunique()}",
        f"- 失败股票数：{len(failed)}",
        f"- 行情区间：{start} 至 {end}",
        f"- 调仓频率：{config.rebalance_freq}",
        "- 因子：动量、反转、波动率、成交额规模代理、流动性、价值代理、质量代理",
        "- 说明：本次使用当前沪深300成分股回测，未处理历史成分调整，存在幸存者偏差；历史市值优先用 `close * outstanding_share`，缺失时用成交额中期均值代理。",
        "",
        "## 指标",
        "",
        _markdown_table(metrics.round(4)),
        "",
        "## 严格时间序列验证",
        "",
        "新增 walk-forward 输出见 `walk_forward_summary.md`。该结果使用滞后一日的因子收益/市场收益构造择时信号，并按训练-验证-测试窗口滚动选择候选择时规则。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = ["portfolio"] + list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for index, row in frame.iterrows():
        values = [index] + [f"{value:.4f}" for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
