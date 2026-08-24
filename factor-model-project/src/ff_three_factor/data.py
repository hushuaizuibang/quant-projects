from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class MarketData:
    close: pd.DataFrame
    outstanding_share: pd.DataFrame
    net_asset_per_share: pd.Series | pd.DataFrame
    benchmark_close: pd.Series
    stock_names: pd.Series
    industry: pd.Series | None = None
    roe: pd.Series | pd.DataFrame | None = None
    operating_cashflow_to_assets: pd.Series | pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    metadata: dict[str, object] | None = None


def load_market_data(config: BacktestConfig) -> MarketData:
    if config.use_sample_data or config.data_source == "sample":
        return make_sample_market_data(config)
    if config.data_source == "yfinance":
        return load_yfinance_market_data(config)
    if config.data_source != "akshare":
        raise ValueError(f"Unsupported data source: {config.data_source}")
    return load_akshare_market_data(config)


def load_akshare_market_data(config: BacktestConfig) -> MarketData:
    """Load prices from AKShare and static metadata from the local universe file.

    Fundamental values in the local file are intentionally treated as static.
    For a production study, replace them with point-in-time, publication-date
    aligned data; the factor engine accepts dated DataFrames for that purpose.
    """
    cache_path = Path(__file__).resolve().parents[2] / "data" / "akshare" / "csi300_prices.csv"
    if cache_path.exists():
        return load_local_akshare_cache(config, cache_path)

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare is not installed. Run `pip install -r requirements.txt`.") from exc

    local = _load_local_constituents(config.stock_count)
    codes = local["code"].tolist()
    close_frames: list[pd.Series] = []
    for code in codes:
        try:
            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=config.start_date,
                end_date=config.resolved_end_date,
                adjust="qfq",
            )
            date_col = _first_existing_column(hist, ["日期", "date"])
            close_col = _first_existing_column(hist, ["收盘", "close"])
            values = hist.assign(**{date_col: pd.to_datetime(hist[date_col])})
            close_frames.append(
                pd.to_numeric(values.set_index(date_col)[close_col], errors="coerce").rename(code)
            )
        except Exception as exc:
            print(f"Warning: skipped {code}: {exc}")

    if len(close_frames) < min(10, config.stock_count):
        raise RuntimeError(
            "Too few stocks were downloaded. Check network access or use "
            "`python main.py --data-source sample` for an offline validation."
        )
    close = pd.concat(close_frames, axis=1).sort_index().ffill()
    meta = local.set_index("code").reindex(close.columns)
    shares = _static_frame(meta["outstanding_share"], close.index)

    benchmark = ak.stock_zh_index_daily_em(
        symbol="sh000300",
        start_date=config.start_date,
        end_date=config.resolved_end_date,
    )
    date_col = _first_existing_column(benchmark, ["date", "日期"])
    close_col = _first_existing_column(benchmark, ["close", "收盘"])
    benchmark_close = pd.to_numeric(
        benchmark.assign(**{date_col: pd.to_datetime(benchmark[date_col])})
        .set_index(date_col)[close_col],
        errors="coerce",
    ).rename("CSI300")
    return _assemble_market_data(close, shares, benchmark_close, meta)


def load_local_akshare_cache(config: BacktestConfig, price_path: Path | None = None) -> MarketData:
    """Load the canonical long-form AKShare cache without network access.

    The cache contains qfq OHLCV data for the current CSI 300 constituents.
    Because no official index series or historical constituent membership is
    stored alongside it, the benchmark is an equal-weight current-constituent
    proxy and the metadata explicitly records the resulting research limits.
    """
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "akshare"
    price_path = price_path or cache_dir / "csi300_prices.csv"
    constituent_path = cache_dir / "csi300_constituents.csv"
    if not price_path.exists() or not constituent_path.exists():
        raise FileNotFoundError(
            "Local AKShare cache is incomplete. Expected csi300_prices.csv and "
            f"csi300_constituents.csv under {cache_dir}."
        )

    prices = pd.read_csv(
        price_path,
        usecols=["date", "ticker", "close", "market_cap"],
        dtype={"ticker": str},
    )
    prices["ticker"] = prices["ticker"].str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["market_cap"] = pd.to_numeric(prices["market_cap"], errors="coerce")
    start = pd.to_datetime(config.start_date)
    end = pd.to_datetime(config.resolved_end_date)
    prices = prices.loc[prices["date"].between(start, end)].dropna(
        subset=["date", "ticker", "close"]
    )
    prices = prices.loc[prices["close"] > 0].drop_duplicates(
        ["date", "ticker"], keep="last"
    )

    constituents = pd.read_csv(constituent_path, dtype=str)
    code_col = _first_existing_column(constituents, ["成分券代码", "code", "ticker"])
    name_col = _first_existing_column(constituents, ["成分券名称", "name"])
    constituents[code_col] = constituents[code_col].str.zfill(6)
    requested = min(config.stock_count, len(constituents))
    codes = constituents[code_col].head(requested).tolist()
    prices = prices.loc[prices["ticker"].isin(codes)]
    if prices["ticker"].nunique() < min(10, requested):
        raise RuntimeError("The local AKShare cache has too few usable stocks for the requested period.")

    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    market_cap = prices.pivot(index="date", columns="ticker", values="market_cap").sort_index()
    available = [code for code in codes if code in close.columns]
    close = close.reindex(columns=available)
    market_cap = market_cap.reindex(index=close.index, columns=available)
    # Only bridge short suspensions; never backfill pre-listing history.
    close = close.ffill(limit=5)
    market_cap = market_cap.ffill(limit=5)
    implied_shares = market_cap.div(close).replace([np.inf, -np.inf], np.nan)

    daily_equal_weight_return = close.pct_change(fill_method=None).mean(axis=1, skipna=True)
    benchmark_close = (1000.0 * (1.0 + daily_equal_weight_return.fillna(0)).cumprod()).rename(
        "CSI300_equal_weight_proxy"
    )
    meta = constituents.set_index(code_col).reindex(available)
    names = meta[name_col].rename("name")
    constituent_date_col = _first_existing_column(
        constituents, ["日期", "date"], required=False
    )
    constituent_date = (
        str(constituents[constituent_date_col].iloc[0]) if constituent_date_col else "unknown"
    )
    market_cap_coverage = float(market_cap.notna().sum().sum() / market_cap.size)
    return MarketData(
        close=close,
        outstanding_share=implied_shares,
        net_asset_per_share=pd.Series(np.nan, index=available, dtype=float),
        benchmark_close=benchmark_close,
        stock_names=names,
        industry=pd.Series("Unknown", index=available, dtype=object),
        roe=None,
        operating_cashflow_to_assets=None,
        market_cap=market_cap,
        metadata={
            "data_source": "AKShare local qfq cache",
            "price_adjustment": "qfq",
            "requested_stocks": requested,
            "loaded_stocks": len(available),
            "observations": len(prices),
            "start_date": str(close.index.min().date()),
            "end_date": str(close.index.max().date()),
            "constituent_snapshot_date": constituent_date,
            "benchmark": "equal-weight current CSI300 constituent proxy",
            "historical_membership": False,
            "industry_coverage": 0.0,
            "point_in_time_fundamentals": False,
            "market_cap_coverage": market_cap_coverage,
            "market_cap_definition": "cached field; close*shares when available, otherwise rolling amount proxy",
            "source_file": str(price_path.relative_to(project_root)),
        },
    )


def load_yfinance_market_data(config: BacktestConfig) -> MarketData:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run `pip install -r requirements.txt`.") from exc

    cache_dir = Path.cwd() / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    local = _load_local_constituents(config.stock_count)
    codes = local["code"].tolist()
    tickers = [_to_yahoo_ticker(code) for code in codes]
    start = _date_yyyymmdd_to_iso(config.start_date)
    end = _date_yyyymmdd_to_iso(config.resolved_end_date)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    close = _extract_yfinance_close(raw, tickers, codes).sort_index().ffill()
    if close.shape[1] < min(10, config.stock_count):
        raise RuntimeError("Too few yfinance prices were downloaded; try the sample data source.")
    meta = local.set_index("code").reindex(close.columns)
    shares = _static_frame(meta["outstanding_share"], close.index)
    benchmark_close = _download_yfinance_benchmark(yf, start, end)
    return _assemble_market_data(close, shares, benchmark_close, meta)


def _assemble_market_data(
    close: pd.DataFrame,
    shares: pd.DataFrame,
    benchmark_close: pd.Series,
    meta: pd.DataFrame,
) -> MarketData:
    def optional(name: str) -> pd.Series | None:
        return pd.to_numeric(meta[name], errors="coerce") if name in meta else None

    industry = meta["industry"].fillna("Unknown") if "industry" in meta else pd.Series("Unknown", index=meta.index)
    return MarketData(
        close=close,
        outstanding_share=shares,
        net_asset_per_share=pd.to_numeric(meta["net_asset_per_share"], errors="coerce"),
        benchmark_close=benchmark_close,
        stock_names=meta["name"],
        industry=industry,
        roe=optional("roe"),
        operating_cashflow_to_assets=optional("operating_cashflow_to_assets"),
    )


def _download_yfinance_benchmark(yf_module: object, start: str, end: str) -> pd.Series:
    errors = []
    for ticker in ["510300.SS", "000300.SS", "399300.SZ", "ASHR"]:
        try:
            raw = yf_module.download(
                tickers=ticker, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if not raw.empty:
                return _extract_yfinance_benchmark_close(raw, ticker)
            errors.append(f"{ticker}: empty")
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
    raise RuntimeError("yfinance benchmark download failed. " + " | ".join(errors))


def _load_local_constituents(stock_count: int) -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "data" / "csi300_constituents.csv"
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"code", "name", "outstanding_share", "net_asset_per_share"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    frame["code"] = frame["code"].str.zfill(6)
    return frame.head(stock_count).copy()


def _static_frame(values: pd.Series, index: pd.Index) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.DataFrame(
        np.tile(numeric.to_numpy(), (len(index), 1)),
        index=index,
        columns=numeric.index,
    )


def _to_yahoo_ticker(code: str) -> str:
    return f"{code}.SS" if code.startswith(("6", "5", "9")) else f"{code}.SZ"


def _date_yyyymmdd_to_iso(value: str) -> str:
    return datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d")


def _extract_yfinance_close(raw: pd.DataFrame, tickers: list[str], codes: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for ticker, code in zip(tickers, codes, strict=True):
            if ticker in raw.columns.get_level_values(0) and "Close" in raw[ticker]:
                frames.append(raw[ticker]["Close"].rename(code))
            elif ("Close", ticker) in raw.columns:
                frames.append(raw[("Close", ticker)].rename(code))
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()
    return raw["Close"].rename(codes[0]).to_frame() if "Close" in raw else pd.DataFrame()


def _extract_yfinance_benchmark_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[(ticker, "Close")] if (ticker, "Close") in raw else raw[("Close", ticker)]
    else:
        close = raw["Close"]
    return pd.to_numeric(close, errors="coerce").rename("CSI300").sort_index()


def _first_existing_column(
    frame: pd.DataFrame, names: Iterable[str], required: bool = True
) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise KeyError(f"None of these columns exist: {', '.join(names)}")
    return None


def make_sample_market_data(config: BacktestConfig) -> MarketData:
    """Create deterministic data with known factor premia for offline tests."""
    rng = np.random.default_rng(config.random_seed)
    dates = pd.bdate_range(pd.to_datetime(config.start_date), pd.to_datetime(config.resolved_end_date))
    if len(dates) < 800:
        dates = pd.bdate_range(end=pd.to_datetime(config.resolved_end_date), periods=1000)
    n = max(config.stock_count, 60)
    codes = [f"S{i:04d}" for i in range(n)]
    industries = pd.Series(
        np.resize(["Financials", "Industrials", "Technology", "Consumer", "Healthcare", "Materials"], n),
        index=codes,
        name="industry",
    )
    shares_static = rng.lognormal(20.5, 0.8, n)
    navps = pd.Series(rng.uniform(2, 18, n), index=codes)
    roe = pd.Series(rng.normal(0.13, 0.05, n).clip(0.01, 0.35), index=codes)
    cashflow = pd.Series((roe.to_numpy() + rng.normal(0, 0.04, n)).clip(-0.05, 0.35), index=codes)
    quality = (roe.rank(pct=True) + cashflow.rank(pct=True)).to_numpy() - 1
    size_alpha = -pd.Series(shares_static).rank(pct=True).to_numpy() + 0.5
    market = rng.normal(0.0002, 0.010, len(dates))
    industry_shocks = rng.normal(0, 0.004, (len(dates), industries.nunique()))
    idio = rng.normal(0, 0.014, (len(dates), n))
    returns = market[:, None] * rng.normal(1, 0.12, n) + idio
    returns += industry_shocks[:, pd.Categorical(industries).codes]
    returns += 0.00008 * quality + 0.00005 * size_alpha
    close = pd.DataFrame(30 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=codes)
    shares = pd.DataFrame(np.tile(shares_static, (len(dates), 1)), index=dates, columns=codes)
    benchmark = pd.Series(3500 * np.exp(np.cumsum(market)), index=dates, name="CSI300")
    names = pd.Series([f"Sample {i:04d}" for i in range(n)], index=codes)
    return MarketData(close, shares, navps, benchmark, names, industries, roe, cashflow)
