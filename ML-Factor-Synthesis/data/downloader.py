"""Data providers for reproducible demos and free A-share data."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DataConfig
from data.cache import load_frame, save_frame

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "date", "symbol", "open", "high", "low", "close", "volume", "vwap", "market_cap", "industry"
}


def _call_with_retry(method: object, **kwargs: object) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(3):
        try:
            return method(**kwargs)
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])
    numeric = ["open", "high", "low", "close", "volume", "vwap", "market_cap"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["open", "high", "low", "close", "volume"])
    if (result[["open", "high", "low", "close", "volume"]].le(0)).any().any():
        raise ValueError("OHLCV values must be positive")
    return result.reset_index(drop=True)


def make_synthetic_data(config: DataConfig) -> pd.DataFrame:
    """Create a deterministic market-like panel with learnable nonlinear structure."""
    rng = np.random.default_rng(config.seed)
    # Warm-up history is required by long rolling Alpha101 expressions.
    warm_start = (pd.Timestamp(config.start_date) - pd.offsets.BDay(280)).date()
    dates = pd.bdate_range(warm_start, config.end_date)
    n_dates, n_symbols = len(dates), config.synthetic_symbols
    symbols = np.array([f"SIM{i:04d}" for i in range(n_symbols)])
    industries = np.array([f"Industry-{i % 8}" for i in range(n_symbols)])

    market = rng.normal(0.00015, 0.009, n_dates)
    styles = rng.normal(0, 0.004, (n_dates, 8))
    idio = rng.normal(0, 0.013, (n_dates, n_symbols))
    returns = market[:, None] + styles[:, np.arange(n_symbols) % 8] + idio
    # A small nonlinear, lagged component makes the synthesis exercise meaningful.
    for t in range(8, n_dates):
        returns[t] += -0.10 * returns[t - 1] + 0.0015 * np.tanh(returns[t - 5] * 50)
    close = 20 * np.exp(np.cumsum(returns, axis=0))
    overnight = rng.normal(0, 0.004, close.shape)
    open_ = np.vstack([close[0], close[:-1]]) * np.exp(overnight)
    spread = np.abs(rng.normal(0.012, 0.005, close.shape))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * np.maximum(0.5, 1 - spread)
    vwap = (open_ + high + low + 2 * close) / 5
    base_volume = rng.lognormal(15.2, 0.65, n_symbols)
    volume = base_volume[None, :] * np.exp(rng.normal(0, 0.35, close.shape))
    shares = rng.lognormal(18.0, 0.6, n_symbols)

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.values, n_symbols),
            "symbol": np.tile(symbols, n_dates),
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "vwap": vwap.ravel(),
            "market_cap": (close * shares[None, :]).ravel(),
            "industry": np.tile(industries, n_dates),
        }
    )
    return _validate(frame)


def _constituents(config: DataConfig) -> list[str]:
    if config.constituents_file:
        supplied = pd.read_csv(config.constituents_file, dtype={"symbol": str})
        if "symbol" not in supplied:
            raise ValueError("Constituents file must contain a 'symbol' column")
        return supplied["symbol"].str.zfill(6).unique().tolist()
    import akshare as ak

    table = ak.index_stock_cons_csindex(symbol="000300")
    for candidate in ("成分券代码", "品种代码", "code"):
        if candidate in table:
            LOGGER.warning("Using current CSI 300 members; provide --constituents-file for point-in-time membership.")
            return table[candidate].astype(str).str.zfill(6).tolist()
    raise ValueError(f"Could not identify constituent code column: {table.columns.tolist()}")


def download_akshare(config: DataConfig) -> pd.DataFrame:
    import akshare as ak

    rows: list[pd.DataFrame] = []
    start, end = config.start_date.replace("-", ""), config.end_date.replace("-", "")
    for position, symbol in enumerate(_constituents(config), 1):
        LOGGER.info("Downloading %s (%d)", symbol, position)
        try:
            raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        except Exception as exc:  # one suspended/delisted stock must not abort the universe
            LOGGER.warning("Skipping %s: %s", symbol, exc)
            continue
        if raw.empty:
            continue
        rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        raw = raw.rename(columns=rename)
        raw["symbol"] = symbol
        raw["vwap"] = (raw["high"] + raw["low"] + raw["close"]) / 3
        raw["market_cap"] = np.nan
        raw["industry"] = "Unknown"
        rows.append(raw[list(REQUIRED_COLUMNS)])
    if not rows:
        raise RuntimeError("AkShare returned no price data")
    # AkShare daily prices do not provide point-in-time industry or market cap.
    # Keep these fields missing rather than manufacturing a look-ahead-prone proxy.
    return _validate(pd.concat(rows, ignore_index=True))


def _tushare_membership(pro: object, config: DataConfig, start: str, end: str) -> pd.DataFrame:
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        frame = _call_with_retry(
            pro.index_weight,
            index_code=config.index_code,
            start_date=f"{year}0101",
            end_date=f"{year}1231",
        )
        if not frame.empty:
            frames.append(frame[["con_code", "trade_date", "weight"]])
    if not frames:
        raise RuntimeError(f"Tushare returned no membership for {config.index_code}")
    membership = pd.concat(frames, ignore_index=True).rename(columns={"con_code": "symbol"})
    membership["date"] = pd.to_datetime(membership["trade_date"])
    membership["month"] = membership["date"].dt.to_period("M")
    return membership.sort_values("date").drop_duplicates(["month", "symbol"], keep="last")


def _tushare_industry_history(pro: object, symbol: str) -> pd.DataFrame:
    frames = []
    for is_new in ("Y", "N"):
        try:
            frame = _call_with_retry(pro.index_member_all, ts_code=symbol, is_new=is_new)
        except Exception as exc:
            LOGGER.warning("Industry lookup failed for %s: %s", symbol, exc)
            continue
        if not frame.empty:
            frames.append(frame[["l1_name", "in_date", "out_date"]])
    if not frames:
        return pd.DataFrame(columns=["industry", "in_date", "out_date"])
    history = pd.concat(frames, ignore_index=True).drop_duplicates()
    history = history.rename(columns={"l1_name": "industry"})
    history["in_date"] = pd.to_datetime(history["in_date"], errors="coerce")
    history["out_date"] = pd.to_datetime(history["out_date"], errors="coerce")
    return history.sort_values("in_date")


def _current_csi300_membership(
    start: str, end: str, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Expand the current CSI 300 list when historical membership is unavailable."""
    import akshare as ak

    local_file = cache_dir / "current_csi300.csv" if cache_dir is not None else None
    if local_file is not None and local_file.exists():
        table = pd.read_csv(local_file, dtype=str)
    else:
        table = ak.index_stock_cons_csindex(symbol="000300")
        if local_file is not None:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(local_file, index=False, encoding="utf-8")
    code_column = next(
        (column for column in ("成分券代码", "品种代码", "code") if column in table),
        None,
    )
    if code_column is None:
        raise RuntimeError(f"Could not identify CSI 300 code column: {table.columns.tolist()}")
    symbols = []
    for code in table[code_column].astype(str).str.zfill(6).unique():
        suffix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        symbols.append(f"{code}.{suffix}")
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    rows = [
        {
            "symbol": symbol,
            "trade_date": month.to_timestamp("M").strftime("%Y%m%d"),
            "weight": np.nan,
            "date": month.to_timestamp("M"),
            "month": month,
        }
        for month in months
        for symbol in symbols
    ]
    LOGGER.warning(
        "Using %d current CSI 300 constituents for all dates; results have survivorship bias",
        len(symbols),
    )
    return pd.DataFrame(rows)


def _assign_industry(dates: pd.Series, history: pd.DataFrame) -> pd.Series:
    result = pd.Series("Unknown", index=dates.index, dtype=object)
    for row in history.itertuples(index=False):
        active = dates.ge(row.in_date) & (dates.le(row.out_date) if pd.notna(row.out_date) else True)
        result.loc[active] = row.industry
    return result


def download_tushare(config: DataConfig, refresh: bool = False) -> pd.DataFrame:
    """Download a point-in-time CSI 300 panel from Tushare Pro."""
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install the 'tushare' package to use --source tushare") from exc
    token = os.environ.get("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        raise RuntimeError(
            "No Tushare token is configured. Use TUSHARE_TOKEN or tushare.set_token()."
        )

    pro = ts.pro_api(token)
    warm_start = (pd.Timestamp(config.start_date) - pd.offsets.BDay(280)).strftime("%Y%m%d")
    end = pd.Timestamp(config.end_date).strftime("%Y%m%d")
    membership_source = "tushare.index_weight"
    try:
        membership = _tushare_membership(pro, config, warm_start, end)
    except Exception as exc:
        LOGGER.warning(
            "Tushare index membership unavailable (%s); using current CSI 300 fallback",
            exc,
        )
        membership = _current_csi300_membership(warm_start, end, config.cache_dir)
        membership_source = "akshare.current_csi300_survivorship_biased"

    industry_available = True
    probe_symbol = str(membership["symbol"].iloc[0])
    try:
        _call_with_retry(pro.index_member_all, ts_code=probe_symbol, is_new="Y")
    except Exception as exc:
        LOGGER.warning("Tushare historical industry is unavailable: %s", exc)
        industry_available = False
    rows = []
    symbols = sorted(membership["symbol"].unique())
    checkpoint_dir = (
        config.cache_dir
        / "tushare_symbols"
        / f"{config.index_code.replace('.', '_')}_{warm_start}_{end}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for position, symbol in enumerate(symbols, 1):
        LOGGER.info("Downloading Tushare %s (%d/%d)", symbol, position, len(symbols))
        checkpoint = checkpoint_dir / f"{symbol.replace('.', '_')}.csv.gz"
        if checkpoint.exists() and not refresh:
            cached_symbol = pd.read_csv(checkpoint, parse_dates=["date"])
            rows.append(cached_symbol)
            continue
        try:
            daily = _call_with_retry(
                pro.daily, ts_code=symbol, start_date=warm_start, end_date=end
            )
            adjustment = _call_with_retry(
                pro.adj_factor, ts_code=symbol, start_date=warm_start, end_date=end
            )
            basic = _call_with_retry(
                pro.daily_basic,
                ts_code=symbol,
                start_date=warm_start,
                end_date=end,
                fields="ts_code,trade_date,circ_mv",
            )
        except Exception as exc:
            LOGGER.warning("Skipping %s after Tushare error: %s", symbol, exc)
            continue
        if (
            daily.empty
            or adjustment.empty
            or basic.empty
            or not {"trade_date", "circ_mv"}.issubset(basic.columns)
        ):
            LOGGER.warning("Skipping %s because a required Tushare table is empty", symbol)
            continue
        frame = daily.merge(
            adjustment[["trade_date", "adj_factor"]], on="trade_date", how="left"
        ).merge(basic[["trade_date", "circ_mv"]], on="trade_date", how="left")
        frame["date"] = pd.to_datetime(frame["trade_date"])
        frame["month"] = frame["date"].dt.to_period("M")
        active_months = membership.loc[membership["symbol"] == symbol, "month"]
        frame = frame.loc[frame["month"].isin(active_months)]
        if frame.empty:
            continue
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] * frame["adj_factor"]
        # Tushare amount is thousand CNY and vol is hundred-share lots.
        raw_vwap = frame["amount"] * 10 / frame["vol"].replace(0, np.nan)
        frame["vwap"] = raw_vwap * frame["adj_factor"]
        frame["volume"] = frame["vol"] * 100
        frame["market_cap"] = frame["circ_mv"] * 10_000
        frame["symbol"] = symbol
        industry = (
            _tushare_industry_history(pro, symbol)
            if industry_available
            else pd.DataFrame(columns=["industry", "in_date", "out_date"])
        )
        frame["industry"] = _assign_industry(frame["date"], industry)
        frame["_membership_source"] = membership_source
        standardized = frame[list(REQUIRED_COLUMNS) + ["_membership_source"]]
        standardized.to_csv(checkpoint, index=False, compression="gzip")
        rows.append(standardized)
    if not rows:
        raise RuntimeError("Tushare returned no usable CSI 300 observations")
    result = _validate(pd.concat(rows, ignore_index=True))
    market_cap_coverage = result["market_cap"].gt(0).mean()
    industry_coverage = result["industry"].ne("Unknown").mean()
    if market_cap_coverage < 0.95:
        raise RuntimeError(
            f"Point-in-time market-cap coverage is insufficient: {market_cap_coverage:.1%}"
        )
    if industry_coverage < 0.95:
        LOGGER.warning(
            "Point-in-time industry coverage is only %.1f%%; industry neutralization will reject it",
            industry_coverage * 100,
        )
    return result


def download_baostock(config: DataConfig, refresh: bool = False) -> pd.DataFrame:
    """Download real adjusted OHLCV for the current CSI 300 universe."""
    import baostock as bs

    warm_start = (pd.Timestamp(config.start_date) - pd.offsets.BDay(280)).strftime("%Y-%m-%d")
    end = pd.Timestamp(config.end_date).strftime("%Y-%m-%d")
    membership = _current_csi300_membership(
        warm_start.replace("-", ""), end.replace("-", ""), config.cache_dir
    )
    symbols = sorted(membership["symbol"].unique())
    checkpoint_dir = (
        config.cache_dir
        / "baostock_symbols"
        / f"{config.index_code.replace('.', '_')}_{warm_start}_{end}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    rows = []
    fields = "date,code,open,high,low,close,volume,amount,tradestatus,isST"
    try:
        for position, symbol in enumerate(symbols, 1):
            LOGGER.info("Downloading BaoStock %s (%d/%d)", symbol, position, len(symbols))
            checkpoint = checkpoint_dir / f"{symbol.replace('.', '_')}.csv.gz"
            if checkpoint.exists() and not refresh:
                rows.append(pd.read_csv(checkpoint, parse_dates=["date"]))
                continue
            code, suffix = symbol.split(".")
            provider_code = f"{suffix.lower()}.{code}"
            result = bs.query_history_k_data_plus(
                provider_code,
                fields,
                start_date=warm_start,
                end_date=end,
                frequency="d",
                # BaoStock 1 is backward-adjusted, avoiding a future endpoint normalization.
                adjustflag="1",
            )
            values = []
            while result.error_code == "0" and result.next():
                values.append(result.get_row_data())
            if result.error_code != "0":
                LOGGER.warning("Skipping %s: %s", symbol, result.error_msg)
                continue
            frame = pd.DataFrame(values, columns=result.fields)
            if frame.empty:
                continue
            frame = frame.loc[frame["tradestatus"] == "1"].copy()
            frame["symbol"] = symbol
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
            for column in ("open", "high", "low", "close"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["vwap"] = (frame["high"] + frame["low"] + frame["close"]) / 3
            frame["market_cap"] = np.nan
            frame["industry"] = "Unknown"
            frame["_membership_source"] = "akshare.current_csi300_survivorship_biased"
            standardized = frame[list(REQUIRED_COLUMNS) + ["_membership_source"]]
            standardized.to_csv(checkpoint, index=False, compression="gzip")
            rows.append(standardized)
    finally:
        bs.logout()
    if not rows:
        raise RuntimeError("BaoStock returned no usable CSI 300 observations")
    return _validate(pd.concat(rows, ignore_index=True))


def load_market_data(config: DataConfig, refresh: bool = False) -> pd.DataFrame:
    suffix = (
        f"_{config.synthetic_symbols}_{config.seed}"
        if config.source == "synthetic"
        else f"_{config.index_code.replace('.', '_')}"
    )
    cache_key = config.cache_dir / f"{config.source}_{config.start_date}_{config.end_date}{suffix}"
    cached = None if refresh else load_frame(cache_key)
    if cached is not None:
        LOGGER.info("Loaded market data from cache")
        return _validate(cached)
    if config.source == "synthetic":
        frame = make_synthetic_data(config)
    elif config.source == "akshare":
        frame = download_akshare(config)
    elif config.source == "tushare":
        frame = download_tushare(config, refresh=refresh)
    elif config.source == "baostock":
        frame = download_baostock(config, refresh=refresh)
    else:
        raise ValueError("source must be 'synthetic', 'akshare', 'tushare', or 'baostock'")
    saved = save_frame(frame, cache_key)
    LOGGER.info("Cached market data at %s", saved)
    return frame
