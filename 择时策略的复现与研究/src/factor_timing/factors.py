from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factor_exposures(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["ticker", "date"]).copy()
    group = df.groupby("ticker", group_keys=False)
    df["ret_1d"] = group["close"].pct_change()
    df["momentum_20"] = group["close"].pct_change(20)
    df["reversal_5"] = -group["close"].pct_change(5)
    df["volatility_20"] = -group["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    df["size"] = -np.log(df["market_cap"].clip(lower=1))
    df["liquidity"] = -np.log((df["amount"] if "amount" in df else df["close"] * df["volume"]).clip(lower=1))
    df["value_proxy"] = df["bm"] if "bm" in df else -group["close"].pct_change(252)
    df["quality_proxy"] = df["roe"] if "roe" in df else group["ret_1d"].rolling(60).mean().reset_index(level=0, drop=True)
    df["next_ret"] = group["close"].pct_change().shift(-1)

    factor_cols = get_factor_columns()
    df[factor_cols] = df.groupby("date", group_keys=False)[factor_cols].apply(_winsorize_zscore)
    return df.dropna(subset=factor_cols + ["next_ret"]).reset_index(drop=True)


def compute_factor_returns(exposures: pd.DataFrame, long_q: float, short_q: float) -> pd.DataFrame:
    factor_returns = []
    for date, frame in exposures.groupby("date"):
        row = {"date": date}
        for factor in get_factor_columns():
            values = frame[["ticker", factor, "next_ret"]].dropna()
            if len(values) < 20:
                row[factor] = np.nan
                continue
            high = values[factor].quantile(1 - long_q)
            low = values[factor].quantile(short_q)
            long_ret = values.loc[values[factor] >= high, "next_ret"].mean()
            short_ret = values.loc[values[factor] <= low, "next_ret"].mean()
            row[factor] = long_ret - short_ret
        factor_returns.append(row)
    out = pd.DataFrame(factor_returns).sort_values("date")
    return out.set_index("date").dropna(how="all")


def get_factor_columns() -> list[str]:
    return [
        "momentum_20",
        "reversal_5",
        "volatility_20",
        "size",
        "liquidity",
        "value_proxy",
        "quality_proxy",
    ]


def _winsorize_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    clipped = frame.copy()
    for col in clipped.columns:
        lower = clipped[col].quantile(0.01)
        upper = clipped[col].quantile(0.99)
        clipped[col] = clipped[col].clip(lower, upper)
        std = clipped[col].std()
        clipped[col] = 0.0 if std == 0 or np.isnan(std) else (clipped[col] - clipped[col].mean()) / std
    return clipped
