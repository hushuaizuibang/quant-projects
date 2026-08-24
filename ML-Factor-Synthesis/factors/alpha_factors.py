"""A diverse, reproducible subset of formulas from the Alpha101 paper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.operators import (
    correlation, covariance, delay, delta, rank, safe_divide, stddev, ts_max, ts_min, ts_rank, ts_sum
)

FACTOR_NAMES = [
    "alpha002", "alpha003", "alpha004", "alpha005", "alpha006",
    "alpha007", "alpha009", "alpha010", "alpha011", "alpha012",
    "alpha013", "alpha014", "alpha018", "alpha022", "alpha033",
    "alpha038", "alpha040", "alpha041", "alpha043", "alpha101",
]


def calculate_factors(market: pd.DataFrame) -> pd.DataFrame:
    """Calculate 20 Alpha101 signals without using data after each row's date."""
    data = market.sort_values(["symbol", "date"]).set_index(["date", "symbol"])
    o, h, low, c, v, w = (data[x].astype(float) for x in ("open", "high", "low", "close", "volume", "vwap"))
    returns = safe_divide(c, delay(c, 1)) - 1
    adv20 = ts_sum(v, 20) / 20
    dvwap = w - c
    dclose = delta(c, 1)

    out = pd.DataFrame(index=data.index)
    out["alpha002"] = -correlation(rank(delta(np.log(v), 2)), rank(safe_divide(c - o, o)), 6)
    out["alpha003"] = -correlation(rank(o), rank(v), 10)
    out["alpha004"] = -ts_rank(rank(low), 9)
    out["alpha005"] = rank(o - ts_sum(w, 10) / 10) * -rank((c - w).abs())
    out["alpha006"] = -correlation(o, v, 10)
    out["alpha007"] = (-ts_rank(delta(c, 7).abs(), 60) * np.sign(delta(c, 7))).where(v > adv20, -1.0)
    min_d, max_d = ts_min(dclose, 5), ts_max(dclose, 5)
    out["alpha009"] = dclose.where((min_d > 0) | (max_d < 0), -dclose)
    min_d4, max_d4 = ts_min(dclose, 4), ts_max(dclose, 4)
    out["alpha010"] = rank(dclose.where((min_d4 > 0) | (max_d4 < 0), -dclose))
    out["alpha011"] = (rank(ts_max(dvwap, 3)) + rank(ts_min(dvwap, 3))) * rank(delta(v, 3))
    out["alpha012"] = np.sign(delta(v, 1)) * -dclose
    out["alpha013"] = -rank(covariance(rank(c), rank(v), 5))
    out["alpha014"] = -rank(delta(returns, 3)) * correlation(o, v, 10)
    out["alpha018"] = -rank(stddev((c - o).abs(), 5) + (c - o) + correlation(c, o, 10))
    out["alpha022"] = -delta(correlation(h, v, 5), 5) * rank(stddev(c, 20))
    out["alpha033"] = rank(o / c - 1)
    out["alpha038"] = -rank(ts_rank(c, 10)) * rank(safe_divide(c, o))
    out["alpha040"] = -rank(stddev(h, 10)) * correlation(h, v, 10)
    out["alpha041"] = np.sqrt(h * low) - w
    out["alpha043"] = ts_rank(safe_divide(v, adv20), 20) * ts_rank(-delta(c, 7), 8)
    out["alpha101"] = safe_divide(c - o, (h - low) + 0.001)
    return out.replace([np.inf, -np.inf], np.nan).sort_index()
