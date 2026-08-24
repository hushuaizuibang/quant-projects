import pandas as pd
import pytest

from config import DataConfig
from data.downloader import _assign_industry, download_tushare


def test_tushare_requires_token_without_exposing_a_fallback(monkeypatch):
    import tushare as ts

    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(ts, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN"):
        download_tushare(DataConfig(source="tushare"))


def test_point_in_time_industry_assignment_respects_exit_date():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    history = pd.DataFrame(
        {
            "industry": ["Old", "New"],
            "in_date": pd.to_datetime(["2019-01-01", "2021-06-01"]),
            "out_date": pd.to_datetime(["2021-05-31", None]),
        }
    )
    assert _assign_industry(dates, history).tolist() == ["Old", "Old", "New"]
