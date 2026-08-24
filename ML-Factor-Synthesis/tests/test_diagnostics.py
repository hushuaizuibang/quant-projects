import numpy as np
import pandas as pd

from model.diagnostics import build_model_diagnostics
from model.train import RidgeFallback, TrainedModel


def test_model_diagnostics_reports_dimension_and_rmse_skill():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    score = pd.Series([0.01, -0.01, 0.02, -0.02], index=index)
    target = pd.Series([0.02, -0.02, 0.01, -0.01], index=index)
    samples = pd.DataFrame({"feature": 1.0, "target": target}, index=index)
    estimator = RidgeFallback().fit(np.ones((4, 1)), target.to_numpy())
    model = TrainedModel(estimator, ["feature"], "ridge-fallback", 0.1, 0.1)
    backtest = pd.DataFrame(
        {"gross_return": [0.01, 0.02], "net_return": [0.009, 0.019], "turnover": [0.3, 0.3]},
        index=dates,
    )
    result = build_model_diagnostics(score, target, samples, {2024: model}, backtest)
    assert result["feature_count"] == 1
    assert result["test_months"] == 2
    assert np.isfinite(result["rmse_skill_vs_zero"])
