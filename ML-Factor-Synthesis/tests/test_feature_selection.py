import numpy as np
import pandas as pd

from config import FeatureConfig
from model.feature_selection import select_features_by_icir


def test_icir_selection_uses_threshold_and_top_absolute_ic_lags():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-31", periods=36, freq="ME")
    symbols = [f"S{i:02d}" for i in range(20)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    target = rng.normal(size=len(index))
    frame = pd.DataFrame(
        {
            "strong__lag_00": target + rng.normal(0, 0.05, len(index)),
            "strong__lag_01": target + rng.normal(0, 0.10, len(index)),
            "strong__lag_02": rng.normal(size=len(index)),
            "weak__lag_00": rng.normal(size=len(index)),
            "weak__lag_01": rng.normal(size=len(index)),
            "target": target,
        },
        index=index,
    )
    config = FeatureConfig(
        factor_icir_threshold=0.5,
        lags_per_factor=2,
        min_selected_features=2,
        max_selected_features=3,
    )
    result = select_features_by_icir(frame, config)
    assert result.factor_summary.loc["strong", "eligible"]
    assert result.features == ["strong__lag_00", "strong__lag_01"]
    assert not any(feature.startswith("weak__") for feature in result.features)
