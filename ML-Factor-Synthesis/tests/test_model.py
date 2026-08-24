import numpy as np

from model.train import RidgeFallback, _is_better_candidate


def test_fallback_model_retains_target_intercept():
    features = np.zeros((8, 3))
    target = np.full(8, 0.125)
    model = RidgeFallback().fit(features, target)
    assert np.allclose(model.predict(features), target)


def test_candidate_selection_can_follow_rmse_or_ic():
    lower_rmse = (0.05, 0.04, -0.10, object())
    higher_ic = (0.06, 0.05, 0.20, object())
    assert _is_better_candidate(lower_rmse, higher_ic, "rmse")
    assert _is_better_candidate(higher_ic, lower_rmse, "ic")
