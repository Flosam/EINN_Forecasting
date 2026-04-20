#!/usr/bin/env python
"""Tests for src/var.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from var import VAR


def _make_var_data(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.array([0.2, -0.1])
    a = np.array([[0.6, 0.1], [0.2, 0.5]])
    y = np.zeros((n, 2))
    for t in range(1, n):
        eps = rng.normal(0.0, 0.15, size=2)
        y[t] = c + a @ y[t - 1] + eps
    return pd.DataFrame(y, columns=["y1", "y2"], index=pd.RangeIndex(n))


def _make_var2_data(n: int = 500, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.array([0.1, -0.05])
    a1 = np.array([[0.5, 0.2], [-0.1, 0.45]])
    a2 = np.array([[0.2, -0.05], [0.08, 0.15]])
    cov = np.array([[0.04, 0.02], [0.02, 0.05]])
    y = np.zeros((n, 2))
    for t in range(2, n):
        eps = rng.multivariate_normal(mean=np.zeros(2), cov=cov)
        y[t] = c + a1 @ y[t - 1] + a2 @ y[t - 2] + eps
    return pd.DataFrame(y, columns=["y1", "y2"], index=pd.RangeIndex(n))


def test_fit_outputs_expected_shapes_and_finite_values():
    data = _make_var_data()
    model = VAR(lags=1).fit(data)
    true_c = np.array([0.2, -0.1])
    true_a = np.array([[0.6, 0.1], [0.2, 0.5]])

    assert model.coefs is not None
    assert model.sigma_u is not None
    assert model.coefs.shape == (1 + data.shape[1] * model.lags, data.shape[1])
    assert model.sigma_u.shape == (data.shape[1], data.shape[1])
    assert np.isfinite(model.coefs).all()
    assert np.isfinite(model.sigma_u).all()
    estimated_c = model.coefs[0]
    estimated_a = model.coefs[1 : 1 + data.shape[1], :].T
    assert np.allclose(estimated_c, true_c, atol=0.08)
    assert np.allclose(estimated_a, true_a, atol=0.12)


def test_fit_recovers_var2_params_under_correlated_shocks():
    data = _make_var2_data()
    model = VAR(lags=2).fit(data)
    true_c = np.array([0.1, -0.05])
    true_a1 = np.array([[0.5, 0.2], [-0.1, 0.45]])
    true_a2 = np.array([[0.2, -0.05], [0.08, 0.15]])

    estimated_c = model.coefs[0]
    estimated_a1 = model.coefs[1:3, :].T
    estimated_a2 = model.coefs[3:5, :].T
    assert np.allclose(estimated_c, true_c, atol=0.08)
    assert np.allclose(estimated_a1, true_a1, atol=0.1)
    assert np.allclose(estimated_a2, true_a2, atol=0.1)
    assert model.sigma_u[0, 1] > 0.0


def test_forecast_returns_per_variable_frames_with_confidence_bands():
    data = _make_var_data()
    model = VAR(lags=1).fit(data)
    horizon = 9
    forecasts = model.forecast(horizon=horizon, alpha=0.1)

    assert set(forecasts.keys()) == set(data.columns)
    for name, frame in forecasts.items():
        assert list(frame.columns) == ["forecast", "lower", "upper"]
        assert len(frame) == horizon
        assert frame.index[0] == data.index[-1] + 1
        assert frame.index[-1] == data.index[-1] + horizon
        assert np.isfinite(frame.to_numpy()).all()
        assert (frame["lower"] <= frame["forecast"]).all()
        assert (frame["forecast"] <= frame["upper"]).all()
        assert ((frame["upper"] - frame["lower"]) >= 0).all()
        assert name in data.columns

    estimated_c = model.coefs[0]
    estimated_a = model.coefs[1 : 1 + data.shape[1], :].T
    expected_step1 = estimated_c + estimated_a @ data.iloc[-1].to_numpy()
    expected_step2 = estimated_c + estimated_a @ expected_step1
    actual_step1 = np.array([forecasts[col]["forecast"].iloc[0] for col in data.columns])
    actual_step2 = np.array([forecasts[col]["forecast"].iloc[1] for col in data.columns])
    assert np.allclose(actual_step1, expected_step1, atol=1e-12)
    assert np.allclose(actual_step2, expected_step2, atol=1e-12)


def test_var2_forecast_follows_two_lag_recursion():
    data = _make_var2_data()
    model = VAR(lags=2).fit(data)
    forecasts = model.forecast(horizon=3, alpha=0.1)

    estimated_c = model.coefs[0]
    estimated_a1 = model.coefs[1:3, :].T
    estimated_a2 = model.coefs[3:5, :].T
    y_t = data.iloc[-1].to_numpy()
    y_tm1 = data.iloc[-2].to_numpy()
    expected_step1 = estimated_c + estimated_a1 @ y_t + estimated_a2 @ y_tm1
    expected_step2 = estimated_c + estimated_a1 @ expected_step1 + estimated_a2 @ y_t
    actual_step1 = np.array([forecasts[col]["forecast"].iloc[0] for col in data.columns])
    actual_step2 = np.array([forecasts[col]["forecast"].iloc[1] for col in data.columns])
    assert np.allclose(actual_step1, expected_step1, atol=1e-12)
    assert np.allclose(actual_step2, expected_step2, atol=1e-12)


def test_confidence_bands_do_not_shrink_with_horizon():
    data = _make_var_data()
    model = VAR(lags=1).fit(data)
    forecasts = model.forecast(horizon=12, alpha=0.05)

    for frame in forecasts.values():
        width = frame["upper"] - frame["lower"]
        assert width.iloc[-1] >= width.iloc[0]


def test_forecast_before_fit_raises_value_error():
    model = VAR(lags=1)
    try:
        model.forecast(horizon=3)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when forecasting before fit")


def test_zero_horizon_forecast_returns_empty_frames():
    data = _make_var_data()
    model = VAR(lags=1).fit(data)
    forecasts = model.forecast(horizon=0)
    assert set(forecasts.keys()) == set(data.columns)
    for frame in forecasts.values():
        assert list(frame.columns) == ["forecast", "lower", "upper"]
        assert frame.empty


if __name__ == "__main__":
    test_fit_outputs_expected_shapes_and_finite_values()
    test_fit_recovers_var2_params_under_correlated_shocks()
    test_forecast_returns_per_variable_frames_with_confidence_bands()
    test_var2_forecast_follows_two_lag_recursion()
    test_confidence_bands_do_not_shrink_with_horizon()
    test_forecast_before_fit_raises_value_error()
    test_zero_horizon_forecast_returns_empty_frames()
    print("\n✅ All VAR tests passed!")
