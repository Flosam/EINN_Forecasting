#!/usr/bin/env python
"""Tests for src/ar1.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ar1 import AR1Model


def _make_ar1_series(
    n: int = 500,
    beta: float = 0.8,
    phi: float = 0.6,
    sigma: float = 0.5,
    seed: int = 123,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = beta + phi * y[t - 1] + rng.normal(0.0, sigma)
    return pd.Series(y, index=pd.RangeIndex(n))


def _make_persistent_ar1_series(
    n: int = 1200,
    beta: float = 0.2,
    phi: float = 0.97,
    sigma: float = 0.2,
    seed: int = 7,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = beta + phi * y[t - 1] + rng.normal(0.0, sigma)
    return pd.Series(y, index=pd.RangeIndex(n))


def test_fit_recovers_ar1_parameters_on_synthetic_data():
    series = _make_ar1_series()
    model = AR1Model().fit(series)

    assert np.isfinite(model.phi)
    assert np.isfinite(model.beta)
    assert np.isfinite(model.sigma2)
    assert model.sigma2 > 0.0
    assert abs(model.phi - 0.6) < 0.1
    assert abs(model.beta - 0.8) < 0.15
    assert abs((model.beta / (1 - model.phi)) - 2.0) < 0.25


def test_forecast_returns_expected_shape_and_ci_ordering():
    series = _make_ar1_series()
    model = AR1Model().fit(series)
    horizon = 12
    forecast = model.forecast(horizon=horizon, alpha=0.1)

    assert list(forecast.columns) == ["forecast", "lower", "upper"]
    assert len(forecast) == horizon
    assert forecast.index[0] == series.index[-1] + 1
    assert forecast.index[-1] == series.index[-1] + horizon
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
    assert ((forecast["upper"] - forecast["lower"]) >= 0).all()
    expected = []
    value = series.iloc[-1]
    for _ in range(horizon):
        value = model.beta + model.phi * value
        expected.append(value)
    assert np.allclose(forecast["forecast"].to_numpy(), np.array(expected), atol=1e-12)


def test_persistent_ar1_long_horizon_moves_toward_long_run_mean():
    series = _make_persistent_ar1_series()
    model = AR1Model().fit(series)
    horizon = 40
    forecast = model.forecast(horizon=horizon, alpha=0.1)
    long_run_mean = model.beta / (1 - model.phi)

    assert abs(model.phi - 0.97) < 0.03
    assert model.phi < 1.0
    assert abs(forecast["forecast"].iloc[-1] - long_run_mean) < abs(
        forecast["forecast"].iloc[0] - long_run_mean
    )

    expected = []
    value = series.iloc[-1]
    for _ in range(horizon):
        value = model.beta + model.phi * value
        expected.append(value)
    assert np.allclose(forecast["forecast"].to_numpy(), np.array(expected), atol=1e-12)


def test_lower_alpha_produces_wider_confidence_intervals():
    series = _make_ar1_series()
    model = AR1Model().fit(series)
    wide = model.forecast(horizon=8, alpha=0.05)
    narrow = model.forecast(horizon=8, alpha=0.2)
    wide_width = (wide["upper"] - wide["lower"]).mean()
    narrow_width = (narrow["upper"] - narrow["lower"]).mean()
    assert wide_width > narrow_width


def test_forecast_before_fit_raises_value_error():
    model = AR1Model()
    try:
        model.forecast(horizon=5)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when forecasting before fit")


def test_zero_horizon_forecast_returns_empty_dataframe():
    series = _make_ar1_series()
    model = AR1Model().fit(series)
    forecast = model.forecast(horizon=0)
    assert list(forecast.columns) == ["forecast", "lower", "upper"]
    assert forecast.empty


if __name__ == "__main__":
    test_fit_recovers_ar1_parameters_on_synthetic_data()
    test_forecast_returns_expected_shape_and_ci_ordering()
    test_persistent_ar1_long_horizon_moves_toward_long_run_mean()
    test_lower_alpha_produces_wider_confidence_intervals()
    test_forecast_before_fit_raises_value_error()
    test_zero_horizon_forecast_returns_empty_dataframe()
    print("\n✅ All AR1 tests passed!")
