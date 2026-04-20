#!/usr/bin/env python
"""Tests for src/base_nn.py."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from base_nn import BaseNN


def _make_multihorizon_data(n: int = 320, seed: int = 13, sigma: float = 0.12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x1[t] = 0.65 * x1[t - 1] + rng.normal(0.0, 0.6)
        x2[t] = 0.3 * x2[t - 1] + 0.2 * x1[t - 1] + rng.normal(0.0, 0.5)
        y[t] = (
            0.2
            + 0.75 * y[t - 1]
            + 0.28 * x1[t - 1]
            - 0.22 * x2[t]
            + 0.05 * (x1[t - 1] ** 2)
            + rng.normal(0.0, sigma)
        )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=pd.RangeIndex(n))


def _build_model(data: pd.DataFrame, epochs: int = 240) -> BaseNN:
    lags = 1
    din = data.shape[1] * (lags + 1)
    return BaseNN(
        din=din,
        dmid=(24, 24),
        lags=lags,
        learning_rate=0.01,
        epochs=epochs,
        rngs=nnx.Rngs(0),
    )


def test_fit_learns_multihorizon_structure_and_residuals_are_finite():
    data = _make_multihorizon_data()
    model = _build_model(data).fit(data, target="y")

    X = pd.concat([data.shift(lag) for lag in range(model.lags + 1)], axis=1).dropna().values[:-12]
    y = pd.concat([data["y"].shift(-h) for h in [1, 3, 6, 12]], axis=1).dropna().values[model.lags:]
    preds = np.asarray(model(jnp.asarray(X, dtype=jnp.float32)))

    assert model.residuals is not None
    assert model.residuals.shape == y.shape
    assert np.isfinite(model.residuals).all()
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.mean((y - y.mean(axis=0)) ** 2)
    assert mse < baseline_mse * 0.8

    corr_h1 = np.corrcoef(preds[:, 0], y[:, 0])[0, 1]
    corr_h12 = np.corrcoef(preds[:, 3], y[:, 3])[0, 1]
    assert corr_h1 > 0.75
    assert corr_h12 > 0.45


def test_forecast_returns_expected_horizons_and_confidence_band_ordering():
    data = _make_multihorizon_data(seed=41)
    model = _build_model(data, epochs=260).fit(data, target="y")
    forecast = model.forecast(data, n_bootstrap=300, alpha=0.1)

    assert list(forecast.columns) == ["forecast", "lower", "upper"]
    assert len(forecast) == 4
    assert forecast.index.tolist() == [data.index[-1] + h for h in [1, 3, 6, 12]]
    assert np.isfinite(forecast.to_numpy()).all()
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
    assert ((forecast["upper"] - forecast["lower"]) >= 0).all()
    assert forecast["forecast"].std() > 0.01


def test_persistent_and_noisy_data_still_yields_nontrivial_fit():
    data = _make_multihorizon_data(n=420, seed=99, sigma=0.25)
    model = _build_model(data, epochs=280).fit(data, target="y")

    X = pd.concat([data.shift(lag) for lag in range(model.lags + 1)], axis=1).dropna().values[:-12]
    y = pd.concat([data["y"].shift(-h) for h in [1, 3, 6, 12]], axis=1).dropna().values[model.lags:]
    preds = np.asarray(model(jnp.asarray(X, dtype=jnp.float32)))

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.mean((y - y.mean(axis=0)) ** 2)
    assert mse < baseline_mse * 0.9
    assert np.std(preds[:, 0]) > 0.15


def test_fit_raises_for_missing_target():
    data = _make_multihorizon_data(n=120)
    model = _build_model(data, epochs=10)
    try:
        model.fit(data, target="missing")
    except ValueError:
        return
    raise AssertionError("Expected ValueError when target column does not exist")


def test_fit_raises_for_too_short_series():
    data = _make_multihorizon_data(n=12)
    model = _build_model(data, epochs=10)
    try:
        model.fit(data, target="y")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for data too short for lags and 12-step targets")


def test_forecast_before_fit_raises_value_error():
    data = _make_multihorizon_data(n=80)
    model = _build_model(data, epochs=10)
    try:
        model.forecast(data)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when forecasting before fit")


def test_forecast_rejects_nonpositive_bootstrap_draws():
    data = _make_multihorizon_data(n=120)
    model = _build_model(data, epochs=60).fit(data, target="y")
    try:
        model.forecast(data, n_bootstrap=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when n_bootstrap <= 0")


if __name__ == "__main__":
    test_fit_learns_multihorizon_structure_and_residuals_are_finite()
    test_forecast_returns_expected_horizons_and_confidence_band_ordering()
    test_persistent_and_noisy_data_still_yields_nontrivial_fit()
    test_fit_raises_for_missing_target()
    test_fit_raises_for_too_short_series()
    test_forecast_before_fit_raises_value_error()
    test_forecast_rejects_nonpositive_bootstrap_draws()
    print("\n✅ All BaseNN tests passed!")
