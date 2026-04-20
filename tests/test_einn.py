#!/usr/bin/env python
"""Tests for src/einn.py."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from einn import EINN


def _build_targets(data: pd.DataFrame, target: str, horizons: list[int], lags: int) -> np.ndarray:
    return (
        pd.concat([data[target].shift(-h) for h in horizons], axis=1)
        .dropna()
        .values[lags:]
    )


def _make_economic_data(
    n: int = 340,
    seed: int = 11,
    sigma: float = 0.1,
    persistence: float = 0.75,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cpi = np.zeros(n)
    natural_u = np.zeros(n)
    u = np.zeros(n)
    exp_inf = np.zeros(n)
    activity = np.zeros(n)

    natural_u[0] = 4.7
    u[0] = 5.0
    exp_inf[0] = 2.1
    cpi[0] = 2.0
    activity[0] = 0.0

    for t in range(1, n):
        activity[t] = 0.7 * activity[t - 1] + rng.normal(0.0, 0.35)
        natural_u[t] = 4.7 + 0.96 * (natural_u[t - 1] - 4.7) + rng.normal(0.0, 0.05)
        u[t] = 5.0 + 0.88 * (u[t - 1] - 5.0) - 0.08 * activity[t - 1] + rng.normal(0.0, 0.12)
        gap = u[t] - natural_u[t]
        exp_inf[t] = 0.68 * exp_inf[t - 1] + 0.25 * cpi[t - 1] + rng.normal(0.0, 0.08)
        cpi[t] = (
            0.15
            + persistence * cpi[t - 1]
            + 0.28 * exp_inf[t - 1]
            - 0.22 * gap
            + 0.04 * (gap**2)
            + 0.06 * activity[t]
            + rng.normal(0.0, sigma)
        )

    return pd.DataFrame(
        {
            "cpi_all_items": cpi,
            "unemployment_rate": u,
            "natural_rate_unemployment": natural_u,
            "inflation_expectations_umich": exp_inf,
            "activity_factor": activity,
        },
        index=pd.RangeIndex(n),
    )


def _build_model(
    data: pd.DataFrame, epochs: int = 220, learning_rate: float = 0.01, pc_weight: float = 0.3
) -> EINN:
    lags = 1
    din = data.shape[1] * (lags + 1)
    return EINN(
        din=din,
        dmid=(24, 24),
        lags=lags,
        learning_rate=learning_rate,
        epochs=epochs,
        pc_weight=pc_weight,
        rngs=nnx.Rngs(0),
    )


def test_fit_learns_multihorizon_structure_and_residuals_are_finite():
    data = _make_economic_data()
    model = _build_model(data).fit(data, target="cpi_all_items")

    max_h = max(model.horizons)
    X = pd.concat([data.shift(lag) for lag in range(model.lags + 1)], axis=1).dropna().values[:-max_h]
    y = _build_targets(data, target="cpi_all_items", horizons=model.horizons, lags=model.lags)
    preds = np.asarray(model(jnp.asarray(X, dtype=jnp.float32)))

    assert model.residuals is not None
    assert model.residuals.shape == y.shape
    assert np.isfinite(model.residuals).all()
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.mean((y - y.mean(axis=0)) ** 2)
    assert mse < baseline_mse * 0.82

    corr_h1 = np.corrcoef(preds[:, 0], y[:, 0])[0, 1]
    corr_h12 = np.corrcoef(preds[:, 3], y[:, 3])[0, 1]
    assert corr_h1 > 0.75
    assert corr_h12 > 0.45


def test_forecast_returns_expected_horizons_and_confidence_band_ordering():
    data = _make_economic_data(seed=33)
    model = _build_model(data, epochs=240).fit(data, target="cpi_all_items")
    forecast = model.forecast(data, n_bootstrap=350, alpha=0.1)

    assert list(forecast.columns) == ["forecast", "lower", "upper"]
    assert len(forecast) == len(model.horizons)
    assert forecast.index.tolist() == [data.index[-1] + h for h in model.horizons]
    assert np.isfinite(forecast.to_numpy()).all()
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
    assert ((forecast["upper"] - forecast["lower"]) >= 0).all()
    assert forecast["forecast"].std() > 0.01


def test_lower_alpha_produces_wider_confidence_intervals():
    data = _make_economic_data(seed=49)
    model = _build_model(data, epochs=230).fit(data, target="cpi_all_items")
    wide = model.forecast(data, n_bootstrap=1200, alpha=0.05)
    narrow = model.forecast(data, n_bootstrap=1200, alpha=0.2)

    wide_width = (wide["upper"] - wide["lower"]).mean()
    narrow_width = (narrow["upper"] - narrow["lower"]).mean()
    assert wide_width > narrow_width


def test_persistent_and_noisy_data_still_yields_nontrivial_fit():
    data = _make_economic_data(n=440, seed=91, sigma=0.18, persistence=0.72)
    model = _build_model(data, epochs=220, learning_rate=0.008, pc_weight=0.0).fit(
        data, target="cpi_all_items"
    )

    max_h = max(model.horizons)
    X = pd.concat([data.shift(lag) for lag in range(model.lags + 1)], axis=1).dropna().values[:-max_h]
    y = _build_targets(data, target="cpi_all_items", horizons=model.horizons, lags=model.lags)
    preds = np.asarray(model(jnp.asarray(X, dtype=jnp.float32)))

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.mean((y - y.mean(axis=0)) ** 2)
    assert mse < baseline_mse * 0.94
    assert np.std(preds[:, 0]) > 0.08


def test_fit_raises_for_missing_target():
    data = _make_economic_data(n=140)
    model = _build_model(data, epochs=10)
    try:
        model.fit(data, target="missing")
    except ValueError:
        return
    raise AssertionError("Expected ValueError when target column does not exist")


def test_fit_raises_for_too_short_series():
    data = _make_economic_data(n=12)
    model = _build_model(data, epochs=10)
    try:
        model.fit(data, target="cpi_all_items")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for data too short for lags and 12-step targets")


def test_forecast_before_fit_raises_value_error():
    data = _make_economic_data(n=90)
    model = _build_model(data, epochs=10)
    try:
        model.forecast(data)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when forecasting before fit")


def test_forecast_rejects_nonpositive_bootstrap_draws():
    data = _make_economic_data(n=180)
    model = _build_model(data, epochs=80).fit(data, target="cpi_all_items")
    try:
        model.forecast(data, n_bootstrap=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when n_bootstrap <= 0")


def test_forecast_rejects_too_short_forecast_history():
    data = _make_economic_data(n=180)
    model = _build_model(data, epochs=80).fit(data, target="cpi_all_items")
    try:
        model.forecast(data.iloc[:1])
    except ValueError:
        return
    raise AssertionError("Expected ValueError when lagged forecasting features cannot be built")


if __name__ == "__main__":
    test_fit_learns_multihorizon_structure_and_residuals_are_finite()
    test_forecast_returns_expected_horizons_and_confidence_band_ordering()
    test_lower_alpha_produces_wider_confidence_intervals()
    test_persistent_and_noisy_data_still_yields_nontrivial_fit()
    test_fit_raises_for_missing_target()
    test_fit_raises_for_too_short_series()
    test_forecast_before_fit_raises_value_error()
    test_forecast_rejects_nonpositive_bootstrap_draws()
    test_forecast_rejects_too_short_forecast_history()
    print("\n✅ All EINN tests passed!")
