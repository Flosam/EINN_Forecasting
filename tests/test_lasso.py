#!/usr/bin/env python
"""Tests for src/lasso.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lasso import Lasso


def _make_lasso_data(n: int = 220, seed: int = 321) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.3 + 0.5 * y[t - 1] + 0.7 * x1[t - 1] - 0.2 * x2[t - 1] + rng.normal(0.0, 0.2)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=pd.RangeIndex(n))


def _make_lasso_lag2_correlated_data(n: int = 320, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.85 * x1 + 0.15 * rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.15
            + 0.55 * y[t - 1]
            - 0.25 * y[t - 2]
            + 0.8 * x1[t - 1]
            - 0.35 * x2[t - 2]
            + rng.normal(0.0, 0.15)
        )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}, index=pd.RangeIndex(n))


def _make_lasso_data_datetime(n: int = 220, seed: int = 321) -> pd.DataFrame:
    data = _make_lasso_data(n=n, seed=seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    data.index = idx
    return data


def test_fit_produces_finite_parameters_and_residuals():
    data = _make_lasso_data()
    model = Lasso(lags=1, horizon=1, lmbda=0.05, max_iter=2000, tol=1e-7).fit(data, target="y")

    expected_coef_len = data.shape[1] * (model.lags + 1)
    assert model.coefs is not None
    assert model.coefs.shape == (expected_coef_len,)
    assert np.isfinite(model.coefs).all()
    assert np.isfinite(model.intercept)
    assert model.residuals is not None
    assert np.isfinite(model.residuals).all()
    assert model.residuals.shape[0] == len(data) - model.lags - model.horizon
    assert abs(model.intercept - 0.3) < 0.15
    assert abs(model.coefs[0] - 0.5) < 0.2
    assert abs(model.coefs[1] - 0.7) < 0.2
    assert abs(model.coefs[2] + 0.2) < 0.15
    assert np.max(np.abs(model.coefs[3:])) < 0.2


def test_forecast_returns_expected_shape_and_ci_ordering():
    data = _make_lasso_data()
    horizon = 10
    model = Lasso(lags=1, horizon=horizon, lmbda=0.05, max_iter=2000, tol=1e-7).fit(data, target="y")
    forecast = model.forecast(alpha=0.1, n_bootstrap=250)

    assert list(forecast.columns) == ["forecast", "lower", "upper"]
    assert len(forecast) == 1
    assert forecast.index[0] == data.index[-1] + horizon
    assert np.isfinite(forecast.to_numpy()).all()
    assert (forecast["lower"] <= forecast["upper"]).all()


def test_fit_lag2_recovers_sparse_signal_with_correlated_features():
    data = _make_lasso_lag2_correlated_data()
    model = Lasso(lags=2, horizon=1, lmbda=0.03, max_iter=4000, tol=1e-7).fit(data, target="y")

    assert model.coefs is not None
    assert model.coefs.shape == (data.shape[1] * (model.lags + 1),)
    assert np.isfinite(model.intercept)
    assert np.isfinite(model.coefs).all()
    # x1 is the primary driver in this DGP; Lasso should keep a strong x1 signal
    assert abs(model.coefs[1]) > 0.4
    # x3 is noise and should remain weak across included lags
    assert abs(model.coefs[3]) < 0.2
    assert abs(model.coefs[7]) < 0.2
    assert abs(model.coefs[11]) < 0.2


def test_lag2_forecast_matches_linear_prediction_for_requested_horizon():
    data = _make_lasso_lag2_correlated_data()
    model = Lasso(lags=2, horizon=3, lmbda=0.03, max_iter=4000, tol=1e-7).fit(data, target="y")
    forecast = model.forecast(alpha=0.1, n_bootstrap=150)

    x = data.iloc[-(model.lags + 1) :].values[::-1].flatten()
    expected = model.intercept + x @ model.coefs
    assert len(forecast) == 1
    assert np.allclose(forecast["forecast"].to_numpy(), np.array([expected]), atol=1e-12)


def test_bootstrap_forecast_is_reproducible_with_fixed_seed():
    data = _make_lasso_data()
    model = Lasso(lags=1, horizon=6, lmbda=0.05, max_iter=2000, tol=1e-7).fit(data, target="y")

    np.random.seed(42)
    first = model.forecast(alpha=0.1, n_bootstrap=200)
    np.random.seed(42)
    second = model.forecast(alpha=0.1, n_bootstrap=200)
    assert np.allclose(first.to_numpy(), second.to_numpy())


def test_forecast_before_fit_raises_value_error():
    model = Lasso(lags=1, horizon=4)
    data = _make_lasso_data()
    model.data = data
    try:
        model.forecast()
    except ValueError:
        return
    raise AssertionError("Expected ValueError when forecasting before fit")


def test_fit_rejects_too_short_series_for_requested_lags():
    short = _make_lasso_data(n=2)
    model = Lasso(lags=2, horizon=1)
    try:
        model.fit(short, target="y")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for data too short for requested lags")


def test_forecast_uses_month_end_index_for_datetime_data():
    data = _make_lasso_data_datetime()
    model = Lasso(lags=1, horizon=3, lmbda=0.05, max_iter=2000, tol=1e-7).fit(data, target="y")
    forecast = model.forecast(alpha=0.1, n_bootstrap=150)
    assert forecast.index.tolist() == [pd.Timestamp("2018-07-31")]


if __name__ == "__main__":
    test_fit_produces_finite_parameters_and_residuals()
    test_forecast_returns_expected_shape_and_ci_ordering()
    test_fit_lag2_recovers_sparse_signal_with_correlated_features()
    test_lag2_forecast_matches_linear_prediction_for_requested_horizon()
    test_bootstrap_forecast_is_reproducible_with_fixed_seed()
    test_forecast_before_fit_raises_value_error()
    test_fit_rejects_too_short_series_for_requested_lags()
    test_forecast_uses_month_end_index_for_datetime_data()
    print("\n✅ All Lasso tests passed!")
