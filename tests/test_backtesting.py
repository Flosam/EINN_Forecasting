#!/usr/bin/env python
"""Tests for src/backtesting.py."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting import (
    ModelSpec,
    _add_months,
    build_test_origins,
    load_hyperparameters,
    merge_hyperparameters,
    run_fixed_params_backtest,
    run_validation_once,
    save_hyperparameters,
    window_bounds_for_origin,
)


def _make_data() -> pd.DataFrame:
    idx = pd.date_range("2010-01-31", "2020-12-31", freq="ME")
    t = np.arange(len(idx))
    y = 2.0 + 0.03 * t + 0.2 * np.sin(t / 4.0)
    x = 0.5 + 0.01 * t
    return pd.DataFrame({"cpi_all_items": y, "x1": x}, index=idx)


def _dummy_predict_fn(train_df: pd.DataFrame, params: dict, alpha: float, n_bootstrap: int) -> pd.DataFrame:
    del alpha, n_bootstrap
    horizons = params["horizons"]
    bias = params.get("bias", 0.0)
    base = float(train_df["cpi_all_items"].iloc[-1])
    rows = []
    idx = []
    for h in horizons:
        idx.append(_add_months(train_df.index[-1], h))
        point = base + 0.02 * h + bias
        rows.append({"forecast": point, "lower": point, "upper": point})
    return pd.DataFrame(rows, index=idx)


def test_build_test_origins_respects_horizon_limit():
    val_end = pd.Timestamp("2018-01-31")
    test_end = pd.Timestamp("2018-12-31")
    origins = build_test_origins(val_end=val_end, test_end=test_end, max_horizon=3, step_months=1)
    assert origins[0] == pd.Timestamp("2018-01-31")
    assert origins[-1] == pd.Timestamp("2018-09-30")


def test_window_bounds_shift_with_origin():
    train_end0 = pd.Timestamp("2016-01-31")
    val_end0 = pd.Timestamp("2018-01-31")
    origin = pd.Timestamp("2018-04-30")
    train_end, val_start, val_end = window_bounds_for_origin(train_end0, val_end0, origin)

    assert train_end == pd.Timestamp("2016-04-30")
    assert val_start == pd.Timestamp("2016-05-31")
    assert val_end == pd.Timestamp("2018-04-30")


def test_run_validation_once_returns_best_params_and_validation_rows():
    data = _make_data()
    model_specs = [
        ModelSpec(
            name="dummy",
            param_grid=[
                {"horizons": [1, 3, 6, 12], "bias": 0.0},
                {"horizons": [1, 3, 6, 12], "bias": 1.0},
            ],
            predict_fn=_dummy_predict_fn,
        )
    ]

    best_params, predictions, by_horizon, overall = run_validation_once(
        data,
        model_specs=model_specs,
        target="cpi_all_items",
        horizons=[1, 3, 6, 12],
        train_end="2014-01",
        val_end="2015-01",
        step_months=1,
        alpha=0.1,
        n_bootstrap=0,
    )

    assert best_params["dummy"]["bias"] == 0.0
    assert not predictions.empty
    assert (predictions["split"] == "validation").all()
    assert predictions["inner_fold"].notna().all()
    assert not by_horizon.empty
    assert not overall.empty


def test_run_fixed_params_backtest_outputs_test_split_only():
    data = _make_data()
    model_specs = [
        ModelSpec(
            name="dummy",
            param_grid=[{"horizons": [1, 3, 6, 12]}],
            predict_fn=_dummy_predict_fn,
        )
    ]
    predictions, by_horizon, overall = run_fixed_params_backtest(
        data,
        fixed_hyperparameters={"dummy": {"horizons": [1, 3, 6, 12], "bias": 0.0}},
        model_specs=model_specs,
        target="cpi_all_items",
        horizons=[1, 3, 6, 12],
        train_end="2014-01",
        val_end="2015-01",
        test_end="2017-12",
        step_months=1,
        alpha=0.1,
        n_bootstrap=0,
    )

    assert not predictions.empty
    assert (predictions["split"] == "test").all()
    assert predictions["inner_fold"].isna().all()
    assert not by_horizon.empty
    assert not overall.empty


def test_save_and_load_hyperparameters_roundtrip_restores_types():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "hyperparameters.json"
        save_hyperparameters(
            {
                "einn": {"dmid": (32, 32), "lags": 24},
                "base_nn": {"dmid": (24, 24), "epochs": 300},
                "var": {"use_bic": True},
            },
            path=str(path),
        )
        loaded = load_hyperparameters(str(path))

    assert isinstance(loaded["einn"]["dmid"], tuple)
    assert isinstance(loaded["base_nn"]["dmid"], tuple)
    assert loaded["var"]["use_bic"] is True


def test_manual_hyperparameters_override_saved_values():
    saved = {"einn": {"lags": 1, "epochs": 300}, "var": {"use_bic": True}}
    manual = {"einn": {"lags": 24, "epochs": 200}}
    merged = merge_hyperparameters(saved, manual)

    assert merged["einn"]["lags"] == 24
    assert merged["einn"]["epochs"] == 200
    assert merged["var"]["use_bic"] is True


if __name__ == "__main__":
    test_build_test_origins_respects_horizon_limit()
    test_window_bounds_shift_with_origin()
    test_run_validation_once_returns_best_params_and_validation_rows()
    test_run_fixed_params_backtest_outputs_test_split_only()
    test_save_and_load_hyperparameters_roundtrip_restores_types()
    test_manual_hyperparameters_override_saved_values()
    print("\n✅ All backtesting tests passed!")
