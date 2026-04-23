from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import (
    BACKTEST_STEP_MONTHS,
    BACKTEST_TEST_END,
    BACKTEST_TRAIN_END,
    BACKTEST_VAL_END,
    EXPECTATIONS_VARIABLE,
    GAP_VARIABLES,
    HORIZONS,
    MODEL_PARAM_GRIDS,
    TARGET_VARIABLE,
    VAR_MAX_BIC_LAGS,
)
from .evaluation import summarize_metrics, validation_score
from .logging_utils import get_logger, log_debug, log_info, log_warning, wandb_log
from .models.ar1 import AR1Model
from .models.base_nn import BaseNN
from .models.einn import EINN
from .models.lasso import Lasso
from .models.var import VAR
from .utils import add_months, month_end

DEFAULT_HYPERPARAMETERS_PATH = "data/processed/backtest_hyperparameters.json"
BACKTEST_MODES = {"validate", "run"}
GRID_PROFILES = {"full", "balanced", "fast"}

# Backward-compatibility alias used by older tests.
_add_months = add_months

# Global storage for NKPC parameters during backtesting
_nkpc_parameters_history: list[dict[str, Any]] = []


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_grid: list[dict[str, Any]]
    predict_fn: Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class CVFold:
    fold_id: int
    origin: pd.Timestamp


def build_test_origins(
    val_end: pd.Timestamp, test_end: pd.Timestamp, max_horizon: int, step_months: int
) -> list[pd.Timestamp]:
    # Only use origins that still have every forecast horizon available before test_end.
    last_origin = add_months(test_end, -max_horizon)
    origins: list[pd.Timestamp] = []
    cursor = val_end
    while cursor <= last_origin:
        origins.append(cursor)
        cursor = add_months(cursor, step_months)
    return origins


def window_bounds_for_origin(
    initial_train_end: pd.Timestamp, initial_val_end: pd.Timestamp, origin: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    # Shift the original train/validation split forward by the same number of months as the origin.
    offset = (origin.to_period("M") - initial_val_end.to_period("M")).n
    train_end = add_months(initial_train_end, offset)
    val_start = add_months(train_end, 1)
    val_end = add_months(initial_val_end, offset)
    return train_end, val_start, val_end


def _expand_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, row)) for row in product(*values)]


def _grid_from_config(
    model_name: str, model_param_grids: dict[str, dict[str, list[Any]]]
) -> list[dict[str, Any]]:
    space = model_param_grids.get(model_name, {})
    if not space:
        return [{}]
    return _expand_grid(space)


def _resolve_model_param_grids(
    profile: str = "balanced", model_param_grids: dict[str, dict[str, list[Any]]] | None = None
) -> dict[str, dict[str, list[Any]]]:
    if model_param_grids is None:
        model_param_grids = MODEL_PARAM_GRIDS
    if profile not in GRID_PROFILES:
        raise ValueError(f"Unknown grid profile '{profile}'. Expected one of {sorted(GRID_PROFILES)}.")
    if profile in {"full", "balanced"}:
        return model_param_grids

    fast: dict[str, dict[str, list[Any]]] = {}
    for model_name, space in model_param_grids.items():
        reduced_space: dict[str, list[Any]] = {}
        for key, values in space.items():
            if not values:
                reduced_space[key] = values
                continue
            if key == "epochs":
                reduced_space[key] = [min(values)]
            elif len(values) > 2:
                reduced_space[key] = [values[0], values[-1]]
            else:
                reduced_space[key] = values
        fast[model_name] = reduced_space
    return fast


def _predict_einn_with_tracking(
    train_df: pd.DataFrame,
    params: dict[str, Any],
    alpha: float,
    n_bootstrap: int,
    target: str,
    horizons: list[int],
    nkpc_params_list: list[dict[str, Any]],
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Predict with EINN and capture NKPC parameters for tracking."""
    lags = params.get("lags", 1)
    din = train_df.shape[1] * (lags + 1)
    dmid = params.get("dmid", (24, 24))
    if isinstance(dmid, list):
        dmid = tuple(dmid)
    model = EINN(
        din=din,
        dmid=dmid,
        lags=lags,
        learning_rate=params.get("learning_rate", 0.01),
        epochs=params.get("epochs", 160),
        pc_weight=params.get("pc_weight", 0.3),
        horizons=horizons,
        horizon_weight=params.get("horizon_weight", 0.7),
    )
    model.fit(
        train_df,
        target=target,
        gap_variables=params.get("gap_variables", GAP_VARIABLES),
        expectations_variable=params.get("expectations_variable", EXPECTATIONS_VARIABLE),
    )
    
    # Capture NKPC parameters
    nkpc_params = model.get_nkpc_params()
    nkpc_params_list.append({
        "origin": origin.strftime("%Y-%m"),
        "beta": nkpc_params["beta"],
        "kappa": nkpc_params["kappa"],
    })
    
    return model.forecast(train_df, n_bootstrap=n_bootstrap, alpha=alpha)


def _save_nkpc_parameters(nkpc_params_list: list[dict[str, Any]]) -> None:
    """Save NKPC parameters to CSV in results folder."""
    output_path = Path("data/processed/results/einn_nkpc_parameters.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(nkpc_params_list)
    df.to_csv(output_path, index=False)
    log_info(f"Saved NKPC parameters to {output_path}")



def _train_model(
    model_spec: ModelSpec, train_df: pd.DataFrame, params: dict[str, Any],
    target: str, horizons: list[int], alpha: float, n_bootstrap: int
) -> Any:
    """Train a model without forecasting. Returns the trained model object."""
    if model_spec.name == "einn":
        lags = params.get("lags", 1)
        din = train_df.shape[1] * (lags + 1)
        dmid = params.get("dmid", (24, 24))
        if isinstance(dmid, list):
            dmid = tuple(dmid)
        model = EINN(
            din=din, dmid=dmid, lags=lags,
            learning_rate=params.get("learning_rate", 0.01),
            epochs=params.get("epochs", 160),
            pc_weight=params.get("pc_weight", 0.3),
            horizons=horizons,
            horizon_weight=params.get("horizon_weight", 0.7),
        )
        model.fit(train_df, target=target, gap_variables=params.get("gap_variables", GAP_VARIABLES),
                  expectations_variable=params.get("expectations_variable", EXPECTATIONS_VARIABLE))
        return model
    elif model_spec.name == "base_nn":
        lags = params.get("lags", 1)
        din = train_df.shape[1] * (lags + 1)
        dmid = params.get("dmid", (24, 24))
        if isinstance(dmid, list):
            dmid = tuple(dmid)
        model = BaseNN(
            din=din, dmid=dmid, lags=lags,
            learning_rate=params.get("learning_rate", 0.01),
            epochs=params.get("epochs", 160),
            horizons=horizons,
        )
        model.fit(train_df, target=target)
        return model
    elif model_spec.name == "lasso":
        # For LASSO, we return a list of models (one per horizon)
        models = []
        for h in horizons:
            model = Lasso(
                lags=params.get("lags", 1), horizon=h,
                lmbda=params.get("lmbda", 0.1),
                max_iter=params.get("max_iter", 1000),
                tol=params.get("tol", 1e-6),
            ).fit(train_df, target=target)
            models.append(model)
        return models
    elif model_spec.name == "ar1":
        series = train_df[target]
        model = AR1Model().fit(series)
        return model
    elif model_spec.name == "var":
        model = VAR(max_bic_lags=params.get("max_bic_lags", VAR_MAX_BIC_LAGS)).fit(train_df)
        return model
    else:
        # For test models or unknown models, return a wrapper that calls the predict_fn
        # This allows the forecast functions to work with test models
        return {"_predict_fn": model_spec.predict_fn, "_params": params, "_train_df": train_df}


def _forecast_from_trained_model(
    trained_model: Any, train_df: pd.DataFrame, model_name: str,
    target: str, horizons: list[int], alpha: float, n_bootstrap: int
) -> pd.DataFrame:
    """Generate forecast from a pre-trained model."""
    # Handle test models
    if isinstance(trained_model, dict) and "_predict_fn" in trained_model:
        return trained_model["_predict_fn"](train_df, trained_model["_params"], alpha, n_bootstrap)
    
    if model_name == "einn":
        return trained_model.forecast(train_df, n_bootstrap=n_bootstrap, alpha=alpha)
    elif model_name == "base_nn":
        return trained_model.forecast(train_df, n_bootstrap=n_bootstrap, alpha=alpha)
    elif model_name == "lasso":
        pieces = []
        for model in trained_model:
            pieces.append(model.forecast(alpha=alpha, n_bootstrap=n_bootstrap))
        return pd.concat(pieces).sort_index()
    elif model_name == "ar1":
        max_h = max(horizons)
        fc = trained_model.forecast(horizon=max_h, alpha=alpha)
        fc = fc.loc[[add_months(train_df.index[-1], h) for h in horizons]]
        return fc
    elif model_name == "var":
        return trained_model.forecast(horizon=max(horizons), alpha=alpha, target=target)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _forecast_from_cached_model(
    trained_model: Any, train_df: pd.DataFrame, model_name: str,
    target: str, horizons: list[int], alpha: float, n_bootstrap: int
) -> pd.DataFrame:
    """Generate forecast from a cached pre-trained model (same as non-cached)."""
    return _forecast_from_trained_model(trained_model, train_df, model_name, target, horizons, alpha, n_bootstrap)


def _forecast_from_trained_model_with_tracking(
    trained_model: Any, train_df: pd.DataFrame, model_name: str,
    target: str, horizons: list[int], alpha: float, n_bootstrap: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate forecast and extract NKPC parameters for EINN."""
    if model_name != "einn":
        raise ValueError("NKPC parameter tracking only supported for EINN")
    forecast_df = trained_model.forecast(train_df, n_bootstrap=n_bootstrap, alpha=alpha)
    nkpc_params = trained_model.get_nkpc_params()
    return forecast_df, nkpc_params


def default_model_specs(
    target: str,
    horizons: list[int],
    model_param_grids: dict[str, dict[str, list[Any]]] | None = None,
) -> list[ModelSpec]:
    if model_param_grids is None:
        model_param_grids = MODEL_PARAM_GRIDS

    def predict_einn(
        train_df: pd.DataFrame, params: dict[str, Any], alpha: float, n_bootstrap: int
    ) -> pd.DataFrame:
        lags = params.get("lags", 1)
        din = train_df.shape[1] * (lags + 1)
        dmid = params.get("dmid", (24, 24))
        if isinstance(dmid, list):
            dmid = tuple(dmid)
        model = EINN(
            din=din,
            dmid=dmid,
            lags=lags,
            learning_rate=params.get("learning_rate", 0.01),
            epochs=params.get("epochs", 160),
            pc_weight=params.get("pc_weight", 0.3),
            horizons=horizons,
            horizon_weight=params.get("horizon_weight", 0.7),
        )
        model.fit(
            train_df,
            target=target,
            gap_variables=params.get("gap_variables", GAP_VARIABLES),
            expectations_variable=params.get("expectations_variable", EXPECTATIONS_VARIABLE),
        )
        # Store NKPC parameters for analysis
        nkpc_params = model.get_nkpc_params()
        model._nkpc_params = nkpc_params
        return model.forecast(train_df, n_bootstrap=n_bootstrap, alpha=alpha)

    def predict_base_nn(
        train_df: pd.DataFrame, params: dict[str, Any], alpha: float, n_bootstrap: int
    ) -> pd.DataFrame:
        lags = params.get("lags", 1)
        din = train_df.shape[1] * (lags + 1)
        dmid = params.get("dmid", (24, 24))
        if isinstance(dmid, list):
            dmid = tuple(dmid)
        model = BaseNN(
            din=din,
            dmid=dmid,
            lags=lags,
            learning_rate=params.get("learning_rate", 0.01),
            epochs=params.get("epochs", 160),
            horizons=horizons,
        )
        model.fit(train_df, target=target)
        return model.forecast(train_df, n_bootstrap=n_bootstrap, alpha=alpha)

    def predict_lasso(
        train_df: pd.DataFrame, params: dict[str, Any], alpha: float, n_bootstrap: int
    ) -> pd.DataFrame:
        pieces = []
        for h in horizons:
            model = Lasso(
                lags=params.get("lags", 1),
                horizon=h,
                lmbda=params.get("lmbda", 0.1),
                max_iter=params.get("max_iter", 1000),
                tol=params.get("tol", 1e-6),
            ).fit(train_df, target=target)
            pieces.append(model.forecast(alpha=alpha, n_bootstrap=n_bootstrap))
        return pd.concat(pieces).sort_index()

    def predict_ar1(
        train_df: pd.DataFrame, params: dict[str, Any], alpha: float, n_bootstrap: int
    ) -> pd.DataFrame:
        del params, n_bootstrap
        series = train_df[target]
        model = AR1Model().fit(series)
        max_h = max(horizons)
        fc = model.forecast(horizon=max_h, alpha=alpha)
        fc = fc.loc[[add_months(train_df.index[-1], h) for h in horizons]]
        return fc

    def predict_var(
        train_df: pd.DataFrame, params: dict[str, Any], alpha: float, n_bootstrap: int
    ) -> pd.DataFrame:
        del n_bootstrap
        use_bic = params.get("use_bic", True)
        
        # VAR with all 18 variables is unstable. Use only key economic variables.
        var_cols = [TARGET_VARIABLE, EXPECTATIONS_VARIABLE] + GAP_VARIABLES
        var_cols = [c for c in var_cols if c in train_df.columns]
        var_data = train_df[var_cols]
        
        if var_data.empty or var_data.shape[1] < 2:
            raise ValueError(f"Insufficient VAR columns: need at least 2, got {var_data.shape[1]}")
        
        model = VAR(use_bic=use_bic, max_bic_lags=VAR_MAX_BIC_LAGS).fit(var_data)
        max_h = max(horizons)
        fc = model.forecast(horizon=max_h, alpha=alpha, target=TARGET_VARIABLE)
        fc = fc.loc[[add_months(train_df.index[-1], h) for h in horizons]]
        return fc

    return [
        ModelSpec(
            name="einn",
            param_grid=_grid_from_config("einn", model_param_grids),
            predict_fn=predict_einn,
        ),
        ModelSpec(
            name="base_nn",
            param_grid=_grid_from_config("base_nn", model_param_grids),
            predict_fn=predict_base_nn,
        ),
        ModelSpec(
            name="lasso",
            param_grid=_grid_from_config("lasso", model_param_grids),
            predict_fn=predict_lasso,
        ),
        ModelSpec(name="ar1", param_grid=_grid_from_config("ar1", model_param_grids), predict_fn=predict_ar1),
        ModelSpec(name="var", param_grid=_grid_from_config("var", model_param_grids), predict_fn=predict_var),
    ]


def _rows_from_forecast(
    *,
    model: str,
    split: str,
    origin: pd.Timestamp,
    forecast_df: pd.DataFrame,
    data: pd.DataFrame,
    target: str,
    horizons: list[int],
    params: dict[str, Any],
    outer_fold_id: int,
    inner_fold_id: int | None = None,
    cutoff: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in horizons:
        target_date = add_months(origin, h)
        if target_date not in data.index:
            continue
        if cutoff is not None and target_date > cutoff:
            continue
        if target_date not in forecast_df.index:
            continue
        pred_row = forecast_df.loc[target_date]
        rows.append(
            {
                "model": model,
                "split": split,
                "outer_fold": outer_fold_id,
                "inner_fold": inner_fold_id,
                "origin_date": origin,
                "target_date": target_date,
                "horizon": h,
                "y_true": float(data.loc[target_date, target]),
                "y_pred": float(pred_row["forecast"]),
                "lower": float(pred_row["lower"]),
                "upper": float(pred_row["upper"]),
                "params": str(params),
            }
        )
    return rows


def _score_model_on_validation(
    *,
    data: pd.DataFrame,
    model_spec: ModelSpec,
    params: dict[str, Any],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    horizons: list[int],
    target: str,
    alpha: float,
    n_bootstrap: int,
    outer_fold_id: int,
    step_months: int,
    nkpc_params_list: list[dict[str, Any]] | None = None,
) -> tuple[float, pd.DataFrame]:
    val_rows: list[dict[str, Any]] = []
    # For validation, fit only once at val_end (single point estimate, not expanding window)
    train_df = data.loc[:val_end]
    if len(train_df) < max(horizons) + 20:
        raise ValueError(f"Insufficient training data for validation (need {max(horizons) + 20}, got {len(train_df)})")
    
    log_debug(f"Training {model_spec.name} on {len(train_df)} rows (up to {val_end.strftime('%Y-%m')})")
    
    # Capture NKPC parameters if requested (for EINN only)
    if nkpc_params_list is not None and model_spec.name == "einn":
        forecast_df = _predict_einn_with_tracking(
            train_df, params, alpha, n_bootstrap, target, horizons,
            nkpc_params_list, val_end
        )
    else:
        try:
            forecast_df = model_spec.predict_fn(train_df, params, alpha, n_bootstrap)
        except Exception as e:
            log_debug(f"Forecast error: {type(e).__name__}: {e}")
            raise ValueError(f"Forecast failed: {e}")
    
    # Check for NaN/inf predictions
    if forecast_df.empty:
        raise ValueError("Forecast returned empty dataframe")
    if forecast_df["forecast"].isna().all():
        raise ValueError("All forecasts are NaN")
    if (forecast_df["forecast"] == float("inf")).any() or (forecast_df["forecast"] == float("-inf")).any():
        raise ValueError("Forecast contains infinity values")
    
    val_rows.extend(
        _rows_from_forecast(
            model=model_spec.name,
            split="validation",
            origin=val_end,
            forecast_df=forecast_df,
            data=data,
            target=target,
            horizons=horizons,
            params=params,
            outer_fold_id=outer_fold_id,
            inner_fold_id=1,  # Single validation point
            cutoff=None,  # No cutoff for validation scoring
        )
    )

    val_df = pd.DataFrame(val_rows)
    return validation_score(val_df, horizons), val_df


def _ensure_backtest_inputs(data: pd.DataFrame, target: str) -> None:
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported hyperparameter value type: {type(value)}")


def _restore_param_types(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    restored = dict(params)
    if model_name in {"einn", "base_nn"} and isinstance(restored.get("dmid"), list):
        restored["dmid"] = tuple(restored["dmid"])
    return restored


def _validate_hyperparameter_map(hyperparameters: dict[str, dict[str, Any]]) -> None:
    if not isinstance(hyperparameters, dict):
        raise ValueError("hyperparameters must be a dict[str, dict[str, Any]].")
    for model_name, params in hyperparameters.items():
        if not isinstance(model_name, str):
            raise ValueError("hyperparameter model names must be strings.")
        if not isinstance(params, dict):
            raise ValueError(f"Hyperparameters for model '{model_name}' must be a dictionary.")


def save_hyperparameters(
    hyperparameters: dict[str, dict[str, Any]],
    path: str = DEFAULT_HYPERPARAMETERS_PATH,
) -> None:
    _validate_hyperparameter_map(hyperparameters)
    payload = {model_name: _to_jsonable(params) for model_name, params in hyperparameters.items()}
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_hyperparameters(path: str = DEFAULT_HYPERPARAMETERS_PATH) -> dict[str, dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Hyperparameter file not found at '{path}'. Run validation mode first or provide manual parameters."
        )
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Hyperparameter file must contain a dictionary at top level.")
    loaded: dict[str, dict[str, Any]] = {}
    for model_name, params in raw.items():
        if not isinstance(params, dict):
            raise ValueError(f"Hyperparameters for model '{model_name}' must be a dictionary.")
        loaded[model_name] = _restore_param_types(str(model_name), dict(params))
    return loaded


def merge_hyperparameters(
    saved_hyperparameters: dict[str, dict[str, Any]] | None,
    manual_hyperparameters: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if saved_hyperparameters is not None:
        _validate_hyperparameter_map(saved_hyperparameters)
        merged.update({model_name: dict(params) for model_name, params in saved_hyperparameters.items()})
    if manual_hyperparameters is not None:
        _validate_hyperparameter_map(manual_hyperparameters)
        for model_name, params in manual_hyperparameters.items():
            merged[model_name] = dict(params)
    return merged


def _param_dict_str(params: dict[str, Any], max_len: int = 60) -> str:
    """Format parameter dict concisely for logging."""
    if not params:
        return "{}"
    items = [f"{k}={v!r}" for k, v in sorted(params.items())]
    result = "{" + ", ".join(items) + "}"
    if len(result) > max_len:
        return result[:max_len - 3] + "..."
    return result


def run_validation_once(
    data: pd.DataFrame,
    model_specs: list[ModelSpec] | None = None,
    *,
    target: str = TARGET_VARIABLE,
    horizons: list[int] = HORIZONS,
    train_end: str | pd.Timestamp = BACKTEST_TRAIN_END,
    val_end: str | pd.Timestamp = BACKTEST_VAL_END,
    step_months: int = BACKTEST_STEP_MONTHS,
    alpha: float = 0.1,
    n_bootstrap: int = 0,
    save_nkpc_params: bool = True,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_backtest_inputs(data, target)
    data = data.sort_index()
    train_end_ts = month_end(train_end)
    val_end_ts = month_end(val_end)
    horizons = sorted(horizons)

    if model_specs is None:
        model_specs = default_model_specs(target=target, horizons=horizons)

    rows: list[dict[str, Any]] = []
    best_params_by_model: dict[str, dict[str, Any]] = {}
    validation_fold_id = 1
    nkpc_params_list: list[dict[str, Any]] = []

    log_info("=" * 80)
    log_info("Starting One-Time Validation / Hyperparameter Search")
    log_info(f"Models: {len(model_specs)}, Horizons: {horizons}, Step months: {step_months}")
    log_info(
        f"Window: train_end={train_end_ts.strftime('%Y-%m')}, val_end={val_end_ts.strftime('%Y-%m')}, "
        f"bootstrap={n_bootstrap}"
    )
    log_info("=" * 80)

    for model_spec in model_specs:
        best_score = float("inf")
        best_params: dict[str, Any] | None = None
        best_val_df = pd.DataFrame()
        param_count = len(model_spec.param_grid)

        log_info(f"Evaluating {model_spec.name} ({param_count} param configs)")
        for param_idx, params in enumerate(model_spec.param_grid):
            try:
                score, val_df = _score_model_on_validation(
                    data=data,
                    model_spec=model_spec,
                    params=params,
                    train_end=train_end_ts,
                    val_end=val_end_ts,
                    horizons=horizons,
                    target=target,
                    alpha=alpha,
                    n_bootstrap=n_bootstrap,
                    outer_fold_id=validation_fold_id,
                    step_months=step_months,
                    nkpc_params_list=nkpc_params_list if save_nkpc_params else None,
                )
            except ValueError as e:
                log_info(f"  [{param_idx + 1}/{param_count}] Validation failed: {e}")
                continue

            # Check for invalid scores (NaN or inf)
            if not (0 <= score < float("inf")):
                log_info(f"  [{param_idx + 1}/{param_count}] Invalid RMSE: {score} | Params: {_param_dict_str(params)}")
                continue

            # Log progress: always log first and last, plus checkpoint at 25/50/75 for larger grids
            is_first = param_idx == 0
            is_last = param_idx == param_count - 1
            is_checkpoint = param_count > 4 and (param_idx + 1) in [
                param_count // 4,
                param_count // 2,
                3 * param_count // 4,
            ]
            should_log = is_first or is_last or is_checkpoint
            
            if should_log:
                log_info(f"  [{param_idx + 1}/{param_count}] Val RMSE: {score:.6f} | Params: {_param_dict_str(params)}")
            else:
                log_debug(f"  [{param_idx + 1}/{param_count}] Val RMSE: {score:.6f}")
            
            wandb_log({"model": model_spec.name, "val_rmse": score})
            if score < best_score:
                best_score = score
                best_params = params
                best_val_df = val_df

        if best_params is None:
            raise ValueError(f"No valid parameter configs found for model '{model_spec.name}'.")

        best_params_by_model[model_spec.name] = dict(best_params)
        log_info(f"[OK] {model_spec.name}: Best RMSE={best_score:.6f}")
        log_debug(f"  Best params: {best_params}")
        if not best_val_df.empty:
            rows.extend(best_val_df.to_dict("records"))
    
    # Save NKPC parameters if tracking was enabled
    if save_nkpc_params and nkpc_params_list:
        _save_nkpc_parameters(nkpc_params_list)

    predictions = pd.DataFrame(rows)
    by_horizon, overall = summarize_metrics(predictions)
    return best_params_by_model, predictions, by_horizon, overall


def run_fixed_params_backtest(
    data: pd.DataFrame,
    fixed_hyperparameters: dict[str, dict[str, Any]],
    model_specs: list[ModelSpec] | None = None,
    *,
    target: str = TARGET_VARIABLE,
    horizons: list[int] = HORIZONS,
    train_end: str | pd.Timestamp = BACKTEST_TRAIN_END,
    val_end: str | pd.Timestamp = BACKTEST_VAL_END,
    test_end: str | pd.Timestamp = BACKTEST_TEST_END,
    step_months: int = BACKTEST_STEP_MONTHS,
    retrain_months: int | None = None,
    alpha: float = 0.1,
    n_bootstrap: int = 100,
    save_nkpc_params: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run expanding-window backtest with fixed hyperparameters.
    
    Args:
        data: DataFrame with DatetimeIndex
        fixed_hyperparameters: Pre-tuned hyperparameters for each model
        model_specs: Model specifications (default: all models)
        target: Target variable name
        horizons: Forecast horizons
        train_end: Initial training period end
        val_end: Validation period end
        test_end: Test period end
        step_months: Frequency for test/evaluation (e.g., 1 for monthly)
        retrain_months: Frequency for retraining. If None, retrains at every step_months
                       (default). If set, retrains every retrain_months and reuses
                       cached model between retrain points.
        alpha: Significance level for prediction intervals
        n_bootstrap: Bootstrap samples for confidence intervals
        save_nkpc_params: Whether to save EINN NKPC parameters to CSV
    
    Example:
        # Test monthly, retrain yearly
        run_fixed_params_backtest(data, params, step_months=1, retrain_months=12)
    """
    _ensure_backtest_inputs(data, target)
    _validate_hyperparameter_map(fixed_hyperparameters)

    data = data.sort_index()
    train_end_ts = month_end(train_end)
    val_end_ts = month_end(val_end)
    test_end_ts = month_end(test_end)
    horizons = sorted(horizons)
    
    # If retrain_months not specified, retrain at every test step (original behavior)
    if retrain_months is None:
        retrain_months = step_months

    if model_specs is None:
        model_specs = default_model_specs(target=target, horizons=horizons)

    missing_models = [spec.name for spec in model_specs if spec.name not in fixed_hyperparameters]
    if missing_models:
        raise ValueError(
            f"Missing hyperparameters for models: {missing_models}. "
            "Provide manual values or run validation mode to save them."
        )

    # Build test origins (evaluation frequency)
    test_origins = build_test_origins(
        val_end=val_end_ts,
        test_end=test_end_ts,
        max_horizon=max(horizons),
        step_months=step_months,
    )
    
    # Build retrain origins (training frequency) - subset of test origins
    if retrain_months != step_months:
        retrain_origins = build_test_origins(
            val_end=val_end_ts,
            test_end=test_end_ts,
            max_horizon=max(horizons),
            step_months=retrain_months,
        )
    else:
        retrain_origins = test_origins
    
    outer_folds = [CVFold(fold_id=i + 1, origin=origin) for i, origin in enumerate(test_origins)]
    rows: list[dict[str, Any]] = []
    nkpc_params_list: list[dict[str, Any]] = []
    
    # Cache for trained models to avoid retraining at every test step
    model_cache: dict[tuple[str, pd.Timestamp], Any] = {}

    log_info("=" * 80)
    log_info("Starting Fixed-Parameter Expanding-Window Backtest")
    log_info(f"Config: {len(horizons)} horizons={horizons}, {len(outer_folds)} folds, {len(model_specs)} models")
    log_info(f"Train: {train_end_ts.strftime('%Y-%m')} | Val: {val_end_ts.strftime('%Y-%m')} | Test: {test_end_ts.strftime('%Y-%m')}")
    log_info(f"Target: {target}, Alpha: {alpha}, Bootstrap: {n_bootstrap}")
    if retrain_months != step_months:
        log_info(f"Retrain frequency: every {retrain_months} months | Test frequency: every {step_months} months")
    log_info("=" * 80)

    for outer_fold in outer_folds:
        origin = outer_fold.origin
        _, _, val_end_cur = window_bounds_for_origin(train_end_ts, val_end_ts, origin)
        
        # Determine if we should retrain at this origin
        should_retrain = origin in retrain_origins
        
        log_info(f"\n[Fold {outer_fold.fold_id}/{len(outer_folds)}] Origin: {origin.strftime('%Y-%m')}" +
                 (" [RETRAIN]" if should_retrain else ""))

        for model_spec in model_specs:
            params = _restore_param_types(model_spec.name, fixed_hyperparameters[model_spec.name])
            log_debug(f"  {model_spec.name}: params={params}")

            train_val_df = data.loc[:val_end_cur]
            
            # Find the most recent retrain origin before/at this test origin
            retrain_origin_for_cache = None
            for i, retrain_origin in enumerate(retrain_origins):
                if retrain_origin <= origin:
                    retrain_origin_for_cache = retrain_origin
                else:
                    break
            
            cache_key = (model_spec.name, retrain_origin_for_cache)
            
            # AR1 always retrains because it stores last_value/last_index at fit time
            skip_cache = model_spec.name == "ar1"
            
            # Use cached model or retrain
            if cache_key in model_cache and not should_retrain and not skip_cache:
                # Use cached model and generate forecast
                cached_model = model_cache[cache_key]
                forecast_df = _forecast_from_cached_model(
                    cached_model, train_val_df, model_spec.name, target, horizons, alpha, n_bootstrap
                )
                log_debug(f"  {model_spec.name}: Using cached model from {retrain_origin_for_cache.strftime('%Y-%m')}")
            elif should_retrain or retrain_months != step_months or skip_cache:
                # Retrain and cache the model
                trained_model = _train_model(
                    model_spec, train_val_df, params, target, horizons, alpha, n_bootstrap
                )
                if not skip_cache:
                    model_cache[cache_key] = trained_model
                
                # Generate forecast from trained model
                if save_nkpc_params and model_spec.name == "einn":
                    forecast_df, nkpc_params = _forecast_from_trained_model_with_tracking(
                        trained_model, train_val_df, model_spec.name, target, horizons, alpha, n_bootstrap
                    )
                    nkpc_params["origin"] = origin
                    nkpc_params_list.append(nkpc_params)
                else:
                    forecast_df = _forecast_from_trained_model(
                        trained_model, train_val_df, model_spec.name, target, horizons, alpha, n_bootstrap
                    )
                
                if should_retrain:
                    log_debug(f"  {model_spec.name}: Trained at {retrain_origin_for_cache.strftime('%Y-%m')}")
            else:
                # Fallback (shouldn't happen in normal operation)
                if save_nkpc_params and model_spec.name == "einn":
                    forecast_df = _predict_einn_with_tracking(
                        train_val_df, params, alpha, n_bootstrap, target, horizons,
                        nkpc_params_list, origin
                    )
                else:
                    forecast_df = model_spec.predict_fn(train_val_df, params, alpha, n_bootstrap)
            
            rows.extend(
                _rows_from_forecast(
                    model=model_spec.name,
                    split="test",
                    origin=origin,
                    forecast_df=forecast_df,
                    data=data,
                    target=target,
                    horizons=horizons,
                    params=params,
                    outer_fold_id=outer_fold.fold_id,
                    inner_fold_id=None,
                    cutoff=test_end_ts,
                )
            )

    predictions = pd.DataFrame(rows)
    by_horizon, overall = summarize_metrics(predictions)
    
    # Save NKPC parameters if tracking was enabled
    if save_nkpc_params and nkpc_params_list:
        _save_nkpc_parameters(nkpc_params_list)
    
    return predictions, by_horizon, overall


def run_backtest_from_csv(
    csv_path: str = "data/processed/transformed_fred_data.csv",
    *,
    mode: str = "run",
    hyperparameters_path: str = DEFAULT_HYPERPARAMETERS_PATH,
    manual_hyperparameters: dict[str, dict[str, Any]] | None = None,
    train_end: str | pd.Timestamp = BACKTEST_TRAIN_END,
    val_end: str | pd.Timestamp = BACKTEST_VAL_END,
    test_end: str | pd.Timestamp = BACKTEST_TEST_END,
    step_months: int = BACKTEST_STEP_MONTHS,
    retrain_months: int | None = None,
    alpha: float = 0.1,
    validation_bootstrap: int = 0,
    test_bootstrap: int = 100,
    grid_profile: str = "balanced",
    log_level: str = "INFO",
    models_to_run: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run backtesting pipeline.
    
    Args:
        csv_path: Path to transformed data CSV
        mode: "validate" (tune hyperparameters), "run" (test with tuned params), or "full" (retune per fold)
        hyperparameters_path: Path to save/load hyperparameters JSON
        manual_hyperparameters: Dict of manually-specified hyperparameters (overrides saved)
        train_end, val_end, test_end: Backtest window boundaries
        step_months: Test/evaluation frequency (e.g., 1 for monthly)
        retrain_months: Retraining frequency (e.g., 12 for yearly). If None, retrains at step_months.
                       Only used in "run" mode.
        alpha: Significance level for prediction intervals
        validation_bootstrap: Bootstrap samples during validation
        test_bootstrap: Bootstrap samples during testing
        grid_profile: "full", "balanced", or "fast" hyperparameter grid
        log_level: Logging level
        models_to_run: Subset of models to run (e.g., ["einn", "base_nn"])
    
    Example:
        # Validate hyperparameters
        run_backtest_from_csv(mode="validate")
        
        # Test with monthly evaluation but yearly retraining
        run_backtest_from_csv(mode="run", step_months=1, retrain_months=12)
    """
    if mode not in BACKTEST_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {sorted(BACKTEST_MODES)}.")

    logger = get_logger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    data = pd.read_csv(csv_path, index_col=0, parse_dates=True).dropna()
    param_grids = _resolve_model_param_grids(grid_profile)
    model_specs = default_model_specs(
        target=TARGET_VARIABLE,
        horizons=HORIZONS,
        model_param_grids=param_grids,
    )
    
    # Filter to requested models
    if models_to_run is not None:
        model_specs = [m for m in model_specs if m.name in models_to_run]
        if not model_specs:
            raise ValueError(f"No valid models found. Requested: {models_to_run}")
        log_info(f"Running models: {[m.name for m in model_specs]}")

    if mode == "validate":
        best_params, predictions, by_horizon, overall = run_validation_once(
            data,
            model_specs=model_specs,
            target=TARGET_VARIABLE,
            horizons=HORIZONS,
            train_end=train_end,
            val_end=val_end,
            step_months=step_months,
            alpha=alpha,
            n_bootstrap=validation_bootstrap,
        )
        save_hyperparameters(best_params, path=hyperparameters_path)
        log_info(f"Saved tuned hyperparameters to {hyperparameters_path}")
        return predictions, by_horizon, overall

    if mode == "run":
        saved_hyperparameters = None
        if Path(hyperparameters_path).exists():
            saved_hyperparameters = load_hyperparameters(hyperparameters_path)
        elif manual_hyperparameters is None:
            raise FileNotFoundError(
                f"Hyperparameter file not found at '{hyperparameters_path}'. "
                "Run validation mode first or provide MANUAL_HYPERPARAMETERS."
            )
        merged_hyperparameters = merge_hyperparameters(saved_hyperparameters, manual_hyperparameters)
        return run_fixed_params_backtest(
            data,
            fixed_hyperparameters=merged_hyperparameters,
            model_specs=model_specs,
            target=TARGET_VARIABLE,
            horizons=HORIZONS,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
            step_months=step_months,
            retrain_months=retrain_months,
            alpha=alpha,
            n_bootstrap=test_bootstrap,
        )


def save_backtest_outputs(
    predictions: pd.DataFrame,
    by_horizon: pd.DataFrame,
    overall: pd.DataFrame,
    out_dir: str = "data/processed",
    file_prefix: str = "backtest",
) -> None:
    # For backtest results, save in results subfolder
    if file_prefix == "backtest":
        results_dir = Path(out_dir) / "results"
    else:
        # For validation, keep in main processed folder
        results_dir = Path(out_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert inflation forecast to percentage points (multiply by 100)
    pred_output = predictions.copy()
    value_cols = ["y_true", "y_pred", "lower", "upper"]
    for col in value_cols:
        if col in pred_output.columns:
            pred_output[col] = pred_output[col] * 100
    
    metric_cols = ["rmse", "mae"]
    by_horizon_output = by_horizon.copy()
    overall_output = overall.copy()
    
    for col in metric_cols:
        if col in by_horizon_output.columns:
            by_horizon_output[col] = by_horizon_output[col] * 100
        if col in overall_output.columns:
            overall_output[col] = overall_output[col] * 100
    
    pred_output.to_csv(results_dir / f"{file_prefix}_predictions.csv", index=False)
    by_horizon_output.to_csv(results_dir / f"{file_prefix}_metrics_by_horizon.csv", index=False)
    overall_output.to_csv(results_dir / f"{file_prefix}_metrics_overall.csv", index=False)
    
    # Create RMSE comparison table: models (rows) x horizons (columns) for test split
    test_metrics = by_horizon_output[by_horizon_output["split"] == "test"].copy()
    if not test_metrics.empty and "horizon" in test_metrics.columns and "model" in test_metrics.columns:
        rmse_comparison = test_metrics.pivot_table(
            index="model",
            columns="horizon",
            values="rmse",
            aggfunc="first"
        )
        rmse_comparison.columns.name = None
        rmse_comparison.index.name = "Model"
        rmse_comparison = rmse_comparison.round(4)
        rmse_comparison.to_csv(results_dir / f"{file_prefix}_rmse_by_horizon.csv")
        log_info(f"Saved RMSE comparison table to {results_dir / f'{file_prefix}_rmse_by_horizon.csv'}")

