from dataclasses import dataclass
from typing import Any

# Data specification
@dataclass
class VariableSpec:
    name: str
    candidates: list[str]
    prefer_seasonally_adjusted: bool = True
    interpolation_method: str | None = None
    aggregation_method: str = "last"
    transformation: str | None = None


VARIABLE_SPECS: list[VariableSpec] = [
    VariableSpec("cpi_all_items", ["CPIAUCSL"], True, transformation="log_diff_12"),
    VariableSpec("unemployment_rate", ["UNRATE"], True, aggregation_method="mean"),
    VariableSpec("natural_rate_unemployment", ["NROU"], False, interpolation_method="linear"),
    VariableSpec("inflation_expectations_umich", ["MICH"], False, aggregation_method="mean"),
    VariableSpec("industrial_production", ["INDPRO"], True),
    VariableSpec("wti_oil_price", ["WTISPLC"], False),
    VariableSpec("producer_price_index", ["PPIACO"], True),
    VariableSpec("fed_funds_rate", ["FEDFUNDS"], False, aggregation_method="mean"),
    VariableSpec("financial_conditions", ["NFCI", "ANFCI"], False),
    VariableSpec("payroll_employment", ["PAYEMS"], True),
    VariableSpec("real_personal_income_less_transfers", ["W875RX1"], True),
    VariableSpec("housing_starts", ["HOUST"], True),
    VariableSpec("retail_sales", ["RSAFS"], True),
    VariableSpec("capacity_utilization", ["TCU"], True, aggregation_method="mean"),
    VariableSpec("real_disposable_personal_income", ["DSPIC96"], True),
    VariableSpec("money_stock_m2", ["M2SL"], False),
    VariableSpec("treasury_10y", ["GS10"], False, aggregation_method="mean"),
    VariableSpec("treasury_3m", ["TB3MS"], False, aggregation_method="mean"),
]

# forecast specification
TARGET_VARIABLE = "cpi_all_items"
EXPECTATIONS_VARIABLE = "inflation_expectations_umich"
GAP_VARIABLES = ["unemployment_rate", "natural_rate_unemployment"]
HORIZONS = [1, 3, 6, 12]

# recursive expanding-window backtest configuration
BACKTEST_TRAIN_END = "2016-01"
BACKTEST_VAL_END = "2018-01"
BACKTEST_TEST_END = "2026-01"
BACKTEST_STEP_MONTHS = 1  # Frequency for test/evaluation (e.g., 1 for monthly)
BACKTEST_RETRAIN_MONTHS = 1  # Frequency for retraining (e.g., 12 for yearly). None means retrain at every step.

# prediction interval significance level (alpha=0.1 → 90% confidence, alpha=0.05 → 95% confidence)
ALPHA = 0.05

# BIC lag selection for VAR
VAR_MAX_BIC_LAGS = 24

# model search spaces for backtesting
MODEL_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "einn": {
        "lags": [24],
        "dmid": [(32, 32)],
        "learning_rate": [0.01, 0.005],
        "epochs": [1000],
        "pc_weight": [0.05, 0.5],
        "horizon_weight": [0.5, 1],
    },
    "base_nn": {
        "lags": [24],
        "dmid": [(32, 32)],
        "learning_rate": [0.01, 0.005],
        "epochs": [1000],
    },
    "lasso": {
        "lags": [24],
        "lmbda": [0.05, 0.1],
        "max_iter": [1200],
        "tol": [1e-6],
    },
    "ar1": {},
    "var": {"use_bic": [True]},
}

# main pipeline control settings
PIPELINE_MODE = "run"  # "validate", "run", or "full"
PIPELINE_GRID_PROFILE = "balanced"  # "full", "balanced", or "fast"

# model selection: which models to include in backtesting
# options: "einn", "base_nn", "lasso", "ar1", "var"
MODELS_TO_RUN = ["einn", "base_nn", "lasso", "ar1"]  # Exclude "var" due to instability
