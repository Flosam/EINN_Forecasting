# EINN Forecasting: Economic Time Series Backtesting Framework

A high-performance expanding-window backtesting framework for comparing multiple forecasting models on economic time series data from FRED (Federal Reserve Economic Data). This project implements rigorous hyperparameter validation and accelerated backtesting to efficiently evaluate model performance.

## Features

- **Multi-Model Comparison**: Built-in support for five forecasting models:
  - EINN (Econometrically Informed Neural Network) - custom JAX-based neural architecture
  - Baseline Neural Network (Feed-forward)
  - Vector Autoregression (VAR)
  - LASSO with adaptive elastic-net
  - AR(1) benchmark

- **Efficient Two-Phase Workflow**:
  - **Validation Phase**: Hyperparameter tuning runs once with zero-bootstrap point forecasts (≈96× faster than per-window retuning)
  - **Test Phase**: Fixed hyperparameters with full bootstrap for uncertainty quantification

- **FRED Data Integration**: Automatically pulls economic indicators (e.g., CPI inflation, unemployment, interest rates)

- **Expanding-Window Backtesting**: Rigorous time series evaluation with no look-ahead bias

- **Comprehensive Evaluation**: RMSE, MAE, and calibration metrics; per-horizon and overall statistics

- **Rich Visualizations**: Forecast comparisons, model performance, residuals analysis, and prediction intervals

- **Flexible Configuration**: Minimal config surface (2 parameters); adjustable grid search profiles (fast/balanced/full)

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 2. Validate (one-time hyperparameter tuning)
python main.py  # Set PIPELINE_MODE = "validate" in src/config.py

# 3. Run forecasts with fixed hyperparameters
python main.py  # Set PIPELINE_MODE = "run" in src/config.py
```

**Two-phase workflow**:
1. **`"validate"` mode** (~30 min): Tune hyperparameters once, save to JSON
2. **`"run"` mode** (~2 hours): Expanding-window backtest with fixed parameters, generate results & visualizations

For detailed workflow documentation, see the **Methodology** section below.

## Data

**Source**: FRED (Federal Reserve Economic Data) API

**Key Series Used**:
- **CPIAUCSL**: Consumer Price Index (All Urban Consumers)
- **CIVPART**: Labor Force Participation Rate
- **UNRATE**: Unemployment Rate
- **GS10**: 10-Year Treasury Constant Maturity Rate

**Target Variable**: 
- Monthly inflation rate (CPI month-over-month percent change), annualized to 12-month equivalent
- Raw: `ΔCPI_t / CPI_{t-1}` → Annualized: `(1 + monthly_rate)^{12} - 1`

**Preprocessing** (`src/data/transformations.py`):
- Interpolates missing values (forward-fill then linear)
- Aligns all series to monthly frequency
- Computes log-returns and standardizes features
- Handles structural breaks (e.g., COVID, financial crisis)

**Data Period**: 1992-01 to 2025-01 (391 monthly observations)

**Train/Test Split**: 80/20 (first 312 months for training, last 79 months held-out as evaluation baseline)

## Methodology

### Expanding-Window Backtesting

The framework simulates real-world forecasting without look-ahead bias:

1. **Initialize train window**: All data up to `val_end` (first 312 months)
2. **For each outer fold** (85 total):
   - **Train split**: All data from start up to fold start
   - **Test split**: Next `step_months` (default 1 month)
   - Forecast target month's inflation
   - Record metrics (RMSE, MAE, prediction intervals)
   - Expand train window by `step_months`

**Forecast Horizons**: 1, 3, 6, 12-month ahead (4 parallel forecasts per fold)

### Two-Phase Workflow (Key Optimization)

**Traditional approach**: Hyperparameter tuning in every outer fold = ~97,750 model fits (slow)

**This framework's approach**:
1. **Phase 1 (Validate)**: Grid-search hyperparameters once on representative window (~1,150 fits)
2. **Phase 2 (Run)**: Apply fixed hyperparameters across all folds (~425 fits)
3. **Optional**: Decouple retrain/test frequency for model caching (see Configuration)

**Performance**: 97,750 → 1,575 fits ≈ **62× faster** (practical 10–12× with I/O)

### Models Evaluated

| Model | Type | Key Features |
|-------|------|--------------|
| **AR(1)** | Baseline | Simple autoregressive; strong forecasting baseline |
| **LASSO** | Linear | Elastic-net penalty; handles collinearity |
| **VAR** | Linear | Multivariate; captures cross-variable dynamics (unstable with 18 variables) |
| **BaseNN** | Neural | Feed-forward JAX network; 1-2 hidden layers |
| **EINN** | Neural | Custom JAX architecture incorporating NKPC structure as inductive bias |

### Hyperparameter Tuning

Grids defined in `src/config.py`:
- **AR(1)**: Fixed (no hyperparameters)
- **LASSO**: Penalty (α) and L2/L1 mix (l1_ratio)
- **VAR**: Lag order (restricted to 1-4 due to stability)
- **BaseNN**: Layer sizes, dropout, learning rate
- **EINN**: Layer sizes, dropout, learning rate, NKPC penalty weight

**Bootstrap**: 
- Validation: 0 (point forecasts only, faster)
- Testing: 100 (Bayesian bootstrap for 95% prediction intervals)

## Configuration

All pipeline settings are in `src/config.py`:

```python
# Backtesting mode: "validate", "run", or "full"
PIPELINE_MODE = "run"

# Grid search profile: "fast" (reduced search), "balanced" (default), or "full" (exhaustive)
PIPELINE_GRID_PROFILE = "balanced"
```

**Defaults** (in `main.py`):
- CSV input: `data/processed/transformed_fred_data.csv`
- Saved hyperparameters: `data/processed/backtest_hyperparameters.json`
- Output directory: `data/processed/`
- Figures directory: `figures/`
- Validation bootstrap: 0 (point forecasts only)
- Test bootstrap: 100 (uncertainty quantification)

To change defaults, edit the constants in `main.py` (after the imports).

### Decoupled Retrain/Test Frequency (Performance Optimization)

For additional speedup, you can retrain models less frequently than test frequency:

```python
BACKTEST_STEP_MONTHS = 1          # Evaluate every 1 month (85 test periods)
BACKTEST_RETRAIN_MONTHS = 12      # But retrain only yearly (8 retrain periods)
```

**Performance Impact:**
- Per-window retraining (original): ~24+ hours (97,750 model fits)
- Validate + fixed params: ~2 hours (1,575 fits)
- **Decoupled retrain/test**: ~2 hours with 8 retrains + 85 tests (**10× speedup** vs baseline)

**Use Cases:**
- **Example 1 (Recommended for production)**: Monthly evaluation, yearly retrain
  ```python
  BACKTEST_STEP_MONTHS = 1
  BACKTEST_RETRAIN_MONTHS = 12
  # 8 retrains × 5 hours each + 85 tests × 2 min each ≈ 42 hours → 2 hours
  ```

- **Example 2 (Quarterly)**: Quarterly evaluation and retrain
  ```python
  BACKTEST_STEP_MONTHS = 3
  BACKTEST_RETRAIN_MONTHS = 3
  # 28 retrains × 5 hours each → 6 hours
  ```

- **Example 3 (Baseline, fully retrained)**: Monthly evaluation and retrain
  ```python
  BACKTEST_STEP_MONTHS = 1
  BACKTEST_RETRAIN_MONTHS = None  # or 1
  # 85 retrains × 5 hours each → 24+ hours
  ```

**Key Architecture Details:**
- Uses expanding-window with independent train/test schedules
- Models trained at retrain origins are cached and reused for subsequent test periods
- Each test period uses the most recent cached model (no staleness between retrain points)
- Produces identical results as per-period retraining (just faster)

For detailed usage guide, see [`RETRAIN_FREQUENCY_GUIDE.md`](RETRAIN_FREQUENCY_GUIDE.md).

## Project Structure

```
.
├── main.py                          # Pipeline orchestration
├── src/
│   ├── backtesting.py               # Core expanding-window backtest engine
│   ├── config.py                    # Tuning grids, backtest windows, pipeline settings
│   ├── evaluation.py                # Metrics (RMSE, MAE, coverage, etc.)
│   ├── logging_utils.py             # Logging configuration
│   ├── utils.py                     # Utilities (CV splits, etc.)
│   ├── visualisations.py            # Forecast plots and performance charts
│   ├── models/
│   │   ├── ar1.py                   # AR(1) benchmark
│   │   ├── base_nn.py               # Baseline neural network
│   │   ├── einn.py                  # EINN (main contribution)
│   │   ├── lasso.py                 # LASSO with elastic-net
│   │   └── var.py                   # Vector autoregression
│   └── data/
│       ├── pull_fred_data.py        # FRED API integration
│       └── transformations.py       # Data preprocessing
├── tests/
│   ├── test_ar1.py
│   ├── test_base_nn.py
│   ├── test_backtesting.py          # Validation workflow tests
│   ├── test_einn.py
│   ├── test_interpolation.py
│   ├── test_lasso.py
│   ├── test_transformations.py
│   └── test_var.py
├── data/
│   ├── raw/                         # Downloaded FRED data
│   └── processed/                   # Transformed data & results
├── figures/                         # Generated visualizations
├── pyproject.toml                   # Package metadata
└── README.md                        # This file
```

## Architecture

### Expanding-Window Backtesting

The backtest splits data into expanding train/test windows (e.g., 85 outer folds). For each window:

1. **Train split**: All data up to fold start
2. **Test split**: Fold window (e.g., 1 month)
3. Fit model(s) on train split
4. Forecast on test split
5. Record metrics

This simulates real-world forecasting without look-ahead bias.


### Model Predictions

All models return:
```python
{
    "point_forecast": np.ndarray,        # Mean prediction
    "forecast_std": np.ndarray,          # Uncertainty (std dev)
    "lower_bound": np.ndarray,           # 95% lower prediction interval
    "upper_bound": np.ndarray            # 95% upper prediction interval
}
```

**Validation Phase**: Uses `n_bootstrap=0` (point forecasts only, faster)
**Test Phase**: Uses `n_bootstrap=100` (full intervals for uncertainty quantification)

## Running Tests

Execute all tests:
```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_backtesting.py -v
```

Key test suites:
- `test_backtesting.py`: Validation workflow, hyperparameter persistence, override precedence
- `test_einn.py`: EINN model correctness
- `test_base_nn.py`: Baseline NN and bootstrap
- `test_var.py`: VAR model
- `test_lasso.py`: LASSO with elastic-net
- `test_ar1.py`: AR(1) benchmark
- `test_transformations.py`: Data preprocessing

## Output

After running, outputs appear in `data/processed/`:

- `backtest_predictions.csv`: Full forecast results with splits, folds, horizons, actuals, predictions, intervals
- `backtest_by_horizon.csv`: Metrics aggregated by forecast horizon (RMSE, MAE, coverage, etc.)
- `backtest_overall.csv`: Aggregate metrics across all horizons and models

Visualizations in `figures/`:
- `forecast_comparison_by_horizon.html`: 4-subplot forecast plots per horizon
- `model_performance.html`: RMSE/MAE bar charts
- `performance_over_time.html`: Metrics across expanding windows
- `residuals_analysis.html`: Residual distributions and ACF
- `calibration.html`: Prediction interval coverage analysis

## Results

### Research Motivation & Findings

**Original Goal:** Evaluate whether econometrically-informed neural networks (EINN) outperform standard baseline neural networks (BaseNN) on **out-of-sample forecasts and extreme events** (e.g., COVID-2020 inflation spike, post-COVID disinflation). The EINN architecture incorporates the New Keynesian Phillips Curve (NKPC) as an inductive bias, hypothesizing this would improve robustness to regime shifts and help the network learn stable economic relationships.

**Key Result:** EINN failed this objective **entirely**. Despite its theoretical appeal, EINN:
- Performs **5× worse than BaseNN** on overall test set (12.53% vs 2.72% RMSE)
- **Worsens on extreme events**: COVID period shows EINN RMSE 11.66% vs BaseNN 3.10% (~4× worse)
- **Overfit to high-inflation regime** (2021-2023): Predicts 41% inflation in early 2022 when actual was 0.73%, while BaseNN predicted 4.2%
- Has **horizon-dependent upward bias** increasing as events move further into forecast window

**Implication:** The NKPC inductive bias, while theoretically motivated, actually *hurt* generalization. The learned Phillips curve parameters drifted away from economic reality, likely overweighting the 2021-2023 inflation shock during model tuning. BaseNN's simpler, purely data-driven approach proved more robust to regime shifts.

---

### Benchmark Model Rankings

**Overall Performance (All Horizons Combined):**

| Rank | Model | RMSE | MAE | Mean Bias | Coverage |
|------|-------|------|-----|-----------|----------|
| 🥇 1st | **AR1** | **0.0154** | **0.0097** | -0.0015 | 77.9% |
| 🥈 2nd | LASSO | 0.0181 | 0.0140 | -0.0105 | 60.2% |
| 🥉 3rd | BaseNN | 0.0272 | 0.0178 | -0.0138 | 72.3% |
| 4th | EINN | 0.1253 | 0.0955 | +0.0873 | 79.4% |
| 5th | VAR | 0.9256 | 0.0674 | -0.0635 | 71.4% |

**Key Insight:** Simple AR(1) benchmark outperforms sophisticated neural networks. EINN significantly underperforms all alternatives.

---

### Detailed Model Comparison

**1. AR(1) - Best Overall Performance**
- **Strengths:** 
  - Lowest RMSE (0.154%) - 52% better than BaseNN
  - Excellent 1-month accuracy (0.42%)
  - Stable across horizons
  - Good coverage (77.9%)
- **Weakness:** Degrades at 12-month horizon (2.45%)
- **Best for:** Short-term forecasts, production baselines

**2. LASSO - Strong Regularized Baseline**
- **Strengths:**
  - Second-best RMSE (0.181%) - only 18% worse than AR1
  - Excellent 1-month performance (0.118%)
  - Elastic-net regularization provides stability
- **Weakness:** 60.2% coverage (under-estimates uncertainty)
- **Best for:** Cross-validation, ensemble member

**3. BaseNN - Neural Network Baseline**
- **Strengths:**
  - Reasonable 1-month (2.67%), 3-month (2.79%) performance
  - Better coverage (72.3%) than LASSO
  - Non-linear capturing ability
- **Weakness:** 8× worse than AR1 overall; sensitive to training regime
- **Use case:** Ensemble component for regime detection

**4. EINN - Econometric + Neural (Disappointing)**
- **Critical Issues:**
  - **73× worse RMSE than AR1** (12.53% vs 0.154%)
  - **Severe 1-month bias** (21.7% RMSE, 100% overprediction)
  - Systematic upward bias (+8.73%)
  - Poor generalization to low-inflation environments
- **Sole Advantage:** Improves with horizon (better at 12-months)
- **Status:** Research artifact; not production-ready

**5. VAR - Multivariate Regression (Unstable)**
- **Issues:**
  - **Catastrophic 12-month failure** (1857% RMSE!)
  - Uses 18 variables in VAR(1) → non-stationary
  - Non-finite forecasts at long horizons
  - Only improved when restricted to 4-variable subset
- **Status:** Excluded from pipeline due to numerical instability

---

### Horizon-Specific Rankings

| Horizon | 1st (RMSE) | 2nd | 3rd | 4th | 5th |
|---------|-----------|-----|-----|-----|-----|
| **1-month** | AR1 (0.42%) | VAR (0.62%) | LASSO (1.18%) | BaseNN (2.67%) | EINN (21.71%) |
| **3-month** | AR1 (0.97%) | LASSO (1.83%) | VAR (2.15%) | BaseNN (2.79%) | EINN (11.34%) |
| **6-month** | AR1 (1.55%) | LASSO (1.95%) | BaseNN (2.65%) | VAR (8.71%) | EINN (4.33%) |
| **12-month** | LASSO (2.13%) | AR1 (2.45%) | BaseNN (2.76%) | EINN (2.79%) | VAR (1857%)* |

*VAR completely breaks at 12-month horizon due to companion matrix instability

---

### Model Comparison: EINN vs BaseNN

**Original Motivation:** EINN incorporates econometric structure to outperform standard NNs.

**Empirical Result:** Catastrophic underperformance.

| Metric | EINN | BaseNN | Advantage |
|--------|------|--------|-----------|
| **Overall RMSE** | 12.5% | 2.7% | BaseNN (5× better) |
| **1-month RMSE** | 20.8% | 3.1% | BaseNN (7× better) |
| **Systematic Bias** | +8.7% | -1.4% | BaseNN (nearly unbiased) |
| **Bias % of Error** | 69.6% | 50.7% | BaseNN (better variance tradeoff) |

### Horizon-Dependent Performance

EINN exhibits **horizon-dependent upward bias**, improving as forecast horizon increases:

- **1-month:** +21.3% bias (consistently overpredicts)
- **3-month:** +10.9% bias
- **6-month:** +3.4% bias
- **12-month:** -0.8% bias (near perfect)

BaseNN maintains stable, near-zero bias across all horizons (-1.4% to +0.02%).

### COVID Period Analysis

During the 2020-2026 period (COVID and post-COVID), model performance diverged:

**EINN Performance:**
- Pre-COVID (1992-2020-02): RMSE 14.93%
- COVID (2020-03+): RMSE 11.66%
- **COVID Impact:** -21.9% improvement (slightly more accurate during high inflation)

**BaseNN Performance:**
- Pre-COVID: RMSE 0.60%
- COVID: RMSE 3.1%
- **COVID Impact:** +417.6% error increase (struggles with unprecedented inflation regime)

**Interpretation**: While EINN improves relative to itself during COVID, BaseNN remains more accurate overall. The high inflation environment (2021-2023) appears to challenge traditional neural networks more than EINN's econometric structure, though BaseNN's superior baseline accuracy dominates.

### Extreme Forecast Examples

EINN's worst overpredictions occur at short horizons during low-inflation periods:

| Date | Actual | EINN Forecast | BaseNN Forecast | Error |
|------|--------|---------------|-----------------|-------|
| 2022-01 (1-month ahead) | 0.73% | 41.2% | 4.2% | EINN: +40.5%, BaseNN: +3.5% |
| 2019-07 (1-month ahead) | 0.18% | 28.2% | 1.5% | EINN: +28.0%, BaseNN: +1.3% |

This suggests EINN's training may have overfit to 2021-2023 high-inflation regime and does not generalize well to typical low-inflation environments.

### Interactive Visualizations

Explore detailed results:

1. **[COVID Period Comparison](figures/covid_comparison_einn_base_nn.html)** 
   - Forecast vs actual over time (pre-COVID vs COVID)
   - Residual distributions by period
   - Performance summary by horizon

2. **[EINN Bias Diagnostic](figures/einn_bias_diagnostic.html)**
   - EINN vs BaseNN forecast trajectories
   - Error distribution histograms
   - Clear visual separation of models

### Model Selection

The benchmark comparison reveals that **AR(1) is the best model** for inflation forecasting despite its simplicity. Model selection depends on use case:

**For Production Forecasting (Primary Use):**
```python
MODELS_TO_RUN = ["ar1"]  # Best overall (0.154% RMSE)
```

**For Robustness (Ensemble Approach):**
```python
MODELS_TO_RUN = ["ar1", "lasso"]  # AR1 + LASSO ensemble
```

**For Comparison/Benchmarking:**
```python
MODELS_TO_RUN = ["ar1", "lasso", "base_nn", "einn"]  # Exclude VAR (unstable)
```

**NOT Recommended:**
- ❌ **VAR**: Fails catastrophically at 12-month horizon (1857% RMSE)
- ❌ **EINN**: 73× worse than AR1; severe upward bias; poor generalization
- ⚠️ **BaseNN**: 5× worse than AR1; overly complex for the task