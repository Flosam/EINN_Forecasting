from zipp import Path

from src.backtesting import run_backtest_from_csv, save_backtest_outputs
from src.config import (
    ALPHA,
    BACKTEST_RETRAIN_MONTHS,
    MODELS_TO_RUN,
    PIPELINE_GRID_PROFILE,
    PIPELINE_MODE,
)
from src.data.pull_fred_data import load_fred_data
from src.data.transformations import transform_data
from src.visualisations import save_all_visualizations

# defaults
CSV_PATH = "data/processed/transformed_fred_data.csv"
HYPERPARAMETERS_PATH = "data/processed/backtest_hyperparameters.json"
OUTPUT_DIR = "data/processed"
RESULTS_DIR = "data/processed/results"
FIGURES_DIR = "figures"
TARGET_NAME = "CPI Inflation"
VALIDATION_BOOTSTRAP = 0
TEST_BOOTSTRAP = 100
LOG_LEVEL = "INFO"


def main() -> None:
    # Always pull and transform
    print("Pulling FRED data...")
    if not Path(CSV_PATH).exists():
        load_fred_data()

    print("Transforming data...")
    transform_data()

    print(f"Running backtesting mode: {PIPELINE_MODE}")
    print(f"Models: {', '.join(MODELS_TO_RUN)}")
    if BACKTEST_RETRAIN_MONTHS is not None:
        print(f"Retraining frequency: every {BACKTEST_RETRAIN_MONTHS} months")
    
    predictions, by_horizon, overall = run_backtest_from_csv(
        csv_path=CSV_PATH,
        mode=PIPELINE_MODE,
        hyperparameters_path=HYPERPARAMETERS_PATH,
        manual_hyperparameters=None,
        validation_bootstrap=VALIDATION_BOOTSTRAP,
        test_bootstrap=TEST_BOOTSTRAP,
        grid_profile=PIPELINE_GRID_PROFILE,
        log_level=LOG_LEVEL,
        alpha=ALPHA,
        models_to_run=MODELS_TO_RUN,
        retrain_months=BACKTEST_RETRAIN_MONTHS,
    )

    output_prefix = "validation" if PIPELINE_MODE == "validate" else "backtest"
    save_backtest_outputs(
        predictions,
        by_horizon,
        overall,
        out_dir=OUTPUT_DIR,
        file_prefix=output_prefix,
    )
    
    # Determine actual output directory
    if output_prefix == "backtest":
        actual_output_dir = RESULTS_DIR
    elif output_prefix == "validation":
        actual_output_dir = f"{RESULTS_DIR}/validation"
    else:
        actual_output_dir = OUTPUT_DIR
    
    print(
        f"Saved outputs to {actual_output_dir} with prefix '{output_prefix}': {len(predictions)} predictions"
    )

    should_visualize = (
        PIPELINE_MODE in {"run", "full"}
        and not predictions.empty
        and "split" in predictions.columns
        and (predictions["split"] == "test").any()
    )
    if should_visualize:
        print("Generating visualizations...")
        save_all_visualizations(
            predictions,
            metrics_by_horizon=by_horizon,
            metrics_overall=overall,
            output_dir=FIGURES_DIR,
            target_name=TARGET_NAME,
        )

    print("Complete!")


if __name__ == "__main__":
    main()
