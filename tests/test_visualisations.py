#!/usr/bin/env python
"""Test visualizations module with sample backtest data."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualisations import (
    plot_calibration,
    plot_forecasts_by_horizon,
    plot_model_performance,
    plot_performance_over_time,
    plot_residuals_analysis,
    save_all_visualizations,
)


def create_sample_predictions() -> pd.DataFrame:
    """Create sample predictions data for testing."""
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=100, freq="MS")
    models = ["einn", "base_nn", "lasso", "ar1", "var"]
    horizons = [1, 3, 6, 12]

    rows = []
    fold_id = 1
    for origin_idx, origin in enumerate(dates[:-12]):
        if origin_idx % 10 == 0:
            fold_id = origin_idx // 10 + 1

        for h in horizons:
            target_date = origin + pd.DateOffset(months=h)
            if target_date not in dates:
                continue

            true_value = 100 + np.sin(origin_idx * 0.1) * 10 + np.random.normal(0, 1)

            for model in models:
                pred = true_value + np.random.normal(0, 2)
                lower = pred - 2.5
                upper = pred + 2.5

                rows.append(
                    {
                        "model": model,
                        "split": "test",
                        "origin_date": origin,
                        "target_date": target_date,
                        "horizon": h,
                        "y_true": float(true_value),
                        "y_pred": float(pred),
                        "lower": float(lower),
                        "upper": float(upper),
                        "outer_fold": fold_id,
                        "inner_fold": None,
                        "params": "{}",
                    }
                )

    return pd.DataFrame(rows)


def create_sample_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample metrics data for testing."""
    models = ["einn", "base_nn", "lasso", "ar1", "var"]
    horizons = [1, 3, 6, 12]
    splits = ["validation", "test"]

    rows_by_horizon = []
    for split in splits:
        for horizon in horizons:
            for model in models:
                rows_by_horizon.append(
                    {
                        "split": split,
                        "model": model,
                        "horizon": horizon,
                        "rmse": 1.5 + horizon * 0.1 + abs(hash(model) % 10) * 0.1,
                        "mae": 1.0 + horizon * 0.1 + abs(hash(model) % 10) * 0.1,
                        "n": 20,
                    }
                )

    by_horizon = pd.DataFrame(rows_by_horizon)

    rows_overall = []
    for split in splits:
        for model in models:
            rows_overall.append(
                {
                    "split": split,
                    "model": model,
                    "rmse": 2.0 + abs(hash(model) % 10) * 0.1,
                    "mae": 1.5 + abs(hash(model) % 10) * 0.1,
                    "n": 80,
                }
            )

    overall = pd.DataFrame(rows_overall)

    return by_horizon, overall


def test_forecast_plot() -> None:
    print("\n[Test 1] Forecast comparison plot...")
    predictions = create_sample_predictions()
    try:
        fig = plot_forecasts_by_horizon(predictions, save=False)
        assert fig is not None
        print("  ✓ Forecast plot created successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def test_performance_plot() -> None:
    print("\n[Test 2] Model performance plot...")
    _, by_horizon, _ = create_sample_metrics()[0], *create_sample_metrics()
    try:
        fig = plot_model_performance(by_horizon, save=False)
        assert fig is not None
        print("  ✓ Performance plot created successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def test_time_evolution_plot() -> None:
    print("\n[Test 3] Performance over time plot...")
    predictions = create_sample_predictions()
    try:
        fig = plot_performance_over_time(predictions, save=False)
        assert fig is not None
        print("  ✓ Time evolution plot created successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def test_residuals_plot() -> None:
    print("\n[Test 4] Residuals analysis plot...")
    predictions = create_sample_predictions()
    try:
        fig = plot_residuals_analysis(predictions, save=False)
        assert fig is not None
        print("  ✓ Residuals plot created successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def test_calibration_plot() -> None:
    print("\n[Test 5] Calibration analysis plot...")
    predictions = create_sample_predictions()
    try:
        fig = plot_calibration(predictions, save=False)
        assert fig is not None
        print("  ✓ Calibration plot created successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def test_all_visualizations() -> None:
    print("\n[Test 6] Save all visualizations...")
    predictions = create_sample_predictions()
    by_horizon, overall = create_sample_metrics()
    output_dir = "figures_test"

    try:
        results = save_all_visualizations(
            predictions,
            metrics_by_horizon=by_horizon,
            metrics_overall=overall,
            output_dir=output_dir,
        )
        assert results is not None
        assert all(status == "✓" for status in results.values())
        print(f"  ✓ All visualizations saved to {output_dir}/")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def main() -> None:
    print("=" * 60)
    print("Testing Visualizations Module")
    print("=" * 60)

    try:
        test_forecast_plot()
        test_performance_plot()
        test_time_evolution_plot()
        test_residuals_plot()
        test_calibration_plot()
        test_all_visualizations()

        print("\n" + "=" * 60)
        print("All visualization tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Visualization tests failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
