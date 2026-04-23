#!/usr/bin/env python
"""Tests for src/evaluation.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import mae, rmse, summarize_metrics, validation_score


def test_rmse_and_mae_match_manual_calculation():
    y_true = pd.Series([1.0, 2.0, 4.0, 5.0])
    y_pred = pd.Series([0.0, 2.0, 3.0, 6.0])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(((y_true - y_pred) ** 2).mean()))
    assert np.isclose(mae(y_true, y_pred), (y_true - y_pred).abs().mean())


def test_validation_score_averages_per_horizon_rmse():
    rows = pd.DataFrame(
        {
            "horizon": [1, 1, 3, 3],
            "y_true": [2.0, 4.0, 2.0, 6.0],
            "y_pred": [1.0, 5.0, 1.0, 7.0],
        }
    )
    h1 = np.sqrt((((rows.iloc[:2]["y_true"] - rows.iloc[:2]["y_pred"]) ** 2).mean()))
    h3 = np.sqrt((((rows.iloc[2:]["y_true"] - rows.iloc[2:]["y_pred"]) ** 2).mean()))
    assert np.isclose(validation_score(rows, [1, 3]), (h1 + h3) / 2.0)


def test_summarize_metrics_returns_horizon_and_overall_tables():
    preds = pd.DataFrame(
        {
            "split": ["test", "test", "test", "validation"],
            "model": ["m1", "m1", "m1", "m1"],
            "horizon": [1, 1, 3, 1],
            "y_true": [2.0, 4.0, 6.0, 3.0],
            "y_pred": [1.0, 5.0, 7.0, 2.0],
        }
    )
    by_horizon, overall = summarize_metrics(preds)
    assert set(by_horizon.columns) == {"split", "model", "horizon", "rmse", "mae", "n"}
    assert set(overall.columns) == {"split", "model", "rmse", "mae", "n"}
    assert not by_horizon.empty
    assert not overall.empty


if __name__ == "__main__":
    test_rmse_and_mae_match_manual_calculation()
    test_validation_score_averages_per_horizon_rmse()
    test_summarize_metrics_returns_horizon_and_overall_tables()
    print("\n✅ All evaluation tests passed!")
