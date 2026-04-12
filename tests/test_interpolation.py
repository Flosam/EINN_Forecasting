#!/usr/bin/env python
"""Test interpolation functionality."""

import pandas as pd
import sys
from pathlib import Path

# Add src directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import VariableSpec
from pull_fred_data import _apply_interpolation, _series_to_monthly


def test_variablespec_fields():
    """Test that VariableSpec has new fields."""
    spec = VariableSpec(
        name="test_var",
        candidates=["TEST"],
        interpolation_method="linear",
        aggregation_method="mean",
    )
    assert spec.interpolation_method == "linear"
    assert spec.aggregation_method == "mean"
    print("✓ VariableSpec fields work correctly")


def test_variablespec_defaults():
    """Test that VariableSpec defaults work."""
    spec = VariableSpec(name="test_var", candidates=["TEST"])
    assert spec.interpolation_method is None
    assert spec.aggregation_method == "last"
    print("✓ VariableSpec defaults are correct")


def test_apply_interpolation_none():
    """Test that None interpolation returns series unchanged."""
    dates = pd.date_range("2020-01-01", periods=5, freq="QE")
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    series = pd.Series(values, index=dates)
    
    result = _apply_interpolation(series, None)
    assert (result == series).all()
    print("✓ None interpolation returns series unchanged")


def test_apply_interpolation_linear():
    """Test linear interpolation."""
    dates = pd.date_range("2020-01-01", periods=5, freq="QE")
    values = [10.0, float("nan"), float("nan"), 40.0, 50.0]
    series = pd.Series(values, index=dates)
    
    result = _apply_interpolation(series, "linear")
    assert not result.isna().any()
    assert result.iloc[0] == 10.0
    assert result.iloc[1] == 20.0
    assert result.iloc[2] == 30.0
    assert result.iloc[3] == 40.0
    assert result.iloc[-1] == 50.0
    print("✓ Linear interpolation works correctly")


def test_apply_interpolation_forward_fill():
    """Test forward fill interpolation."""
    dates = pd.date_range("2020-01-01", periods=5, freq="QE")
    values = [10.0, float("nan"), float("nan"), 40.0, 50.0]
    series = pd.Series(values, index=dates)
    
    result = _apply_interpolation(series, "forward_fill")
    assert result.iloc[0] == 10.0
    assert result.iloc[1] == 10.0
    assert result.iloc[2] == 10.0
    assert result.iloc[3] == 40.0
    assert result.iloc[-1] == 50.0
    print("✓ Forward fill interpolation works correctly")


def test_series_to_monthly_no_interpolation():
    """Test that monthly data without interpolation works."""
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    values = list(range(12))
    series = pd.Series(values, index=dates)
    
    result = _series_to_monthly(series, source_freq="M")
    assert len(result) == 12
    assert result.index.name == "date"
    print("✓ Monthly data without interpolation works")


def test_series_to_monthly_quarterly_linear():
    """Test quarterly data with linear interpolation."""
    dates = pd.date_range("2020-01-01", periods=4, freq="QE")
    values = [100.0, 200.0, 300.0, 400.0]
    series = pd.Series(values, index=dates)
    
    result = _series_to_monthly(series, source_freq="Q", interpolation_method="linear")
    # Quarterly data interpolated to monthly should be dense monthly values.
    assert len(result) > len(series)
    assert not result.isna().any()
    print("✓ Quarterly data with linear interpolation works")


def test_series_to_monthly_quarterly_linear_quarter_start_dates():
    """Quarter-start dated quarterly series should still interpolate to monthly."""
    dates = pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"])
    values = [100.0, 200.0, 300.0, 400.0]
    series = pd.Series(values, index=dates)

    result = _series_to_monthly(series, source_freq="Q", interpolation_method="linear")
    assert len(result) > len(series)
    assert not result.isna().all()
    print("✓ Quarter-start quarterly dates interpolate to monthly")


def test_series_to_monthly_quarterly_missing_interpolation_keeps_sparse():
    """Test quarterly data without interpolation keeps sparse monthly output."""
    dates = pd.date_range("2020-01-01", periods=4, freq="QE")
    values = [100.0, 200.0, 300.0, 400.0]
    series = pd.Series(values, index=dates)

    result = _series_to_monthly(series, source_freq="Q")
    assert len(result) > len(series)
    assert result.isna().any()
    print("✓ Quarterly data without interpolation stays sparse")


def test_series_to_monthly_aggregation_mean():
    """Test that aggregation_method='mean' works."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    values = [float(i) for i in range(30)]
    series = pd.Series(values, index=dates)
    
    result = _series_to_monthly(series, source_freq="D", aggregation_method="mean")
    assert len(result) >= 1
    assert result.iloc[0] == sum(values[:31]) / 30  # Mean of first month (Jan)
    print("✓ Aggregation with mean works")


def test_series_to_monthly_aggregation_last():
    """Test that aggregation_method='last' works (default)."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    values = [float(i) for i in range(30)]
    series = pd.Series(values, index=dates)
    
    result = _series_to_monthly(series, source_freq="D")
    assert len(result) >= 1
    assert result.iloc[0] == values[29]  # Last value of first month (Jan)
    print("✓ Aggregation with last works (default)")


if __name__ == "__main__":
    test_variablespec_fields()
    test_variablespec_defaults()
    test_apply_interpolation_none()
    test_apply_interpolation_linear()
    test_apply_interpolation_forward_fill()
    test_series_to_monthly_no_interpolation()
    test_series_to_monthly_quarterly_linear()
    test_series_to_monthly_quarterly_linear_quarter_start_dates()
    test_series_to_monthly_quarterly_missing_interpolation_keeps_sparse()
    test_series_to_monthly_aggregation_mean()
    test_series_to_monthly_aggregation_last()
    print("\n✅ All tests passed!")
