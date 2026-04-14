#!/usr/bin/env python
"""Tests for src/transformations.py."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add src directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import VariableSpec
import transformations


def _make_raw_frame(periods: int = 14) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=periods, freq="ME")
    return pd.DataFrame(
        {
            "var_none": np.arange(1, periods + 1, dtype=float),
            "var_diff": np.arange(10, 10 + periods, dtype=float),
            "var_log": np.exp(np.arange(periods, dtype=float)),
            "var_log_diff": np.exp(np.arange(periods, dtype=float)),
            "var_log_diff_12": np.exp(np.arange(periods, dtype=float)),
        },
        index=idx,
    )


def test_transform_data_applies_all_supported_transformations():
    raw = _make_raw_frame()
    specs = [
        VariableSpec(name="var_none", candidates=["A"], transformation=None),
        VariableSpec(name="var_diff", candidates=["B"], transformation="diff"),
        VariableSpec(name="var_log", candidates=["C"], transformation="log"),
        VariableSpec(name="var_log_diff", candidates=["D"], transformation="log_diff"),
        VariableSpec(name="var_log_diff_12", candidates=["E"], transformation="log_diff_12"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.csv"
        raw.to_csv(raw_path)

        with patch.object(transformations, "VARIABLE_SPECS", specs):
            result = transformations.transform_data(raw_path=str(raw_path))

    assert result["var_none"].equals(raw["var_none"])
    assert pd.isna(result["var_diff"].iloc[0])
    assert result["var_diff"].iloc[1] == 1.0
    assert np.isclose(result["var_log"].iloc[5], 5.0)
    assert pd.isna(result["var_log_diff"].iloc[0])
    assert np.isclose(result["var_log_diff"].iloc[6], 1.0)
    assert pd.isna(result["var_log_diff_12"].iloc[11])
    assert np.isclose(result["var_log_diff_12"].iloc[12], 12.0)


def test_transform_data_skips_columns_missing_from_raw_data():
    raw = _make_raw_frame()[["var_none"]]
    specs = [
        VariableSpec(name="var_none", candidates=["A"], transformation=None),
        VariableSpec(name="missing_var", candidates=["B"], transformation="diff"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.csv"
        raw.to_csv(raw_path)
        with patch.object(transformations, "VARIABLE_SPECS", specs):
            result = transformations.transform_data(raw_path=str(raw_path))

    assert list(result.columns) == ["var_none"]


def test_transform_data_raises_on_unknown_transformation_name():
    raw = _make_raw_frame()[["var_none"]]
    specs = [VariableSpec(name="var_none", candidates=["A"], transformation="not_a_transform")]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.csv"
        raw.to_csv(raw_path)
        with patch.object(transformations, "VARIABLE_SPECS", specs):
            try:
                transformations.transform_data(raw_path=str(raw_path))
            except KeyError as exc:
                assert "not_a_transform" in str(exc)
            else:
                raise AssertionError("Expected KeyError for unknown transformation")


def test_inverse_transform_restores_scale_for_supported_inverse_methods():
    dates = pd.date_range("2021-01-31", periods=12, freq="ME")
    transformed = pd.DataFrame(
        {
            "var_none": np.array([2.0] * 12),
            "var_diff": np.array([1.0] * 12),
            "var_log": np.zeros(12),
            "var_log_diff": np.array([0.0] * 12),
            "var_log_diff_12": np.log(np.arange(2.0, 14.0)),
        },
        index=dates,
    )
    raw = pd.DataFrame(
        {
            "var_none": np.array([5.0] * 24),
            "var_diff": np.arange(1.0, 25.0),
            "var_log": np.arange(1.0, 25.0),
            "var_log_diff": np.array([10.0] * 24),
            "var_log_diff_12": np.arange(100.0, 124.0),
        },
        index=pd.date_range("2019-01-31", periods=24, freq="ME"),
    )
    specs = [
        VariableSpec(name="var_none", candidates=["A"], transformation=None),
        VariableSpec(name="var_diff", candidates=["B"], transformation="diff"),
        VariableSpec(name="var_log", candidates=["C"], transformation="log"),
        VariableSpec(name="var_log_diff", candidates=["D"], transformation="log_diff"),
        VariableSpec(name="var_log_diff_12", candidates=["E"], transformation="log_diff_12"),
    ]

    with patch.object(transformations, "VARIABLE_SPECS", specs):
        result = transformations.inverse_transform(transformed, raw_df=raw)

    assert (result["var_none"] == 2.0).all()
    assert result["var_diff"].iloc[0] == raw["var_diff"].iloc[-1] + 1.0
    assert result["var_diff"].iloc[1] == raw["var_diff"].iloc[-1] + 2.0
    assert (result["var_log"] == 1.0).all()
    assert (result["var_log_diff"] == raw["var_log_diff"].iloc[-1]).all()
    expected_log_diff_12 = np.arange(2.0, 14.0) * raw["var_log_diff_12"].iloc[-12:].values
    assert np.allclose(result["var_log_diff_12"].values, expected_log_diff_12)


def test_inverse_transform_skips_columns_missing_from_inputs():
    transformed = pd.DataFrame(
        {"present": [1.0, 2.0], "missing_in_raw": [3.0, 4.0]},
        index=pd.date_range("2022-01-31", periods=2, freq="ME"),
    )
    raw = pd.DataFrame(
        {"present": [10.0, 11.0]},
        index=pd.date_range("2021-11-30", periods=2, freq="ME"),
    )
    specs = [
        VariableSpec(name="present", candidates=["A"], transformation="none"),
        VariableSpec(name="missing_in_raw", candidates=["B"], transformation="none"),
        VariableSpec(name="missing_in_transformed", candidates=["C"], transformation="none"),
    ]

    with patch.object(transformations, "VARIABLE_SPECS", specs):
        result = transformations.inverse_transform(transformed, raw_df=raw)

    assert list(result.columns) == ["present"]


if __name__ == "__main__":
    test_transform_data_applies_all_supported_transformations()
    test_transform_data_skips_columns_missing_from_raw_data()
    test_transform_data_raises_on_unknown_transformation_name()
    test_inverse_transform_restores_scale_for_supported_inverse_methods()
    test_inverse_transform_skips_columns_missing_from_inputs()
    print("\n✅ All transformation tests passed!")
