#!/usr/bin/env python
"""Pull monthly macro data from FRED with fredapi and export merged raw CSV."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from config import VARIABLE_SPECS, VariableSpec

UPSAMPLING_FREQS = {"Q", "QS", "QE", "A", "AS", "AE", "Y", "YS", "YE", "SA", "BA", "BAS", "BAE"}


def _apply_interpolation(series: pd.Series, interpolation_method: str | None) -> pd.Series:
    if interpolation_method is None:
        return series
    if interpolation_method == "forward_fill":
        return series.ffill()
    if interpolation_method == "linear":
        return series.interpolate(method="linear")
    raise ValueError(f"Unknown interpolation method: {interpolation_method}")


def _pick_best_series(fred: Fred, spec: VariableSpec) -> tuple[str, dict[str, str]]:
    ranked: list[tuple[int, str, dict[str, str]]] = []

    for candidate in spec.candidates:
        info_raw = fred.get_series_info(candidate)
        info = dict(info_raw) if isinstance(info_raw, Mapping) else dict(info_raw.to_dict())
        freq = str(info.get("frequency_short", "")).upper()
        monthly = freq == "M"
        seasonally_adjusted = str(info.get("seasonal_adjustment_short", "")).upper() in {"SA", "SAAR"}

        score = 0
        if monthly:
            score += 100
        if spec.prefer_seasonally_adjusted and seasonally_adjusted:
            score += 10
        ranked.append((score, candidate, info))

    if not ranked:
        raise RuntimeError(f"No valid candidates for variable '{spec.name}'.")

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, selected_code, selected_info = ranked[0]
    return selected_code, selected_info


def _series_to_monthly(
    raw_series: pd.Series,
    source_freq: str,
    interpolation_method: str | None = None,
    aggregation_method: str = "last",
) -> pd.Series:
    series = pd.to_numeric(raw_series, errors="coerce").dropna().sort_index()
    series.index = pd.to_datetime(series.index)
    source_freq = source_freq.upper()

    if source_freq == "M":
        series.index = series.index.to_period("M").to_timestamp("M")
        series.index.name = "date"
        return series

    if source_freq in UPSAMPLING_FREQS:
        # Preserve original anchor timestamps (which may be quarter-start/mid-month),
        # interpolate on a combined timeline, then project to month-end values.
        month_end_index = pd.date_range(series.index.min(), series.index.max(), freq="ME")
        combined_index = series.index.union(month_end_index).sort_values()
        expanded = series.reindex(combined_index)
        interpolated = _apply_interpolation(expanded, interpolation_method)
        monthly = interpolated.reindex(month_end_index)
        monthly.index.name = "date"
        return monthly

    if aggregation_method == "mean":
        monthly = series.resample("ME").mean()
    elif aggregation_method == "last":
        monthly = series.resample("ME").last()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    monthly.index.name = "date"
    return monthly


def build_dataset(fred: Fred, specs: list[VariableSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged: pd.DataFrame | None = None
    manifest_rows: list[dict[str, str]] = []

    for spec in specs:
        code, info = _pick_best_series(fred, spec)
        series = fred.get_series(code)
        source_freq = str(info.get("frequency_short", "")).upper()
        monthly = _series_to_monthly(
            series,
            source_freq=source_freq,
            interpolation_method=spec.interpolation_method,
            aggregation_method=spec.aggregation_method,
        )
        col_df = monthly.rename(spec.name).to_frame()
        merged = col_df if merged is None else merged.join(col_df, how="outer")

        manifest_rows.append(
            {
                "variable_name": spec.name,
                "selected_series_id": code,
                "title": str(info.get("title", "")),
                "source_frequency_short": str(info.get("frequency_short", "")).upper(),
                "seasonal_adjustment_short": str(info.get("seasonal_adjustment_short", "")).upper(),
            }
        )

    if merged is None:
        raise RuntimeError("No series loaded.")

    merged = merged.sort_index().reset_index()
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    manifest = pd.DataFrame(manifest_rows).sort_values("variable_name")
    return merged, manifest


def load_fred_data() -> None:
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FRED_API_KEY. Add it to .env or environment variables.")
    if not re.fullmatch(r"[a-z0-9]{32}", api_key):
        raise RuntimeError(
            "Invalid FRED_API_KEY format. Expected a 32-character lowercase alphanumeric key."
        )

    fred = Fred(api_key=api_key)
    data, manifest = build_dataset(fred, VARIABLE_SPECS)

    out_dir = Path("data") / "raw_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "fred_macro_monthly.csv"
    manifest_path = out_dir / "fred_series_manifest.csv"
    data.to_csv(merged_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    print(f"Saved: {merged_path.resolve()}")
    print(f"Saved: {manifest_path.resolve()}")
    print(f"Rows={len(data)}, Columns={data.shape[1]}")


if __name__ == "__main__":
    load_fred_data()

