from __future__ import annotations

import pandas as pd


def month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.to_period("M").to_timestamp("M")


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return (ts.to_period("M") + months).to_timestamp("M")


def build_forecast_index(last_index: int | float | pd.Timestamp, horizons: list[int]) -> pd.Index:
    if isinstance(last_index, pd.Timestamp):
        return pd.Index([add_months(last_index, h) for h in horizons])
    return pd.Index([last_index + h for h in horizons])
