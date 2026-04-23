from __future__ import annotations

import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((((y_true - y_pred) ** 2).mean()) ** 0.5)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def validation_score(validation_rows: pd.DataFrame, horizons: list[int]) -> float:
    if validation_rows.empty:
        return float("inf")

    horizon_scores: list[float] = []
    for h in horizons:
        subset = validation_rows.loc[validation_rows["horizon"] == h]
        if subset.empty:
            continue
        horizon_scores.append(rmse(subset["y_true"], subset["y_pred"]))

    if not horizon_scores:
        return float("inf")
    return float(sum(horizon_scores) / len(horizon_scores))


def summarize_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        empty = pd.DataFrame(columns=["split", "model", "horizon", "rmse", "mae", "n"])
        return empty, empty

    by_horizon = (
        predictions.groupby(["split", "model", "horizon"], as_index=False)
        .apply(
            lambda df: pd.Series(
                {
                    "rmse": rmse(df["y_true"], df["y_pred"]),
                    "mae": mae(df["y_true"], df["y_pred"]),
                    "n": len(df),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    overall = (
        predictions.groupby(["split", "model"], as_index=False)
        .apply(
            lambda df: pd.Series(
                {
                    "rmse": rmse(df["y_true"], df["y_pred"]),
                    "mae": mae(df["y_true"], df["y_pred"]),
                    "n": len(df),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    return by_horizon, overall
