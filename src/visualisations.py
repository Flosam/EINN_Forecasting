from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _ensure_figures_dir(output_dir: str = "figures") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(
    fig: go.Figure, name: str, output_dir: str = "figures", *, include_png: bool = True
) -> dict[str, str]:
    """Save figure as HTML and optionally PNG."""
    out_path = _ensure_figures_dir(output_dir)
    html_file = out_path / f"{name}.html"
    png_file = out_path / f"{name}.png"

    fig.write_html(str(html_file))
    saved_files = {"html": str(html_file)}

    if include_png:
        try:
            fig.write_image(str(png_file))
            saved_files["png"] = str(png_file)
        except Exception as e:
            print(f"Warning: Could not save PNG for {name}: {e}")

    return saved_files


def _get_test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Filter predictions to test split only, excluding unstable VAR model."""
    if predictions.empty:
        return predictions
    if "split" not in predictions.columns:
        return predictions
    test_df = predictions[predictions["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test split predictions found in data.")
    # Exclude VAR due to numerical instability
    test_df = test_df[test_df["model"] != "var"]
    if test_df.empty:
        raise ValueError("No valid model predictions found (VAR excluded).")
    return test_df


def _get_color_palette(models: list[str]) -> dict[str, str]:
    """Return consistent color palette for models."""
    colors = {
        "einn": "#1f77b4",
        "base_nn": "#ff7f0e",
        "lasso": "#2ca02c",
        "ar1": "#d62728",
        "var": "#9467bd",
    }
    return {model: colors.get(model, "#7f7f7f") for model in models}


def plot_forecasts_by_horizon(
    predictions: pd.DataFrame,
    horizons: list[int] | None = None,
    output_dir: str = "figures",
    target_name: str = "CPI Inflation (pp)",
    save: bool = True,
) -> go.Figure:
    """
    Plot forecasts vs actual values for each horizon.
    Creates 4 subplots (one per horizon) showing model predictions and confidence intervals.
    """
    test_df = _get_test_predictions(predictions)

    if horizons is None:
        horizons = sorted(test_df["horizon"].unique().tolist())

    if not horizons:
        raise ValueError("No horizons found in predictions data.")

    models = sorted(test_df["model"].unique().tolist())
    colors = _get_color_palette(models)
    shown_models = set()  # Track which models have been added to legend

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{h}-Month Horizon" for h in horizons],
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
    )

    for idx, h in enumerate(horizons):
        row = idx // 2 + 1
        col = idx % 2 + 1

        h_data = test_df[test_df["horizon"] == h].sort_values("target_date")

        if h_data.empty:
            continue

        target_dates = h_data["target_date"].unique()
        y_true_by_date = h_data.drop_duplicates("target_date").set_index("target_date")["y_true"]

        # Add actual line only to first subplot legend
        fig.add_trace(
            go.Scatter(
                x=y_true_by_date.index,
                y=y_true_by_date.values,
                name="Actual",
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=idx == 0,
                hovertemplate="<b>Actual</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f} pp<extra></extra>",
            ),
            row=row,
            col=col,
        )

        for model in models:
            model_data = h_data[h_data["model"] == model].sort_values("target_date")

            if model_data.empty:
                continue

            should_show_legend = model not in shown_models
            if should_show_legend:
                shown_models.add(model)

            fig.add_trace(
                go.Scatter(
                    x=model_data["target_date"],
                    y=model_data["y_pred"],
                    name=model.upper(),
                    mode="lines",
                    line=dict(color=colors[model], width=1.5),
                    showlegend=should_show_legend,
                    hovertemplate=f"<b>{model.upper()}</b><br>Date: %{{x|%Y-%m-%d}}<br>Forecast: %{{y:.2f}} pp<extra></extra>",
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(model_data["target_date"]) + list(model_data["target_date"][::-1]),
                    y=list(model_data["upper"]) + list(model_data["lower"][::-1]),
                    name=f"{model.upper()} (90% CI)",
                    fill="toself",
                    fillcolor=colors[model],
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.2,
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text=target_name, row=row, col=col)

    fig.update_layout(
        title_text="Multi-Horizon Inflation Forecast Comparison (Test Split, pp)",
        height=900,
        width=1400,
        hovermode="x unified",
        showlegend=True,
    )

    if save:
        _save_figure(fig, "forecast_comparison_by_horizon", output_dir)

    return fig


def plot_model_performance(
    metrics_by_horizon: pd.DataFrame,
    output_dir: str = "figures",
    save: bool = True,
) -> go.Figure:
    """
    Plot RMSE and MAE by model and horizon for validation and test splits.
    """
    if metrics_by_horizon.empty:
        raise ValueError("metrics_by_horizon is empty.")

    if "split" not in metrics_by_horizon.columns:
        raise ValueError("metrics_by_horizon must have 'split' column.")

    horizons = sorted(metrics_by_horizon["horizon"].unique().tolist())
    models = sorted(metrics_by_horizon["model"].unique().tolist())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("RMSE by Model and Horizon", "MAE by Model and Horizon"),
    )

    for metric, col_idx in [("rmse", 1), ("mae", 2)]:
        for split in ["validation", "test"]:
            split_data = metrics_by_horizon[metrics_by_horizon["split"] == split]

            if split_data.empty:
                continue

            for model in models:
                model_data = split_data[split_data["model"] == model].sort_values("horizon")

                if model_data.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=model_data["horizon"],
                        y=model_data[metric],
                        name=f"{model.upper()} ({split})",
                        mode="lines+markers",
                        hovertemplate=f"<b>{model.upper()} ({split})</b><br>Horizon: %{{x}}<br>{metric.upper()}: %{{y:.4f}}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_xaxes(title_text="Horizon (months)", row=1, col=col_idx)
        fig.update_yaxes(title_text=f"{metric.upper()} (pp)", row=1, col=col_idx)

    fig.update_layout(
        title_text="Model Performance Comparison (pp)",
        height=500,
        width=1200,
        hovermode="x unified",
        showlegend=True,
    )

    if save:
        _save_figure(fig, "model_performance_comparison", output_dir)

    return fig


def plot_performance_over_time(
    predictions: pd.DataFrame,
    horizons: list[int] | None = None,
    output_dir: str = "figures",
    save: bool = True,
) -> go.Figure:
    """
    Plot RMSE evolution over expanding window folds for each horizon and model.
    """
    test_df = _get_test_predictions(predictions)

    if horizons is None:
        horizons = sorted(test_df["horizon"].unique().tolist())

    if not horizons:
        raise ValueError("No horizons found in predictions data.")

    models = sorted(test_df["model"].unique().tolist())
    colors = _get_color_palette(models)
    shown_models = set()  # Track which models have been added to legend

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{h}-Month Horizon" for h in horizons],
    )

    for idx, h in enumerate(horizons):
        row = idx // 2 + 1
        col = idx % 2 + 1

        h_data = test_df[test_df["horizon"] == h]

        for model in models:
            model_data = h_data[h_data["model"] == model]

            if model_data.empty:
                continue

            rmse_by_fold = (
                model_data.groupby("outer_fold")
                .apply(lambda x: ((x["y_true"] - x["y_pred"]) ** 2).mean() ** 0.5, include_groups=False)
                .reset_index()
            )
            rmse_by_fold.columns = ["outer_fold", "rmse"]

            should_show_legend = model not in shown_models
            if should_show_legend:
                shown_models.add(model)

            fig.add_trace(
                go.Scatter(
                    x=rmse_by_fold["outer_fold"],
                    y=rmse_by_fold["rmse"],
                    name=model.upper(),
                    mode="lines+markers",
                    line=dict(color=colors[model]),
                    showlegend=should_show_legend,
                    hovertemplate=f"<b>{model.upper()}</b><br>Fold: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Fold (Expanding Window)", row=row, col=col)
        fig.update_yaxes(title_text="RMSE", row=row, col=col)

    fig.update_layout(
        title_text="Model Performance Evolution Over Expanding Window",
        height=900,
        width=1400,
        hovermode="x unified",
        showlegend=True,
    )

    if save:
        _save_figure(fig, "performance_over_time", output_dir)

    return fig


def plot_residuals_analysis(
    predictions: pd.DataFrame,
    horizons: list[int] | None = None,
    output_dir: str = "figures",
    save: bool = True,
) -> go.Figure:
    """
    Plot residuals (y_true - y_pred) over time for each horizon and model.
    """
    test_df = _get_test_predictions(predictions)

    if horizons is None:
        horizons = sorted(test_df["horizon"].unique().tolist())

    if not horizons:
        raise ValueError("No horizons found in predictions data.")

    models = sorted(test_df["model"].unique().tolist())
    colors = _get_color_palette(models)
    shown_models = set()  # Track which models have been added to legend

    test_df["residual"] = test_df["y_true"] - test_df["y_pred"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{h}-Month Horizon" for h in horizons],
    )

    for idx, h in enumerate(horizons):
        row = idx // 2 + 1
        col = idx % 2 + 1

        h_data = test_df[test_df["horizon"] == h].sort_values("target_date")

        for model in models:
            model_data = h_data[h_data["model"] == model]

            if model_data.empty:
                continue

            should_show_legend = model not in shown_models
            if should_show_legend:
                shown_models.add(model)

            fig.add_trace(
                go.Scatter(
                    x=model_data["target_date"],
                    y=model_data["residual"],
                    name=model.upper(),
                    mode="markers",
                    marker=dict(color=colors[model], size=6),
                    showlegend=should_show_legend,
                    hovertemplate=f"<b>{model.upper()}</b><br>Date: %{{x|%Y-%m-%d}}<br>Residual: %{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col, annotation_text=None)

        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Residual (Actual - Forecast, pp)", row=row, col=col)

    fig.update_layout(
        title_text="Residuals Analysis (Test Split, pp)",
        height=900,
        width=1400,
        hovermode="x unified",
        showlegend=True,
    )

    if save:
        _save_figure(fig, "residuals_analysis", output_dir)

    return fig


def plot_calibration(
    predictions: pd.DataFrame,
    horizons: list[int] | None = None,
    output_dir: str = "figures",
    save: bool = True,
) -> go.Figure:
    """
    Plot actual vs predicted values with confidence intervals (calibration check).
    Shows whether true values fall within predicted intervals.
    """
    test_df = _get_test_predictions(predictions)

    if horizons is None:
        horizons = sorted(test_df["horizon"].unique().tolist())

    if not horizons:
        raise ValueError("No horizons found in predictions data.")

    models = sorted(test_df["model"].unique().tolist())
    colors = _get_color_palette(models)
    shown_models = set()  # Track which models have been added to legend

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{h}-Month Horizon" for h in horizons],
    )

    for idx, h in enumerate(horizons):
        row = idx // 2 + 1
        col = idx % 2 + 1

        h_data = test_df[test_df["horizon"] == h]

        min_val = h_data["y_true"].min()
        max_val = h_data["y_true"].max()
        range_val = max_val - min_val
        bounds = [min_val - 0.1 * range_val, max_val + 0.1 * range_val]

        # Add perfect forecast line only to first subplot
        if idx == 0:
            fig.add_trace(
                go.Scatter(
                    x=bounds,
                    y=bounds,
                    name="Perfect Forecast",
                    mode="lines",
                    line=dict(color="black", dash="dash", width=2),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=bounds,
                    y=bounds,
                    mode="lines",
                    line=dict(color="black", dash="dash", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        for model in models:
            model_data = h_data[h_data["model"] == model]

            if model_data.empty:
                continue

            should_show_legend = model not in shown_models
            if should_show_legend:
                shown_models.add(model)

            fig.add_trace(
                go.Scatter(
                    x=model_data["y_pred"],
                    y=model_data["y_true"],
                    name=model.upper(),
                    mode="markers",
                    marker=dict(
                        color=model_data["y_true"],
                        colorscale="Viridis",
                        size=6,
                        line=dict(color=colors[model], width=1),
                        showscale=False,
                    ),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=model_data["upper"] - model_data["y_true"],
                        arrayminus=model_data["y_true"] - model_data["lower"],
                        color=colors[model],
                        thickness=1,
                    ),
                    showlegend=should_show_legend,
                    hovertemplate=f"<b>{model.upper()}</b><br>Predicted: %{{x:.4f}}<br>Actual: %{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Predicted Inflation (pp)", row=row, col=col, range=bounds)
        fig.update_yaxes(title_text="Actual Inflation (pp)", row=row, col=col, range=bounds)

    fig.update_layout(
        title_text="Calibration Analysis: Actual vs Predicted Inflation (pp, with 90% Confidence Intervals)",
        height=900,
        width=1400,
        hovermode="closest",
        showlegend=True,
    )

    if save:
        _save_figure(fig, "calibration_analysis", output_dir)

    return fig


def save_all_visualizations(
    predictions: pd.DataFrame,
    metrics_by_horizon: pd.DataFrame | None = None,
    metrics_overall: pd.DataFrame | None = None,
    output_dir: str = "figures",
    target_name: str = "CPI",
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """
    Generate and save all visualizations.
    Returns dictionary mapping plot names to file paths.
    """
    if predictions.empty:
        raise ValueError("predictions DataFrame is empty.")

    if horizons is None:
        horizons = sorted(predictions["horizon"].unique().tolist())

    results = {}

    print("Generating forecast comparison plot...")
    try:
        plot_forecasts_by_horizon(predictions, horizons=horizons, output_dir=output_dir, target_name=target_name)
        results["forecast_comparison"] = "✓"
    except Exception as e:
        print(f"  Error: {e}")
        results["forecast_comparison"] = f"✗ {e}"

    if metrics_by_horizon is not None and not metrics_by_horizon.empty:
        print("Generating model performance plot...")
        try:
            plot_model_performance(metrics_by_horizon, output_dir=output_dir)
            results["model_performance"] = "✓"
        except Exception as e:
            print(f"  Error: {e}")
            results["model_performance"] = f"✗ {e}"

    print("Generating performance over time plot...")
    try:
        plot_performance_over_time(predictions, horizons=horizons, output_dir=output_dir)
        results["performance_over_time"] = "✓"
    except Exception as e:
        print(f"  Error: {e}")
        results["performance_over_time"] = f"✗ {e}"

    print("Generating residuals analysis plot...")
    try:
        plot_residuals_analysis(predictions, horizons=horizons, output_dir=output_dir)
        results["residuals_analysis"] = "✓"
    except Exception as e:
        print(f"  Error: {e}")
        results["residuals_analysis"] = f"✗ {e}"

    print("Generating calibration analysis plot...")
    try:
        plot_calibration(predictions, horizons=horizons, output_dir=output_dir)
        results["calibration_analysis"] = "✓"
    except Exception as e:
        print(f"  Error: {e}")
        results["calibration_analysis"] = f"✗ {e}"

    print(f"\nVisualization Summary:")
    print(f"  Output directory: {output_dir}")
    for name, status in results.items():
        print(f"  {name}: {status}")

    return results
