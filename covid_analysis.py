#!/usr/bin/env python
"""Analyze EINN vs BaseNN performance during COVID period."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load predictions
predictions = pd.read_csv("data/processed/backtest_predictions.csv")
predictions["target_date"] = pd.to_datetime(predictions["target_date"])

# Convert to percentage (×100) for display
predictions["y_true"] = predictions["y_true"] * 100
predictions["y_pred"] = predictions["y_pred"] * 100

# Filter to test split
test_pred = predictions[predictions["split"] == "test"].copy()

# Split into pre-COVID and COVID periods
covid_start = pd.Timestamp("2020-03-01")
pre_covid = test_pred[test_pred["target_date"] < covid_start]
covid_period = test_pred[test_pred["target_date"] >= covid_start]

print("=" * 80)
print("EINN vs BaseNN: Pre-COVID vs COVID Period Analysis")
print("=" * 80)
print()

# Calculate metrics by model and period
models = ["einn", "base_nn"]
for model in models:
    pre = pre_covid[pre_covid["model"] == model]
    covid = covid_period[covid_period["model"] == model]
    
    if pre.empty or covid.empty:
        continue
    
    pre_rmse = np.sqrt(((pre["y_true"] - pre["y_pred"]) ** 2).mean())
    covid_rmse = np.sqrt(((covid["y_true"] - covid["y_pred"]) ** 2).mean())
    
    pre_mae = (pre["y_true"] - pre["y_pred"]).abs().mean()
    covid_mae = (covid["y_true"] - covid["y_pred"]).abs().mean()
    
    print(f"{model.upper()}")
    print(f"  Pre-COVID (1992-2020-02):")
    print(f"    RMSE: {pre_rmse:.4f} %    MAE: {pre_mae:.4f} %    N: {len(pre)}")
    print(f"  COVID (2020-03+):")
    print(f"    RMSE: {covid_rmse:.4f} %    MAE: {covid_mae:.4f} %    N: {len(covid)}")
    print(f"  COVID Impact: RMSE {covid_rmse - pre_rmse:+.4f} % ({(covid_rmse/pre_rmse - 1)*100:+.1f}%)")
    print()

# Create comparison plot
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "EINN: Pre-COVID vs COVID",
        "BaseNN: Pre-COVID vs COVID",
        "EINN: Residuals by Period",
        "BaseNN: Residuals by Period",
    ),
    specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "box"}, {"type": "box"}]]
)

colors = {"einn": "#1f77b4", "base_nn": "#ff7f0e"}

for idx, model in enumerate(models, 1):
    col = idx
    
    # Scatter plot: actual vs forecast
    pre = pre_covid[pre_covid["model"] == model].sort_values("target_date")
    covid = covid_period[covid_period["model"] == model].sort_values("target_date")
    
    fig.add_trace(
        go.Scatter(
            x=pre["target_date"],
            y=pre["y_pred"],
            mode="lines+markers",
            name=f"{model.upper()} (Pre-COVID)",
            line=dict(color=colors[model], width=2),
            marker=dict(size=4),
            legendgroup=f"{model}_pre",
        ),
        row=1, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=covid["target_date"],
            y=covid["y_pred"],
            mode="lines+markers",
            name=f"{model.upper()} (COVID)",
            line=dict(color=colors[model], width=2, dash="dash"),
            marker=dict(size=4),
            legendgroup=f"{model}_covid",
        ),
        row=1, col=col
    )
    
    # Add actual as reference (shared across both periods)
    actual = pd.concat([pre, covid]).drop_duplicates("target_date", keep="first").sort_values("target_date")
    fig.add_trace(
        go.Scatter(
            x=actual["target_date"],
            y=actual["y_true"],
            mode="lines",
            name="Actual",
            line=dict(color="black", width=2),
            showlegend=(idx == 1),
            legendgroup="actual",
        ),
        row=1, col=col
    )
    
    # Box plots: residuals
    pre_resid = pre["y_true"] - pre["y_pred"]
    covid_resid = covid["y_true"] - covid["y_pred"]
    
    fig.add_trace(
        go.Box(
            y=pre_resid,
            name="Pre-COVID",
            marker=dict(color=colors[model]),
            boxmean="sd",
            showlegend=False,
        ),
        row=2, col=idx
    )
    
    fig.add_trace(
        go.Box(
            y=covid_resid,
            name="COVID",
            marker=dict(color=colors[model], opacity=0.5),
            boxmean="sd",
            showlegend=False,
        ),
        row=2, col=idx
    )

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_xaxes(title_text="Period", row=2, col=1)
fig.update_xaxes(title_text="Period", row=2, col=2)

fig.update_yaxes(title_text="Inflation Forecast (%)", row=1, col=1)
fig.update_yaxes(title_text="Inflation Forecast (%)", row=1, col=2)
fig.update_yaxes(title_text="Residual (%)", row=2, col=1)
fig.update_yaxes(title_text="Residual (%)", row=2, col=2)

fig.update_layout(
    title_text="EINN vs BaseNN: Forecast Performance During COVID",
    height=900,
    width=1400,
    showlegend=True,
    hovermode="x unified",
)

fig.write_html("figures/covid_comparison_einn_base_nn.html")
print("[OK] Saved comparison plot to figures/covid_comparison_einn_base_nn.html")
print()

# By horizon analysis during COVID
print("COVID Period RMSE by Horizon:")
print("-" * 50)
covid_by_h = covid_period.groupby(["model", "horizon"]).apply(
    lambda x: np.sqrt(((x["y_true"] - x["y_pred"]) ** 2).mean()),
    include_groups=False
).unstack(level="horizon")

print(covid_by_h.round(4))
