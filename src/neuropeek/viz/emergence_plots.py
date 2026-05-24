"""Plotly figures for the emergence grid + scaling-law fit.

Produces:
- `emergence_heatmap(df)`  -- 2D heatmap of max prefix-match over (size, step).
- `scaling_law_figure(fit)` -- log-log scatter of emergence step vs model
  params with the fitted power-law overlay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..scaling import SIZE_TO_PARAMS


def emergence_heatmap(df: pd.DataFrame) -> go.Figure:
    sizes = sorted(df["size"].unique(), key=lambda s: SIZE_TO_PARAMS[s])
    steps = sorted(df["step"].unique())
    z = np.zeros((len(sizes), len(steps)))
    for i, size in enumerate(sizes):
        for j, step in enumerate(steps):
            mask = (df["size"] == size) & (df["step"] == step)
            if mask.any():
                z[i, j] = df.loc[mask, "max_prefix_match"].iloc[0]
            else:
                z[i, j] = float("nan")
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=steps,
            y=sizes,
            colorscale="Viridis",
            colorbar=dict(title="max prefix-match"),
        )
    )
    fig.update_layout(
        title="Induction-head emergence map (max prefix-match per cell)",
        xaxis_title="Training step",
        yaxis_title="Model size",
        xaxis_type="log",
    )
    return fig


def scaling_law_figure(fit: dict) -> go.Figure:
    sizes = list(fit["per_size"].keys())
    xs = [
        SIZE_TO_PARAMS[s]
        for s in sizes
        if fit["per_size"][s]["emerge_step"] is not None
    ]
    ys = [
        fit["per_size"][s]["emerge_step"]
        for s in sizes
        if fit["per_size"][s]["emerge_step"] is not None
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[s for s in sizes if fit["per_size"][s]["emerge_step"] is not None],
            textposition="top center",
            name="Observed emergence step",
        )
    )
    if fit.get("fit") is not None:
        a, b = fit["fit"]["a"], fit["fit"]["b"]
        x_curve = np.geomspace(min(xs) / 2, max(xs) * 2, 100)
        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=a * x_curve**b,
                mode="lines",
                name=f"y = {a:.2e} . N^{b:.3f}",
            )
        )
    fig.update_layout(
        title="Emergence step vs model parameters (power-law fit)",
        xaxis_title="Model parameters (N)",
        yaxis_title="First training step where max prefix-match >= threshold",
        xaxis_type="log",
        yaxis_type="log",
    )
    return fig
