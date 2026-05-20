"""Tests for emergence visualisation (synthetic data, no parquet required)."""

import pandas as pd
import plotly.graph_objects as go

from circuitprobe.viz import emergence_plots


def test_emergence_heatmap_returns_figure_with_heatmap_trace():
    df = pd.DataFrame([
        {"size": "160M", "step": 0, "max_prefix_match": 0.01},
        {"size": "160M", "step": 1000, "max_prefix_match": 0.4},
        {"size": "410M", "step": 0, "max_prefix_match": 0.02},
        {"size": "410M", "step": 1000, "max_prefix_match": 0.6},
    ])
    fig = emergence_plots.emergence_heatmap(df)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Heatmap) for t in fig.data)


def test_scaling_law_figure_handles_missing_emergence():
    fit = {
        "per_size": {
            "160M": {"emerge_step": 5000},
            "410M": {"emerge_step": None},  # never crossed
        },
        "fit": None,
        "threshold": 0.3,
    }
    fig = emergence_plots.scaling_law_figure(fit)
    assert isinstance(fig, go.Figure)
    # Should have just the markers trace, no fit overlay.
    assert len(fig.data) == 1


def test_scaling_law_figure_with_fit_includes_curve():
    fit = {
        "per_size": {
            "160M": {"emerge_step": 5000},
            "410M": {"emerge_step": 8000},
            "1.4B": {"emerge_step": 12000},
        },
        "fit": {"a": 1e-5, "b": 0.5, "b_ci": (0.4, 0.6)},
        "threshold": 0.3,
    }
    fig = emergence_plots.scaling_law_figure(fit)
    assert len(fig.data) == 2  # markers + fit curve
