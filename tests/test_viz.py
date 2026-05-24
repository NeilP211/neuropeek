"""Tests for emergence visualisation (synthetic data, no parquet required)."""

import pandas as pd
import plotly.graph_objects as go

from neuropeek.viz import emergence_plots
from neuropeek.viz.attention import iframe_srcdoc


def test_iframe_srcdoc_wraps_script_bearing_html():
    # CircuitsVis output is a div plus an inline module <script>. Injected as a
    # raw HTML string (e.g. Gradio's gr.HTML), the browser never runs the
    # script. Wrapping in an <iframe srcdoc> makes it execute.
    inner = (
        '<div id="circuits-vis-x"></div>'
        '<script type="module">import {render} from "u";render("circuits-vis-x");</script>'
    )
    out = iframe_srcdoc(inner, height=480)

    assert out.startswith("<iframe")
    assert "srcdoc=" in out
    assert "height:480px" in out
    # The only live markup is the <iframe> itself; the snippet's <script> must
    # be escaped inside srcdoc, never a live top-level tag.
    assert "<script" not in out
    assert "&lt;script" in out
    # Inner content is carried through (escaped), not dropped.
    assert "render(" in out
    assert "circuits-vis-x" in out


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
