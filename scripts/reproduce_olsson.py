"""Reproduce Olsson '22 figure 1-style ICL loss-by-position curve.

Runs Pythia-410M on a Pile sample and dumps the curve + a Plotly figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import torch

from neuropeek import icl, models


def main() -> None:
    print("Loading Pythia-410M (final step)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = models.load_pythia(size="410M", step=None, device=device)

    print("Computing ICL curve on 32 Pile sequences (max_len=512)...")
    curve = icl.icl_curve_on_pile(model, n_seqs=32, max_len=512, seed=0)

    Path("results").mkdir(exist_ok=True)
    Path("results/icl_curve_pythia_410m.json").write_text(
        json.dumps({"loss_by_position": curve.tolist()})
    )

    fig = go.Figure()
    positions = list(range(1, len(curve) + 1))
    fig.add_trace(go.Scatter(x=positions, y=curve.tolist(), mode="lines"))
    fig.update_layout(
        title="ICL loss-by-position — Pythia-410M",
        xaxis_title="Token position",
        yaxis_title="Mean cross-entropy loss",
        xaxis_type="log",
    )
    fig.write_html("results/icl_curve_pythia_410m.html")
    fig.write_image("results/icl_curve_pythia_410m.png", scale=2)
    print("Wrote results/icl_curve_pythia_410m.{json,html,png}")


if __name__ == "__main__":
    main()
