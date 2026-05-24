"""Regenerate all headline figures from cached results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuropeek.viz import emergence_plots


def main() -> None:
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("results/emergence_grid.parquet")
    fit = json.loads(Path("results/scaling_law_fit.json").read_text())

    heatmap = emergence_plots.emergence_heatmap(df)
    heatmap.write_html(out / "emergence_heatmap.html")
    heatmap.write_image(out / "emergence_heatmap.png", scale=2)

    fig = emergence_plots.scaling_law_figure(fit)
    fig.write_html(out / "scaling_law.html")
    fig.write_image(out / "scaling_law.png", scale=2)

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
