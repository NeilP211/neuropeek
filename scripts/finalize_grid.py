"""Run after the emergence grid completes:

1. Consolidate the JSONL into a Parquet (if not already done by the runner).
2. Fit the scaling law and dump ``results/scaling_law_fit.json``.
3. Render headline figures into ``results/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from circuitprobe import emergence, scaling
from circuitprobe.viz import emergence_plots


def main() -> None:
    jsonl = Path("results/emergence_grid.jsonl")
    parquet = Path("results/emergence_grid.parquet")
    fit_json = Path("results/scaling_law_fit.json")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl.exists():
        raise SystemExit("results/emergence_grid.jsonl not found — run the grid first")

    print(f"1. Consolidating {jsonl} -> {parquet}")
    emergence.grid_to_parquet(jsonl, parquet)

    print("2. Fitting scaling law (threshold=0.3)")
    fit = scaling.fit_emergence_law(
        str(parquet),
        threshold=0.3,
        n_boot=500,
        seed=0,
    )
    fit_json.write_text(json.dumps(fit, indent=2, default=str))
    print(f"   wrote {fit_json}")
    if fit.get("fit") is not None:
        a = fit["fit"]["a"]
        b = fit["fit"]["b"]
        lo, hi = fit["fit"]["b_ci"]
        print(f"   power law: y = {a:.2e} * N^{b:.3f}  (95% CI on b: [{lo:.3f}, {hi:.3f}])")
    else:
        print(f"   {fit.get('note', 'no fit available')}")

    print("3. Rendering figures")
    df = pd.read_parquet(parquet)
    heatmap = emergence_plots.emergence_heatmap(df)
    heatmap.write_html(fig_dir / "emergence_heatmap.html")
    heatmap.write_image(fig_dir / "emergence_heatmap.png", scale=2)

    scaling_fig = emergence_plots.scaling_law_figure(fit)
    scaling_fig.write_html(fig_dir / "scaling_law.html")
    scaling_fig.write_image(fig_dir / "scaling_law.png", scale=2)
    print(f"   wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
