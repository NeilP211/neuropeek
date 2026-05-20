"""Run the (model_size x training_step) emergence grid for CircuitProbe.

Output: results/emergence_grid.jsonl  (one cell per line, appended live)
        results/emergence_grid.parquet (consolidated at end)

This is a long-running script (~3-4 hours on Apple Silicon CPU/MPS for the
full 3-size x 12-step sweep). It cleans the HF revision cache after each
cell to keep peak disk usage under ~6 GB.

Usage:
    uv run python scripts/run_emergence_grid.py [--n-steps N] [--device DEV]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from circuitprobe import checkpoints, emergence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=12,
                        help="Number of log-spaced training checkpoints per size")
    parser.add_argument("--sizes", nargs="+", default=["160M", "410M", "1.4B"],
                        help="Pythia sizes to sweep")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: mps if available else cpu)")
    parser.add_argument("--n-seqs", type=int, default=64)
    parser.add_argument("--half-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Disable cache cleanup between cells (uses more disk)")
    args = parser.parse_args()

    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    steps = checkpoints.select_emergence_steps(n=args.n_steps)
    print(f"Running grid: sizes={args.sizes} steps={steps} device={device}")
    print(f"cleanup_after_cell={not args.no_cleanup}")

    output = Path("results/emergence_grid.jsonl")
    cells = list(
        emergence.run_grid(
            sizes=args.sizes,
            steps=steps,
            device=device,
            n_seqs=args.n_seqs,
            half_len=args.half_len,
            seed=args.seed,
            output_path=output,
            cleanup_after_cell=not args.no_cleanup,
        )
    )

    parquet = Path("results/emergence_grid.parquet")
    emergence.grid_to_parquet(output, parquet)

    print(f"\nDone: {len(cells)} cells. Parquet at {parquet}.")


if __name__ == "__main__":
    main()
