"""Resume the emergence grid after a crash by reading the existing JSONL,
determining which (size, step) cells are already done, and running only
the missing ones.

Appends to `results/emergence_grid.jsonl` and rebuilds the Parquet.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from circuitprobe import checkpoints, emergence


def main() -> None:
    sizes = ["160M", "410M", "1.4B"]
    steps = checkpoints.select_emergence_steps(n=12)
    expected = {(s, st) for s in sizes for st in steps}

    jsonl_path = Path("results/emergence_grid.jsonl")
    done: set[tuple[str, int]] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["size"], row["step"]))

    missing = sorted(expected - done, key=lambda x: (sizes.index(x[0]), x[1]))
    print(f"Expected: {len(expected)} cells.  Done: {len(done)}.  Missing: {len(missing)}.")
    if not missing:
        print("Nothing to do.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with jsonl_path.open("a") as f:
        for i, (size, step) in enumerate(missing, start=1):
            # Pre-clean any partial cache from a failed earlier attempt.
            emergence._cleanup_revision_snapshots(size, f"step{step}")
            print(f"[{i}/{len(missing)}] {size:>5} step={step:>6} ... ", end="", flush=True)
            try:
                cell = emergence.run_cell(
                    size=size, step=step, device=device,
                    n_seqs=64, half_len=50, seed=0,
                )
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            f.write(json.dumps(asdict(cell)) + "\n")
            f.flush()
            print(f"max_pm={cell.max_prefix_match:.3f}")

    emergence.grid_to_parquet(jsonl_path, Path("results/emergence_grid.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()
