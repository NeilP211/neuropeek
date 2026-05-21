"""Emergence-grid runner.

Sweeps (model size x training step), measures the per-head prefix-match
score on random-repeat sequences, and persists one row per cell.

Disk-management note: each Pythia checkpoint occupies ~600 MB (160M)
to ~5.6 GB (1.4B) on disk under `~/.cache/huggingface/hub/`. With 36 cells
total, naive caching would exceed 70 GB. When `cleanup_after_cell=True`
(the default), we remove the snapshot directory for the specific revision
we just used after each cell completes, keeping peak disk under ~6 GB.
"""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import induction, models, tracking


@dataclass
class CellResult:
    size: str
    step: int
    max_prefix_match: float
    mean_prefix_match: float
    top_heads: list[tuple[int, int, float]] = field(default_factory=list)
    seed: int = 0
    n_seqs: int = 64
    half_len: int = 50


def summarise_scores(
    scores: torch.Tensor, top_k: int = 5
) -> tuple[float, float, list[tuple[int, int, float]]]:
    top = induction.rank_induction_heads(scores, top_k=top_k)
    # Round to 6 decimal places to suppress float32 round-trip artifacts
    # (e.g., 0.4 -> 0.40000000596...), matching `rank_induction_heads`.
    mx = round(float(scores.max().item()), 6)
    mn = round(float(scores.mean().item()), 6)
    return mx, mn, top


def _hf_cache_dir_for(size: str) -> Path:
    """Return the HF hub cache directory for a Pythia size."""
    home = Path.home() / ".cache" / "huggingface" / "hub"
    return home / f"models--EleutherAI--pythia-{size.lower().replace('.', '_')}"


def _cleanup_revision_snapshots(size: str, revision: str | None) -> None:
    """Remove the snapshot + blob shards for a specific revision after we're done with it."""
    if revision is None:
        return
    cache_dir = _hf_cache_dir_for(size)
    if not cache_dir.exists():
        return
    refs_dir = cache_dir / "refs"
    snapshots_dir = cache_dir / "snapshots"
    blobs_dir = cache_dir / "blobs"

    # Resolve the commit hash for this revision via refs/<revision>.
    ref_file = refs_dir / revision
    if not ref_file.exists():
        return
    commit_hash = ref_file.read_text().strip()
    snapshot_path = snapshots_dir / commit_hash
    if snapshot_path.exists():
        # Resolve each symlink in the snapshot to its blob, delete the blob, then the snapshot.
        for entry in snapshot_path.iterdir():
            try:
                if entry.is_symlink():
                    target = (snapshot_path / entry).resolve()
                    if blobs_dir in target.parents and target.exists():
                        target.unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(snapshot_path, ignore_errors=True)
    ref_file.unlink(missing_ok=True)


def run_cell(
    size: str,
    step: int,
    device: str = "cpu",
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
    top_k: int = 5,
    cleanup_after_cell: bool = True,
) -> CellResult:
    from . import checkpoints as _ckpt
    revision = _ckpt.revision_for_step(step) if step is not None else None

    model = None
    try:
        model = models.load_pythia(size=size, step=step, device=device)
        scores = induction.prefix_match_score(
            model, n_seqs=n_seqs, half_len=half_len, seed=seed
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cleanup_after_cell:
            # Clean even when load_pythia itself failed mid-download —
            # otherwise partial blobs accumulate and exhaust disk.
            _cleanup_revision_snapshots(size, revision)

    mx, mn, top = summarise_scores(scores, top_k=top_k)
    return CellResult(
        size=size,
        step=step,
        max_prefix_match=mx,
        mean_prefix_match=mn,
        top_heads=top,
        seed=seed,
        n_seqs=n_seqs,
        half_len=half_len,
    )


def run_grid(
    sizes: list[str],
    steps: list[int],
    device: str = "cpu",
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
    output_path: Path | str = "results/emergence_grid.jsonl",
    cleanup_after_cell: bool = True,
) -> Iterator[CellResult]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracking.init_run(
        project="circuitprobe",
        name="emergence_grid",
        config={
            "sizes": sizes,
            "steps": steps,
            "n_seqs": n_seqs,
            "half_len": half_len,
            "seed": seed,
        },
    )

    with output_path.open("w") as f:
        for size in sizes:
            for step in steps:
                cell = run_cell(
                    size=size, step=step, device=device,
                    n_seqs=n_seqs, half_len=half_len, seed=seed,
                    cleanup_after_cell=cleanup_after_cell,
                )
                f.write(json.dumps(asdict(cell)) + "\n")
                f.flush()
                tracking.log({
                    f"max_prefix_match/{size}": cell.max_prefix_match,
                    f"mean_prefix_match/{size}": cell.mean_prefix_match,
                    "step": step,
                })
                yield cell

    tracking.finish()


def grid_to_parquet(jsonl_path: Path | str, parquet_path: Path | str) -> None:
    import pandas as pd
    rows = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
