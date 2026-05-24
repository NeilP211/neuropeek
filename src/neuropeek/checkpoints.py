"""Pythia checkpoint enumeration and revision-pinning helpers.

Pythia (EleutherAI) ships dense training checkpoints. The published HF
revisions follow `step{N}` where N ranges over a log-spaced set of training
steps plus every 1000 steps from 1000..143000. We expose:

- `pythia_log_steps()`        — the full set of log-spaced + final checkpoints
- `select_emergence_steps(n)` — pick n log-spaced steps spanning the run
- `revision_for_step(step)`   — HF revision string
- `cache_root()`              — local checkpoint cache directory
"""

from __future__ import annotations

import os
from pathlib import Path

# Log-spaced early checkpoints published by EleutherAI for Pythia.
# Source: https://huggingface.co/EleutherAI/pythia-160m (Branches tab).
_PYTHIA_EARLY_STEPS: list[int] = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
]

# The remainder are every 1000 steps up to 143_000.
_PYTHIA_LATE_STEPS: list[int] = list(range(1000, 144000, 1000))


def pythia_log_steps() -> list[int]:
    return sorted(set(_PYTHIA_EARLY_STEPS + _PYTHIA_LATE_STEPS))


def select_emergence_steps(n: int = 12) -> list[int]:
    """Return n log-spaced steps from {0, 1, ..., 143_000} matching the published set."""
    if n < 2:
        raise ValueError("n must be >= 2")
    all_steps = pythia_log_steps()
    # We want endpoints fixed (0 and 143_000), and the interior log-spaced.
    import numpy as np

    log_targets = np.logspace(0, np.log10(143000), n - 1)
    chosen = [0]
    for t in log_targets:
        # snap to closest available checkpoint > previous
        candidates = [s for s in all_steps if s > chosen[-1]]
        if not candidates:
            break
        closest = min(candidates, key=lambda s: abs(s - t))
        chosen.append(closest)
    # Ensure last is 143_000.
    if chosen[-1] != 143000:
        chosen[-1] = 143000
    return chosen


def revision_for_step(step: int) -> str:
    if step < 0:
        raise ValueError("step must be >= 0")
    return f"step{step}"


def cache_root() -> Path:
    env = os.environ.get("CIRCUITPROBE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "neuropeek"
