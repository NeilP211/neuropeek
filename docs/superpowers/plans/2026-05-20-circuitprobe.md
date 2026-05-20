# CircuitProbe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce induction-head identification (Olsson '22) on Pythia 160M / 410M / 1.4B, then map the (model scale × training step) emergence grid, fit a scaling law, ship a Triton kernel for fused prefix-match score extraction, and write up the result LessWrong-style.

**Architecture:** A small Python package (`src/circuitprobe/`) wrapping TransformerLens. Modules are single-responsibility (`models.py` for loaders, `induction.py` for the scoring metrics, `patching.py` for interventions, `kernels/` for the Triton fused op, `viz/` for figures). A single emergence-grid runner (`emergence.py`) drives the (size × step) sweep and persists Parquet. A scaling-law fit module consumes that Parquet. CI runs CPU-only against the smallest model.

**Tech Stack:** Python 3.11+, uv, PyTorch, TransformerLens, Triton (CUDA-only kernel + Torch fallback), Plotly, CircuitsVis, wandb, pytest, ruff, GitHub Actions.

**Spec:** [`docs/superpowers/specs/2026-05-20-circuitprobe-design.md`](../specs/2026-05-20-circuitprobe-design.md)

---

## File Structure (Final State)

```
src/circuitprobe/
├── __init__.py                     # version, public re-exports
├── models.py                       # load_pythia / load_gpt2 → HookedTransformer
├── checkpoints.py                  # PYTHIA_LOG_STEPS, cache root, revision pinning
├── data.py                         # random_repeat_seqs, pile_sample, ioi_prompts
├── induction.py                    # prefix_match_score, copying_score, rank
├── patching.py                     # activation patching, head ablation context mgr
├── faithfulness.py                 # circuit_faithfulness, circuit_completeness
├── emergence.py                    # CellResult, run_cell, run_grid
├── scaling.py                      # fit_emergence_law, bootstrap CIs
├── tracking.py                     # wandb wrapper with offline fallback
├── kernels/
│   ├── __init__.py                 # backend selector (triton if CUDA else torch)
│   ├── prefix_match_torch.py       # PyTorch reference
│   └── prefix_match_triton.py      # Triton kernel
└── viz/
    ├── __init__.py
    ├── attention.py                # CircuitsVis wrappers
    └── emergence_plots.py          # Plotly emergence heatmap + scaling-law fig

scripts/
├── reproduce_olsson.py             # P4 driver
├── run_emergence_grid.py           # P7 driver
├── bench_kernel.py                 # P6 driver (Colab)
└── make_figures.py                 # P9/P10 driver

notebooks/
├── 01_olsson_reproduction.ipynb
├── 02_emergence_grid.ipynb
└── 03_kernel_benchmark.ipynb

tests/
├── conftest.py                     # shared fixtures, deterministic seeds
├── test_models.py
├── test_data.py
├── test_induction.py
├── test_patching.py
├── test_faithfulness.py
├── test_kernels.py
├── test_emergence.py
└── test_scaling.py

docs/writeup.md                     # LessWrong-style post (P10)
docs/METHODOLOGY.md                 # exact protocols + repro checklist (P10)
.github/workflows/ci.yml
Makefile
pyproject.toml
ruff.toml
```

---

# Phase 0 — Scaffolding

## Task 0.1: Initialise uv + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `ruff.toml`

- [ ] **Step 1:** Create `pyproject.toml`:

```toml
[project]
name = "circuitprobe"
version = "0.1.0"
description = "Mechanistic interpretability: reproducing + extending Olsson '22 induction heads on Pythia."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Neil Patel" }]
dependencies = [
    "torch>=2.3",
    "transformer-lens>=2.0",
    "transformers>=4.40",
    "datasets>=2.18",
    "einops>=0.7",
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=15",
    "plotly>=5.20",
    "circuitsvis>=1.43",
    "tqdm>=4.66",
    "wandb>=0.17",
    "scipy>=1.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "ruff>=0.4",
    "ipykernel>=6.29",
    "jupyter>=1.0",
]
triton = [
    "triton>=2.3 ; platform_system == 'Linux'",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/circuitprobe"]
```

- [ ] **Step 2:** Create `ruff.toml`:

```toml
line-length = 100
target-version = "py311"

[lint]
select = ["E", "F", "I", "B", "UP", "SIM", "W"]
ignore = ["E501"]

[lint.per-file-ignores]
"tests/*" = ["E402"]
"notebooks/*" = ["E402", "F401"]
```

- [ ] **Step 3:** Run uv sync:

```bash
cd ~/projects/circuitprobe && uv sync --extra dev
```

Expected: virtualenv created at `.venv/`, all deps installed, no errors.

- [ ] **Step 4:** Commit:

```bash
git add pyproject.toml ruff.toml
git commit -m "P0: pyproject + ruff config + initial uv sync"
git push
```

## Task 0.2: Source layout + Makefile

**Files:**
- Create: `src/circuitprobe/__init__.py`
- Create: `Makefile`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1:** Create `src/circuitprobe/__init__.py`:

```python
"""CircuitProbe: mechanistic interpretability on small LMs."""

__version__ = "0.1.0"
```

- [ ] **Step 2:** Create `tests/__init__.py` as an empty file.

- [ ] **Step 3:** Create `tests/conftest.py`:

```python
"""Shared pytest fixtures. Deterministic seeds for reproducibility."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    seed = 1623
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4:** Create `Makefile`:

```makefile
.PHONY: help install test lint format reproduce bench clean

help:
	@echo "Targets:"
	@echo "  install   - uv sync (dev extras)"
	@echo "  test      - run pytest"
	@echo "  lint      - ruff check"
	@echo "  format    - ruff format"
	@echo "  reproduce - regenerate headline figures from cached results"
	@echo "  bench     - run Triton kernel benchmark (requires CUDA)"
	@echo "  clean     - remove caches"

install:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

reproduce:
	uv run python scripts/make_figures.py

bench:
	uv run python scripts/bench_kernel.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
```

- [ ] **Step 5:** Add `results/.gitkeep` and `data/.gitkeep`:

```bash
mkdir -p results data && touch results/.gitkeep data/.gitkeep
```

- [ ] **Step 6:** Smoke-test:

```bash
cd ~/projects/circuitprobe && uv run python -c "import circuitprobe; print(circuitprobe.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 7:** Commit:

```bash
git add src/ tests/ Makefile results/.gitkeep data/.gitkeep
git commit -m "P0: src layout, Makefile, conftest with deterministic seeds"
git push
```

## Task 0.3: CI skeleton

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1:** Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Lint
        run: uv run ruff check src tests

      - name: Test
        run: uv run pytest -q -m "not slow and not gpu"
```

- [ ] **Step 2:** Commit:

```bash
git add .github/
git commit -m "P0: GitHub Actions CI (lint + fast tests)"
git push
```

- [ ] **Step 3:** Verify CI runs on GitHub. Expected: workflow shows on Actions tab; first run may fail because no tests exist yet — that's fine, we'll add one in Task 0.4.

## Task 0.4: First test (sanity)

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1:** Write the test:

```python
"""Smoke test: package importable and version accessible."""

import circuitprobe


def test_package_version():
    assert circuitprobe.__version__ == "0.1.0"


def test_package_importable():
    import circuitprobe  # noqa: F401
```

- [ ] **Step 2:** Run it:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: 2 passed.

- [ ] **Step 3:** Commit:

```bash
git add tests/test_smoke.py
git commit -m "P0: smoke test (package importable, version exposed)"
git push
```

---

# Phase 1 — Model + Checkpoint Loaders

## Task 1.1: Checkpoint enumeration

**Files:**
- Create: `src/circuitprobe/checkpoints.py`
- Create: `tests/test_checkpoints.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for Pythia checkpoint enumeration."""

import pytest

from circuitprobe import checkpoints


def test_pythia_log_steps_returns_non_empty_sorted_list():
    steps = checkpoints.pythia_log_steps()
    assert len(steps) >= 12
    assert steps == sorted(steps)
    assert steps[0] == 0


def test_pythia_log_steps_includes_final_step():
    # Pythia models train for 143_000 steps.
    steps = checkpoints.pythia_log_steps()
    assert 143000 in steps


def test_select_emergence_steps_returns_n_steps():
    selected = checkpoints.select_emergence_steps(n=12)
    assert len(selected) == 12
    assert selected[0] == 0
    assert selected[-1] == 143000


def test_select_emergence_steps_log_spaced():
    selected = checkpoints.select_emergence_steps(n=8)
    # Excluding the first (0), gaps should grow.
    gaps = [selected[i + 1] - selected[i] for i in range(1, len(selected) - 1)]
    assert all(g2 >= g1 for g1, g2 in zip(gaps[:-1], gaps[1:])), gaps


def test_revision_for_step_formats_correctly():
    assert checkpoints.revision_for_step(1000) == "step1000"
    assert checkpoints.revision_for_step(0) == "step0"


def test_cache_root_under_home():
    path = checkpoints.cache_root()
    assert ".cache" in str(path)
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_checkpoints.py -v
```

Expected: import error (`circuitprobe.checkpoints` does not exist).

- [ ] **Step 3:** Implement `src/circuitprobe/checkpoints.py`:

```python
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
    return Path.home() / ".cache" / "circuitprobe"
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_checkpoints.py -v
```

Expected: 6 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/checkpoints.py tests/test_checkpoints.py
git commit -m "P1: Pythia checkpoint enumeration + emergence-step selector"
git push
```

## Task 1.2: Model loader

**Files:**
- Create: `src/circuitprobe/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for TransformerLens model loaders.

These tests mark the network-dependent loader test as `slow` so CI skips it.
We test the size→hf-id mapping and validation on CPU without downloading.
"""

import pytest

from circuitprobe import models


def test_pythia_size_to_hf_id_known():
    assert models.pythia_hf_id("160M") == "EleutherAI/pythia-160m"
    assert models.pythia_hf_id("410M") == "EleutherAI/pythia-410m"
    assert models.pythia_hf_id("1.4B") == "EleutherAI/pythia-1.4b"


def test_pythia_size_to_hf_id_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown Pythia size"):
        models.pythia_hf_id("999B")


@pytest.mark.slow
def test_load_pythia_smallest_step_0_smoke():
    """Actually loads Pythia-160M @ step 0 from HF. Network + ~600MB. Marked slow."""
    model = models.load_pythia(size="160M", step=0, device="cpu")
    assert model.cfg.n_layers == 12
    assert model.cfg.d_model == 768
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_models.py -v -m "not slow"
```

Expected: import error.

- [ ] **Step 3:** Add `slow` marker to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: requires network or model download",
    "gpu: requires CUDA GPU",
]
```

- [ ] **Step 4:** Implement `src/circuitprobe/models.py`:

```python
"""TransformerLens model loaders for Pythia and GPT-2-small.

Pythia checkpoints are loaded via the `checkpoint_index` argument to
`HookedTransformer.from_pretrained`. We allow specifying a numeric step
(matching `revision="step{N}"` on HuggingFace) for emergence-grid sweeps.
"""

from __future__ import annotations

from typing import Literal

import torch
from transformer_lens import HookedTransformer

from . import checkpoints

PythiaSize = Literal["160M", "410M", "1.4B"]

_PYTHIA_HF_IDS: dict[str, str] = {
    "160M": "EleutherAI/pythia-160m",
    "410M": "EleutherAI/pythia-410m",
    "1.4B": "EleutherAI/pythia-1.4b",
}


def pythia_hf_id(size: str) -> str:
    if size not in _PYTHIA_HF_IDS:
        raise ValueError(f"Unknown Pythia size: {size!r}. Allowed: {list(_PYTHIA_HF_IDS)}")
    return _PYTHIA_HF_IDS[size]


def load_pythia(
    size: PythiaSize,
    step: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load Pythia-{size} at training step `step` (or final if None).

    Notes:
    - `step` must be a valid Pythia checkpoint (see `checkpoints.pythia_log_steps`).
    - Loads with `revision="step{N}"` via TransformerLens's HF passthrough.
    """
    hf_id = pythia_hf_id(size)
    revision = checkpoints.revision_for_step(step) if step is not None else None

    model = HookedTransformer.from_pretrained(
        hf_id,
        revision=revision,
        device=str(device),
        torch_dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


def load_gpt2_small(device: str | torch.device = "cpu") -> HookedTransformer:
    return HookedTransformer.from_pretrained("gpt2", device=str(device))
```

- [ ] **Step 5:** Run tests:

```bash
uv run pytest tests/test_models.py -v -m "not slow"
```

Expected: 2 passed (the smoke test is skipped).

- [ ] **Step 6:** Manually verify the slow test on M-series (once):

```bash
uv run pytest tests/test_models.py::test_load_pythia_smallest_step_0_smoke -v -m slow
```

Expected: passes, downloads ~600 MB first time.

- [ ] **Step 7:** Commit:

```bash
git add src/circuitprobe/models.py tests/test_models.py pyproject.toml
git commit -m "P1: Pythia + GPT-2 model loaders (TransformerLens)"
git push
```

---

# Phase 2 — Probe Data

## Task 2.1: Random-repeat sequences

**Files:**
- Create: `src/circuitprobe/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for probe data generators."""

import torch

from circuitprobe import data


def test_random_repeat_seqs_shape():
    seqs = data.random_repeat_seqs(vocab_size=1000, half_len=32, n_seqs=8, seed=0)
    assert seqs.shape == (8, 64)
    assert seqs.dtype == torch.long


def test_random_repeat_seqs_actually_repeats():
    seqs = data.random_repeat_seqs(vocab_size=1000, half_len=16, n_seqs=4, seed=0)
    first_half = seqs[:, :16]
    second_half = seqs[:, 16:]
    assert torch.equal(first_half, second_half)


def test_random_repeat_seqs_deterministic_with_seed():
    a = data.random_repeat_seqs(1000, 16, 4, seed=42)
    b = data.random_repeat_seqs(1000, 16, 4, seed=42)
    assert torch.equal(a, b)


def test_random_repeat_seqs_different_seeds_differ():
    a = data.random_repeat_seqs(1000, 16, 4, seed=1)
    b = data.random_repeat_seqs(1000, 16, 4, seed=2)
    assert not torch.equal(a, b)


def test_random_repeat_seqs_respects_vocab_bounds():
    seqs = data.random_repeat_seqs(vocab_size=100, half_len=16, n_seqs=4, seed=0)
    assert seqs.min().item() >= 0
    assert seqs.max().item() < 100
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_data.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/data.py`:

```python
"""Probe data generators for induction-head analysis.

The canonical induction-head probe is a sequence of the form
    r1 r2 r3 ... rN | r1 r2 r3 ... rN
where r_i are random tokens (sampled uniformly from a vocab range). Induction
heads should attend (in the second half) from position N+k to position k, and
copy the token there to the output. This produces a sharp "prefix-matching
score" that distinguishes induction heads from other heads.

We also expose a Pile sample for in-context-learning loss curves (P4).
"""

from __future__ import annotations

import hashlib
import random

import torch


def random_repeat_seqs(
    vocab_size: int,
    half_len: int,
    n_seqs: int,
    seed: int = 0,
    bos_token: int | None = None,
) -> torch.Tensor:
    """Random-repeat sequences: [r_1 ... r_N | r_1 ... r_N], shape (n_seqs, 2N).

    If `bos_token` is provided, it is prepended once at position 0 (shape becomes
    (n_seqs, 2N + 1)).
    """
    if half_len < 1 or n_seqs < 1 or vocab_size < 2:
        raise ValueError("require half_len>=1, n_seqs>=1, vocab_size>=2")

    g = torch.Generator()
    g.manual_seed(seed)
    half = torch.randint(0, vocab_size, (n_seqs, half_len), generator=g, dtype=torch.long)
    full = torch.cat([half, half], dim=1)
    if bos_token is not None:
        bos = torch.full((n_seqs, 1), bos_token, dtype=torch.long)
        full = torch.cat([bos, full], dim=1)
    return full


def pile_sample(n_seqs: int, max_len: int, seed: int = 0) -> list[str]:
    """Deterministic Pile sample for in-context-learning loss curves.

    Uses `monology/pile-uncopyrighted` (smaller, license-clean Pile subset).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )
    # Deterministic skip-based subsample.
    rng = random.Random(seed)
    skip = rng.randint(0, 10_000)
    picked: list[str] = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        text = row["text"]
        if len(text) < 200:
            continue
        picked.append(text[: max_len * 8])  # rough char->token ratio
        if len(picked) >= n_seqs:
            break
    return picked


def stable_hash(*args: object) -> str:
    """Deterministic hash of inputs for caching."""
    h = hashlib.sha256()
    for a in args:
        h.update(repr(a).encode())
    return h.hexdigest()[:16]
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_data.py -v
```

Expected: 5 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/data.py tests/test_data.py
git commit -m "P2: random-repeat probe sequences + Pile sample loader"
git push
```

---

# Phase 3 — Induction-Head Identification

## Task 3.1: Prefix-matching score

**Files:**
- Create: `src/circuitprobe/induction.py`
- Create: `tests/test_induction.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for induction-head identification.

We test the metric on a synthetic, hand-constructed attention tensor where
we know which heads "should" score high. No model loading required.
"""

import torch

from circuitprobe import induction


def make_synthetic_pattern(half_len: int, induction_head: bool) -> torch.Tensor:
    """Construct a (1, n_heads=1, seq_len=2N, seq_len=2N) attention pattern.

    If induction_head=True, attention from each position N+k attends to position k.
    Otherwise it attends uniformly.
    """
    seq_len = 2 * half_len
    pat = torch.zeros(1, 1, seq_len, seq_len)
    for q in range(seq_len):
        if induction_head and q >= half_len:
            target = q - half_len + 1  # offset-by-one induction (attends to next token of prev occurrence)
            target = min(target, seq_len - 1)
            pat[0, 0, q, target] = 1.0
        else:
            pat[0, 0, q, : q + 1] = 1.0 / (q + 1)
    return pat


def test_prefix_match_score_high_for_induction_pattern():
    pat = make_synthetic_pattern(half_len=8, induction_head=True)
    score = induction.prefix_match_score_from_pattern(pat, half_len=8)
    assert score.shape == (1, 1)
    assert score.item() > 0.8


def test_prefix_match_score_low_for_uniform_pattern():
    pat = make_synthetic_pattern(half_len=8, induction_head=False)
    score = induction.prefix_match_score_from_pattern(pat, half_len=8)
    assert score.item() < 0.2


def test_rank_induction_heads_orders_by_score():
    scores = torch.tensor([[0.1, 0.9], [0.5, 0.3]])
    ranked = induction.rank_induction_heads(scores, top_k=3)
    assert ranked[0] == (0, 1, 0.9)
    assert ranked[1] == (1, 0, 0.5)
    assert ranked[2] == (1, 1, 0.3)
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_induction.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/induction.py`:

```python
"""Induction-head identification metrics (prefix-matching + copying scores).

Definitions follow Olsson et al. (2022, "In-context Learning and Induction
Heads", appendix B).

prefix_match_score(layer, head):
    For random-repeat sequences r_1..r_N | r_1..r_N:
    measure the mean attention, in the second half, from position N+k+1 to
    position k+1 (i.e., the position that holds the token immediately after
    a repeat). Higher score means the head is implementing the induction
    pattern "look at where this token previously appeared, then look at the
    token right after it".

copying_score(layer, head):
    Approximate measure: probability that the OV-circuit of this head, applied
    to a fixed embedding, increases the logit of the same token. Implemented
    as the diagonal-fraction of W_E @ W_OV @ W_U for this head.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from . import data


def prefix_match_score_from_pattern(
    pattern: torch.Tensor,
    half_len: int,
) -> torch.Tensor:
    """Compute prefix-match score from a pre-computed attention pattern.

    pattern: (batch, n_heads, seq_len, seq_len) where seq_len == 2*half_len
             (assumes NO BOS token; if BOS used, strip it before calling).

    Returns: (batch, n_heads) — mean attention from query position q in the
             second half (q >= half_len) to key position q - half_len + 1.
    """
    batch, n_heads, seq_len, _ = pattern.shape
    assert seq_len == 2 * half_len, f"seq_len {seq_len} != 2*half_len {2 * half_len}"

    # For each q in [half_len, 2*half_len - 1], the target key is q - half_len + 1
    # (clamped). We gather attention[:, :, q, q - half_len + 1] across all q in
    # the second half.
    q_indices = torch.arange(half_len, seq_len - 1)  # exclude last (target would overflow)
    k_indices = q_indices - half_len + 1

    # Gather: result[b, h, i] = pattern[b, h, q_i, k_i]
    gathered = pattern[:, :, q_indices, k_indices]  # (batch, n_heads, len(q_indices))
    return gathered.mean(dim=-1)


@torch.no_grad()
def prefix_match_score(
    model: HookedTransformer,
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
) -> torch.Tensor:
    """Per-(layer, head) prefix-matching score on random-repeat sequences.

    Returns: (n_layers, n_heads) tensor on CPU.
    """
    vocab = model.cfg.d_vocab
    seqs = data.random_repeat_seqs(
        vocab_size=vocab,
        half_len=half_len,
        n_seqs=n_seqs,
        seed=seed,
        bos_token=model.tokenizer.bos_token_id if model.tokenizer is not None else None,
    ).to(model.cfg.device)

    bos_offset = 1 if model.tokenizer is not None and model.tokenizer.bos_token_id is not None else 0

    _, cache = model.run_with_cache(
        seqs,
        return_type=None,
        names_filter=lambda name: name.endswith(".hook_pattern"),
    )

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        pat = cache[f"blocks.{layer}.attn.hook_pattern"]  # (batch, n_heads, seq, seq)
        # Strip BOS if present.
        if bos_offset:
            pat = pat[:, :, bos_offset:, bos_offset:]
        s = prefix_match_score_from_pattern(pat, half_len=half_len)  # (batch, n_heads)
        scores[layer] = s.mean(dim=0).cpu()

    return scores


@torch.no_grad()
def copying_score(model: HookedTransformer) -> torch.Tensor:
    """Per-(layer, head) copying score.

    Defined as the fraction of the diagonal of (W_E @ W_OV @ W_U) that is
    positive (Olsson '22, appendix B.2). This is a model-only computation —
    no data required.

    Returns: (n_layers, n_heads) tensor on CPU.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)

    W_E = model.W_E  # (vocab, d_model)
    W_U = model.W_U  # (d_model, vocab)

    for layer in range(n_layers):
        W_V = model.W_V[layer]  # (n_heads, d_model, d_head)
        W_O = model.W_O[layer]  # (n_heads, d_head, d_model)
        for head in range(n_heads):
            W_OV = W_V[head] @ W_O[head]  # (d_model, d_model)
            full = W_E @ W_OV @ W_U  # (vocab, vocab)
            diag = full.diag()
            scores[layer, head] = (diag > 0).float().mean().cpu()

    return scores


def rank_induction_heads(
    scores: torch.Tensor,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Return the top-k (layer, head, score) tuples sorted by score descending."""
    n_layers, n_heads = scores.shape
    flat = [(layer, head, scores[layer, head].item())
            for layer in range(n_layers) for head in range(n_heads)]
    flat.sort(key=lambda x: x[2], reverse=True)
    return flat[:top_k]
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_induction.py -v
```

Expected: 3 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/induction.py tests/test_induction.py
git commit -m "P3: induction-head metrics (prefix-match + copying score + ranking)"
git push
```

## Task 3.2: End-to-end identification on real Pythia-160M (manual smoke)

**Files:**
- Create: `scripts/identify_pythia_160m.py`

- [ ] **Step 1:** Create the script:

```python
"""Identify induction heads on Pythia-160M (final checkpoint). Manual smoke test.

Run: uv run python scripts/identify_pythia_160m.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from circuitprobe import induction, models


def main() -> None:
    print("Loading Pythia-160M (final step)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = models.load_pythia(size="160M", step=None, device=device)

    print("Computing prefix-matching score (64 seqs of half_len=50)...")
    pm = induction.prefix_match_score(model, n_seqs=64, half_len=50, seed=0)

    print("Computing copying score...")
    cp = induction.copying_score(model)

    top = induction.rank_induction_heads(pm, top_k=10)
    print("\nTop 10 induction heads by prefix-matching score:")
    for layer, head, score in top:
        cp_h = cp[layer, head].item()
        print(f"  L{layer}H{head}: prefix_match={score:.3f}, copying={cp_h:.3f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/pythia_160m_heads.json").write_text(
        json.dumps(
            {
                "prefix_match": pm.tolist(),
                "copying": cp.tolist(),
                "top_10": [{"layer": l, "head": h, "score": s} for l, h, s in top],
            },
            indent=2,
        )
    )
    print("\nResults written to results/pythia_160m_heads.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run it (one-time manual; expect ~5 min):

```bash
uv run python scripts/identify_pythia_160m.py
```

Expected: prints top-10 heads; at least one head has prefix_match > 0.5
(Pythia-160M has known induction heads around L4-L6).

- [ ] **Step 3:** Commit (the script + the resulting JSON):

```bash
git add scripts/identify_pythia_160m.py results/pythia_160m_heads.json
git commit -m "P3: identification script + first Pythia-160M induction-head dump"
git push
```

---

# Phase 4 — Olsson Baseline Reproduction

## Task 4.1: In-context-learning loss-by-position

**Files:**
- Create: `src/circuitprobe/icl.py`
- Create: `tests/test_icl.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for in-context-learning loss-by-position."""

import torch

from circuitprobe import icl


def test_loss_by_position_shape():
    # Fake logits + tokens.
    n_seqs, seq_len, vocab = 4, 16, 100
    torch.manual_seed(0)
    logits = torch.randn(n_seqs, seq_len, vocab)
    tokens = torch.randint(0, vocab, (n_seqs, seq_len))
    loss = icl.loss_by_position(logits, tokens)
    assert loss.shape == (seq_len - 1,)


def test_loss_by_position_lower_for_repeated_token():
    # Construct a deterministic case: vocab=2, second half repeats the first.
    seq_len = 8
    half = torch.tensor([[0, 1, 0, 1]])
    tokens = torch.cat([half, half], dim=1)  # (1, 8)
    # Logits that predict perfectly in second half.
    logits = torch.full((1, seq_len, 2), -10.0)
    for t in range(seq_len - 1):
        next_tok = tokens[0, t + 1].item()
        logits[0, t, next_tok] = 10.0
    loss = icl.loss_by_position(logits, tokens)
    assert loss[3:].mean().item() < 0.01
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_icl.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/icl.py`:

```python
"""In-context-learning curves: per-position cross-entropy loss.

The diagnostic plot from Olsson '22 figure 1: average per-token loss as a
function of position in the sequence. As context accumulates, well-trained
models develop a sharp drop in loss at certain positions — interpreted as
in-context-learning kicking in. We reproduce this curve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from . import data


def loss_by_position(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy loss averaged over the batch.

    logits: (batch, seq, vocab) — predictions for positions 0..seq-1.
    tokens: (batch, seq)        — the actual tokens.

    Returns (seq - 1,) — mean CE loss at each predicted position (1..seq-1).
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq-1, vocab)
    targets = tokens[:, 1:]  # (batch, seq-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq-1)
    return nll.mean(dim=0)


@torch.no_grad()
def icl_curve_on_pile(
    model: HookedTransformer,
    n_seqs: int = 32,
    max_len: int = 512,
    seed: int = 0,
) -> torch.Tensor:
    """Compute the in-context-learning loss-by-position curve on a Pile sample."""
    texts = data.pile_sample(n_seqs=n_seqs, max_len=max_len, seed=seed)
    tokens = model.to_tokens(texts, prepend_bos=True)
    tokens = tokens[:, :max_len]
    logits = model(tokens)
    return loss_by_position(logits, tokens).cpu()
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_icl.py -v
```

Expected: 2 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/icl.py tests/test_icl.py
git commit -m "P4: in-context-learning loss-by-position metric"
git push
```

## Task 4.2: Reproduce-Olsson script + figure

**Files:**
- Create: `scripts/reproduce_olsson.py`

- [ ] **Step 1:** Create the driver:

```python
"""Reproduce Olsson '22 figure 1-style ICL loss-by-position curve.

Runs Pythia-410M on a Pile sample and dumps the curve + a Plotly figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import torch

from circuitprobe import icl, models


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
```

- [ ] **Step 2:** Add `kaleido` to dev deps for static PNG export:

```bash
uv add --dev kaleido
```

- [ ] **Step 3:** Run it:

```bash
uv run python scripts/reproduce_olsson.py
```

Expected: curve shows the characteristic drop in loss after position ~50,
consistent with in-context learning kicking in.

- [ ] **Step 4:** Commit:

```bash
git add scripts/reproduce_olsson.py results/icl_curve_pythia_410m.{json,html,png} pyproject.toml uv.lock
git commit -m "P4: reproduce Olsson '22-style ICL loss-by-position on Pythia-410M"
git push
```

---

# Phase 5 — Patching + Faithfulness

## Task 5.1: Head ablation context manager + activation patching

**Files:**
- Create: `src/circuitprobe/patching.py`
- Create: `tests/test_patching.py`

- [ ] **Step 1:** Write the failing test (with a real but tiny model — GPT-2-small):

```python
"""Tests for patching + ablation. Uses GPT-2-small for speed."""

import pytest
import torch

from circuitprobe import patching


@pytest.fixture(scope="module")
def gpt2_small():
    from circuitprobe import models
    return models.load_gpt2_small(device="cpu")


def test_head_ablate_changes_logits(gpt2_small):
    model = gpt2_small
    tokens = model.to_tokens("The quick brown fox")
    base_logits = model(tokens)

    with patching.ablate_heads(model, heads=[(5, 1)]):
        ablated_logits = model(tokens)

    diff = (base_logits - ablated_logits).abs().sum().item()
    assert diff > 0.0, "ablating a head should change logits"


def test_head_ablate_restores_after_context(gpt2_small):
    model = gpt2_small
    tokens = model.to_tokens("The quick brown fox")
    base_logits = model(tokens)

    with patching.ablate_heads(model, heads=[(5, 1)]):
        _ = model(tokens)

    after_logits = model(tokens)
    assert torch.allclose(base_logits, after_logits, atol=1e-5)


def test_zero_ablate_zeroes_head_z(gpt2_small):
    model = gpt2_small
    tokens = model.to_tokens("hi")
    with patching.ablate_heads(model, heads=[(0, 0)], mode="zero"):
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
    assert cache["blocks.0.attn.hook_z"][:, :, 0, :].abs().sum().item() == 0.0
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_patching.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/patching.py`:

```python
"""Activation patching, head ablation, and path patching utilities.

We expose two primary primitives:

- `ablate_heads(model, heads, mode="zero")` — context manager that registers
  forward hooks zeroing (or mean-ablating) the `z` output of specified heads.
  Cleans up on exit.
- `path_patch(model, clean_tokens, corrupt_tokens, sender, receiver)` — runs
  the model on clean tokens, caches the sender activation, then runs on
  corrupt tokens replacing the sender's activation with the cached clean
  value, and returns the receiver activation under that patch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


@contextmanager
def ablate_heads(
    model: HookedTransformer,
    heads: list[tuple[int, int]],
    mode: Literal["zero", "mean"] = "zero",
    mean_activations: dict[int, torch.Tensor] | None = None,
) -> Iterator[None]:
    """Context manager that zero- or mean-ablates the listed (layer, head)s.

    Hooks the `hook_z` output of each head.
    """
    if mode == "mean" and mean_activations is None:
        raise ValueError("mode='mean' requires `mean_activations` keyed by layer")

    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    handles = []
    try:
        for layer, head_list in by_layer.items():
            def make_hook(heads_in_layer: list[int], layer_idx: int):
                def hook(z: torch.Tensor, hook: HookPoint):
                    # z shape: (batch, seq, n_heads, d_head)
                    if mode == "zero":
                        z[:, :, heads_in_layer, :] = 0.0
                    else:
                        mean_z = mean_activations[layer_idx]  # (n_heads, d_head)
                        for h in heads_in_layer:
                            z[:, :, h, :] = mean_z[h]
                    return z
                return hook

            hp = model.blocks[layer].attn.hook_z
            handle = hp.add_hook(make_hook(head_list, layer))
            handles.append((hp, handle))
        yield
    finally:
        for hp, _ in handles:
            hp.remove_hooks()


@torch.no_grad()
def path_patch(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    sender_hook: str,
    receiver_hook: str,
) -> torch.Tensor:
    """Path-patch: replace `sender_hook` on the corrupt run with its clean value,
    then return the activation at `receiver_hook`.

    Both hook names are TransformerLens hook strings (e.g.
    "blocks.5.attn.hook_z").
    """
    _, clean_cache = model.run_with_cache(
        clean_tokens, names_filter=lambda n: n == sender_hook
    )
    clean_sender = clean_cache[sender_hook]

    def patch_hook(activation: torch.Tensor, hook: HookPoint):
        return clean_sender

    receiver_value: list[torch.Tensor] = []

    def grab_receiver(activation: torch.Tensor, hook: HookPoint):
        receiver_value.append(activation.detach().clone())
        return activation

    model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(sender_hook, patch_hook), (receiver_hook, grab_receiver)],
        return_type=None,
    )
    assert receiver_value, "receiver hook never fired"
    return receiver_value[0]
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_patching.py -v
```

Expected: 3 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/patching.py tests/test_patching.py
git commit -m "P5: head ablation context mgr + path patching primitive"
git push
```

## Task 5.2: Circuit faithfulness metric

**Files:**
- Create: `src/circuitprobe/faithfulness.py`
- Create: `tests/test_faithfulness.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for circuit faithfulness."""

import torch

from circuitprobe import faithfulness


def test_faithfulness_from_losses():
    # Faithfulness = (loss_ablate_others - loss_full) / (loss_ablate_all - loss_full)
    # If ablating just-non-circuit-heads ≈ ablating-everything, faithfulness ≈ 1.0
    f = faithfulness.faithfulness_from_losses(
        loss_full=2.0,
        loss_ablate_others=10.0,
        loss_ablate_all=10.0,
    )
    assert abs(f - 1.0) < 1e-6


def test_faithfulness_zero_when_others_match_full():
    f = faithfulness.faithfulness_from_losses(
        loss_full=2.0,
        loss_ablate_others=2.0,
        loss_ablate_all=10.0,
    )
    assert abs(f - 0.0) < 1e-6
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_faithfulness.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/faithfulness.py`:

```python
"""Circuit faithfulness scoring.

Given:
- L_full   = loss with the full model
- L_others = loss when we ablate everything *except* the proposed circuit
- L_all    = loss when we ablate everything (i.e., remove all of attention)

Define:
    faithfulness = (L_others - L_full) / (L_all - L_full)

Intuition: if our proposed circuit fully captures the behaviour, ablating
non-circuit components shouldn't matter (L_others ≈ L_full) → faithfulness ≈ 0.

NOTE: we use the *complement*-style definition used by Wang et al. 2022 and
Conmy et al. 2023 — faithfulness near 0 means "circuit suffices"; near 1
means "circuit insufficient". (Some authors invert this; we follow the
Conmy convention.)

We additionally expose a "recovery" form: 1 - faithfulness, so that closer
to 1 means "circuit captures behaviour", which reads better in figures.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from .induction import prefix_match_score


def faithfulness_from_losses(
    loss_full: float,
    loss_ablate_others: float,
    loss_ablate_all: float,
) -> float:
    denom = loss_ablate_all - loss_full
    if abs(denom) < 1e-9:
        return 0.0
    return (loss_ablate_others - loss_full) / denom


def recovery(faithfulness: float) -> float:
    """1 - faithfulness, clamped to [0, 1]. Reads as 'circuit captures behaviour'."""
    return max(0.0, min(1.0, 1.0 - faithfulness))


@torch.no_grad()
def circuit_prefix_match_recovery(
    model: HookedTransformer,
    circuit_heads: list[tuple[int, int]],
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Measure how much of the prefix-matching behaviour is recovered by
    the proposed circuit.

    Returns (mean_full, mean_circuit_only, mean_ablated_circuit).

    - mean_full: average prefix-match score across all heads (baseline reference).
    - mean_circuit_only: same metric but with NON-circuit heads ablated.
    - mean_ablated_circuit: same metric but with circuit heads ablated (sanity).
    """
    from .patching import ablate_heads

    pm_full = prefix_match_score(model, n_seqs=n_seqs, half_len=half_len, seed=seed)
    circuit_mean_full = float(sum(pm_full[l, h].item() for l, h in circuit_heads)) / max(
        len(circuit_heads), 1
    )

    all_heads = [
        (l, h)
        for l in range(model.cfg.n_layers)
        for h in range(model.cfg.n_heads)
        if (l, h) not in set(circuit_heads)
    ]
    with ablate_heads(model, heads=all_heads, mode="zero"):
        pm_circuit_only = prefix_match_score(model, n_seqs=n_seqs, half_len=half_len, seed=seed)
    circuit_only_mean = float(sum(pm_circuit_only[l, h].item() for l, h in circuit_heads)) / max(
        len(circuit_heads), 1
    )

    with ablate_heads(model, heads=list(circuit_heads), mode="zero"):
        pm_ablated = prefix_match_score(model, n_seqs=n_seqs, half_len=half_len, seed=seed)
    ablated_mean = float(pm_ablated.mean().item())

    return circuit_mean_full, circuit_only_mean, ablated_mean
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_faithfulness.py -v
```

Expected: 2 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/faithfulness.py tests/test_faithfulness.py
git commit -m "P5: circuit faithfulness + prefix-match recovery"
git push
```

---

# Phase 6 — Triton Kernel

## Task 6.1: PyTorch reference impl + benchmark harness

**Files:**
- Create: `src/circuitprobe/kernels/__init__.py`
- Create: `src/circuitprobe/kernels/prefix_match_torch.py`
- Create: `tests/test_kernels.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for the prefix-match kernel backends."""

import pytest
import torch

from circuitprobe.kernels import prefix_match_torch


def test_torch_backend_correct_on_synthetic():
    # (batch=1, n_heads=1, seq=4, seq=4); seq=4, half_len=2
    pat = torch.zeros(1, 1, 4, 4)
    pat[0, 0, 2, 1] = 1.0  # q=2 -> k=2-2+1=1 ✓
    pat[0, 0, 3, 2] = 1.0  # q=3 -> k=3-2+1=2 ✓
    score = prefix_match_torch.prefix_match_score(pat, half_len=2)
    # Last valid q is seq_len - 1 = 3; q range is half_len=2 to seq_len-2=2.
    # So only q=2 counted; k=1; score = 1.0
    assert score.shape == (1, 1)
    assert score.item() == pytest.approx(1.0)


def test_torch_backend_zero_pattern_gives_zero_score():
    pat = torch.zeros(1, 4, 16, 16)
    score = prefix_match_torch.prefix_match_score(pat, half_len=8)
    assert torch.allclose(score, torch.zeros(1, 4))


def test_torch_backend_matches_induction_module_impl():
    # The kernel backend MUST agree with the reference impl in induction.py.
    from circuitprobe.induction import prefix_match_score_from_pattern
    torch.manual_seed(0)
    pat = torch.rand(2, 8, 32, 32)
    a = prefix_match_score_from_pattern(pat, half_len=16)
    b = prefix_match_torch.prefix_match_score(pat, half_len=16)
    assert torch.allclose(a, b, atol=1e-6)
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_kernels.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/kernels/prefix_match_torch.py`:

```python
"""PyTorch eager reference for the fused prefix-match score kernel.

Signature mirrors the Triton variant: takes an attention pattern tensor and
half-length, returns (batch, n_heads) score. This is the implementation the
Triton kernel must match for correctness tests.
"""

from __future__ import annotations

import torch


def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:
    """Reference PyTorch impl.

    pattern: (batch, n_heads, seq_len, seq_len), seq_len == 2 * half_len.
    returns: (batch, n_heads).
    """
    batch, n_heads, seq_len, _ = pattern.shape
    assert seq_len == 2 * half_len

    q_indices = torch.arange(half_len, seq_len - 1, device=pattern.device)
    k_indices = q_indices - half_len + 1
    gathered = pattern[:, :, q_indices, k_indices]
    return gathered.mean(dim=-1)
```

- [ ] **Step 4:** Implement `src/circuitprobe/kernels/__init__.py`:

```python
"""Kernel-backend selector.

`prefix_match_score(pattern, half_len)` dispatches to the Triton kernel when
CUDA is available, otherwise to the PyTorch reference.
"""

from __future__ import annotations

import torch

from .prefix_match_torch import prefix_match_score as _torch_impl


def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:
    if pattern.is_cuda:
        try:
            from .prefix_match_triton import prefix_match_score as _triton_impl
            return _triton_impl(pattern, half_len)
        except ImportError:
            pass
    return _torch_impl(pattern, half_len)
```

- [ ] **Step 5:** Run tests:

```bash
uv run pytest tests/test_kernels.py -v
```

Expected: 3 passed.

- [ ] **Step 6:** Commit:

```bash
git add src/circuitprobe/kernels/ tests/test_kernels.py
git commit -m "P6: PyTorch reference for prefix-match score + backend selector"
git push
```

## Task 6.2: Triton kernel

**Files:**
- Create: `src/circuitprobe/kernels/prefix_match_triton.py`
- Create: `scripts/bench_kernel.py`
- Create: `notebooks/03_kernel_benchmark.ipynb` (Colab)

- [ ] **Step 1:** Implement `src/circuitprobe/kernels/prefix_match_triton.py`:

```python
"""Triton kernel for fused prefix-match score extraction.

Fuses the index-gather + mean-reduction along the (q, k) diagonal in a
single launch, eliminating the intermediate (batch, n_heads, n_q) tensor.
Runs only on CUDA. Falls back to PyTorch via the backend selector.

Layout:
    pattern: (B, H, S, S), float32 or float16, contiguous.
    output : (B, H), float32.

Each program instance handles one (batch, head) pair. We stride over the
n_q = half_len - 1 positions and accumulate, then divide.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _prefix_match_kernel(
        pattern_ptr,      # *fp16/fp32, shape (B*H, S*S)
        out_ptr,          # *fp32, shape (B*H,)
        n_q,              # int — number of q positions to average over
        half_len,         # int — sequence half length
        S,                # int — full seq_len (= 2*half_len)
        BLOCK_Q: tl.constexpr,
    ):
        bh_id = tl.program_id(0)

        # Per-program accumulator.
        acc = tl.zeros((), dtype=tl.float32)

        # Stride through n_q positions in chunks of BLOCK_Q.
        for q_start in tl.range(0, n_q, BLOCK_Q):
            q_offsets = q_start + tl.arange(0, BLOCK_Q)
            mask = q_offsets < n_q

            # For each q in [half_len, half_len + n_q - 1], gather pattern[q, q-half_len+1].
            q_idx = q_offsets + half_len
            k_idx = q_offsets + 1  # = q_idx - half_len + 1

            flat_offsets = bh_id * (S * S) + q_idx * S + k_idx
            vals = tl.load(pattern_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(vals, axis=0)

        mean = acc / n_q
        tl.store(out_ptr + bh_id, mean)


    def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:
        assert pattern.is_cuda, "Triton kernel requires CUDA"
        B, H, S, _ = pattern.shape
        assert S == 2 * half_len
        n_q = half_len - 1  # q runs in [half_len, S - 2]; len = half_len - 1

        pattern_c = pattern.contiguous()
        out = torch.empty(B * H, device=pattern.device, dtype=torch.float32)

        grid = (B * H,)
        _prefix_match_kernel[grid](
            pattern_c.view(-1),
            out,
            n_q,
            half_len,
            S,
            BLOCK_Q=64,
            num_warps=2,
        )
        return out.view(B, H)

else:

    def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:  # pragma: no cover
        raise ImportError("Triton is not installed")
```

- [ ] **Step 2:** Implement `scripts/bench_kernel.py`:

```python
"""Benchmark the Triton vs PyTorch prefix-match kernel on CUDA.

Run on a Colab T4: `uv run python scripts/bench_kernel.py`.
Reports median + IQR ms across N trials at several shapes, plus the speedup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from circuitprobe.kernels import prefix_match_torch


def _benchmark(fn, *args, warmup: int = 5, trials: int = 50) -> tuple[float, float]:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    iqr = samples[3 * len(samples) // 4] - samples[len(samples) // 4]
    return median, iqr


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for benchmarking.")
    from circuitprobe.kernels import prefix_match_triton  # noqa: F401

    results = []
    shapes = [
        (16, 12, 128),
        (8, 16, 256),
        (4, 32, 512),
    ]
    for B, H, S in shapes:
        half = S // 2
        pat = torch.rand(B, H, S, S, device="cuda", dtype=torch.float16)

        # Correctness first.
        ref = prefix_match_torch.prefix_match_score(pat.float(), half)
        tri = prefix_match_triton.prefix_match_score(pat, half)
        max_err = (ref - tri).abs().max().item()
        assert max_err < 1e-3, f"correctness failed: max_err={max_err}"

        t_torch, iqr_torch = _benchmark(
            prefix_match_torch.prefix_match_score, pat.float(), half
        )
        t_tri, iqr_tri = _benchmark(
            prefix_match_triton.prefix_match_score, pat, half
        )
        speedup = t_torch / t_tri
        results.append(
            {
                "shape": (B, H, S),
                "torch_ms": t_torch,
                "torch_iqr_ms": iqr_torch,
                "triton_ms": t_tri,
                "triton_iqr_ms": iqr_tri,
                "speedup": speedup,
                "max_err": max_err,
            }
        )
        print(f"shape={(B, H, S)}  torch={t_torch:.4f}ms  triton={t_tri:.4f}ms  "
              f"speedup={speedup:.2f}x  err={max_err:.2e}")

    Path("results").mkdir(exist_ok=True)
    Path("results/kernel_benchmark.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** Create `notebooks/03_kernel_benchmark.ipynb` (Colab notebook, see template in repo). It clones the repo, `uv pip install -e .[triton]`, and runs `scripts/bench_kernel.py`.

(See the README for the Colab one-cell snippet; this notebook is generated by the script template in `scripts/make_figures.py` once results are present.)

- [ ] **Step 4:** Run benchmark on Colab T4 (manual, ~5 min). Capture `results/kernel_benchmark.json`.

- [ ] **Step 5:** Locally verify the file exists and parses; check speedup ≥ 1.5×:

```bash
uv run python -c "import json; r=json.load(open('results/kernel_benchmark.json')); print([x['speedup'] for x in r]); assert all(x['speedup'] >= 1.5 for x in r), 'speedup target missed'"
```

Expected: prints speedups ≥ 1.5, no assertion error.

- [ ] **Step 6:** Commit:

```bash
git add src/circuitprobe/kernels/prefix_match_triton.py scripts/bench_kernel.py \
        notebooks/03_kernel_benchmark.ipynb results/kernel_benchmark.json
git commit -m "P6: Triton fused prefix-match kernel + Colab T4 benchmark"
git push
```

---

# Phase 7 — Emergence Grid

## Task 7.1: Tracking wrapper

**Files:**
- Create: `src/circuitprobe/tracking.py`
- Create: `tests/test_tracking.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for the wandb tracking wrapper."""

import os

from circuitprobe import tracking


def test_init_offline_when_no_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    run = tracking.init_run(project="circuitprobe-test", config={"foo": 1})
    assert run is not None
    tracking.log({"metric": 0.5})
    tracking.finish()


def test_log_noops_without_init():
    tracking.finish()  # ensure no active run
    tracking.log({"x": 1})  # must not raise
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_tracking.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/tracking.py`:

```python
"""Thin wandb wrapper with graceful offline fallback.

If `WANDB_API_KEY` is not set, runs in `offline` mode (data persisted under
`./wandb/` for later sync). `log()` and `finish()` are no-ops if no run is
active.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import wandb
    _HAVE_WANDB = True
except ImportError:
    _HAVE_WANDB = False

_RUN: Any = None


def init_run(project: str, config: dict[str, Any] | None = None, name: str | None = None) -> Any:
    global _RUN
    if not _HAVE_WANDB:
        return None
    if "WANDB_API_KEY" not in os.environ and "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    _RUN = wandb.init(project=project, config=config or {}, name=name, reinit=True)
    return _RUN


def log(payload: dict[str, Any]) -> None:
    if _RUN is None or not _HAVE_WANDB:
        return
    wandb.log(payload)


def finish() -> None:
    global _RUN
    if _RUN is None or not _HAVE_WANDB:
        return
    wandb.finish()
    _RUN = None
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_tracking.py -v
```

Expected: 2 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/tracking.py tests/test_tracking.py
git commit -m "P7: wandb tracking wrapper with offline fallback"
git push
```

## Task 7.2: Emergence-grid runner

**Files:**
- Create: `src/circuitprobe/emergence.py`
- Create: `tests/test_emergence.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for emergence-grid runner (mocked model loads)."""

from dataclasses import asdict

import torch

from circuitprobe import emergence


def test_cell_result_is_serialisable():
    cell = emergence.CellResult(
        size="160M",
        step=1000,
        max_prefix_match=0.42,
        mean_prefix_match=0.05,
        top_heads=[(4, 7, 0.42), (5, 1, 0.41)],
        seed=0,
    )
    d = asdict(cell)
    assert d["size"] == "160M"
    assert d["step"] == 1000
    assert len(d["top_heads"]) == 2


def test_summarise_scores_picks_max_and_mean():
    scores = torch.tensor([[0.1, 0.3], [0.4, 0.2]])
    mx, mn, top = emergence.summarise_scores(scores, top_k=2)
    assert mx == 0.4
    assert mn == 0.25
    assert top[0] == (1, 0, 0.4)
    assert top[1] == (0, 1, 0.3)
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_emergence.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/emergence.py`:

```python
"""Emergence-grid runner.

Sweeps (model size × training step), measures the per-head prefix-match
score on random-repeat sequences, and persists one row per cell.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

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
    return float(scores.max().item()), float(scores.mean().item()), top


def run_cell(
    size: str,
    step: int,
    device: str = "cpu",
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
    top_k: int = 5,
) -> CellResult:
    model = models.load_pythia(size=size, step=step, device=device)
    try:
        scores = induction.prefix_match_score(
            model, n_seqs=n_seqs, half_len=half_len, seed=seed
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_emergence.py -v
```

Expected: 2 passed.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/emergence.py tests/test_emergence.py
git commit -m "P7: emergence-grid runner (CellResult, run_cell, run_grid)"
git push
```

## Task 7.3: Run the actual grid

**Files:**
- Create: `scripts/run_emergence_grid.py`

- [ ] **Step 1:** Implement `scripts/run_emergence_grid.py`:

```python
"""Run the (model_size x training_step) emergence grid for CircuitProbe.

Output: results/emergence_grid.jsonl  (one cell per line)
        results/emergence_grid.parquet (consolidated)
"""

from __future__ import annotations

from pathlib import Path

import torch

from circuitprobe import checkpoints, emergence


def main() -> None:
    sizes = ["160M", "410M", "1.4B"]
    steps = checkpoints.select_emergence_steps(n=12)
    print(f"Running grid: sizes={sizes} steps={steps}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    output = Path("results/emergence_grid.jsonl")
    cells = list(
        emergence.run_grid(
            sizes=sizes, steps=steps, device=device,
            n_seqs=64, half_len=50, seed=0,
            output_path=output,
        )
    )

    parquet = Path("results/emergence_grid.parquet")
    emergence.grid_to_parquet(output, parquet)

    print(f"\nDone: {len(cells)} cells. Parquet at {parquet}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run it (long-running; ~3-4 hours total):

```bash
uv run python scripts/run_emergence_grid.py
```

Expected: 36 cells written; final parquet exists; wandb run page populated
(or `wandb/` offline dir if no API key set).

- [ ] **Step 3:** Commit:

```bash
git add scripts/run_emergence_grid.py results/emergence_grid.jsonl results/emergence_grid.parquet
git commit -m "P7: run emergence grid (3 sizes x 12 steps); persist Parquet"
git push
```

---

# Phase 8 — Scaling-Law Analysis

## Task 8.1: Bootstrap CI + scaling-law fit

**Files:**
- Create: `src/circuitprobe/scaling.py`
- Create: `tests/test_scaling.py`

- [ ] **Step 1:** Write the failing test:

```python
"""Tests for scaling-law fits."""

import numpy as np

from circuitprobe import scaling


def test_bootstrap_ci_returns_expected_shape():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=1.0, size=200)
    mean, lo, hi = scaling.bootstrap_ci(samples, statistic=np.mean, n_boot=200, alpha=0.05, seed=0)
    assert lo < mean < hi
    assert abs(mean - 5.0) < 0.5


def test_emergence_threshold_step_finds_first_crossing():
    # Curve: 0.0 0.1 0.2 0.4 0.6 — threshold 0.3 first crossed at index 3.
    steps = [0, 100, 1000, 10_000, 100_000]
    values = [0.0, 0.1, 0.2, 0.4, 0.6]
    idx = scaling.emergence_threshold_step(steps, values, threshold=0.3)
    assert idx == 10_000


def test_emergence_threshold_step_none_when_never_crossed():
    steps = [0, 100, 1000]
    values = [0.0, 0.1, 0.2]
    idx = scaling.emergence_threshold_step(steps, values, threshold=0.5)
    assert idx is None


def test_fit_power_law_recovers_exponent():
    xs = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    ys = 3.0 * xs ** -0.5
    a, b = scaling.fit_power_law(xs, ys)
    assert abs(a - 3.0) < 1e-3
    assert abs(b - (-0.5)) < 1e-3
```

- [ ] **Step 2:** Run to verify failure:

```bash
uv run pytest tests/test_scaling.py -v
```

Expected: import error.

- [ ] **Step 3:** Implement `src/circuitprobe/scaling.py`:

```python
"""Scaling-law fits + bootstrap confidence intervals.

We answer: at which training step does each model size first cross a
prefix-match threshold (e.g. 0.3)? Then we fit a power law to
    emergence_step  vs  model_size_params
and report a 95% bootstrap CI on the exponent.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


SIZE_TO_PARAMS: dict[str, int] = {
    "160M": 162_000_000,
    "410M": 405_000_000,
    "1.4B": 1_414_000_000,
}


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    boot_stats = np.empty(n_boot)
    n = len(samples)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = statistic(samples[idx])
    point = float(statistic(samples))
    lo = float(np.quantile(boot_stats, alpha / 2))
    hi = float(np.quantile(boot_stats, 1 - alpha / 2))
    return point, lo, hi


def emergence_threshold_step(
    steps: list[int],
    values: list[float],
    threshold: float,
) -> int | None:
    """Return the first step at which `values` crosses `threshold` (>=)."""
    for s, v in zip(steps, values):
        if v >= threshold:
            return s
    return None


def fit_power_law(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Fit y = a * x^b via log-log linear regression. Returns (a, b)."""
    log_x = np.log(xs)
    log_y = np.log(ys)
    b, log_a = np.polyfit(log_x, log_y, 1)
    return float(np.exp(log_a)), float(b)


def fit_emergence_law(
    grid_parquet_path: str,
    threshold: float = 0.3,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Load the grid, compute the per-size emergence step, fit a power law.

    Returns a dict with: per-size emergence steps + the fit + bootstrap CIs.
    """
    import pandas as pd

    df = pd.read_parquet(grid_parquet_path)
    sizes = sorted(df["size"].unique(), key=lambda s: SIZE_TO_PARAMS[s])

    per_size: dict[str, dict] = {}
    for size in sizes:
        sub = df[df["size"] == size].sort_values("step")
        steps = sub["step"].tolist()
        vals = sub["max_prefix_match"].tolist()
        emerge = emergence_threshold_step(steps, vals, threshold)
        per_size[size] = {"steps": steps, "max_prefix_match": vals, "emerge_step": emerge}

    points = [(SIZE_TO_PARAMS[s], per_size[s]["emerge_step"]) for s in sizes
              if per_size[s]["emerge_step"] is not None]
    if len(points) < 2:
        return {
            "per_size": per_size,
            "fit": None,
            "threshold": threshold,
            "note": "Insufficient data to fit power law (need >=2 emergence points).",
        }

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    a, b = fit_power_law(xs, ys)

    # Bootstrap CI on the exponent via resampling the (xs, ys) pairs.
    rng = np.random.default_rng(seed)
    exps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), size=len(xs))
        if len(np.unique(xs[idx])) < 2:
            continue
        _, b_boot = fit_power_law(xs[idx], ys[idx])
        exps.append(b_boot)
    exps_arr = np.asarray(exps)
    exp_lo = float(np.quantile(exps_arr, 0.025)) if len(exps_arr) else float("nan")
    exp_hi = float(np.quantile(exps_arr, 0.975)) if len(exps_arr) else float("nan")

    return {
        "per_size": per_size,
        "fit": {"a": a, "b": b, "b_ci": (exp_lo, exp_hi)},
        "threshold": threshold,
    }
```

- [ ] **Step 4:** Run tests:

```bash
uv run pytest tests/test_scaling.py -v
```

Expected: 4 passed.

- [ ] **Step 5:** Run on real grid + dump the fit:

```bash
uv run python -c "from circuitprobe import scaling; import json; \
  fit = scaling.fit_emergence_law('results/emergence_grid.parquet', threshold=0.3); \
  json.dump(fit, open('results/scaling_law_fit.json', 'w'), indent=2, default=str); \
  print(json.dumps(fit, indent=2, default=str))"
```

Expected: prints the per-size emergence steps and the fitted (a, b) + 95% CI.

- [ ] **Step 6:** Commit:

```bash
git add src/circuitprobe/scaling.py tests/test_scaling.py results/scaling_law_fit.json
git commit -m "P8: scaling-law fit + bootstrap CI on emergence grid"
git push
```

---

# Phase 9 — Visualization

## Task 9.1: Emergence-grid heatmap + scaling-law figure

**Files:**
- Create: `src/circuitprobe/viz/__init__.py`
- Create: `src/circuitprobe/viz/emergence_plots.py`
- Create: `scripts/make_figures.py`

- [ ] **Step 1:** Create `src/circuitprobe/viz/__init__.py` as empty.

- [ ] **Step 2:** Implement `src/circuitprobe/viz/emergence_plots.py`:

```python
"""Plotly figures for the emergence grid + scaling-law fit.

Produces:
- `emergence_heatmap(df)`  — 2D heatmap of max prefix-match over (size, step).
- `scaling_law_figure(fit)` — log-log scatter of emergence step vs model
  params with the fitted power-law overlay.
"""

from __future__ import annotations

import math

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
            z=z, x=steps, y=sizes, colorscale="Viridis",
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
    xs = [SIZE_TO_PARAMS[s] for s in sizes if fit["per_size"][s]["emerge_step"] is not None]
    ys = [fit["per_size"][s]["emerge_step"] for s in sizes if fit["per_size"][s]["emerge_step"] is not None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[s for s in sizes if fit["per_size"][s]["emerge_step"] is not None],
        textposition="top center",
        name="Observed emergence step",
    ))
    if fit.get("fit") is not None:
        a, b = fit["fit"]["a"], fit["fit"]["b"]
        x_curve = np.geomspace(min(xs) / 2, max(xs) * 2, 100)
        fig.add_trace(go.Scatter(
            x=x_curve, y=a * x_curve ** b, mode="lines",
            name=f"y = {a:.2e} · N^{b:.3f}",
        ))
    fig.update_layout(
        title="Emergence step vs model parameters (power-law fit)",
        xaxis_title="Model parameters (N)",
        yaxis_title="First training step where max prefix-match >= threshold",
        xaxis_type="log",
        yaxis_type="log",
    )
    return fig
```

- [ ] **Step 3:** Implement `scripts/make_figures.py`:

```python
"""Regenerate all headline figures from cached results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from circuitprobe.viz import emergence_plots


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
```

- [ ] **Step 4:** Run:

```bash
uv run python scripts/make_figures.py
```

Expected: 4 files in `results/figures/`.

- [ ] **Step 5:** Commit:

```bash
git add src/circuitprobe/viz/ scripts/make_figures.py results/figures/
git commit -m "P9: emergence heatmap + scaling-law figure (Plotly)"
git push
```

## Task 9.2: Attention-pattern figure via CircuitsVis

**Files:**
- Create: `src/circuitprobe/viz/attention.py`
- Create: `notebooks/02_emergence_grid.ipynb`

- [ ] **Step 1:** Implement `src/circuitprobe/viz/attention.py`:

```python
"""CircuitsVis wrappers for attention-pattern figures."""

from __future__ import annotations

import circuitsvis as cv
import torch
from transformer_lens import HookedTransformer

from ..data import random_repeat_seqs


@torch.no_grad()
def induction_head_attention_html(
    model: HookedTransformer,
    layer: int,
    head: int,
    half_len: int = 32,
    seed: int = 0,
) -> str:
    """Return an HTML snippet visualising the (layer, head) attention pattern
    on a random-repeat sequence — the canonical induction-head figure.
    """
    bos = model.tokenizer.bos_token_id if model.tokenizer is not None else None
    seqs = random_repeat_seqs(
        vocab_size=model.cfg.d_vocab, half_len=half_len, n_seqs=1, seed=seed,
        bos_token=bos,
    ).to(model.cfg.device)

    _, cache = model.run_with_cache(
        seqs, names_filter=lambda n: n == f"blocks.{layer}.attn.hook_pattern"
    )
    pat = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # (seq, seq)

    tokens = [model.to_string(t) for t in seqs[0]]
    html = cv.attention.attention_pattern(tokens=tokens, attention=pat.cpu())
    return str(html)
```

- [ ] **Step 2:** Create `notebooks/02_emergence_grid.ipynb` (one cell + a few markdown notes) that loads `results/emergence_grid.parquet`, renders the heatmap inline, and saves an HTML of a canonical induction head's attention via the function above.

(The notebook can be a minimal stub — the figure-generation logic lives in `scripts/make_figures.py`; the notebook is for recruiter-friendly inline viewing on GitHub.)

- [ ] **Step 3:** Commit:

```bash
git add src/circuitprobe/viz/attention.py notebooks/02_emergence_grid.ipynb
git commit -m "P9: CircuitsVis attention-pattern figures + emergence-grid notebook"
git push
```

---

# Phase 10 — Writeup

## Task 10.1: METHODOLOGY.md

**Files:**
- Create: `docs/METHODOLOGY.md`

- [ ] **Step 1:** Write the methodology document covering:
  - Exact definitions of prefix-match and copying scores (with formulas)
  - Random-repeat probe construction (vocab range, half_len, seeds)
  - How Pythia checkpoints are pinned (HF `revision="step{N}"`)
  - Emergence-step extraction (threshold choice, sensitivity analysis)
  - Bootstrap-CI protocol (n_boot, resampling unit)
  - Triton kernel correctness check + benchmark protocol
  - What we did NOT do (cross-architecture, refusal directions, training new models)
  - What would falsify the headline (e.g., "emergence step shows no scale dependence after controlling for compute")

  The full text is to be written when results are in. Stub the file now with section headers and an "(authored after P11)" note for `## Headline finding`.

- [ ] **Step 2:** Commit:

```bash
git add docs/METHODOLOGY.md
git commit -m "P10: methodology document scaffold (sections defined; results filled in P11)"
git push
```

## Task 10.2: docs/writeup.md (LessWrong-style)

**Files:**
- Create: `docs/writeup.md`

- [ ] **Step 1:** Write the post with these sections:
  - **TL;DR** — one paragraph: what was reproduced, what was extended, headline number
  - **Background** — induction heads in one paragraph (citing Olsson '22)
  - **Reproduction** — Pythia-410M numbers vs the paper
  - **The (size × step) emergence grid** — the heatmap, with interpretation
  - **Scaling law** — the fit, its CI, and the interpretation
  - **Triton kernel** — what was fused, the benchmark number
  - **Limitations** — what we didn't check; what would falsify the headline
  - **Reproducing this** — clone, `uv sync`, `make reproduce`
  - **Acknowledgements** — Olsson et al., EleutherAI for Pythia, TransformerLens

  All numbers must be filled in from `results/`. Embed figures via `![](figures/...)` paths.

- [ ] **Step 2:** Update README.md to link the writeup + show the headline figure inline.

- [ ] **Step 3:** Commit:

```bash
git add docs/writeup.md README.md
git commit -m "P10: LessWrong-style writeup + README hero figure"
git push
```

---

# Phase 11 — Polish & Verify

## Task 11.1: Lint + full test pass

- [ ] **Step 1:** Run ruff and fix:

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
```

- [ ] **Step 2:** Full test pass (skip slow + gpu):

```bash
uv run pytest -q -m "not slow and not gpu"
```

Expected: all tests pass.

- [ ] **Step 3:** Run slow tests once locally (model download):

```bash
uv run pytest -q -m slow
```

Expected: passes (after first run, cached).

- [ ] **Step 4:** Commit any auto-fixes:

```bash
git add -A
git commit -m "P11: ruff format + fix" || echo "nothing to commit"
git push
```

## Task 11.2: `make reproduce` end-to-end

- [ ] **Step 1:** From a clean shell, verify reproduce flow:

```bash
cd ~/projects/circuitprobe && make reproduce
```

Expected: writes `results/figures/*.png` and `results/figures/*.html`
without re-running the long grid (uses cached results).

- [ ] **Step 2:** Verify the headline figures look correct (manual). Open the HTMLs in a browser.

- [ ] **Step 3:** Verify CI on GitHub is green.

## Task 11.3: README hero section

- [ ] **Step 1:** Update `README.md` to include:
  - Hero emergence-heatmap PNG embedded.
  - The headline number from the writeup.
  - Triton kernel speedup number.
  - Reproduction quickstart.
  - Links: spec, plan, writeup, wandb run page.

- [ ] **Step 2:** Commit:

```bash
git add README.md
git commit -m "P11: README hero section with headline numbers + figures"
git push
```

## Task 11.4: Tag a release

- [ ] **Step 1:** Tag v0.1.0:

```bash
git tag -a v0.1.0 -m "CircuitProbe v0.1.0 — reproduction + emergence grid + Triton kernel"
git push origin v0.1.0
```

- [ ] **Step 2:** On GitHub, create a release from the tag with the writeup TL;DR as the release notes.

---

## Spec-Coverage Self-Review

| Spec section | Covered by |
|---|---|
| §3 Headline finding targets | P3 (identification) + P4 (Olsson baseline) + P7 (grid) + P8 (scaling fit) |
| §4.1 Repo layout | P0 (scaffolding) + every subsequent phase adds modules |
| §4.2 Tech stack locked | P0 `pyproject.toml` |
| §4.3 Data flow | P1–P9 each implement one box in the diagram |
| §4.4 Module boundaries | One module per task in P1–P9 |
| §5 Phase plan | This document, P0–P11 |
| §6 Disk + time budget | P1 caching + P7 grid runtime |
| §7 Risks | P1 Task 1.2 (revision pinning), P3 (seed control), P6 (kernel fallback), P10 (honest negative writeup) |
| §8 Honesty stance | P0 conftest seeding + P7 wandb logging + P8 bootstrap CIs + P10 writeup limitations |
| §9 Success criteria | P11 polish + tagged release |
| §10 Non-goals | Nothing in the plan attempts training, multi-arch, refusal directions, or paid cloud |
