# CircuitProbe

Mechanistic interpretability on small language models — reproducing
induction heads (Olsson et al., 2022) on Pythia, then mapping their
emergence across the **(model scale × training step)** grid using Pythia's
public training checkpoints.

> **Status:** ✅ complete. Headline: induction-head emergence is
> **scale-invariant in training-step terms** — all three Pythia sizes
> (160M / 410M / 1.4B) cross the `max prefix-match ≥ 0.3` threshold at
> training step 1000.
> Spec: [`docs/superpowers/specs/2026-05-20-circuitprobe-design.md`](docs/superpowers/specs/2026-05-20-circuitprobe-design.md).
> Plan: [`docs/superpowers/plans/2026-05-20-circuitprobe.md`](docs/superpowers/plans/2026-05-20-circuitprobe.md).
> Writeup: [`docs/writeup.md`](docs/writeup.md).

![Induction-head emergence map](results/figures/emergence_heatmap.png)

*All three model sizes flip from no-induction (dark) to mature-induction (yellow) inside the same training-step window — between step 256 and step 1000.*

---

## What this project is

**Reproduction.** Identify induction heads on Pythia-160M / 410M / 1.4B using
TransformerLens. On Pythia-160M, the canonical induction head **L4H6** has
prefix-match score **0.985** (top-10 in `results/pythia_160m_heads.json`).
The in-context-learning loss-by-position curve on Pythia-410M shows the
characteristic drop between positions ~5 and ~50 — Olsson '22 figure 1
qualitatively reproduced.

**Extension.** Pythia is the only major model family that ships training
checkpoints across its full training run. CircuitProbe sweeps a 3-size ×
12-checkpoint emergence grid (36 cells), with per-cell prefix-match scoring
plus cache-cleanup-between-cells (peak disk ~6 GB instead of ~70 GB).
A power law fit to the emergence step vs model parameters gives exponent
**b ≈ 0** (machine-precision zero) with 95% bootstrap CI covering zero —
induction-head emergence is locked to the *training schedule*, not the
*model's capacity*, across 8.7× in parameter count.

**Custom kernel.** A single Triton kernel fuses gather + mean-reduce along
the prefix-match diagonal in the attention pattern tensor — the hot inner
loop of induction-head identification. Targets 1.5–3× over PyTorch eager on
a Colab T4 (benchmark in `notebooks/03_kernel_benchmark.ipynb`).

**Writeup.** LessWrong-style post in [`docs/writeup.md`](docs/writeup.md) with
methodology in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Quickstart

```bash
uv sync --extra dev
make test                                   # full pytest suite (33+ tests)
make lint                                   # ruff check
uv run python scripts/identify_pythia_160m.py    # P3 — top induction heads on Pythia-160M
uv run python scripts/reproduce_olsson.py        # P4 — ICL loss-by-position curve on Pythia-410M
uv run python scripts/run_emergence_grid.py      # P7 — the headline 36-cell sweep (~3-4 hrs)
make reproduce                              # regenerate headline figures from results/
```

The Triton kernel benchmark requires CUDA — open
`notebooks/03_kernel_benchmark.ipynb` in Colab (T4 runtime) and run all cells.

## What's in here

| Path | What it is |
|---|---|
| `src/circuitprobe/models.py` | TransformerLens loaders for Pythia 160M/410M/1.4B + GPT-2 small |
| `src/circuitprobe/checkpoints.py` | Pythia checkpoint enumeration + emergence-step selector |
| `src/circuitprobe/data.py` | Random-repeat sequences + Pile sample |
| `src/circuitprobe/induction.py` | Prefix-matching score, copying score, head ranking |
| `src/circuitprobe/icl.py` | In-context-learning loss-by-position |
| `src/circuitprobe/patching.py` | Activation patching + head-ablation context manager |
| `src/circuitprobe/faithfulness.py` | Circuit faithfulness / recovery metrics |
| `src/circuitprobe/kernels/` | PyTorch reference + Triton kernel + backend selector |
| `src/circuitprobe/emergence.py` | The (size × step) grid runner with cache-cleanup-between-cells |
| `src/circuitprobe/scaling.py` | Power-law fit + bootstrap CIs |
| `src/circuitprobe/viz/` | Plotly emergence heatmap + scaling-law figure + CircuitsVis attention |
| `src/circuitprobe/tracking.py` | wandb wrapper with offline fallback |
| `scripts/` | Runnable drivers for each phase |
| `tests/` | 33+ unit tests, CPU-only, runs in CI |
| `docs/writeup.md` | LessWrong-style writeup |
| `docs/METHODOLOGY.md` | Exact protocols, definitions, and reproducibility checklist |

## Why this exists

Part of a numbered portfolio series:
[Crypt](https://github.com/NeilP211/crypt),
[Exfil](https://github.com/NeilP211/exfil),
[FitGraph](https://github.com/NeilP211/fitgraph),
[DistKV](https://github.com/NeilP211/distkv).
This one (Project 2) sits at the intersection of mechanistic interpretability
research and low-level GPU programming — the niche where Anthropic-style
circuit-finding meets systems engineering.

## License

MIT. See [LICENSE](LICENSE).
