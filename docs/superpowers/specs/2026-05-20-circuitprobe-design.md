# CircuitProbe: Design Spec

**Author:** Neil Patel
**Date:** 2026-05-20
**Status:** Approved (brainstorming complete)
**Series:** Project 2 in Neil's numbered portfolio series (see Crypt, Exfil, FitGraph, DistKV).

---

## 1. Story

Olsson et al. (2022, "In-context Learning and Induction Heads") identified
*induction heads*, small attention-head circuits inside transformer language
models that implement in-context pattern matching of the form `[A][B] ... [A] → [B]`.
The paper showed that induction-head formation coincides with a sharp phase
transition in the in-context-learning loss curve, but did not fully map *when*
this transition happens as a joint function of **model scale** and **training
compute**.

CircuitProbe:

1. **Reproduces** the induction-head identification protocol on Pythia-160M,
   Pythia-410M, and Pythia-1.4B using TransformerLens.
2. **Extends** the result by leveraging Pythia's public training checkpoints
   (~143 logged steps across training) to build the full
   **(model scale × training step) emergence map** for induction heads.
3. **Fits a scaling law** to the emergence boundary, characterising whether
   induction heads form earlier (in steps), later, or at a consistent
   compute-equivalent point in larger models.
4. **Ships a Triton kernel** for fused per-head prefix-matching score
   extraction, targeting 1.5-3× speedup over PyTorch eager on a Colab T4.
5. **Writes up** the result in LessWrong/Alignment Forum format with
   reproducible figures and code.

### Headline resume bullet (draft)

> CircuitProbe, Reproduced Olsson '22 induction-head circuit on Pythia
> 160M-1.4B (TransformerLens, activation/path patching), then mapped the full
> emergence boundary across 3 model scales × 12 training checkpoints, fit a
> scaling law to induction-head formation as a joint function of scale and
> training compute. Wrote a Triton kernel for fused prefix-matching score
> extraction (target 1.5-3× over PyTorch eager). Writeup published on
> LessWrong / personal blog.

---

## 2. Scope

**In scope:**

- Faithful reproduction of induction-head identification on Pythia 160M /
  410M / 1.4B (TransformerLens).
- Activation and path patching infrastructure for measuring circuit
  faithfulness/completeness.
- Emergence-grid sweep over (3 sizes × ~12 training checkpoints).
- Scaling-law fit with bootstrap confidence intervals.
- One Triton kernel (fused prefix-match score extraction) with PyTorch
  eager reference + correctness tests + benchmark.
- Interactive Plotly emergence heatmap; CircuitsVis attention-pattern
  figures for the canonical induction heads.
- LessWrong-style writeup in `docs/writeup.md`.
- Reproducibility: `make reproduce` regenerates headline figures from a
  clean clone (with cached checkpoints).

**Out of scope:**

- Training any model from scratch. We only use pre-trained Pythia +
  GPT-2-small weights.
- Comparison across architectures (Mamba/SSM, RWKV), single direction
  (Pythia + GPT-2) keeps scope tight.
- Refusal directions, sycophancy steering, IOI circuits, these are
  alternative interpretability findings that were explicitly considered
  during brainstorming and not chosen.
- Production deployment / serving. CircuitProbe is a research artifact.
- Paid cloud compute. Local M-series Mac + free Colab T4 (kernel benchmark
  only). The project ships without billing anything.

---

## 3. Headline Findings (Targets)

| Target                                                         | Source / threshold                                       |
|----------------------------------------------------------------|----------------------------------------------------------|
| Prefix-matching + copying scores on Pythia-410M's top induction heads | Within ~10% of Olsson '22's reported values              |
| Emergence-grid cells completed                                 | 36 (3 sizes × 12 ckpts) with 95% bootstrap CIs           |
| Triton kernel correctness                                      | `atol=1e-5` vs PyTorch reference                         |
| Triton kernel speedup on Colab T4                              | ≥1.5× over PyTorch eager (target band 1.5-3×)            |
| Scaling-law fit                                                | Functional form fit + R² + 95% bootstrap CIs reported    |
| Reproducibility                                                | `make reproduce` regenerates headline figures            |

A negative or noisier-than-expected scaling-law result is acceptable and
will be reported honestly (matching the FitGraph-style "we caught the
AUC=0.99 leak" honesty posture).

---

## 4. Architecture

### 4.1 Repository layout

```
~/projects/circuitprobe/
├── README.md                       # story + headline figures + reproduce
├── pyproject.toml                  # uv-managed, Python 3.11+
├── Makefile                        # `make reproduce`, `make test`, `make bench`
├── ruff.toml, .gitignore, LICENSE
├── src/circuitprobe/
│   ├── __init__.py
│   ├── models.py                   # TransformerLens loaders for Pythia + GPT-2
│   ├── data.py                     # random-repeat, IOI, Pile sample probes
│   ├── induction.py                # prefix-match score, copying score, ranking
│   ├── patching.py                 # activation + path patching
│   ├── faithfulness.py             # circuit-recovery metrics
│   ├── checkpoints.py              # Pythia checkpoint enumeration + cache
│   ├── emergence.py                # (scale × step) grid runner
│   ├── scaling.py                  # scaling-law fits + bootstrap CIs
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── prefix_match_triton.py  # Triton fused kernel (CUDA only)
│   │   └── prefix_match_torch.py   # PyTorch eager reference
│   ├── tracking.py                 # wandb wrapper with offline fallback
│   └── viz/
│       ├── __init__.py
│       ├── attention.py            # CircuitsVis wrappers
│       └── emergence_plots.py      # Plotly heatmaps + scaling-law figs
├── scripts/
│   ├── reproduce_olsson.py
│   ├── run_emergence_grid.py
│   ├── bench_kernel.py
│   └── make_figures.py
├── notebooks/
│   ├── 01_olsson_reproduction.ipynb
│   ├── 02_emergence_grid.ipynb
│   └── 03_kernel_benchmark.ipynb
├── tests/
│   ├── test_induction.py
│   ├── test_patching.py
│   ├── test_kernels.py
│   ├── test_emergence.py
│   ├── test_scaling.py
│   └── conftest.py
├── data/                           # cached probe sets (gitignored)
├── results/                        # JSON + parquet + figures (gitignored)
├── docs/
│   ├── writeup.md                  # LessWrong-style post
│   ├── METHODOLOGY.md              # exact protocols + repro checklist
│   └── superpowers/
│       ├── specs/                  # this file
│       └── plans/                  # implementation plan(s)
└── .github/workflows/ci.yml
```

### 4.2 Tech stack (locked)

- **Language:** Python 3.11+
- **Env:** [uv](https://github.com/astral-sh/uv)
- **Models / probing:** [TransformerLens](https://github.com/neelnanda-io/TransformerLens),
  PyTorch (MPS for Mac, CUDA on Colab)
- **Custom kernel:** [Triton](https://github.com/triton-lang/triton), single
  fused kernel, gated on `torch.cuda.is_available()` with PyTorch fallback
- **Visualization:** [Plotly](https://plotly.com/python/) for emergence
  heatmaps + scaling-law figures;
  [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis) for
  attention-pattern figures
- **Tracking:** Weights & Biases (free personal tier) with offline-mode
  fallback when no API key is configured
- **Datasets:** the canonical "random-repeat" sequences for induction-head
  scoring; a Pile sample (`monology/pile-uncopyrighted`, deterministic
  subsample) for in-context-learning-loss curves; IOI dataset (for
  contrasting / sanity checks only)
- **Testing:** pytest, ruff
- **CI:** GitHub Actions (CPU-only, smallest model, fast tests only)
- **Repo hosting:** private GitHub repo `NeilP211/circuitprobe` (matches
  Neil's preference)
- **No cloud bills.** Triton benchmark on free Colab T4. Everything else
  on local Mac.

### 4.3 Data flow

```
                           Pythia HF repo (revisions)
                                    │
                                    ▼
                          checkpoints.py  ←──── revision pins
                                    │
                                    ▼
                          TransformerLens HookedTransformer
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
          induction.py        patching.py         faithfulness.py
       (per-head scores)   (clean/corrupt)   (circuit recovery)
                │                   │                   │
                └─────────┬─────────┴───────────────────┘
                          ▼
                     emergence.py
                  (cells: size × step)
                          │
                          ▼
              results/emergence_grid.parquet
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
            scaling.py        viz/emergence_plots.py
        (fit + bootstrap)        (Plotly + CircuitsVis)
                │                   │
                └─────────┬─────────┘
                          ▼
                 docs/writeup.md + README hero figs
```

### 4.4 Module boundaries (interfaces, one-line contracts)

- `models.py`: `load_pythia(size: str, step: int | None) -> HookedTransformer`
- `checkpoints.py`: `pythia_steps() -> list[int]`; `cache_root() -> Path`
- `data.py`: `random_repeat_seqs(vocab_size, seq_len, n_seqs, seed) -> Tensor`;
  `pile_sample(n_seqs, seed) -> list[str]`
- `induction.py`: `prefix_match_score(model, seqs) -> Tensor[layers, heads]`;
  `copying_score(model, seqs) -> Tensor[layers, heads]`;
  `rank_induction_heads(scores) -> list[(layer, head, score)]`
- `patching.py`: `path_patch(model, clean, corrupt, sender, receiver) -> Tensor`;
  `head_ablate(model, head_set) -> ContextManager`
- `faithfulness.py`: `circuit_faithfulness(model, circuit, task) -> float`
- `emergence.py`: `run_cell(size, step, n_seqs, seed) -> CellResult`;
  `run_grid(sizes, steps, n_seqs, seed) -> Iterator[CellResult]`
- `scaling.py`: `fit_emergence_law(grid) -> ScalingLawFit` (carries bootstrap CIs)
- `kernels/prefix_match_triton.py`: `prefix_match_kernel(attn: Tensor, idx: Tensor) -> Tensor`
- `kernels/prefix_match_torch.py`: same signature, reference implementation
- `tracking.py`: thin `init_run(...)` + `log(...)` that no-ops gracefully offline
- `viz/*`: pure functions: `(data) -> Figure`

Each module is independently testable and has a small surface.

---

## 5. Phase Plan

| Phase | Name                          | Ships                                                                                              |
|-------|-------------------------------|----------------------------------------------------------------------------------------------------|
| P0    | Scaffolding                   | repo init, uv, ruff, pyproject, Makefile, CI skeleton, README v0, private GitHub repo, first push  |
| P1    | Model + checkpoint loaders    | TransformerLens loaders for Pythia 160M/410M/1.4B + GPT-2 small; Pythia checkpoint enumeration; cache; smoke tests |
| P2    | Probe data                    | random-repeat sequences, IOI dataset wrapper, Pile sample; deterministic seeds; tests              |
| P3    | Induction-head identification | prefix-matching + copying scores; rank all heads; canonical induction heads identified per model   |
| P4    | Olsson baseline               | in-context-learning loss-by-position curve + phase-transition plot reproduced on one Pythia size   |
| P5    | Patching + faithfulness       | activation + path patching, head ablation; faithfulness/completeness metrics on induction circuit  |
| P6    | Triton kernel                 | fused prefix-match kernel + PyTorch reference + correctness tests + Colab T4 benchmark             |
| P7    | Emergence grid                | sweep (3 sizes × 12 ckpts); persist `results/emergence_grid.parquet`; full wandb logging           |
| P8    | Scaling-law analysis          | fit functional form; bootstrap CIs; headline-finding decision (positive or honest negative)        |
| P9    | Visualization                 | Plotly interactive emergence heatmap; CircuitsVis attention pattern figs; scaling-law figure       |
| P10   | Writeup                       | `docs/writeup.md` LessWrong-style with embedded figures; `METHODOLOGY.md`; reproducibility check   |
| P11   | Polish & verify               | all tests pass, ruff clean, CI green, README hero section finalised, `make reproduce` works        |

**Cadence:** for each phase, the agent will explain the phase in 3-5 bullets,
wait for Neil's "go ahead", build, run the test suite, commit + push, then
move on. Matches the Exfil / FitGraph cadence.

---

## 6. Disk & Time Budget

- **Pythia checkpoints cache:** ~70 GB total for 12 steps × 3 sizes
  (160M ≈ 6 GB, 410M ≈ 16 GB, 1.4B ≈ 50 GB). Cached under `~/.cache/circuitprobe/`
  with explicit manifests.
- **Emergence-grid wall time** on M-series: each cell 2-10 min depending on
  size; 36 cells ≈ 3-4 hours total. Safe to run overnight.
- **CI runtime budget:** under 5 min, CPU-only, smallest model.
- **Triton benchmark:** one-shot Colab T4 notebook, ~5 minutes.

---

## 7. Risks & Mitigations

| Risk                                                                   | Mitigation                                                                                            |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| HuggingFace Pythia revision naming changes                             | Pin `revision="step{N}"` and verify in P1 before depending on it                                       |
| Induction-head identification sensitive to seed / sequence length      | Fix all seeds; report N ≥ 256 sequences per cell; report variance                                      |
| Triton kernel doesn't beat eager on T4                                 | Ship anyway; document the loss honestly; attempt a Modal one-shot **only with explicit Neil approval** |
| Headline finding is null ("emergence doesn't depend cleanly on scale") | That's still a publishable negative result; writeup angle adjusts                                      |
| MPS missing kernels for some TransformerLens ops                       | Fall back to CPU for those steps; document slowdown                                                    |

---

## 8. Honesty Stance (FitGraph posture)

- All RNG seeds fixed and logged in `results/*.json`.
- Per-cell bootstrap 95% CIs reported, not just point estimates.
- The set of sequences used to *identify* canonical induction heads is
  disjoint from the set used to *measure* their score (no selection leak).
- Writeup explicitly lists:
  - What I did not check.
  - What evidence would falsify the headline.
  - Where my numbers diverge from Olsson '22 and why.

---

## 9. Success Criteria (definition of done)

A reasonable recruiter or interpretability researcher can:

1. Land on the README and within 30 seconds understand the project,
   the headline finding, and how to reproduce.
2. Click the wandb run page and see real emergence-grid metrics.
3. `git clone && make reproduce` and regenerate the headline figures from
   cached checkpoints with no further intervention.
4. Read `docs/writeup.md` as a self-contained LessWrong-style post.
5. See in CI that the test suite is green and ruff is clean.

---

## 10. Non-goals (explicit)

- We do **not** claim a new circuit. The circuit is Olsson's.
- We do **not** retrain Pythia. We only consume HF checkpoints.
- We do **not** ship a generic interpretability library. CircuitProbe is
  a focused experiment, not a framework.
- We do **not** burn money on cloud compute. Free local + free Colab T4
  only. Any paid escalation requires an explicit, unmissable "this is
  billed" disclaimer and Neil's specific approval.
