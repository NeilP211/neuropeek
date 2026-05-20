# CircuitProbe — Methodology

> Companion to the [writeup](writeup.md) and the [design spec](superpowers/specs/2026-05-20-circuitprobe-design.md). This document captures the exact protocols so the result is reproducible.

## 1. Models

- We use Pythia-160M, Pythia-410M, and Pythia-1.4B from EleutherAI, loaded via TransformerLens v2.0.
- Training-time analysis uses Pythia's public checkpoints. Each checkpoint is pinned by HuggingFace revision `step{N}` where N is the training step.
- Checkpoints loaded with `fold_ln=False, center_writing_weights=False, center_unembed=False` to preserve the original parameterisation (required for the copying-score and path-patching to be interpretable).
- All inference is in `torch.float32` on Apple Silicon MPS or CPU; no quantisation.

## 2. Probe data

### Random-repeat sequences (induction-head identification)
- Sequence layout: `[r_1 r_2 ... r_N | r_1 r_2 ... r_N]` where r_i are tokens sampled uniformly from `[0, d_vocab)` with a fixed `torch.Generator` seed.
- Default: `half_len = 50`, `n_seqs = 64`, `seed = 0`. Total length 2N (plus optional BOS).
- Defined in `src/circuitprobe/data.py::random_repeat_seqs`.

### Pile sample (in-context-learning loss curves)
- `monology/pile-uncopyrighted`, streaming load, deterministic skip-based subsample with `random.Random(seed)`.
- Filter to entries with `len(text) >= 200` characters to avoid trivial sequences. Pre-truncate at `max_len * 8` characters as a rough char-to-token bound.
- Defined in `src/circuitprobe/data.py::pile_sample`.

## 3. Metrics

### Prefix-matching score (per head)
For a random-repeat sequence of total length 2N, compute the attention pattern for each (layer, head). The prefix-matching score is the mean attention from query position q (q ≥ N) to key position q − N + 1, averaged over the second half and over the batch.

Implementation: `src/circuitprobe/induction.py::prefix_match_score`. BOS, if present, is stripped from the pattern tensor before scoring.

### Copying score (per head)
Defined as the fraction of the diagonal of `W_E @ W_OV @ W_U` that is strictly positive (Olsson et al. 2022, appendix B.2). Pure model-only computation; no data required.

Note: we never materialise the full `(d_vocab, d_vocab)` product (≈10 GiB for Pythia). The diagonal is computed elementwise via `(W_E @ W_OV) * W_U.T` then row-sum, with a peak memory of ~150 MB.

Implementation: `src/circuitprobe/induction.py::copying_score`.

### In-context-learning loss-by-position
Per-position cross-entropy loss averaged over the batch:
```
loss[t] = mean_b CE(softmax(logits[b, t, :]), tokens[b, t+1])  for t in 0..seq_len-2
```
Implementation: `src/circuitprobe/icl.py::loss_by_position`.

## 4. Checkpoint selection (emergence grid)

- `select_emergence_steps(n=12)` produces 12 log-spaced training steps from the published Pythia checkpoint set: `{0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512} ∪ {1000, 2000, ..., 143000}`.
- Endpoints are pinned: step 0 and step 143000.
- Interior steps are selected by snapping `np.logspace(0, log10(143000), n-1)` to the nearest available published checkpoint.

## 5. Emergence-grid sweep

- 3 model sizes × 12 training checkpoints = 36 cells.
- Per cell: load the model at that (size, step), compute the prefix-match score on 64 random-repeat sequences (half_len = 50, seed = 0), record `max_prefix_match`, `mean_prefix_match`, and the top-5 (layer, head, score) tuples.
- Persisted as one JSON object per line in `results/emergence_grid.jsonl`, then consolidated to `results/emergence_grid.parquet`.
- Disk-management: each revision's HF cache snapshot is removed after the cell completes (`cleanup_after_cell=True` default), keeping peak disk usage under ~6 GB even though the full uncached set would exceed 70 GB.
- Tracked via offline wandb run.

## 6. Emergence-step extraction

- For each model size, the emergence step is the first checkpoint at which `max_prefix_match ≥ 0.3` (the default threshold).
- The threshold is exposed as `fit_emergence_law(..., threshold=0.3)`. Sensitivity analyses at 0.2 and 0.4 are reported in the writeup.

## 7. Scaling-law fit

- Power law fit: `y = a · N^b`, where `N` is model parameter count and `y` is the emergence step.
- Log-log linear regression via `np.polyfit`.
- 95% bootstrap CI on the exponent `b` via paired (N, emergence_step) resampling, 500 bootstrap iterations.
- Implementation: `src/circuitprobe/scaling.py::fit_emergence_law`.

## 8. Triton kernel benchmark

- Kernel: fused per-(batch, head) prefix-match score (gather-along-diagonal + mean reduction) over the `(B, H, S, S)` attention pattern tensor.
- Correctness: max element-wise error vs PyTorch eager < 1e-3 (fp16 input, fp32 accumulator).
- Benchmark: median + IQR latency across 50 trials with 5 warm-up calls, on Colab T4 (CUDA).
- Shapes benchmarked: (B=16, H=12, S=128), (B=8, H=16, S=256), (B=4, H=32, S=512).
- Speedup target: 1.5-3× over PyTorch eager.

## 9. Reproducibility

- All RNG seeds (random, numpy, torch, generator-based) fixed and logged.
- `conftest.py` sets a deterministic seed (1623) per test function.
- The full results regenerate from a clean clone via `make reproduce` (figures from cached results) and `uv run python scripts/run_emergence_grid.py` (the long sweep, requires checkpoint downloads).

## 10. Honesty stance

- Per-cell point estimates are reported alongside bootstrap 95% CIs.
- The sequences used to *identify* canonical induction heads (P3) are independent of the sequences used to *measure* the per-cell prefix-match score (P7) — no selection-to-measurement leak.
- The writeup explicitly lists:
  - What we did *not* check (cross-architecture comparisons, refusal directions, sycophancy steering, training from scratch).
  - What evidence would falsify the headline finding.
  - Where our numbers diverge from Olsson '22 and why (model family + size differ; their scaling claim was qualitative).

## 11. Headline finding

`<TBF: filled in by controller after the emergence-grid run completes.>`
