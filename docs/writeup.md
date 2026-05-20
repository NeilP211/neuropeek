# CircuitProbe — Reproducing induction heads on Pythia and mapping their emergence across scale and training compute

**TL;DR.** I reproduced the induction-head identification from Olsson et al. (2022, *"In-context Learning and Induction Heads"*) on Pythia 160M / 410M / 1.4B using TransformerLens, then leveraged Pythia's public training checkpoints to build the full `(model scale × training step)` emergence grid — 3 sizes × 12 checkpoints. A power law fit to the emergence boundary gives <TBF: scaling exponent + CI>. I also shipped a Triton kernel for fused prefix-matching score extraction, targeting <TBF: speedup> over PyTorch eager on a T4.

## Background

The induction head is a small attention circuit that implements in-context pattern matching of the form `[A] [B] ... [A] → [B]`. Olsson et al. showed that these heads emerge during training in a sharp phase transition, coincident with a "bump" in in-context-learning loss. The paper identified the heads and the transition but did not fully map *when* this transition occurs as a joint function of model scale and training compute.

Pythia is uniquely well-suited to filling in that map: EleutherAI ships dense per-step training checkpoints alongside each model size. This lets us swap out training-time arguments without retraining anything ourselves.

## Reproduction

### Induction-head identification on Pythia-160M

On the final Pythia-160M checkpoint, the canonical induction heads are clustered in layers 4–5 (with secondary heads in 8–9). The top three by prefix-match score:

- L4H6 — prefix_match = 0.985, copying = 0.588 (the textbook induction head)
- L8H2 — prefix_match = 0.929, copying = 0.633
- L4H10 — prefix_match = 0.922, copying = 0.626

Top-10 heads at `prefix_match > 0.5`. (Full table: `results/pythia_160m_heads.json`.)

### In-context-learning loss curve on Pythia-410M

I computed the per-position cross-entropy loss on 32 Pile sequences (max_len = 512) using Pythia-410M's final checkpoint. The loss drops sharply between positions ~5 and ~50 — the characteristic in-context-learning signature.

- Loss at position 1: 7.44
- Loss at position 10: 2.35
- Minimum loss (1.49) at position 59
- (Tail positions inflated by variable-length Pile entries — see Methodology §3.)

![ICL loss curve](figures/icl_curve_pythia_410m.png)

## The (scale × step) emergence grid

To map the emergence boundary I swept 3 model sizes × 12 log-spaced training checkpoints — 36 cells. Each cell loads the corresponding (Pythia, step) revision via HuggingFace and computes the prefix-matching score on 64 random-repeat sequences.

![Emergence heatmap](figures/emergence_heatmap.png)

The headline pattern: <TBF: 2-3 sentence qualitative description of the grid — does emergence happen earlier (in steps) for bigger models, later, or at a consistent compute-equivalent step?>

## Scaling law

Fitting `emergence_step = a · N^b` to the per-size emergence points gives:

- Exponent `b` = <TBF>, 95% bootstrap CI [<TBF>, <TBF>]
- Threshold for emergence: `prefix_match >= 0.3` (sensitivity at 0.2 / 0.4 shown in supplement).

![Scaling law](figures/scaling_law.png)

Interpretation: <TBF: 1-2 paragraphs interpreting the sign + magnitude of the exponent. If b < 0, larger models reach the same prefix-match threshold sooner in absolute steps. If b > 0, larger models take longer (in steps). If |b| is small the timing is roughly compute-equivalent.>

## Triton kernel

The hot inner loop of induction-head scoring is "gather pattern[b, h, q, q-N+1] for each q in the second half, then mean". PyTorch eager performs this as advanced indexing (a copy) plus a mean (another pass over the gathered tensor). I wrote a single Triton kernel that fuses the gather + accumulate + mean per (batch, head) program, eliminating the intermediate tensor:

- Correctness: max error vs PyTorch eager < 1e-3 (fp16 input, fp32 accumulator).
- Speedup on Colab T4: <TBF>× median across (B, H, S) ∈ {(16, 12, 128), (8, 16, 256), (4, 32, 512)} shapes.

(Code: `src/circuitprobe/kernels/prefix_match_triton.py`. Benchmark: `scripts/bench_kernel.py`, runnable end-to-end via `notebooks/03_kernel_benchmark.ipynb`.)

## Limitations

What I did not check:
- Cross-architecture (Mamba/SSM, RWKV) — single direction kept scope tight.
- Refusal directions, sycophancy steering, or other interpretability circuits.
- Training new models from scratch — only consumed published Pythia weights.

What would falsify the headline:
- A larger sweep showing the trend flips sign at a fourth model size.
- A different threshold giving a qualitatively different boundary.
- A check that the emergence step is dominated by tokenisation/architecture artifacts rather than scale (controlled by holding architecture fixed across sizes — which Pythia does, so this concern is bounded).

## Reproducing this

```bash
git clone https://github.com/NeilP211/circuitprobe
cd circuitprobe
uv sync --extra dev
make reproduce      # regenerates figures from cached results
make test           # runs the test suite
```

The full emergence-grid sweep takes ~3-4 hours on Apple Silicon and downloads ~5.6 GB per checkpoint (cleaned between cells). To rerun it:

```bash
uv run python scripts/run_emergence_grid.py --sizes 160M 410M 1.4B --n-steps 12
```

The Triton kernel benchmark is gated on CUDA — run it on Colab T4 via `notebooks/03_kernel_benchmark.ipynb`.

## Acknowledgements

- Olsson et al. (2022) for the original induction-head work.
- EleutherAI for publishing the Pythia training checkpoints.
- Neel Nanda and the TransformerLens contributors for the probing infrastructure.

---

*Part of a numbered portfolio series: [Crypt](https://github.com/NeilP211/crypt), [Exfil](https://github.com/NeilP211/exfil), [FitGraph](https://github.com/NeilP211/fitgraph), [DistKV](https://github.com/NeilP211/distkv). Source: https://github.com/NeilP211/circuitprobe.*
