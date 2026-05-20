# CircuitProbe

Mechanistic interpretability on a small language model — reproducing
induction heads (Olsson et al., 2022) on Pythia, then mapping their
emergence across the **(model scale × training step)** grid using Pythia's
public training checkpoints.

> **Status: 🚧 scaffolding.** Design spec lives at
> [`docs/superpowers/specs/2026-05-20-circuitprobe-design.md`](docs/superpowers/specs/2026-05-20-circuitprobe-design.md).
> Phases are being built one at a time.

---

## What this project is

**Reproduction.** Identify induction heads on Pythia-160M / 410M / 1.4B
using TransformerLens. Match Olsson '22 numbers within ~10%.

**Extension.** Pythia is the only major model family that ships training
checkpoints across its full training run. CircuitProbe sweeps a 3-size ×
12-checkpoint grid and asks: **does the timing of induction-head
emergence depend cleanly on scale, training compute, or both?** A
scaling law is fit to the emergence boundary with bootstrap 95% CIs.

**Custom kernel.** A single Triton kernel for fused prefix-matching
score extraction — the hot inner loop of induction-head identification.
Targets 1.5–3× over PyTorch eager on a T4.

**Writeup.** LessWrong-style post in `docs/writeup.md` with reproducible
figures and code.

## Reproduce

```bash
uv sync
make reproduce        # regenerates the headline figures from cached checkpoints
make test             # runs the full pytest suite
make bench            # runs the Triton kernel benchmark (CUDA only)
```

## Why this exists

Part of a numbered portfolio series:
[Crypt](https://github.com/NeilP211/crypt),
[Exfil](https://github.com/NeilP211/exfil),
[FitGraph](https://github.com/NeilP211/fitgraph),
[DistKV](https://github.com/NeilP211/distkv).
This one (Project 2) covers ML interpretability + low-level GPU
programming — the niche where Anthropic-style interpretability research
meets systems engineering.

## License

MIT. See [LICENSE](LICENSE).
