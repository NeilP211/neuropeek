"""Benchmark the Triton vs PyTorch prefix-match kernel on CUDA.

Run on a Colab T4: `uv run python scripts/bench_kernel.py`.
Reports median + IQR ms across N trials at several shapes, plus the speedup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from neuropeek.kernels import prefix_match_torch


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
    from neuropeek.kernels import prefix_match_triton  # noqa: F401

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
