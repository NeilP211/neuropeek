"""Scaling-law fits + bootstrap confidence intervals.

We answer: at which training step does each model size first cross a
prefix-match threshold (e.g. 0.3)? Then we fit a power law to
    emergence_step  vs  model_size_params
and report a 95% bootstrap CI on the exponent.
"""

from __future__ import annotations

from collections.abc import Callable

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
    for s, v in zip(steps, values, strict=True):
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

    points = [
        (SIZE_TO_PARAMS[s], per_size[s]["emerge_step"])
        for s in sizes
        if per_size[s]["emerge_step"] is not None
    ]
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
