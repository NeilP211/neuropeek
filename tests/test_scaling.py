"""Tests for scaling-law fits."""

import numpy as np

from neuropeek import scaling


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
