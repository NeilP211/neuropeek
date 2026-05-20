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
