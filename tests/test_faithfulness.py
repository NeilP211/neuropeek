"""Tests for circuit faithfulness."""

import torch  # noqa: F401

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
