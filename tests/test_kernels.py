"""Tests for the prefix-match kernel backends."""

import pytest
import torch

from circuitprobe.kernels import prefix_match_torch


def test_torch_backend_correct_on_synthetic():
    # (batch=1, n_heads=1, seq=4, seq=4); seq=4, half_len=2
    pat = torch.zeros(1, 1, 4, 4)
    pat[0, 0, 2, 1] = 1.0  # q=2 -> k=2-2+1=1 ✓
    pat[0, 0, 3, 2] = 1.0  # q=3 -> k=3-2+1=2 ✓
    score = prefix_match_torch.prefix_match_score(pat, half_len=2)
    # Last valid q is seq_len - 1 = 3; q range is half_len=2 to seq_len-2=2.
    # So only q=2 counted; k=1; score = 1.0
    assert score.shape == (1, 1)
    assert score.item() == pytest.approx(1.0)


def test_torch_backend_zero_pattern_gives_zero_score():
    pat = torch.zeros(1, 4, 16, 16)
    score = prefix_match_torch.prefix_match_score(pat, half_len=8)
    assert torch.allclose(score, torch.zeros(1, 4))


def test_torch_backend_matches_induction_module_impl():
    # The kernel backend MUST agree with the reference impl in induction.py.
    from circuitprobe.induction import prefix_match_score_from_pattern
    torch.manual_seed(0)
    pat = torch.rand(2, 8, 32, 32)
    a = prefix_match_score_from_pattern(pat, half_len=16)
    b = prefix_match_torch.prefix_match_score(pat, half_len=16)
    assert torch.allclose(a, b, atol=1e-6)
