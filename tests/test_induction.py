"""Tests for induction-head identification.

We test the metric on a synthetic, hand-constructed attention tensor where
we know which heads "should" score high. No model loading required.
"""

import torch

from circuitprobe import induction


def make_synthetic_pattern(half_len: int, induction_head: bool) -> torch.Tensor:
    """Construct a (1, n_heads=1, seq_len=2N, seq_len=2N) attention pattern.

    If induction_head=True, attention from each position N+k attends to position k.
    Otherwise it attends uniformly.
    """
    seq_len = 2 * half_len
    pat = torch.zeros(1, 1, seq_len, seq_len)
    for q in range(seq_len):
        if induction_head and q >= half_len:
            target = q - half_len + 1  # offset-by-one induction (attends to next token of prev occurrence)
            target = min(target, seq_len - 1)
            pat[0, 0, q, target] = 1.0
        else:
            pat[0, 0, q, : q + 1] = 1.0 / (q + 1)
    return pat


def test_prefix_match_score_high_for_induction_pattern():
    pat = make_synthetic_pattern(half_len=8, induction_head=True)
    score = induction.prefix_match_score_from_pattern(pat, half_len=8)
    assert score.shape == (1, 1)
    assert score.item() > 0.8


def test_prefix_match_score_low_for_uniform_pattern():
    pat = make_synthetic_pattern(half_len=8, induction_head=False)
    score = induction.prefix_match_score_from_pattern(pat, half_len=8)
    assert score.item() < 0.2


def test_rank_induction_heads_orders_by_score():
    scores = torch.tensor([[0.1, 0.9], [0.5, 0.3]])
    ranked = induction.rank_induction_heads(scores, top_k=3)
    assert ranked[0] == (0, 1, 0.9)
    assert ranked[1] == (1, 0, 0.5)
    assert ranked[2] == (1, 1, 0.3)
