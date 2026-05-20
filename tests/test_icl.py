"""Tests for in-context-learning loss-by-position."""

import torch

from circuitprobe import icl


def test_loss_by_position_shape():
    # Fake logits + tokens.
    n_seqs, seq_len, vocab = 4, 16, 100
    torch.manual_seed(0)
    logits = torch.randn(n_seqs, seq_len, vocab)
    tokens = torch.randint(0, vocab, (n_seqs, seq_len))
    loss = icl.loss_by_position(logits, tokens)
    assert loss.shape == (seq_len - 1,)


def test_loss_by_position_lower_for_repeated_token():
    # Construct a deterministic case: vocab=2, second half repeats the first.
    seq_len = 8
    half = torch.tensor([[0, 1, 0, 1]])
    tokens = torch.cat([half, half], dim=1)  # (1, 8)
    # Logits that predict perfectly in second half.
    logits = torch.full((1, seq_len, 2), -10.0)
    for t in range(seq_len - 1):
        next_tok = tokens[0, t + 1].item()
        logits[0, t, next_tok] = 10.0
    loss = icl.loss_by_position(logits, tokens)
    assert loss[3:].mean().item() < 0.01
