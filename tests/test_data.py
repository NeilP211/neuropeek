"""Tests for probe data generators."""

import torch

from circuitprobe import data


def test_random_repeat_seqs_shape():
    seqs = data.random_repeat_seqs(vocab_size=1000, half_len=32, n_seqs=8, seed=0)
    assert seqs.shape == (8, 64)
    assert seqs.dtype == torch.long


def test_random_repeat_seqs_actually_repeats():
    seqs = data.random_repeat_seqs(vocab_size=1000, half_len=16, n_seqs=4, seed=0)
    first_half = seqs[:, :16]
    second_half = seqs[:, 16:]
    assert torch.equal(first_half, second_half)


def test_random_repeat_seqs_deterministic_with_seed():
    a = data.random_repeat_seqs(1000, 16, 4, seed=42)
    b = data.random_repeat_seqs(1000, 16, 4, seed=42)
    assert torch.equal(a, b)


def test_random_repeat_seqs_different_seeds_differ():
    a = data.random_repeat_seqs(1000, 16, 4, seed=1)
    b = data.random_repeat_seqs(1000, 16, 4, seed=2)
    assert not torch.equal(a, b)


def test_random_repeat_seqs_respects_vocab_bounds():
    seqs = data.random_repeat_seqs(vocab_size=100, half_len=16, n_seqs=4, seed=0)
    assert seqs.min().item() >= 0
    assert seqs.max().item() < 100
