"""Tests for patching + ablation. Uses GPT-2-small for speed."""

import pytest
import torch

from neuropeek import patching


@pytest.fixture(scope="module")
def gpt2_small():
    from neuropeek import models
    return models.load_gpt2_small(device="cpu")


def test_head_ablate_changes_logits(gpt2_small):
    model = gpt2_small
    tokens = model.to_tokens("The quick brown fox")
    base_logits = model(tokens)

    with patching.ablate_heads(model, heads=[(5, 1)]):
        ablated_logits = model(tokens)

    diff = (base_logits - ablated_logits).abs().sum().item()
    assert diff > 0.0, "ablating a head should change logits"


def test_head_ablate_restores_after_context(gpt2_small):
    model = gpt2_small
    tokens = model.to_tokens("The quick brown fox")
    base_logits = model(tokens)

    with patching.ablate_heads(model, heads=[(5, 1)]):
        _ = model(tokens)

    after_logits = model(tokens)
    assert torch.allclose(base_logits, after_logits, atol=1e-5)


def test_zero_ablate_zeroes_head_z(gpt2_small):
    model = gpt2_small
    tokens = model.to_tokens("hi")
    with patching.ablate_heads(model, heads=[(0, 0)], mode="zero"):
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "hook_z" in n)
    assert cache["blocks.0.attn.hook_z"][:, :, 0, :].abs().sum().item() == 0.0
