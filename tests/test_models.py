"""Tests for TransformerLens model loaders.

These tests mark the network-dependent loader test as `slow` so CI skips it.
We test the size→hf-id mapping and validation on CPU without downloading.
"""

import pytest

from neuropeek import models


def test_pythia_size_to_hf_id_known():
    assert models.pythia_hf_id("160M") == "EleutherAI/pythia-160m"
    assert models.pythia_hf_id("410M") == "EleutherAI/pythia-410m"
    assert models.pythia_hf_id("1.4B") == "EleutherAI/pythia-1.4b"


def test_pythia_size_to_hf_id_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown Pythia size"):
        models.pythia_hf_id("999B")


@pytest.mark.slow
def test_load_pythia_smallest_step_0_smoke():
    """Actually loads Pythia-160M @ step 0 from HF. Network + ~600MB. Marked slow."""
    model = models.load_pythia(size="160M", step=0, device="cpu")
    assert model.cfg.n_layers == 12
    assert model.cfg.d_model == 768
