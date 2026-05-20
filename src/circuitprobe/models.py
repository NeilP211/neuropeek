"""TransformerLens model loaders for Pythia and GPT-2-small.

Pythia checkpoints are loaded by passing ``revision="step{N}"`` to
``HookedTransformer.from_pretrained``, pinning the HuggingFace branch
that corresponds to training step ``N``. Used for the emergence-grid sweep.
"""

from __future__ import annotations

from typing import Literal

import torch
from transformer_lens import HookedTransformer

from . import checkpoints

PythiaSize = Literal["160M", "410M", "1.4B"]

_PYTHIA_HF_IDS: dict[str, str] = {
    "160M": "EleutherAI/pythia-160m",
    "410M": "EleutherAI/pythia-410m",
    "1.4B": "EleutherAI/pythia-1.4b",
}


def pythia_hf_id(size: str) -> str:
    if size not in _PYTHIA_HF_IDS:
        raise ValueError(f"Unknown Pythia size: {size!r}. Allowed: {list(_PYTHIA_HF_IDS)}")
    return _PYTHIA_HF_IDS[size]


def load_pythia(
    size: PythiaSize,
    step: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load Pythia-{size} at training step `step` (or final if None).

    Notes:
    - `step` must be a valid Pythia checkpoint (see `checkpoints.pythia_log_steps`).
    - Loads with `revision="step{N}"` via TransformerLens's HF passthrough.
    """
    hf_id = pythia_hf_id(size)
    revision = checkpoints.revision_for_step(step) if step is not None else None

    model = HookedTransformer.from_pretrained(
        hf_id,
        revision=revision,
        device=str(device),
        torch_dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


def load_gpt2_small(device: str | torch.device = "cpu") -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2", device=str(device))
    model.eval()
    return model
