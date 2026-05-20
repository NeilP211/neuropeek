"""Circuit faithfulness scoring.

Given:
- L_full   = loss with the full model
- L_others = loss when we ablate everything *except* the proposed circuit
- L_all    = loss when we ablate everything (i.e., remove all of attention)

Define:
    faithfulness = (L_others - L_full) / (L_all - L_full)

Intuition: if our proposed circuit fully captures the behaviour, ablating
non-circuit components shouldn't matter (L_others ~ L_full) -> faithfulness ~ 0.

NOTE: we use the *complement*-style definition used by Wang et al. 2022 and
Conmy et al. 2023 - faithfulness near 0 means "circuit suffices"; near 1
means "circuit insufficient". (Some authors invert this; we follow the
Conmy convention.)

We additionally expose a "recovery" form: 1 - faithfulness, so that closer
to 1 means "circuit captures behaviour", which reads better in figures.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from .induction import prefix_match_score


def faithfulness_from_losses(
    loss_full: float,
    loss_ablate_others: float,
    loss_ablate_all: float,
) -> float:
    denom = loss_ablate_all - loss_full
    if abs(denom) < 1e-9:
        return 0.0
    return (loss_ablate_others - loss_full) / denom


def recovery(faithfulness: float) -> float:
    """1 - faithfulness, clamped to [0, 1]. Reads as 'circuit captures behaviour'."""
    return max(0.0, min(1.0, 1.0 - faithfulness))


@torch.no_grad()
def circuit_prefix_match_recovery(
    model: HookedTransformer,
    circuit_heads: list[tuple[int, int]],
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Measure how much of the prefix-matching behaviour is recovered by
    the proposed circuit.

    Returns (mean_full, mean_circuit_only, mean_ablated_circuit).

    - mean_full: average prefix-match score across all heads (baseline reference).
    - mean_circuit_only: same metric but with NON-circuit heads ablated.
    - mean_ablated_circuit: same metric but with circuit heads ablated (sanity).
    """
    from .patching import ablate_heads

    pm_full = prefix_match_score(model, n_seqs=n_seqs, half_len=half_len, seed=seed)
    circuit_mean_full = float(sum(pm_full[layer, head].item() for layer, head in circuit_heads)) / max(
        len(circuit_heads), 1
    )

    all_heads = [
        (layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
        if (layer, head) not in set(circuit_heads)
    ]
    with ablate_heads(model, heads=all_heads, mode="zero"):
        pm_circuit_only = prefix_match_score(model, n_seqs=n_seqs, half_len=half_len, seed=seed)
    circuit_only_mean = float(
        sum(pm_circuit_only[layer, head].item() for layer, head in circuit_heads)
    ) / max(len(circuit_heads), 1)

    with ablate_heads(model, heads=list(circuit_heads), mode="zero"):
        pm_ablated = prefix_match_score(model, n_seqs=n_seqs, half_len=half_len, seed=seed)
    ablated_mean = float(pm_ablated.mean().item())

    return circuit_mean_full, circuit_only_mean, ablated_mean
