"""In-context-learning curves: per-position cross-entropy loss.

The diagnostic plot from Olsson '22 figure 1: average per-token loss as a
function of position in the sequence. As context accumulates, well-trained
models develop a sharp drop in loss at certain positions — interpreted as
in-context-learning kicking in. We reproduce this curve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from . import data


def loss_by_position(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy loss averaged over the batch.

    logits: (batch, seq, vocab) — predictions for positions 0..seq-1.
    tokens: (batch, seq)        — the actual tokens.

    Returns (seq - 1,) — mean CE loss at each predicted position (1..seq-1).
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq-1, vocab)
    targets = tokens[:, 1:]  # (batch, seq-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq-1)
    return nll.mean(dim=0)


@torch.no_grad()
def icl_curve_on_pile(
    model: HookedTransformer,
    n_seqs: int = 32,
    max_len: int = 512,
    seed: int = 0,
) -> torch.Tensor:
    """Compute the in-context-learning loss-by-position curve on a Pile sample."""
    texts = data.pile_sample(n_seqs=n_seqs, max_len=max_len, seed=seed)
    tokens = model.to_tokens(texts, prepend_bos=True)
    tokens = tokens[:, :max_len]
    logits = model(tokens)
    return loss_by_position(logits, tokens).cpu()
