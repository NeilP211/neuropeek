"""Induction-head identification metrics (prefix-matching + copying scores).

Definitions follow Olsson et al. (2022, "In-context Learning and Induction
Heads", appendix B).

prefix_match_score(layer, head):
    For random-repeat sequences r_1..r_N | r_1..r_N:
    measure the mean attention, in the second half, from position N+k+1 to
    position k+1 (i.e., the position that holds the token immediately after
    a repeat). Higher score means the head is implementing the induction
    pattern "look at where this token previously appeared, then look at the
    token right after it".

copying_score(layer, head):
    Approximate measure: probability that the OV-circuit of this head, applied
    to a fixed embedding, increases the logit of the same token. Implemented
    as the diagonal-fraction of W_E @ W_OV @ W_U for this head.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from . import data


def prefix_match_score_from_pattern(
    pattern: torch.Tensor,
    half_len: int,
) -> torch.Tensor:
    """Compute prefix-match score from a pre-computed attention pattern.

    pattern: (batch, n_heads, seq_len, seq_len) where seq_len == 2*half_len
             (assumes NO BOS token; if BOS used, strip it before calling).

    Returns: (batch, n_heads) — mean attention from query position q in the
             second half (q >= half_len) to key position q - half_len + 1.
    """
    batch, n_heads, seq_len, _ = pattern.shape
    assert seq_len == 2 * half_len, f"seq_len {seq_len} != 2*half_len {2 * half_len}"

    # For each q in [half_len, 2*half_len - 1], the target key is q - half_len + 1
    # (clamped). We gather attention[:, :, q, q - half_len + 1] across all q in
    # the second half.
    q_indices = torch.arange(half_len, seq_len - 1)  # exclude last (target would overflow)
    k_indices = q_indices - half_len + 1

    # Gather: result[b, h, i] = pattern[b, h, q_i, k_i]
    gathered = pattern[:, :, q_indices, k_indices]  # (batch, n_heads, len(q_indices))
    return gathered.mean(dim=-1)


@torch.no_grad()
def prefix_match_score(
    model: HookedTransformer,
    n_seqs: int = 64,
    half_len: int = 50,
    seed: int = 0,
) -> torch.Tensor:
    """Per-(layer, head) prefix-matching score on random-repeat sequences.

    Returns: (n_layers, n_heads) tensor on CPU.
    """
    vocab = model.cfg.d_vocab
    bos_token_id = None
    if model.tokenizer is not None:
        bos_token_id = model.tokenizer.bos_token_id

    seqs = data.random_repeat_seqs(
        vocab_size=vocab,
        half_len=half_len,
        n_seqs=n_seqs,
        seed=seed,
        bos_token=bos_token_id,
    ).to(str(model.cfg.device))

    bos_offset = 1 if bos_token_id is not None else 0

    _, cache = model.run_with_cache(
        seqs,
        return_type=None,
        names_filter=lambda name: name.endswith(".hook_pattern"),
    )

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        pat = cache[f"blocks.{layer}.attn.hook_pattern"]  # (batch, n_heads, seq, seq)
        # Strip BOS if present.
        if bos_offset:
            pat = pat[:, :, bos_offset:, bos_offset:]
        s = prefix_match_score_from_pattern(pat, half_len=half_len)  # (batch, n_heads)
        scores[layer] = s.mean(dim=0).cpu()

    return scores


@torch.no_grad()
def copying_score(model: HookedTransformer) -> torch.Tensor:
    """Per-(layer, head) copying score.

    Defined as the fraction of the diagonal of (W_E @ W_OV @ W_U) that is
    positive (Olsson '22, appendix B.2). This is a model-only computation —
    no data required.

    Returns: (n_layers, n_heads) tensor on CPU.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)

    W_E = model.W_E  # (vocab, d_model)
    W_U = model.W_U  # (d_model, vocab)

    for layer in range(n_layers):
        W_V = model.W_V[layer]  # (n_heads, d_model, d_head)
        W_O = model.W_O[layer]  # (n_heads, d_head, d_model)
        for head in range(n_heads):
            W_OV = W_V[head] @ W_O[head]  # (d_model, d_model)
            full = W_E @ W_OV @ W_U  # (vocab, vocab)
            diag = full.diag()
            scores[layer, head] = (diag > 0).float().mean().cpu()

    return scores


def rank_induction_heads(
    scores: torch.Tensor,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Return the top-k (layer, head, score) tuples sorted by score descending.

    Scores are rounded to 6 decimal places so float32 round-trip artifacts
    (e.g. 0.9 -> 0.8999999...) do not leak into the returned tuples.
    """
    n_layers, n_heads = scores.shape
    flat = [
        (layer, head, round(float(scores[layer, head].item()), 6))
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    flat.sort(key=lambda x: x[2], reverse=True)
    return flat[:top_k]
