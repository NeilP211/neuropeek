"""PyTorch eager reference for the fused prefix-match score kernel.

Signature mirrors the Triton variant: takes an attention pattern tensor and
half-length, returns (batch, n_heads) score. This is the implementation the
Triton kernel must match for correctness tests.
"""

from __future__ import annotations

import torch


def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:
    """Reference PyTorch impl.

    pattern: (batch, n_heads, seq_len, seq_len), seq_len == 2 * half_len.
    returns: (batch, n_heads).
    """
    batch, n_heads, seq_len, _ = pattern.shape
    assert seq_len == 2 * half_len

    q_indices = torch.arange(half_len, seq_len - 1, device=pattern.device)
    k_indices = q_indices - half_len + 1
    gathered = pattern[:, :, q_indices, k_indices]
    return gathered.mean(dim=-1)
