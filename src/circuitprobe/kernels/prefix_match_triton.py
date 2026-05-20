"""Triton kernel for fused prefix-match score extraction.

Fuses the index-gather + mean-reduction along the (q, k) diagonal in a
single launch, eliminating the intermediate (batch, n_heads, n_q) tensor.
Runs only on CUDA. Falls back to PyTorch via the backend selector.

Layout:
    pattern: (B, H, S, S), float32 or float16, contiguous.
    output : (B, H), float32.

Each program instance handles one (batch, head) pair. We stride over the
n_q = half_len - 1 positions and accumulate, then divide.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _prefix_match_kernel(
        pattern_ptr,      # *fp16/fp32, shape (B*H, S*S)
        out_ptr,          # *fp32, shape (B*H,)
        n_q,              # int — number of q positions to average over
        half_len,         # int — sequence half length
        S,                # int — full seq_len (= 2*half_len)
        BLOCK_Q: tl.constexpr,
    ):
        bh_id = tl.program_id(0)

        # Per-program accumulator.
        acc = tl.zeros((), dtype=tl.float32)

        # Stride through n_q positions in chunks of BLOCK_Q.
        for q_start in tl.range(0, n_q, BLOCK_Q):
            q_offsets = q_start + tl.arange(0, BLOCK_Q)
            mask = q_offsets < n_q

            # For each q in [half_len, half_len + n_q - 1], gather pattern[q, q-half_len+1].
            q_idx = q_offsets + half_len
            k_idx = q_offsets + 1  # = q_idx - half_len + 1

            flat_offsets = bh_id * (S * S) + q_idx * S + k_idx
            vals = tl.load(pattern_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(vals, axis=0)

        mean = acc / n_q
        tl.store(out_ptr + bh_id, mean)


    def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:
        assert pattern.is_cuda, "Triton kernel requires CUDA"
        B, H, S, _ = pattern.shape
        assert 2 * half_len == S
        n_q = half_len - 1  # q runs in [half_len, S - 2]; len = half_len - 1

        pattern_c = pattern.contiguous()
        out = torch.empty(B * H, device=pattern.device, dtype=torch.float32)

        grid = (B * H,)
        _prefix_match_kernel[grid](
            pattern_c.view(-1),
            out,
            n_q,
            half_len,
            S,
            BLOCK_Q=64,
            num_warps=2,
        )
        return out.view(B, H)

else:

    def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:  # pragma: no cover
        raise ImportError("Triton is not installed")
