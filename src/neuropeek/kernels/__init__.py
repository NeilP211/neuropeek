"""Kernel-backend selector.

`prefix_match_score(pattern, half_len)` dispatches to the Triton kernel when
CUDA is available, otherwise to the PyTorch reference.
"""

from __future__ import annotations

import torch

from .prefix_match_torch import prefix_match_score as _torch_impl


def prefix_match_score(pattern: torch.Tensor, half_len: int) -> torch.Tensor:
    if pattern.is_cuda:
        try:
            from .prefix_match_triton import prefix_match_score as _triton_impl
            return _triton_impl(pattern, half_len)
        except ImportError:
            pass
    return _torch_impl(pattern, half_len)
