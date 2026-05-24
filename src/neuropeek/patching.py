"""Activation patching, head ablation, and path patching utilities.

We expose two primary primitives:

- `ablate_heads(model, heads, mode="zero")` - context manager that registers
  forward hooks zeroing (or mean-ablating) the `z` output of specified heads.
  Cleans up on exit.
- `path_patch(model, clean_tokens, corrupt_tokens, sender, receiver)` - runs
  the model on clean tokens, caches the sender activation, then runs on
  corrupt tokens replacing the sender's activation with the cached clean
  value, and returns the receiver activation under that patch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


@contextmanager
def ablate_heads(
    model: HookedTransformer,
    heads: list[tuple[int, int]],
    mode: Literal["zero", "mean"] = "zero",
    mean_activations: dict[int, torch.Tensor] | None = None,
) -> Iterator[None]:
    """Context manager that zero- or mean-ablates the listed (layer, head)s.

    Hooks the `hook_z` output of each head.
    """
    if mode == "mean" and mean_activations is None:
        raise ValueError("mode='mean' requires `mean_activations` keyed by layer")

    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    hook_points: list[HookPoint] = []
    try:
        for layer, head_list in by_layer.items():

            def make_hook(heads_in_layer: list[int], layer_idx: int):
                def hook(z: torch.Tensor, hook: HookPoint):  # noqa: ARG001
                    # z shape: (batch, seq, n_heads, d_head)
                    if mode == "zero":
                        z[:, :, heads_in_layer, :] = 0.0
                    else:
                        assert mean_activations is not None
                        mean_z = mean_activations[layer_idx]  # (n_heads, d_head)
                        for h in heads_in_layer:
                            z[:, :, h, :] = mean_z[h]
                    return z

                return hook

            hp = model.blocks[layer].attn.hook_z
            hp.add_hook(make_hook(head_list, layer))
            hook_points.append(hp)
        yield
    finally:
        for hp in hook_points:
            hp.remove_hooks()


@torch.no_grad()
def path_patch(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    sender_hook: str,
    receiver_hook: str,
) -> torch.Tensor:
    """Path-patch: replace `sender_hook` on the corrupt run with its clean value,
    then return the activation at `receiver_hook`.

    Both hook names are TransformerLens hook strings (e.g.
    "blocks.5.attn.hook_z").
    """
    _, clean_cache = model.run_with_cache(
        clean_tokens, names_filter=lambda n: n == sender_hook
    )
    clean_sender = clean_cache[sender_hook]

    def patch_hook(activation: torch.Tensor, hook: HookPoint):  # noqa: ARG001
        return clean_sender

    receiver_value: list[torch.Tensor] = []

    def grab_receiver(activation: torch.Tensor, hook: HookPoint):  # noqa: ARG001
        receiver_value.append(activation.detach().clone())
        return activation

    model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(sender_hook, patch_hook), (receiver_hook, grab_receiver)],
        return_type=None,
    )
    assert receiver_value, "receiver hook never fired"
    return receiver_value[0]
