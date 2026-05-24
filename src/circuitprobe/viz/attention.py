"""CircuitsVis wrappers for attention-pattern figures.

Note: CircuitsVis is the standard Anthropic-mechinterp visualisation library;
this module returns HTML/string-able objects from `circuitsvis.attention.attention_pattern`
that can be embedded in notebooks or written to disk.
"""

from __future__ import annotations

import circuitsvis as cv
import torch
from transformer_lens import HookedTransformer

from ..data import random_repeat_seqs


@torch.no_grad()
def induction_head_attention_html(
    model: HookedTransformer,
    layer: int,
    head: int,
    half_len: int = 32,
    seed: int = 0,
) -> str:
    """Return an HTML snippet visualising the (layer, head) attention pattern
    on a random-repeat sequence -- the canonical induction-head figure.
    """
    bos = model.tokenizer.bos_token_id if model.tokenizer is not None else None
    seqs = random_repeat_seqs(
        vocab_size=model.cfg.d_vocab,
        half_len=half_len,
        n_seqs=1,
        seed=seed,
        bos_token=bos,
    ).to(model.cfg.device)

    _, cache = model.run_with_cache(
        seqs, names_filter=lambda n: n == f"blocks.{layer}.attn.hook_pattern"
    )
    pat = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # (seq, seq)

    tokens = [model.to_string(t) for t in seqs[0]]
    html = cv.attention.attention_pattern(tokens=tokens, attention=pat.cpu())
    return str(html)


@torch.no_grad()
def text_attention_html(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
    prepend_bos: bool = True,
) -> str:
    """Return a CircuitsVis attention-pattern snippet for arbitrary input text.

    Renders how the given (layer, head) attends across the tokens of `text`.
    On text with a repeated phrase, an induction head shows a visible stripe
    of attention from the second occurrence back to the token that followed
    the first occurrence.
    """
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n == f"blocks.{layer}.attn.hook_pattern"
    )
    pat = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # (seq, seq)
    str_tokens = model.to_str_tokens(tokens[0])
    html = cv.attention.attention_pattern(tokens=str_tokens, attention=pat.cpu())
    return str(html)
