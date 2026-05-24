"""CircuitsVis wrappers for attention-pattern figures.

Note: CircuitsVis is the standard Anthropic-mechinterp visualisation library;
this module returns HTML/string-able objects from `circuitsvis.attention.attention_pattern`
that can be embedded in notebooks or written to disk.
"""

from __future__ import annotations

import html as _html

import circuitsvis as cv
import torch
from transformer_lens import HookedTransformer

from ..data import random_repeat_seqs


def iframe_srcdoc(inner_html: str, height: int = 560) -> str:
    """Wrap an HTML snippet in an ``<iframe srcdoc>`` so its inline scripts run.

    CircuitsVis renders its attention widget via an inline module ``<script>``.
    Hosts like Gradio's ``gr.HTML`` inject content the way ``innerHTML`` does,
    and the browser deliberately does *not* execute ``<script>`` tags inserted
    that way -- so the widget stays a blank, zero-height div. Embedding the
    snippet as an iframe ``srcdoc`` makes the browser parse it as a real
    document, so the script executes and the figure draws.
    """
    # The attention grid's height grows with the number of tokens, so a fixed
    # iframe height clips longer inputs. A tiny resize script inside the (same-
    # origin, un-sandboxed) srcdoc grows the host iframe to fit its content.
    resize = (
        "<script>(function(){function f(){try{var h=document.body.scrollHeight;"
        "if(window.frameElement)window.frameElement.style.height=(h+8)+'px';}"
        "catch(e){}}new ResizeObserver(f).observe(document.body);"
        "window.addEventListener('load',f);})();</script>"
    )
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;font-family:system-ui,-apple-system,sans-serif;}</style>"
        f"</head><body>{inner_html}{resize}</body></html>"
    )
    srcdoc = _html.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%; height:{height}px; border:0;" '
        'title="attention pattern"></iframe>'
    )


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
