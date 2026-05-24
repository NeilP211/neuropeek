"""NeuroPeek interactive demo.

A small Gradio app that makes induction heads visible. Type a sentence with
a repeated phrase, and the app loads Pythia-160M, shows the attention pattern
of the canonical induction head (L4H6) over your text, and shows the model's
top next-token predictions. A second tab shows the headline emergence map
from the (model scale x training step) sweep.

Run:
    uv run python app/demo.py
Then open the printed local URL (default http://127.0.0.1:7860).
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch

from neuropeek import models
from neuropeek.viz.attention import iframe_srcdoc, text_attention_html

# The canonical induction head identified for Pythia-160M in P3.
DEFAULT_LAYER = 4
DEFAULT_HEAD = 6

EXAMPLES = [
    "Mr and Mrs Dursley of number four Privet Drive. Mr and Mrs",
    "The cat sat on the mat. The cat sat on the",
    "Alice gave the book to Bob. Alice gave the book to",
    "1 2 3 4 5 1 2 3 4",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
HEATMAP_PNG = REPO_ROOT / "results" / "figures" / "emergence_heatmap.png"


# --- pixel-art brain logo -------------------------------------------------
# Hand-drawn pixel matrix. O = outline, P = fill, L = highlight.
_BRAIN = [
    "   OOO  OOO   ",
    "  OPPPOOPPPO  ",
    " OPPPPPPPPPPO ",
    " OPLPPOPPLPPO ",
    "OPPPPPPPPPPPPO",
    "OPPOPPPPPPOPPO",
    "OPPPPPOPPPPPPO",
    " OPPOPPPOPPPO ",
    " OPPPPPPPPPPO ",
    "  OPPPPPPPPO  ",
    "   OOPPPPOO   ",
    "     OOOO     ",
]
_BRAIN_COLORS = {"O": "#b3105f", "P": "#ff2e88", "L": "#ffb3d9"}


def pixel_brain_svg(px: int = 7) -> str:
    """Return a blocky pixel-art brain as a standalone inline SVG string."""
    w = max(len(r) for r in _BRAIN)
    h = len(_BRAIN)
    cells = "".join(
        f'<rect x="{x}" y="{y}" width="1" height="1" fill="{_BRAIN_COLORS[ch]}"/>'
        for y, row in enumerate(_BRAIN)
        for x, ch in enumerate(row)
        if ch in _BRAIN_COLORS
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w * px}" height="{h * px}" shape-rendering="crispEdges">'
        f"{cells}</svg>"
    )


HERO_HTML = f"""
<div class="np-hero">
  <div class="np-brain">{pixel_brain_svg(7)}</div>
  <div>
    <div class="np-title">NeuroPeek</div>
    <div class="np-tag">&#9658; peek inside a neural net's brain</div>
  </div>
</div>
"""

INTRO_MD = (
    "Type a sentence with a **repeated phrase**. I load Pythia-160M and show "
    "how one attention head (the induction head **L4H6**) attends across your "
    "tokens. On repeated text you should see a bright stripe of attention "
    "pointing from the second occurrence of a word back to the token that came "
    "after the first occurrence — that stripe is the induction circuit "
    "doing its job: predict what came next last time."
)

HELP_MD = (
    "**Rows** = the token doing the looking. **Columns** = the token being "
    "looked at. A bright cell means *this row's token is paying attention to "
    "that column's token*.\n\n"
    "- **The induction stripe** sits in the lower-left: each repeated word "
    "lights up the cell pointing back to *whatever came right after it last "
    "time*. That is the circuit predicting the repeat.\n"
    "- **The bright first column** (the `<|endoftext|>` token) is a normal "
    "\"attention sink\" — ignore it; the real signal is the diagonal "
    "stripe.\n"
    "- **Hover** over any row to highlight exactly where that token is looking.\n\n"
    "Tip: drag the **Head** slider off **6** and the stripe disappears — "
    "proof that it is *that one specific head* doing the work."
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

.gradio-container, .gradio-container.dark {
  --body-background-fill:#0b0710;
  --background-fill-primary:#150e22;
  --background-fill-secondary:#1d1433;
  --block-background-fill:#150e22;
  --block-border-color:#ff2e8855;
  --border-color-primary:#ff2e8855;
  --border-color-accent:#ff2e88;
  --block-label-text-color:#ff9ecb;
  --block-title-text-color:#ff9ecb;
  --body-text-color:#f6e9f1;
  --body-text-color-subdued:#c98ab5;
  --button-primary-background-fill:#ff2e88;
  --button-primary-background-fill-hover:#ff5ca3;
  --button-primary-text-color:#0b0710;
  --button-secondary-background-fill:#2a1a3d;
  --button-secondary-text-color:#ff9ecb;
  --input-background-fill:#0f0a18;
  --input-border-color:#ff2e8855;
  --link-color:#ff9ecb;
  --color-accent:#ff2e88;
  --color-accent-soft:#2a1030;
  --table-border-color:#ff2e8833;
  background:#0b0710 !important;
}

/* readable retro body everywhere */
.gradio-container, .gradio-container * {
  font-family:'VT323', ui-monospace, monospace !important;
}
.gradio-container p, .gradio-container li, .gradio-container textarea,
.gradio-container input, .gradio-container .prose {
  font-size:1.18rem !important; letter-spacing:.4px; line-height:1.3;
}

/* pixel font for the chrome (title, headings, tabs, labels, action button) */
.np-title { font-family:'Press Start 2P', monospace !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.tab-nav button, .block-label, label span, #np-analyze button {
  font-family:'Press Start 2P', monospace !important;
  letter-spacing:0 !important;
}
.gradio-container h2 { font-size:1rem !important; color:#ff2e88 !important;
  text-shadow:0 0 6px #ff2e8866; line-height:1.7; }
.tab-nav button { font-size:.62rem !important; }
.block-label, label span { font-size:.6rem !important; color:#ff9ecb !important; }
#np-analyze button { font-size:.8rem !important; padding:16px !important;
  box-shadow:0 0 14px #ff2e8899; border:2px solid #ff7ab8 !important; }

/* hero + floating brain */
.np-hero { display:flex; align-items:center; gap:20px; margin:6px 0 0; }
.np-brain { animation:np-float 3s ease-in-out infinite;
  filter:drop-shadow(0 0 6px #ff2e88cc); }
@keyframes np-float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
.np-title { font-size:2.1rem; color:#ff2e88;
  text-shadow:0 0 10px #ff2e88, 4px 4px 0 #5e0a33; line-height:1.1; }
.np-tag { font-family:'VT323',monospace !important; font-size:1.4rem;
  color:#ff9ecb; margin-top:10px; }

/* frame the attention figure so it reads as a bright island on the dark page */
#np-attn iframe { border:3px solid #ff2e88 !important; border-radius:4px;
  background:#fff; box-shadow:0 0 16px #ff2e8855; }

/* subtle CRT scanlines */
.gradio-container::after { content:""; position:fixed; inset:0; z-index:9999;
  pointer-events:none; opacity:.18;
  background:repeating-linear-gradient(0deg, transparent 0 2px, rgba(0,0,0,.6) 2px 3px); }
"""


@lru_cache(maxsize=1)
def _get_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return models.load_pythia(size="160M", step=None, device=device)


@torch.no_grad()
def analyze(text: str, layer: int, head: int):
    text = (text or "").strip()
    if not text:
        return "<p>Type some text above, then press Analyze.</p>", "Nothing to predict yet."

    model = _get_model()

    # Attention pattern HTML for the chosen head. CircuitsVis renders via an
    # inline <script>, which gr.HTML will not execute -- wrap it in an iframe
    # srcdoc so the browser runs the script and the figure actually draws.
    attn_html = text_attention_html(model, text, layer=int(layer), head=int(head))
    attn_html = iframe_srcdoc(attn_html, height=560)

    # Next-token prediction (what the model thinks comes next).
    tokens = model.to_tokens(text, prepend_bos=True)
    logits = model(tokens)
    last = logits[0, -1, :]
    probs = torch.softmax(last, dim=-1)
    top_probs, top_ids = probs.topk(8)
    lines = ["Top next-token predictions:\n"]
    for p, tid in zip(top_probs.tolist(), top_ids.tolist(), strict=True):
        tok = model.to_string(torch.tensor([tid])).replace("\n", "\\n")
        lines.append(f"  {tok!r}: {p:.1%}")
    prediction = "\n".join(lines)

    return attn_html, prediction


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NeuroPeek - see an induction head fire") as demo:
        gr.HTML(HERO_HTML)
        gr.Markdown(INTRO_MD)

        with gr.Tab("Watch the head fire"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Input text",
                        value=EXAMPLES[0],
                        lines=2,
                    )
                    with gr.Row():
                        layer_in = gr.Slider(
                            0, 11, value=DEFAULT_LAYER, step=1, label="Layer"
                        )
                        head_in = gr.Slider(
                            0, 11, value=DEFAULT_HEAD, step=1, label="Head"
                        )
                    run_btn = gr.Button(
                        "▶ ANALYZE", variant="primary", elem_id="np-analyze"
                    )
                    gr.Examples(examples=EXAMPLES, inputs=text_in)
                with gr.Column(scale=1):
                    prediction_out = gr.Textbox(
                        label="Model prediction", lines=10
                    )

            with gr.Accordion("❔ HOW TO READ THE MAP BELOW", open=False):
                gr.Markdown(HELP_MD)
            attn_out = gr.HTML(elem_id="np-attn")

            run_btn.click(
                analyze,
                inputs=[text_in, layer_in, head_in],
                outputs=[attn_out, prediction_out],
            )

        with gr.Tab("The research finding"):
            gr.Markdown(
                "## Induction-head emergence is scale-invariant\n"
                "I swept 3 Pythia sizes (160M, 410M, 1.4B) across 12 "
                "training checkpoints and measured when each one grows its "
                "induction head. All three sizes flip from no-induction "
                "(dark) to mature-induction (yellow) at the same training "
                "step (1000), regardless of how big the model is."
            )
            if HEATMAP_PNG.exists():
                gr.Image(value=str(HEATMAP_PNG), label="Emergence map")
            else:
                gr.Markdown(
                    "_Emergence map not found. Run "
                    "`uv run python scripts/finalize_grid.py` to generate it._"
                )

    return demo


def _write_favicon() -> str:
    """Write the pixel-brain SVG to a temp file for use as the browser favicon."""
    path = Path(tempfile.gettempdir()) / "neuropeek_favicon.svg"
    path.write_text(pixel_brain_svg(8), encoding="utf-8")
    return str(path)


def main() -> None:
    app = build_app()
    app.launch(theme=gr.themes.Base(), css=CSS, favicon_path=_write_favicon())


if __name__ == "__main__":
    main()
