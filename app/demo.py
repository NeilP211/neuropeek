"""CircuitProbe interactive demo.

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

from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch

from circuitprobe import models
from circuitprobe.viz.attention import text_attention_html

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

    # Attention pattern HTML for the chosen head.
    attn_html = text_attention_html(model, text, layer=int(layer), head=int(head))

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
    with gr.Blocks(title="CircuitProbe: see an induction head fire") as demo:
        gr.Markdown(
            "# CircuitProbe\n"
            "Type a sentence with a repeated phrase. The app loads "
            "Pythia-160M and shows how one attention head (the induction "
            "head **L4H6**) attends across your tokens. On repeated text you "
            "should see a bright stripe of attention pointing from the second "
            "occurrence of a word back to the token that came after the first "
            "occurrence. That stripe is the induction circuit doing its job: "
            "predict what came next last time."
        )

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
                    run_btn = gr.Button("Analyze", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=text_in)
                with gr.Column(scale=1):
                    prediction_out = gr.Textbox(
                        label="Model prediction", lines=10
                    )
            attn_out = gr.HTML(label="Attention pattern")

            run_btn.click(
                analyze,
                inputs=[text_in, layer_in, head_in],
                outputs=[attn_out, prediction_out],
            )

        with gr.Tab("The research finding"):
            gr.Markdown(
                "## Induction-head emergence is scale-invariant\n"
                "We swept 3 Pythia sizes (160M, 410M, 1.4B) across 12 "
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


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
