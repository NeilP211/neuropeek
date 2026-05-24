"""Identify induction heads on Pythia-160M (final checkpoint). Manual smoke test.

Run: uv run python scripts/identify_pythia_160m.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from neuropeek import induction, models


def main() -> None:
    print("Loading Pythia-160M (final step)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = models.load_pythia(size="160M", step=None, device=device)

    print("Computing prefix-matching score (64 seqs of half_len=50)...")
    pm = induction.prefix_match_score(model, n_seqs=64, half_len=50, seed=0)

    print("Computing copying score...")
    cp = induction.copying_score(model)

    top = induction.rank_induction_heads(pm, top_k=10)
    print("\nTop 10 induction heads by prefix-matching score:")
    for layer, head, score in top:
        cp_h = cp[layer, head].item()
        print(f"  L{layer}H{head}: prefix_match={score:.3f}, copying={cp_h:.3f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/pythia_160m_heads.json").write_text(
        json.dumps(
            {
                "prefix_match": pm.tolist(),
                "copying": cp.tolist(),
                "top_10": [{"layer": layer, "head": head, "score": s} for layer, head, s in top],
            },
            indent=2,
        )
    )
    print("\nResults written to results/pythia_160m_heads.json")


if __name__ == "__main__":
    main()
