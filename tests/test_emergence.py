"""Tests for emergence-grid runner (mocked model loads)."""

from dataclasses import asdict

import torch

from circuitprobe import emergence


def test_cell_result_is_serialisable():
    cell = emergence.CellResult(
        size="160M",
        step=1000,
        max_prefix_match=0.42,
        mean_prefix_match=0.05,
        top_heads=[(4, 7, 0.42), (5, 1, 0.41)],
        seed=0,
    )
    d = asdict(cell)
    assert d["size"] == "160M"
    assert d["step"] == 1000
    assert len(d["top_heads"]) == 2


def test_summarise_scores_picks_max_and_mean():
    scores = torch.tensor([[0.1, 0.3], [0.4, 0.2]])
    mx, mn, top = emergence.summarise_scores(scores, top_k=2)
    assert mx == 0.4
    assert mn == 0.25
    assert top[0] == (1, 0, 0.4)
    assert top[1] == (0, 1, 0.3)
