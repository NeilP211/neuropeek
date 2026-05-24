"""Probe data generators for induction-head analysis.

The canonical induction-head probe is a sequence of the form
    r1 r2 r3 ... rN | r1 r2 r3 ... rN
where r_i are random tokens (sampled uniformly from a vocab range). Induction
heads should attend (in the second half) from position N+k to position k, and
copy the token there to the output. This produces a sharp "prefix-matching
score" that distinguishes induction heads from other heads.

We also expose a Pile sample for in-context-learning loss curves (P4).
"""

from __future__ import annotations

import hashlib
import random

import torch


def random_repeat_seqs(
    vocab_size: int,
    half_len: int,
    n_seqs: int,
    seed: int = 0,
    bos_token: int | None = None,
) -> torch.Tensor:
    """Random-repeat sequences: [r_1 ... r_N | r_1 ... r_N], shape (n_seqs, 2N).

    If `bos_token` is provided, it is prepended once at position 0 (shape becomes
    (n_seqs, 2N + 1)).
    """
    if half_len < 1 or n_seqs < 1 or vocab_size < 2:
        raise ValueError("require half_len>=1, n_seqs>=1, vocab_size>=2")

    g = torch.Generator()
    g.manual_seed(seed)
    half = torch.randint(0, vocab_size, (n_seqs, half_len), generator=g, dtype=torch.long)
    full = torch.cat([half, half], dim=1)
    if bos_token is not None:
        bos = torch.full((n_seqs, 1), bos_token, dtype=torch.long)
        full = torch.cat([bos, full], dim=1)
    return full


def pile_sample(n_seqs: int, max_len: int, seed: int = 0) -> list[str]:
    """Deterministic Pile sample for in-context-learning loss curves.

    Uses `monology/pile-uncopyrighted` (smaller, license-clean Pile subset).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )
    # Deterministic skip-based subsample.
    rng = random.Random(seed)
    skip = rng.randint(0, 10_000)
    picked: list[str] = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        text = row["text"]
        if len(text) < 200:
            continue
        picked.append(text[: max_len * 8])  # rough char->token ratio
        if len(picked) >= n_seqs:
            break
    return picked


def stable_hash(*args: object) -> str:
    """Deterministic hash of inputs for caching."""
    h = hashlib.sha256()
    for a in args:
        h.update(repr(a).encode())
    return h.hexdigest()[:16]
