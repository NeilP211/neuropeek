"""Thin wandb wrapper with graceful no-tracking fallback.

If neither ``WANDB_API_KEY`` nor ``WANDB_MODE`` is set we default to
``WANDB_MODE=disabled`` so wandb is a strict no-op. Offline mode is opt-in
via ``WANDB_MODE=offline`` — it can hang under long sweeps on some wandb
versions, and the JSONL/Parquet outputs are already authoritative.

``log()`` and ``finish()`` are no-ops if no run is active or wandb isn't
installed.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import wandb
    _HAVE_WANDB = True
except ImportError:
    _HAVE_WANDB = False

_RUN: Any = None


def init_run(project: str, config: dict[str, Any] | None = None, name: str | None = None) -> Any:
    global _RUN
    if not _HAVE_WANDB:
        return None
    if "WANDB_API_KEY" not in os.environ and "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "disabled"
    _RUN = wandb.init(project=project, config=config or {}, name=name, reinit=True)
    return _RUN


def log(payload: dict[str, Any]) -> None:
    if _RUN is None or not _HAVE_WANDB:
        return
    wandb.log(payload)


def finish() -> None:
    global _RUN
    if _RUN is None or not _HAVE_WANDB:
        return
    wandb.finish()
    _RUN = None
