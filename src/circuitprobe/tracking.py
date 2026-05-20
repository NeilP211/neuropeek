"""Thin wandb wrapper with graceful offline fallback.

If `WANDB_API_KEY` is not set, runs in `offline` mode (data persisted under
`./wandb/` for later sync). `log()` and `finish()` are no-ops if no run is
active.
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
        os.environ["WANDB_MODE"] = "offline"
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
