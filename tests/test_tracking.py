"""Tests for the wandb tracking wrapper."""

from neuropeek import tracking


def test_init_offline_when_no_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    run = tracking.init_run(project="neuropeek-test", config={"foo": 1})
    assert run is not None
    tracking.log({"metric": 0.5})
    tracking.finish()


def test_log_noops_without_init():
    tracking.finish()  # ensure no active run
    tracking.log({"x": 1})  # must not raise
