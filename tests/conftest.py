"""Shared pytest fixtures. Deterministic seeds for reproducibility."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    seed = 1623
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
