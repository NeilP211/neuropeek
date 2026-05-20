"""Smoke test: package importable and version accessible."""

import circuitprobe


def test_package_version():
    assert circuitprobe.__version__ == "0.1.0"


def test_package_exposes_version_attribute():
    assert hasattr(circuitprobe, "__version__")
