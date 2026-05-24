"""Smoke test: package importable and version accessible."""

import neuropeek


def test_package_version():
    assert neuropeek.__version__ == "0.1.0"


def test_package_exposes_version_attribute():
    assert hasattr(neuropeek, "__version__")
