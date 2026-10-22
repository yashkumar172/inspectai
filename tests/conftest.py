"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_synthetic import SyntheticVisA, build_synthetic_visa


@pytest.fixture
def synthetic_visa(tmp_path: Path) -> SyntheticVisA:
    """A small VisA-style tree under a folder name with a space in it."""
    return build_synthetic_visa(tmp_path / "data root")
