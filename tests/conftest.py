"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from fake_backbone import FakeBackbone
from make_synthetic import SyntheticVisA, build_synthetic_visa
from make_world import World, build_world


@pytest.fixture
def synthetic_visa(tmp_path: Path) -> SyntheticVisA:
    """A small VisA-style tree under a folder name with a space in it."""
    return build_synthetic_visa(tmp_path / "data root")


@pytest.fixture
def fake_backbone() -> FakeBackbone:
    return FakeBackbone()


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    """Synthetic data, reports, and artifacts for all models; tests must copy before changing."""
    return build_world(tmp_path_factory.mktemp("shared world"))
