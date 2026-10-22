"""Milestone 0 smoke tests: the package imports and the environment is CPU-only."""

import importlib.metadata
from pathlib import Path

import torch

import inspectai


def test_package_imports_with_version() -> None:
    assert inspectai.__version__ == importlib.metadata.version("inspectai")


def test_package_is_loaded_from_src_layout() -> None:
    package_dir = Path(inspectai.__file__).resolve().parent
    assert package_dir.name == "inspectai"
    assert package_dir.parent.name == "src"


def test_torch_runs_on_cpu_without_cuda() -> None:
    assert not torch.cuda.is_available()
    result = torch.ones(2, 2) @ torch.ones(2, 2)
    assert result.device.type == "cpu"
    assert result.tolist() == [[2.0, 2.0], [2.0, 2.0]]
