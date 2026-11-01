"""The fast coreset and search kernels against brute-force references on synthetic vectors."""

import pytest
import torch

from inspectai.errors import ModelError
from inspectai.models.patchcore import (
    PatchCoreConfig,
    greedy_coreset,
    nearest_neighbor_distances,
)


def _brute_force_coreset(
    features: torch.Tensor, n_select: int, projection_dim: int, seed: int
) -> list[int]:
    """Greedy k-center that recomputes every point's distance to ALL selected points each step.

    It draws the projection and the start point exactly like `greedy_coreset`, but shares no
    update logic with it: no running minimum, no pruning.
    """
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(features.shape[1], projection_dim, generator=generator)
    projected = features @ (projection / projection_dim**0.5)
    start = int(torch.randint(features.shape[0], (1,), generator=generator))
    chosen = [start]
    while len(chosen) < n_select:
        distances = torch.cdist(projected, projected[chosen]).min(dim=1).values
        distances[chosen] = -1.0
        chosen.append(int(torch.argmax(distances)))
    return chosen


@pytest.fixture
def clustered() -> torch.Tensor:
    """Points around a few centers: the structure the pruning is meant to exploit."""
    generator = torch.Generator().manual_seed(7)
    centers = torch.randn(12, 24, generator=generator) * 4.0
    labels = torch.randint(12, (600,), generator=generator)
    return centers[labels] + torch.randn(600, 24, generator=generator)


@pytest.mark.parametrize("backend", ["exact", "tiled"])
def test_both_backends_match_a_brute_force_greedy_selection(
    clustered: torch.Tensor, backend: str
) -> None:
    expected = _brute_force_coreset(clustered, 60, 16, seed=3)
    selected = greedy_coreset(clustered, 60, projection_dim=16, seed=3, backend=backend)
    assert selected.tolist() == expected
    assert len(set(selected.tolist())) == 60


def test_the_pruned_selection_equals_the_scan_on_unstructured_vectors() -> None:
    features = torch.randn(900, 40, generator=torch.Generator().manual_seed(11))
    scan = greedy_coreset(features, 90, seed=5, backend="exact")
    pruned = greedy_coreset(features, 90, seed=5, backend="tiled")
    assert torch.equal(scan, pruned)


def test_selecting_everything_returns_every_index_for_both_backends() -> None:
    features = torch.randn(20, 8)
    for backend in ("exact", "tiled"):
        assert greedy_coreset(features, 20, backend=backend).tolist() == list(range(20))


def test_the_time_limit_stops_both_backends_without_a_partial_result() -> None:
    features = torch.randn(400, 8)
    for backend in ("exact", "tiled"):
        with pytest.raises(ModelError, match="time limit"):
            greedy_coreset(features, 300, max_seconds=0.0, backend=backend)


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ModelError, match="backend"):
        greedy_coreset(torch.randn(10, 4), 3, backend="hnsw")
    with pytest.raises(ModelError, match="backend"):
        nearest_neighbor_distances(torch.randn(3, 4), torch.randn(5, 4), backend="hnsw")
    with pytest.raises(ModelError, match="index_backend"):
        PatchCoreConfig(index_backend="hnsw")


@pytest.mark.parametrize("chunk", [1, 7, 64, 10_000])
def test_the_tiled_search_agrees_with_cdist_to_float_rounding(chunk: int) -> None:
    generator = torch.Generator().manual_seed(2)
    bank = torch.randn(130, 32, generator=generator)
    queries = torch.randn(40, 32, generator=generator) + 0.5
    exact = nearest_neighbor_distances(queries, bank, chunk_size=chunk, backend="exact")
    tiled = nearest_neighbor_distances(queries, bank, chunk_size=chunk, backend="tiled")
    assert torch.allclose(tiled, exact, rtol=1e-4, atol=1e-5)
    brute_force = torch.cdist(queries, bank).min(dim=1).values  # all pairs at once
    assert torch.allclose(exact, brute_force)


def test_a_query_equal_to_a_bank_vector_has_distance_zero_in_both_kernels() -> None:
    bank = torch.randn(50, 16, generator=torch.Generator().manual_seed(4))
    for backend in ("exact", "tiled"):
        distances = nearest_neighbor_distances(bank[:5], bank, backend=backend)
        assert torch.all(distances >= 0.0) and torch.all(distances < 1e-2)


def test_the_index_backend_is_a_runtime_setting_that_leaves_default_snapshots_unchanged() -> None:
    default, tiled = PatchCoreConfig(), PatchCoreConfig(index_backend="tiled")
    assert default.snapshot() == PatchCoreConfig().snapshot()
    assert "index_backend" not in default.runtime_dict()  # earlier artifacts stay valid
    assert tiled.runtime_dict()["index_backend"] == "tiled"
    assert tiled.to_dict() == default.to_dict()  # not part of the fitted model's definition
