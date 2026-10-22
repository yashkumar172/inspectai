"""Model C: coreset selection, nearest-neighbor search, and the PatchCore-inspired model."""

import logging

import numpy as np
import pytest
import torch
from fake_backbone import FakeBackbone

from inspectai.errors import ModelError
from inspectai.models.base import Prediction
from inspectai.models.features import FeatureExtractor, WeightsInfo
from inspectai.models.patchcore import (
    PatchCoreConfig,
    PatchCoreModel,
    estimate_bank_gb,
    greedy_coreset,
    nearest_neighbor_distances,
    patchcore_config_for,
)

WEIGHTS = WeightsInfo("fake", "injected", None)


# ----- greedy coreset ----------------------------------------------------------------------


def _clusters() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    centers = torch.zeros(3, 8)
    centers[1, 0], centers[2, 1] = 30.0, 30.0
    return torch.cat([c + 0.1 * torch.randn(20, 8, generator=generator) for c in centers])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_coreset_takes_one_point_from_each_far_apart_cluster(seed: int) -> None:
    indices = greedy_coreset(_clusters(), 3, projection_dim=8, seed=seed)
    assert sorted((indices // 20).tolist()) == [0, 1, 2]


def test_greedy_k_center_on_a_line_follows_the_farthest_first_rule() -> None:
    """With a 1-D projection (a pure rescale), the selection is exactly farthest-first."""
    points = torch.arange(1000, dtype=torch.float32).unsqueeze(1)
    chosen = greedy_coreset(points, 11, projection_dim=1, seed=0).tolist()
    assert chosen[1] in (0, 999)  # the point farthest from any start is an end of the line
    radius = torch.cdist(points, points[chosen]).min(dim=1).values.max().item()
    optimal = 1000 / (2 * 11)  # best possible covering radius of 11 centers on this line
    assert radius <= 2 * optimal  # farthest-first is a 2-approximation of k-center


def test_coreset_indices_are_valid_unique_and_reproducible() -> None:
    points = torch.randn(300, 12, generator=torch.Generator().manual_seed(2))
    first = greedy_coreset(points, 40, seed=5)
    assert first.shape == (40,) and len(set(first.tolist())) == 40
    assert first.min() >= 0 and first.max() < 300
    assert torch.equal(first, greedy_coreset(points, 40, seed=5))
    assert not torch.equal(first, greedy_coreset(points, 40, seed=6))


def test_selecting_everything_returns_every_index() -> None:
    assert greedy_coreset(torch.randn(10, 4), 10).tolist() == list(range(10))


def test_coreset_selection_stops_at_its_time_limit_without_a_partial_result() -> None:
    points = torch.randn(500, 8)
    with pytest.raises(ModelError, match="random subsampling"):
        greedy_coreset(points, 200, max_seconds=-1.0)


@pytest.mark.parametrize("n_select", [0, 11, -3])
def test_an_invalid_coreset_size_is_rejected(n_select: int) -> None:
    with pytest.raises(ModelError):
        greedy_coreset(torch.randn(10, 4), n_select)


# ----- nearest-neighbor search ---------------------------------------------------------------


def test_nearest_neighbor_distances_match_a_brute_force_computation() -> None:
    generator = torch.Generator().manual_seed(3)
    queries, bank = (
        torch.randn(30, 6, generator=generator),
        torch.randn(100, 6, generator=generator),
    )
    expected = np.sqrt(((queries.numpy()[:, None, :] - bank.numpy()[None]) ** 2).sum(-1)).min(1)
    for chunk in (1, 7, 100, 1000):
        result = nearest_neighbor_distances(queries, bank, chunk_size=chunk)
        assert np.allclose(result.numpy(), expected, atol=1e-4), chunk


def test_a_query_that_is_in_the_bank_has_distance_zero() -> None:
    bank = torch.randn(50, 5, generator=torch.Generator().manual_seed(4))
    assert nearest_neighbor_distances(bank[:5], bank, 8).abs().max() < 1e-3


def test_nearest_neighbor_search_rejects_bad_shapes() -> None:
    with pytest.raises(ModelError):
        nearest_neighbor_distances(torch.zeros(3, 4), torch.zeros(5, 6))
    with pytest.raises(ModelError):
        nearest_neighbor_distances(torch.zeros(3), torch.zeros(5, 3))
    with pytest.raises(ModelError):
        nearest_neighbor_distances(torch.zeros(3, 4), torch.zeros(5, 4), chunk_size=0)


# ----- configuration -------------------------------------------------------------------------


def test_the_bank_estimate_follows_the_spec_formula() -> None:
    # SPEC.md Section 5.5: 450 images x 784 patches x 1536 dims x 4 bytes is about 2.2 GB (GiB 2.02).
    assert estimate_bank_gb(450, 784, 1536) == pytest.approx(450 * 784 * 1536 * 4 / 1024**3)
    assert 2.0 < estimate_bank_gb(450, 784, 1536) < 2.3


def test_profiles_select_the_backbones_and_spec_defaults() -> None:
    dev, full = patchcore_config_for("cashew", "dev"), patchcore_config_for("cashew", "full")
    assert (dev.backbone, dev.image_size, dev.layers) == ("resnet18", 224, ("layer2", "layer3"))
    assert full.backbone == "wide_resnet50_2"
    assert dev.coreset_ratio == 0.10 and dev.projection_dim == 128 and dev.neighborhood == 3
    assert dev.max_bank_gb == 4.0 and dev.max_fit_minutes == 30.0 and dev.num_threads == 6
    assert dev.resize_mode == "letterbox"


def test_the_snapshot_holds_model_settings_and_not_resource_limits() -> None:
    config = PatchCoreConfig(max_bank_gb=1.0, chunk_size=99, max_fit_minutes=5.0)
    assert PatchCoreConfig(max_bank_gb=2.0).to_dict() == PatchCoreConfig().to_dict()
    snapshot = config.to_dict()
    assert "max_bank_gb" not in snapshot and "chunk_size" not in snapshot
    assert config.runtime_dict() == {
        "chunk_size": 99,
        "batch_size": 8,
        "num_threads": 6,
        "max_bank_gb": 1.0,
        "max_fit_minutes": 5.0,
    }
    assert PatchCoreConfig.from_dict(snapshot, config.runtime_dict()) == config
    assert PatchCoreConfig(coreset_ratio=0.05).to_dict() != PatchCoreConfig().to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 4},
        {"resize_mode": "crop"},
        {"coreset_ratio": 0.0},
        {"coreset_ratio": 1.5},
        {"projection_dim": 0},
        {"neighborhood": 2},
        {"neighborhood": 0},
        {"smoothing_sigma": -1.0},
        {"chunk_size": 0},
        {"max_bank_gb": 0.0},
        {"max_fit_minutes": 0.0},
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ModelError):
        PatchCoreConfig(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ModelError):
        PatchCoreConfig.from_dict({"unknown": 1})


# ----- model ---------------------------------------------------------------------------------


def _images(n: int = 14, height: int = 48, width: int = 64, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(80, 120, (height, width, 3), dtype=np.uint8) for _ in range(n)]


def _model(mode: str = "stretch", **kwargs: object) -> PatchCoreModel:
    settings = {"layers": ("layer2", "layer3"), "image_size": 64, "resize_mode": mode}
    config = PatchCoreConfig(**{**settings, **kwargs})  # type: ignore[arg-type]
    return PatchCoreModel(config, FeatureExtractor(FakeBackbone(), config.layers), WEIGHTS)


def _fitted(mode: str = "stretch", **kwargs: object) -> PatchCoreModel:
    model = _model(mode, **kwargs)
    model.fit(_images())
    return model


@pytest.mark.parametrize("mode", ["stretch", "letterbox"])
@pytest.mark.parametrize("shape", [(48, 64), (64, 48), (50, 50)])
def test_the_map_has_the_input_size_and_the_score_is_its_maximum(
    mode: str, shape: tuple[int, int]
) -> None:
    prediction = _fitted(mode).predict(_images(1, *shape, seed=5)[0])
    assert isinstance(prediction, Prediction)
    assert prediction.anomaly_map.shape == shape and prediction.anomaly_map.dtype == np.float32
    assert prediction.score == pytest.approx(float(prediction.anomaly_map.max()))
    assert np.isfinite(prediction.anomaly_map).all()


def test_the_bank_is_reduced_to_the_coreset_ratio_of_all_patches() -> None:
    model = _fitted(coreset_ratio=0.1)
    # 14 images x 16 x 16 patches; layer2 (16 ch) + layer3 (24 ch) = 40 dims.
    assert model.n_candidates == 14 * 256 and model.n_fit_images == 14
    assert model.bank_size == round(0.1 * 14 * 256) and model.feature_dim == 40
    assert model.tensors()["bank"].shape == (model.bank_size, 40)
    assert model.coreset_seconds >= 0 and model.feature_map_shape == (16, 16)


def test_a_ratio_of_one_keeps_every_patch_and_fit_images_score_zero() -> None:
    """Exactness check: with the full bank, a fit image is its own nearest neighbor."""
    model = _fitted(coreset_ratio=1.0, neighborhood=1, smoothing_sigma=0.0)
    assert model.bank_size == model.n_candidates
    assert model.predict(_images()[3]).score == pytest.approx(0.0, abs=1e-3)
    assert model.predict(_images(1, seed=99)[0]).score > 0.01  # an unseen image does not


def test_a_defect_raises_the_score_and_the_map_peaks_near_it() -> None:
    model = _fitted(smoothing_sigma=2.0)
    normal = _images(1, seed=9)[0]
    defective = normal.copy()
    defective[10:26, 36:52] = 250
    clean, flagged = model.predict(normal), model.predict(defective)
    assert flagged.score > 2 * clean.score
    peak = np.unravel_index(flagged.anomaly_map.argmax(), flagged.anomaly_map.shape)
    assert 4 <= peak[0] <= 32 and 30 <= peak[1] <= 58


def test_fitting_and_prediction_are_deterministic_and_follow_the_seed() -> None:
    image = _images(1, seed=3)[0]
    first, second = _fitted(), _fitted()
    assert np.array_equal(first.tensors()["bank"], second.tensors()["bank"])
    assert first.predict(image).score == second.predict(image).score
    assert not np.array_equal(first.tensors()["bank"], _fitted(seed=7).tensors()["bank"])


def test_the_search_chunk_size_does_not_change_the_result() -> None:
    image = _images(1, seed=4)[0]
    small, large = _fitted(chunk_size=7), _fitted(chunk_size=100000)
    assert np.allclose(
        small.predict(image).anomaly_map, large.predict(image).anomaly_map, atol=1e-5
    )


def test_the_neighborhood_averaging_changes_the_features() -> None:
    image = _images(1, seed=4)[0]
    plain, pooled = _fitted(neighborhood=1), _fitted(neighborhood=3)
    assert not np.allclose(plain.predict(image).anomaly_map, pooled.predict(image).anomaly_map)
    assert plain.bank_size == pooled.bank_size


def test_the_memory_bank_estimate_is_logged_before_fitting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="inspectai.models.patchcore"):
        model = _fitted()
    message = next(
        r.getMessage() for r in caplog.records if "Memory bank estimate" in r.getMessage()
    )
    assert "14 images x 256 patches x 40 dims x 4 bytes" in message
    assert model.estimated_bank_gb == pytest.approx(estimate_bank_gb(14, 256, 40))


def test_a_bank_over_the_limit_stops_the_fit_with_guidance() -> None:
    model = _model(max_bank_gb=1e-9)
    with pytest.raises(ModelError) as caught:
        model.fit(_images())  # a list reports its length, so the estimate is checked first
    assert "max_bank_gb" in str(caught.value) and "max_fit_images" in str(caught.value)
    assert model.bank_size == 0
    with pytest.raises(ModelError, match="not been fitted"):
        model.predict(_images(1)[0])


def test_the_bank_limit_is_also_enforced_for_images_of_unknown_length() -> None:
    model = _model(max_bank_gb=1e-9)
    with pytest.raises(ModelError, match="max_bank_gb"):
        model.fit(image for image in _images())


def test_the_estimate_is_checked_before_any_image_beyond_the_first_batch() -> None:
    seen: list[int] = []

    class Counting:
        def __len__(self) -> int:
            return 14

        def __iter__(self):  # type: ignore[no-untyped-def]
            for index, image in enumerate(_images()):
                seen.append(index)
                yield image

    with pytest.raises(ModelError, match="max_bank_gb"):
        _model(max_bank_gb=1e-9, batch_size=4).fit(Counting())
    assert len(seen) <= 4  # stopped after the first batch was read


def test_an_exhausted_time_limit_stops_the_fit_without_a_silent_fallback() -> None:
    model = _model(max_fit_minutes=1e-9)
    with pytest.raises(ModelError, match="random subsampling"):
        model.fit(_images())
    assert model.bank_size == 0


def test_a_sized_input_that_yields_more_images_than_it_reports_is_rejected() -> None:
    class Lying:
        def __len__(self) -> int:
            return 4

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(_images(8))

    with pytest.raises(ModelError, match="More images"):
        _model().fit(Lying())


def test_unfitted_and_underfitted_models_are_rejected() -> None:
    with pytest.raises(ModelError, match="not been fitted"):
        _model().predict(_images(1)[0])
    with pytest.raises(ModelError, match="not been fitted"):
        _model().tensors()
    with pytest.raises(ModelError, match="at least 2"):
        _model().fit(_images(1))
    with pytest.raises(ModelError, match="at least 1"):
        _model().fit([])


def test_reference_aspect_ratio_is_the_median_of_the_fit_images() -> None:
    model = _model()
    model.fit(_images(4, 40, 40) + _images(4, 30, 60, seed=1) + _images(4, 30, 60, seed=2))
    assert model.reference_aspect_ratio == pytest.approx(2.0)


def test_predict_does_not_modify_the_input() -> None:
    model = _fitted()
    image = _images(1, seed=6)[0]
    before = image.copy()
    model.predict(image)
    assert np.array_equal(image, before)


def test_tensors_round_trip_through_from_tensors() -> None:
    model = _fitted()
    clone = PatchCoreModel.from_tensors(
        model.config,
        model.tensors(),
        FeatureExtractor(FakeBackbone(), model.config.layers),
        WEIGHTS,
        model.feature_map_shape,  # type: ignore[arg-type]
        model.reference_aspect_ratio,
        model.n_fit_images,
        model.n_candidates,
    )
    image = _images(1, seed=8)[0]
    assert model.predict(image).score == clone.predict(image).score
    assert clone.bank_size == model.bank_size and clone.feature_dim == model.feature_dim


@pytest.mark.parametrize(
    "bank", [None, np.zeros(5, dtype=np.float32), np.zeros((0, 4), np.float32)]
)
def test_from_tensors_rejects_a_missing_or_malformed_bank(bank: object) -> None:
    tensors = {} if bank is None else {"bank": bank}
    with pytest.raises(ModelError, match="bank"):
        PatchCoreModel.from_tensors(
            PatchCoreConfig(layers=("layer2",)),
            tensors,  # type: ignore[arg-type]
            FeatureExtractor(FakeBackbone(), ("layer2",)),
            WEIGHTS,
            (16, 16),
            None,
            2,
            10,
        )


def test_inference_runs_on_the_cpu() -> None:
    model = _fitted()
    assert model._bank is not None and model._bank.device.type == "cpu"
    assert model.predict(_images(1)[0]).score >= 0.0


def test_the_snapshot_holds_the_exact_configuration_used() -> None:
    config = PatchCoreConfig(max_bank_gb=1.5, chunk_size=99)
    snapshot = config.snapshot()
    assert snapshot["runtime"] == config.runtime_dict()
    assert {k: v for k, v in snapshot.items() if k != "runtime"} == config.to_dict()
    restored = PatchCoreConfig.from_dict(
        {k: v for k, v in snapshot.items() if k != "runtime"},
        snapshot["runtime"],  # type: ignore[arg-type]
    )
    assert restored == config
