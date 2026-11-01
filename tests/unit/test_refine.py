"""Edge-aware map refinement, the multiscale grid, and the added settings' compatibility."""

import numpy as np
import pytest
import torch
from fake_backbone import FakeBackbone

from inspectai.errors import ModelError
from inspectai.models.features import FeatureExtractor, WeightsInfo, images_to_batch
from inspectai.models.gaussian import GaussianConfig, GaussianModel, gaussian_config_for
from inspectai.models.patchcore import PatchCoreConfig, PatchCoreModel, patchcore_config_for
from inspectai.models.refine import guided_filter, luminance_guide, refine_map

WEIGHTS = WeightsInfo("fake", "injected", None)


def _step_image(size: int = 64) -> np.ndarray:
    guide = np.zeros((size, size), dtype=np.float32)
    guide[:, size // 2 :] = 1.0  # a vertical edge in the middle
    return guide


# ----- guided filter ---------------------------------------------------------------------------


def test_the_guided_filter_keeps_a_constant_map_constant() -> None:
    guide = np.random.default_rng(0).random((40, 40)).astype(np.float32)
    source = np.full((40, 40), 3.5, dtype=np.float32)
    assert np.allclose(guided_filter(guide, source, 5, 0.01), 3.5, atol=1e-4)


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    side = 2 * radius + 1
    padded = np.pad(image, radius, mode="symmetric")  # cv2.BORDER_REFLECT repeats the edge pixel
    out = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = padded[y : y + side, x : x + side].mean()
    return out


def test_a_flat_guide_gives_a_box_average_of_a_box_average() -> None:
    # flat guide: a = 0 and b = box(source), so the output is box(b) = box(box(source))
    source = np.random.default_rng(1).random((30, 30)).astype(np.float32)
    flat = np.full((30, 30), 0.4, dtype=np.float32)
    out = guided_filter(flat, source, 3, 0.01)
    assert np.allclose(out, _box(_box(source, 3), 3), atol=1e-4)


def test_the_guided_filter_follows_the_guide_edge_where_gaussian_blurs_it() -> None:
    guide = _step_image()
    # a score map whose boundary coincides with the image edge
    source = guide.copy()
    guided = refine_map(source, guide, "guided", sigma=4.0, radius=8, eps=0.01)
    gaussian = refine_map(source, guide, "gaussian", sigma=4.0)
    row = 32
    # one pixel either side of the edge: guided stays on its own side, Gaussian smears across
    guided_jump = guided[row, 32] - guided[row, 31]
    gaussian_jump = gaussian[row, 32] - gaussian[row, 31]
    assert guided_jump > 0.9
    assert gaussian_jump < 0.5 * guided_jump
    # far from the edge the guided output is still the source
    assert abs(guided[row, 5]) < 0.05 and abs(guided[row, 60] - 1.0) < 0.05


def test_the_guided_filter_does_not_sharpen_beyond_the_source_range() -> None:
    rng = np.random.default_rng(3)
    guide = rng.random((48, 48)).astype(np.float32)
    source = rng.random((48, 48)).astype(np.float32)
    out = guided_filter(guide, source, 4, 0.05)
    assert np.isfinite(out).all() and out.dtype == np.float32
    assert out.min() > source.min() - 0.5 and out.max() < source.max() + 0.5


def test_the_gaussian_path_is_exactly_the_original_blur() -> None:
    import cv2

    source = np.random.default_rng(4).random((50, 50)).astype(np.float32)
    expected = cv2.GaussianBlur(source, (0, 0), 4.0, borderType=cv2.BORDER_REFLECT)
    assert np.array_equal(refine_map(source, None, "gaussian", 4.0), expected)
    assert np.array_equal(refine_map(source, None, "gaussian", 0.0), source)


def test_invalid_refinement_settings_are_rejected() -> None:
    source = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ModelError, match="refinement_filter"):
        refine_map(source, source, "median", 1.0)
    with pytest.raises(ModelError, match="guide"):
        refine_map(source, None, "guided", 1.0)
    with pytest.raises(ModelError, match="guided_radius"):
        guided_filter(source, source, 0, 0.01)
    with pytest.raises(ModelError, match="guided_eps"):
        guided_filter(source, source, 2, 0.0)
    with pytest.raises(ModelError, match="same 2-D shape"):
        guided_filter(source, np.zeros((9, 9), dtype=np.float32), 2, 0.01)


def test_the_luminance_guide_inverts_the_model_normalization() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, 8:] = (255, 255, 255)
    batch, _ = images_to_batch([image], 16, "stretch")
    guide = luminance_guide(batch[0])
    assert guide.shape == (16, 16) and guide.dtype == np.float32
    assert np.allclose(guide[:, :8], 0.0, atol=1e-5) and np.allclose(guide[:, 8:], 1.0, atol=1e-5)


# ----- multiscale grid -------------------------------------------------------------------------


def test_layer1_is_pooled_to_the_layer2_grid_without_adding_patches() -> None:
    backbone = FakeBackbone()
    batch = torch.randn(2, 3, 64, 64)
    legacy = FeatureExtractor(backbone, ("layer2", "layer3")).embed(batch)
    multi = FeatureExtractor(backbone, ("layer1", "layer2", "layer3"), "layer2").embed(batch)
    assert legacy.shape == (2, 16 + 24, 16, 16)
    assert multi.shape == (2, 8 + 16 + 24, 16, 16)  # same grid, 8 more channels
    # the layer2 and layer3 channels are untouched; layer1 is the exact 2x2 mean of its map
    assert torch.equal(multi[:, 8:], legacy)
    layer1 = backbone.layer1(batch)
    assert torch.allclose(multi[:, :8], torch.nn.functional.avg_pool2d(layer1, 2), atol=1e-6)


def test_the_default_grid_is_still_the_first_layer() -> None:
    batch = torch.randn(1, 3, 64, 64)
    embedding = FeatureExtractor(FakeBackbone(), ("layer1", "layer2")).embed(batch)
    assert embedding.shape[-2:] == (32, 32)  # unchanged legacy behavior: first layer's grid


def test_a_grid_layer_must_be_one_of_the_layers() -> None:
    with pytest.raises(ModelError, match="grid_layer"):
        FeatureExtractor(FakeBackbone(), ("layer2", "layer3"), "layer1")
    with pytest.raises(ModelError, match="grid_layer"):
        PatchCoreConfig(layers=("layer2", "layer3"), grid_layer="layer1")
    with pytest.raises(ModelError, match="grid_layer"):
        GaussianConfig(layers=("layer2", "layer3"), grid_layer="layer1")


# ----- configuration compatibility -------------------------------------------------------------


def test_default_configurations_serialize_exactly_as_before() -> None:
    patchcore = PatchCoreConfig().to_dict()
    gaussian = GaussianConfig().to_dict()
    for added in ("refinement_filter", "guided_radius", "guided_eps", "grid_layer"):
        assert added not in patchcore and added not in gaussian
    assert PatchCoreConfig.from_dict(patchcore) == PatchCoreConfig()
    assert GaussianConfig.from_dict(gaussian) == GaussianConfig()


def test_non_default_refinement_and_multiscale_are_recorded_and_round_trip() -> None:
    config = patchcore_config_for("cashew", multiscale=True, refinement_filter="guided")
    data = config.to_dict()
    assert data["layers"] == ["layer1", "layer2", "layer3"] and data["grid_layer"] == "layer2"
    assert data["refinement_filter"] == "guided"
    assert PatchCoreConfig.from_dict(data) == config
    gaussian = gaussian_config_for("cashew", multiscale=True, refinement_filter="guided")
    assert GaussianConfig.from_dict(gaussian.to_dict()) == gaussian
    assert patchcore_config_for("cashew") == PatchCoreConfig(resize_mode="letterbox")


# ----- end to end with the fake backbone -------------------------------------------------------


def _images() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    base = np.linspace(40, 200, 64, dtype=np.float32)[None, :, None]
    return [
        np.clip(base + rng.normal(0, 4, (64, 64, 3)), 0, 255).astype(np.uint8) for _ in range(10)
    ]


@pytest.mark.parametrize("kind", ["patchcore", "gaussian"])
def test_a_guided_multiscale_model_fits_predicts_and_keeps_the_map_geometry(kind: str) -> None:
    layers, grid = ("layer1", "layer2", "layer3"), "layer2"
    common = {
        "layers": layers,
        "grid_layer": grid,
        "image_size": 64,
        "resize_mode": "stretch",
        "refinement_filter": "guided",
        "guided_radius": 4,
        "batch_size": 4,
        "num_threads": 1,
    }
    extractor = FeatureExtractor(FakeBackbone(), layers, grid)
    if kind == "patchcore":
        model = PatchCoreModel(PatchCoreConfig(**common, coreset_ratio=0.2), extractor, WEIGHTS)
    else:
        model = GaussianModel(GaussianConfig(**common, d=20), extractor, WEIGHTS)
    model.fit(_images())
    test = _images()[0].copy()
    test[20:40, 20:40] = 255  # a bright defect
    prediction = model.predict(test)
    assert prediction.anomaly_map.shape == test.shape[:2]
    assert np.isfinite(prediction.anomaly_map).all()
    assert prediction.score == pytest.approx(float(prediction.anomaly_map.max()))
    assert prediction.score > model.predict(_images()[1]).score  # the defect scores higher
