"""Model B: the Gaussian feature model, with the fake backbone."""

import numpy as np
import pytest
import torch
from fake_backbone import FakeBackbone

from inspectai.errors import ConfigError, ModelError
from inspectai.models.base import Prediction
from inspectai.models.features import FeatureExtractor, WeightsInfo, images_to_batch
from inspectai.models.gaussian import GaussianConfig, GaussianModel, gaussian_config_for

WEIGHTS = WeightsInfo("fake", "injected", None)


def _images(n: int = 14, height: int = 48, width: int = 64, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(80, 120, (height, width, 3), dtype=np.uint8) for _ in range(n)]


def _model(mode: str = "stretch", **kwargs: object) -> GaussianModel:
    settings = {"layers": ("layer2", "layer3"), "image_size": 64, "d": 20, "resize_mode": mode}
    config = GaussianConfig(**{**settings, **kwargs})  # type: ignore[arg-type]
    return GaussianModel(config, FeatureExtractor(FakeBackbone(), config.layers), WEIGHTS)


def _fitted(mode: str = "stretch", **kwargs: object) -> GaussianModel:
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


def test_a_defect_raises_the_score_and_the_map_peaks_near_it() -> None:
    model = _fitted(smoothing_sigma=2.0)
    normal = _images(1, seed=9)[0]
    defective = normal.copy()
    defective[10:26, 36:52] = 250
    clean, flagged = model.predict(normal), model.predict(defective)
    assert flagged.score > 2 * clean.score
    peak = np.unravel_index(flagged.anomaly_map.argmax(), flagged.anomaly_map.shape)
    assert 4 <= peak[0] <= 32 and 30 <= peak[1] <= 58


def test_fitting_and_prediction_are_deterministic() -> None:
    image = _images(1, seed=3)[0]
    first, second = _fitted().predict(image), _fitted().predict(image)
    assert first.score == second.score and np.array_equal(first.anomaly_map, second.anomaly_map)


def test_channel_selection_follows_the_seed() -> None:
    same = _fitted().tensors()["channel_index"], _fitted().tensors()["channel_index"]
    other = _fitted(seed=7).tensors()["channel_index"]
    assert np.array_equal(*same) and not np.array_equal(same[0], other)
    assert len(set(same[0].tolist())) == 20 and same[0].max() < 40


def test_means_and_inverse_covariances_match_an_independent_computation() -> None:
    model = _fitted(epsilon=0.01)
    extractor = FeatureExtractor(FakeBackbone(), ("layer2", "layer3"))
    batch, _ = images_to_batch(_images(), 64, "stretch")
    embedding = extractor.embed(batch)[:, torch.from_numpy(model.tensors()["channel_index"])]
    samples = embedding.permute(0, 2, 3, 1).reshape(14, 16 * 16, 20).double().numpy()
    tensors = model.tensors()
    for position in (0, 37, 255):
        values = samples[:, position, :]
        covariance = np.cov(values, rowvar=False) + 0.01 * np.eye(20)
        assert np.allclose(tensors["mean"][position], values.mean(axis=0), atol=1e-4)
        assert np.allclose(
            tensors["inv_cov"][position], np.linalg.inv(covariance), rtol=1e-2, atol=1e-3
        )


def test_score_of_an_image_equals_the_mahalanobis_distance_of_its_features() -> None:
    model = _fitted(smoothing_sigma=0.0)
    image = _images(1, seed=4)[0]
    batch, _ = images_to_batch([image], 64, "stretch")
    embedding = FeatureExtractor(FakeBackbone(), ("layer2", "layer3")).embed(batch)
    index = torch.from_numpy(model.tensors()["channel_index"])
    x = embedding[0, index].permute(1, 2, 0).reshape(256, 20).double().numpy()
    mean, inverse = model.tensors()["mean"].astype(np.float64), model.tensors()["inv_cov"]
    delta = x - mean
    distance = np.sqrt(np.einsum("pi,pij,pj->p", delta, inverse.astype(np.float64), delta))
    # The map is the bilinear upsampling of this 16x16 grid, so its maximum cannot exceed the
    # grid maximum and must be close to it.
    score = model.predict(image).score
    assert score <= distance.max() + 1e-4 and score > 0.5 * distance.max()


def test_the_reference_aspect_ratio_is_the_median_of_the_fit_images() -> None:
    model = _model()
    model.fit(_images(4, 40, 40) + _images(4, 30, 60, seed=1) + _images(4, 30, 60, seed=2))
    assert model.reference_aspect_ratio == pytest.approx(2.0)
    assert model.n_fit_images == 12


def test_predict_does_not_modify_the_input() -> None:
    model = _fitted()
    image = _images(1, seed=6)[0]
    before = image.copy()
    model.predict(image)
    assert np.array_equal(image, before)


def test_tensors_round_trip_through_from_tensors() -> None:
    model = _fitted()
    clone = GaussianModel.from_tensors(
        model.config,
        model.tensors(),
        FeatureExtractor(FakeBackbone(), model.config.layers),
        WEIGHTS,
        model.feature_map_shape,  # type: ignore[arg-type]
        model.total_channels,  # type: ignore[arg-type]
        model.reference_aspect_ratio,
        model.n_fit_images,
    )
    image = _images(1, seed=8)[0]
    assert model.predict(image).score == clone.predict(image).score


def test_from_tensors_rejects_wrong_shapes() -> None:
    model = _fitted()
    tensors = model.tensors()
    tensors["mean"] = tensors["mean"][:, :5]
    with pytest.raises(ModelError, match="mean"):
        GaussianModel.from_tensors(
            model.config,
            tensors,
            FeatureExtractor(FakeBackbone(), model.config.layers),
            WEIGHTS,
            (16, 16),
            40,
            None,
            14,
        )
    tensors = model.tensors()
    tensors["channel_index"] = tensors["channel_index"] + 100
    with pytest.raises(ModelError, match="out-of-range"):
        GaussianModel.from_tensors(
            model.config,
            tensors,
            FeatureExtractor(FakeBackbone(), model.config.layers),
            WEIGHTS,
            (16, 16),
            40,
            None,
            14,
        )


def test_unfitted_and_underfitted_models_are_rejected() -> None:
    with pytest.raises(ModelError, match="not been fitted"):
        _model().predict(_images(1)[0])
    with pytest.raises(ModelError, match="not been fitted"):
        _model().tensors()
    with pytest.raises(ModelError, match="at least 2"):
        _model().fit(_images(1))
    with pytest.raises(ModelError, match="at least 2"):
        _model().fit([])


def test_d_larger_than_the_channel_count_is_rejected() -> None:
    with pytest.raises(ModelError, match="exceeds"):
        _model(d=41).fit(_images())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 4},
        {"resize_mode": "crop"},
        {"d": 0},
        {"epsilon": 0.0},
        {"smoothing_sigma": -1.0},
        {"batch_size": 0},
        {"num_threads": 0},
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ModelError):
        GaussianConfig(**kwargs)  # type: ignore[arg-type]


def test_the_config_round_trips_through_a_dict() -> None:
    config = GaussianConfig(d=30, layers=("layer2",))
    assert GaussianConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ModelError):
        GaussianConfig.from_dict({"unknown": 1})


def test_profiles_select_the_backbones_of_the_spec() -> None:
    dev, full = gaussian_config_for("cashew", "dev"), gaussian_config_for("cashew", "full")
    assert (dev.backbone, dev.image_size, dev.layers) == ("resnet18", 224, ("layer2", "layer3"))
    assert full.backbone == "wide_resnet50_2" and full.d == 100
    assert dev.resize_mode == "letterbox"
    with pytest.raises(ConfigError, match="profile"):
        gaussian_config_for("cashew", "huge")


def test_inference_runs_on_the_cpu() -> None:
    model = _fitted()
    assert all(t.device.type == "cpu" for t in (model._mean, model._inv_cov))  # type: ignore[union-attr]
    assert model.predict(_images(1)[0]).score >= 0.0
