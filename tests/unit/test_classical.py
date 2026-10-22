"""The classical baseline (SPEC.md Section 5.1)."""

from pathlib import Path

import numpy as np
import pytest

from inspectai.errors import ModelError
from inspectai.models.base import Prediction
from inspectai.models.classical import ClassicalConfig, ClassicalModel


def _normal_images(n: int = 12, height: int = 30, width: int = 40, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [
        np.clip(100 + rng.normal(0, 8, (height, width, 3)), 0, 255).astype(np.uint8)
        for _ in range(n)
    ]


def _fitted(mode: str = "stretch", **kwargs: float) -> ClassicalModel:
    model = ClassicalModel(ClassicalConfig(image_size=32, resize_mode=mode, **kwargs))
    model.fit(_normal_images())
    return model


@pytest.mark.parametrize("mode", ["stretch", "letterbox"])
@pytest.mark.parametrize("shape", [(30, 40), (40, 30), (31, 31)])
def test_the_map_has_the_input_image_size_and_the_score_is_its_maximum(
    mode: str, shape: tuple[int, int]
) -> None:
    model = _fitted(mode)
    image = _normal_images(1, *shape, seed=5)[0]
    prediction = model.predict(image)
    assert isinstance(prediction, Prediction)
    assert prediction.anomaly_map.shape == shape and prediction.anomaly_map.dtype == np.float32
    assert prediction.score == pytest.approx(float(prediction.anomaly_map.max()))


def test_a_defect_raises_the_score_and_is_localized() -> None:
    model = _fitted()
    normal = _normal_images(1, seed=9)[0]
    defective = normal.copy()
    defective[10:20, 15:25] = 250
    normal_prediction, defect_prediction = model.predict(normal), model.predict(defective)
    assert defect_prediction.score > 3 * normal_prediction.score
    peak = np.unravel_index(
        defect_prediction.anomaly_map.argmax(), defect_prediction.anomaly_map.shape
    )
    assert 8 <= peak[0] <= 22 and 13 <= peak[1] <= 27


def test_fitting_and_prediction_are_deterministic() -> None:
    image = _normal_images(1, seed=3)[0]
    first, second = _fitted().predict(image), _fitted().predict(image)
    assert first.score == second.score
    assert np.array_equal(first.anomaly_map, second.anomaly_map)


def test_predict_does_not_modify_the_input() -> None:
    model = _fitted()
    image = _normal_images(1, seed=4)[0]
    before = image.copy()
    model.predict(image)
    assert np.array_equal(image, before)


def test_unfitted_and_underfitted_models_are_rejected() -> None:
    image = _normal_images(1)[0]
    with pytest.raises(ModelError, match="not been fitted"):
        ClassicalModel().predict(image)
    with pytest.raises(ModelError, match="at least 2"):
        ClassicalModel(ClassicalConfig(image_size=32)).fit(_normal_images(1))


def test_the_fit_reflects_only_the_images_it_was_given() -> None:
    bright = [np.full((30, 40, 3), 200, dtype=np.uint8) + i for i in range(4)]
    dark = [np.full((30, 40, 3), 50, dtype=np.uint8) + i for i in range(4)]
    model = ClassicalModel(ClassicalConfig(image_size=32, resize_mode="stretch"))
    model.fit(bright)
    assert model.n_fit_images == 4
    assert model.predict(dark[0]).score > model.predict(bright[0]).score


def test_smoothing_can_be_disabled() -> None:
    model = _fitted(smoothing_sigma=0.0)
    assert model.predict(_normal_images(1, seed=6)[0]).anomaly_map.shape == (30, 40)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 4},
        {"resize_mode": "crop"},
        {"smoothing_sigma": -1.0},
        {"std_floor": 0.0},
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ModelError):
        ClassicalConfig(**kwargs)  # type: ignore[arg-type]


def test_a_prediction_with_invalid_values_is_rejected() -> None:
    with pytest.raises(ModelError):
        Prediction(score=1.0, anomaly_map=np.zeros((3, 3, 3), dtype=np.float32))
    with pytest.raises(ModelError):
        Prediction(score=float("nan"), anomaly_map=np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ModelError):
        Prediction(score=1.0, anomaly_map=np.full((3, 3), np.inf, dtype=np.float32))


def test_the_baseline_runs_on_the_cpu_with_numpy_and_opencv_only() -> None:
    # The model module imports no deep-learning framework, so it cannot use a GPU.
    import inspectai.models.classical as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source and "from torch" not in source
    assert _fitted().predict(_normal_images(1)[0]).score >= 0.0


# ----- tensors and aspect ratio (for the saved artifact) -------------------------------------


def test_the_reference_aspect_ratio_is_the_median_of_the_fit_images() -> None:
    model = ClassicalModel(ClassicalConfig(image_size=32, resize_mode="stretch"))
    wide = _normal_images(4, 30, 60, seed=1) + _normal_images(4, 30, 60, seed=2)
    model.fit(_normal_images(4, 40, 40) + wide)
    assert model.reference_aspect_ratio == pytest.approx(2.0)


def test_tensors_round_trip_and_give_identical_predictions() -> None:
    model = _fitted("letterbox")
    tensors = model.tensors()
    assert set(tensors) == {"mean", "std"} and tensors["mean"].shape == (32, 32, 3)
    assert tensors["mean"].dtype == np.float32 and (tensors["std"] > 0).all()
    clone = ClassicalModel.from_tensors(
        model.config, tensors, model.n_fit_images, model.reference_aspect_ratio
    )
    image = _normal_images(1, seed=8)[0]
    assert model.predict(image).score == clone.predict(image).score
    assert np.array_equal(model.predict(image).anomaly_map, clone.predict(image).anomaly_map)
    tensors["mean"][0, 0, 0] = -123.0  # the returned tensors are copies of the model state
    assert model.tensors()["mean"][0, 0, 0] != -123.0


def test_from_tensors_rejects_wrong_shapes_and_non_positive_std() -> None:
    model = _fitted()
    config = model.config
    good = model.tensors()
    for bad in (
        {"mean": good["mean"][:, :5], "std": good["std"]},
        {"mean": good["mean"]},
        {"mean": good["mean"], "std": np.zeros_like(good["std"])},
    ):
        with pytest.raises(ModelError):
            ClassicalModel.from_tensors(config, bad, 12, None)
    with pytest.raises(ModelError, match="not been fitted"):
        ClassicalModel().tensors()
