"""The predictor: model registry, loading through the artifact store, and inspection."""

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fake_backbone import FakeBackbone
from make_world import World

from inspectai.artifacts.store import MANIFEST, TENSORS
from inspectai.data.dataset import VisADataset
from inspectai.errors import ArtifactError, ModelUnavailableError, WeightsNotFoundError
from inspectai.evaluation.comparison import RUN_SPECS
from inspectai.inference.predictor import (
    NOT_A_PROBABILITY,
    STATUS_ANOMALY,
    STATUS_NORMAL,
    WRONG_CATEGORY_WARNING,
    CalibrationRange,
    Inspection,
    Predictor,
    available_categories,
    available_options,
    describe_score,
    details_dict,
    details_json,
    inspect_image,
    is_available,
    load_predictor,
    model_options,
    option_by_key,
    panel_png,
)
from inspectai.models.base import Prediction
from inspectai.visualization.heatmaps import anomaly_map_to_heatmap

OPTIONS = {o.key: o for o in model_options()}
CLASSICAL, GAUSSIAN, PATCHCORE = OPTIONS["classical"], OPTIONS["gaussian"], OPTIONS["patchcore"]


def _dataset(world: World) -> VisADataset:
    return VisADataset(world.synthetic.split_csv, world.synthetic.images_root)


def _load(world: World, option, category: str = "cashew") -> Predictor:
    return load_predictor(world.artifacts, option, category, backbone=FakeBackbone())


# ----- registry ------------------------------------------------------------------------------


def test_the_options_follow_the_comparison_report_registry() -> None:
    assert [o.label for o in model_options()] == [s.label for s in RUN_SPECS]
    assert [o.key for o in model_options()] == [
        "classical",
        "gaussian",
        "gaussian-full",
        "patchcore",
        "patchcore-full",
    ]
    assert option_by_key("patchcore-full").profile == "full"
    with pytest.raises(ModelUnavailableError, match="Unknown"):
        option_by_key("nope")


def test_available_models_and_categories_come_from_the_saved_artifacts(world: World) -> None:
    keys = [o.key for o in available_options(world.artifacts)]
    assert keys == ["classical", "gaussian", "patchcore"]  # no `full` artifacts exist here
    assert available_categories(world.artifacts, GAUSSIAN) == ["cashew", "pipe_fryum"]
    assert is_available(world.artifacts, PATCHCORE, "cashew")
    assert not is_available(world.artifacts, OPTIONS["patchcore-full"], "cashew")
    assert not is_available(world.artifacts, GAUSSIAN, "fryum")
    assert available_options(world.artifacts.parent / "nowhere") == []


# ----- loading -------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["classical", "gaussian", "patchcore"])
def test_a_loaded_predictor_carries_the_stored_threshold_and_calibration(
    world: World, key: str
) -> None:
    option = OPTIONS[key]
    predictor = _load(world, option)
    manifest = json.loads(
        (world.artifacts / option.artifact_folder / "cashew" / MANIFEST).read_text("utf-8")
    )
    summary = json.loads(
        (world.reports / option.key / "cashew" / "summary.json").read_text("utf-8")
    )
    cal = manifest["calibration"]["scores"]
    assert predictor.threshold == manifest["threshold"]["value"]
    assert predictor.threshold == summary["threshold"]["image"]["value"]  # as evaluated
    assert predictor.threshold_policy == "normal_percentile"
    assert predictor.calibration == CalibrationRange(
        cal["n"], cal["min"], cal["median"], cal["max"]
    )
    assert predictor.calibration.minimum == summary["calibration_scores"]["min"]
    assert predictor.calibration.maximum == summary["calibration_scores"]["max"]
    assert predictor.pixel_threshold == manifest["calibration"]["pixel"]["value"]
    assert predictor.reference_aspect_ratio == pytest.approx(64 / 48)  # the synthetic images


@pytest.mark.parametrize("key", ["classical", "gaussian", "patchcore"])
def test_inspection_reproduces_every_saved_prediction_and_decision(world: World, key: str) -> None:
    """The live decision equals the evaluation report: same score, same stored threshold."""
    dataset = _dataset(world)
    predictor = _load(world, OPTIONS[key])
    by_rel = {s.image_rel: s for s in dataset.get_samples("cashew")}
    path = world.reports / OPTIONS[key].key / "cashew" / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    for row in rows:
        result = inspect_image(predictor, dataset.load_image(by_rel[row["image"]]))
        assert result.score == pytest.approx(float(row["score"]), abs=1e-6)
        assert result.threshold == float(row["threshold"])
        assert result.status == (STATUS_ANOMALY if int(row["predicted"]) else STATUS_NORMAL)
        assert result.is_anomaly == (result.score > result.threshold)


def test_a_missing_artifact_raises_instead_of_producing_anything(world: World) -> None:
    with pytest.raises(ModelUnavailableError) as caught:
        load_predictor(world.artifacts, OPTIONS["patchcore-full"], "cashew")
    assert "C:" not in str(caught.value) and "\\" not in str(caught.value)
    with pytest.raises(ModelUnavailableError):
        load_predictor(world.artifacts, GAUSSIAN, "fryum")


def test_a_corrupted_artifact_is_rejected_by_the_integrity_checks(
    world: World, tmp_path: Path
) -> None:
    shutil.copytree(world.artifacts, tmp_path / "artifacts")
    tensors = tmp_path / "artifacts" / "gaussian" / "cashew" / TENSORS
    data = bytearray(tensors.read_bytes())
    data[-1] ^= 0xFF
    tensors.write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="Checksum"):
        load_predictor(tmp_path / "artifacts", GAUSSIAN, "cashew", backbone=FakeBackbone())
    # the classical baseline goes through the same checks
    classical = tmp_path / "artifacts" / "classical" / "cashew" / TENSORS
    classical.write_bytes(classical.read_bytes()[:-8])
    with pytest.raises(ArtifactError):
        load_predictor(tmp_path / "artifacts", CLASSICAL, "cashew")


def test_an_artifact_filed_under_the_wrong_profile_is_rejected(
    world: World, tmp_path: Path
) -> None:
    shutil.copytree(world.artifacts / "gaussian", tmp_path / "gaussian-full")
    with pytest.raises(ArtifactError, match="another profile"):
        load_predictor(tmp_path, OPTIONS["gaussian-full"], "cashew", backbone=FakeBackbone())


def test_missing_pretrained_weights_are_reported_not_replaced(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty torch home"))
    monkeypatch.delenv("INSPECTAI_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    with pytest.raises(WeightsNotFoundError):  # no injected backbone: the cache is consulted
        load_predictor(world.artifacts, GAUSSIAN, "cashew")
    assert _load(world, CLASSICAL).option is CLASSICAL  # the baseline needs no weights at all


# ----- inspection ----------------------------------------------------------------------------


def test_inspection_outputs_have_the_image_size_and_a_fixed_color_scale(world: World) -> None:
    dataset = _dataset(world)
    predictor = _load(world, GAUSSIAN)
    image = dataset.load_image(dataset.get_samples("cashew", "test", "anomaly")[0])
    before = image.copy()
    result = inspect_image(predictor, image)
    assert isinstance(result, Inspection) and np.array_equal(image, before)
    assert result.anomaly_map.shape == image.shape[:2]
    assert result.heatmap.shape == image.shape and result.overlay.shape == image.shape
    assert result.color_scale_max == 2.0 * predictor.pixel_threshold
    assert np.array_equal(
        result.heatmap, anomaly_map_to_heatmap(result.anomaly_map, 0.0, result.color_scale_max)
    )
    assert result.seconds > 0 and result.score == pytest.approx(float(result.anomaly_map.max()))
    again = inspect_image(predictor, image)
    assert again.score == result.score and np.array_equal(again.overlay, result.overlay)


def test_the_color_scale_does_not_depend_on_the_image_shown(world: World) -> None:
    dataset = _dataset(world)
    predictor = _load(world, PATCHCORE)
    scales = {
        inspect_image(predictor, dataset.load_image(s)).color_scale_max
        for s in dataset.get_samples("cashew", "test")
    }
    assert len(scales) == 1


def test_the_decision_uses_the_stored_threshold_not_any_other(world: World) -> None:
    class Constant:
        def __init__(self, score: float) -> None:
            self.score = score

        def predict(self, image):  # type: ignore[no-untyped-def]
            return Prediction(self.score, np.full(image.shape[:2], self.score, np.float32))

    base = _load(world, GAUSSIAN)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    t = base.threshold
    for score, expected in (
        (t * 0.5, STATUS_NORMAL),
        (t, STATUS_NORMAL),
        (t * 1.0001, STATUS_ANOMALY),
    ):
        predictor = Predictor(**{**base.__dict__, "model": Constant(score)})
        assert inspect_image(predictor, image).status == expected, score
    other = Predictor(**{**base.__dict__, "model": Constant(t * 0.9), "threshold": t * 0.5})
    assert inspect_image(other, image).status == STATUS_ANOMALY  # the stored value is what counts


def test_a_model_returning_a_wrong_sized_map_is_rejected() -> None:
    class Wrong:
        def predict(self, image):  # type: ignore[no-untyped-def]
            return Prediction(1.0, np.ones((3, 3), np.float32))

    predictor = Predictor(
        CLASSICAL, "cashew", Wrong(), 1.0, "normal_percentile", 1.0,
        CalibrationRange(5, 0.1, 0.5, 0.9), None, None, None, None,
    )  # fmt: skip
    with pytest.raises(ArtifactError, match="wrong size"):
        inspect_image(predictor, np.zeros((20, 20, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("score", "phrase"),
    [
        (3.0, "higher than every normal calibration image"),
        (1.5, "above the threshold but within the range of normal scores"),
        (1.2, "above the typical (median) normal score"),
        (0.4, "at or below the typical (median) normal score"),
    ],
)
def test_the_score_is_described_relative_to_the_normal_scores(score: float, phrase: str) -> None:
    text = describe_score(score, CalibrationRange(45, 0.3, 1.0, 2.0), threshold=1.4)
    assert phrase in text and "decision threshold (1.4)" in text
    assert "between 0.3 and 2" in text and "median 1" in text
    assert f"{score / 1.4:.2f}x" in text


# ----- downloads -----------------------------------------------------------------------------


def test_the_downloadable_visualization_is_a_png_with_three_panels(world: World) -> None:
    dataset = _dataset(world)
    predictor = _load(world, GAUSSIAN)
    image = dataset.load_image(dataset.get_samples("cashew", "test", "normal")[0])
    data = panel_png(image, inspect_image(predictor, image))
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (image.shape[0], 3 * image.shape[1], 3)


def test_the_prediction_details_are_complete_honest_and_contain_no_paths(world: World) -> None:
    dataset = _dataset(world)
    predictor = _load(world, PATCHCORE)
    image = dataset.load_image(dataset.get_samples("cashew", "test", "anomaly")[0])
    result = inspect_image(predictor, image)
    details = details_dict(predictor, result, image.shape[:2], "upload")
    assert details["status"] == result.status and details["anomaly_score"] == result.score
    assert details["threshold"] == predictor.threshold and details["score_is_probability"] is False
    assert details["category"] == "cashew" and details["model"] == "C PatchCore-inspired dev"
    assert details["decision_rule"] == "anomaly if score > threshold"
    assert NOT_A_PROBABILITY in details["notes"] and WRONG_CATEGORY_WARNING in details["notes"]
    assert details["calibration_normal_scores"]["n_images"] == predictor.calibration.n
    assert details["image"] == {"source": "upload", "height": 48, "width": 64}
    text = details_json(predictor, result, image.shape[:2], "upload")
    assert json.loads(text) == json.loads(json.dumps(details))
    for forbidden in ("C:\\", "\\\\", "/c/", "Users", str(world.artifacts), str(world.reports)):
        assert forbidden not in text, forbidden
