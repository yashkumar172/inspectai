"""Loading fitted models and inspecting single images (SPEC.md Sections 2, 8, and 11).

Everything the app needs from the models lives here, free of any Streamlit code so it can be
tested directly: which fitted models exist, loading one through the artifact store (with all
integrity checks), and running it on an image. The decision always uses the threshold stored in
the model's artifact for that category (SPEC.md Section 3.5, rule 7), and an anomaly score is
never presented as a probability (SPEC.md Section 2).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from torch import nn

from inspectai.artifacts.classical_io import load_classical
from inspectai.artifacts.gaussian_io import load_gaussian
from inspectai.artifacts.patchcore_io import load_patchcore
from inspectai.artifacts.store import MANIFEST
from inspectai.config import PROFILES
from inspectai.data.dataset import SUPPORTED_OBJECTS
from inspectai.errors import ArtifactError, ModelUnavailableError
from inspectai.evaluation.comparison import RUN_SPECS
from inspectai.evaluation.metrics import classify
from inspectai.models.base import AnomalyModel
from inspectai.preprocessing.images import ImageArray
from inspectai.visualization.heatmaps import (
    anomaly_map_to_heatmap,
    overlay_heatmap,
    prediction_panel,
)

STATUS_NORMAL = "Normal"
STATUS_ANOMALY = "Anomaly detected"
STATUS_UNAVAILABLE = "Model unavailable"
WRONG_CATEGORY_WARNING = (
    "Scores are only meaningful if the uploaded object matches the selected category."
)
NOT_A_PROBABILITY = (
    "The anomaly score is a distance-like value, not a probability of a defect, and its scale "
    "differs between models and categories."
)


@dataclass(frozen=True)
class ModelOption:
    """One selectable model configuration, e.g. 'C PatchCore-inspired dev'."""

    key: str
    label: str
    model: str
    profile: str | None
    artifact_folder: str


def model_options() -> tuple[ModelOption, ...]:
    """All model configurations of the project, in the order of the comparison report."""
    return tuple(
        ModelOption(spec.report_folder, spec.label, spec.model, spec.profile, spec.artifact_folder)  # type: ignore[arg-type]
        for spec in RUN_SPECS
    )


def option_by_key(key: str) -> ModelOption:
    for option in model_options():
        if option.key == key:
            return option
    raise ModelUnavailableError("Unknown model selection.")


def artifact_directory(artifact_root: Path, option: ModelOption, category: str) -> Path:
    """Where the artifact of `option` for `category` lives."""
    return Path(artifact_root) / option.artifact_folder / category


def is_available(artifact_root: Path, option: ModelOption, category: str) -> bool:
    """True if a saved artifact exists (its integrity is checked only when it is loaded)."""
    return (artifact_directory(artifact_root, option, category) / MANIFEST).is_file()


def available_categories(artifact_root: Path, option: ModelOption) -> list[str]:
    """The categories for which `option` has a saved artifact."""
    return [c for c in SUPPORTED_OBJECTS if is_available(artifact_root, option, c)]


def available_options(artifact_root: Path) -> list[ModelOption]:
    """The model configurations that have an artifact for at least one category."""
    return [o for o in model_options() if available_categories(artifact_root, o)]


@dataclass(frozen=True)
class CalibrationRange:
    """Scores of the held-out normal (`calibration_normal`) images."""

    n: int
    minimum: float
    median: float
    maximum: float


@dataclass(frozen=True)
class Predictor:
    """A loaded model together with everything the app shows next to a prediction."""

    option: ModelOption
    category: str
    model: AnomalyModel
    threshold: float
    threshold_policy: str
    pixel_threshold: float
    calibration: CalibrationRange
    reference_aspect_ratio: float | None
    created_utc: str | None
    weights_identifier: str | None
    weights_sha256: str | None


def _positive(value: object, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number <= 0:
        raise ArtifactError(f"The stored {name} is not a positive number")
    return number


def load_predictor(
    artifact_root: Path,
    option: ModelOption,
    category: str,
    backbone: nn.Module | None = None,
) -> Predictor:
    """Load a fitted model through the artifact store.

    Raises ModelUnavailableError if no artifact exists, ArtifactError if it fails any integrity
    check (checksums, shapes, NaN, split overlap) or belongs to another profile, and
    WeightsNotFoundError if the pretrained weights are not cached. Nothing is ever substituted.
    """
    directory = artifact_directory(artifact_root, option, category)
    if not is_available(artifact_root, option, category):
        raise ModelUnavailableError(f"No fitted '{option.label}' model for {category}.")
    if option.model == "classical":
        model, artifact = load_classical(directory, category)
    elif option.model == "gaussian":
        model, artifact = load_gaussian(directory, category, backbone=backbone)
    else:
        model, artifact = load_patchcore(directory, category, backbone=backbone)

    manifest = artifact.manifest
    if (
        option.profile is not None
        and manifest.get("backbone") != PROFILES[option.profile]["backbone"]
    ):
        raise ArtifactError("The saved model was fitted with another profile than selected")
    try:
        calibration = manifest["calibration"]["scores"]  # type: ignore[index]
        pixel = manifest["calibration"]["pixel"]  # type: ignore[index]
        threshold = manifest["threshold"]  # type: ignore[assignment]
        reference = manifest["reference_aspect_ratio"]
        weights = manifest.get("weights")
        predictor = Predictor(
            option=option,
            category=category,
            model=model,
            threshold=_positive(threshold["value"], "threshold"),  # type: ignore[index]
            threshold_policy=str(threshold["policy"]),  # type: ignore[index]
            pixel_threshold=_positive(pixel["value"], "pixel threshold"),
            calibration=CalibrationRange(
                int(calibration["n"]),
                float(calibration["min"]),
                float(calibration["median"]),
                float(calibration["max"]),
            ),
            reference_aspect_ratio=None
            if reference is None
            else _positive(reference, "aspect ratio"),
            created_utc=manifest.get("created_utc"),  # type: ignore[arg-type]
            weights_identifier=None if not weights else str(weights.get("identifier")),  # type: ignore[union-attr]
            weights_sha256=None if not weights else weights.get("sha256"),  # type: ignore[union-attr]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("The manifest lacks thresholds or calibration scores") from exc
    return predictor


@dataclass(frozen=True)
class Inspection:
    """The result of running a model on one image."""

    status: str
    score: float
    threshold: float
    threshold_policy: str
    calibration: CalibrationRange
    score_vs_normal: str
    anomaly_map: NDArray[np.float32]
    heatmap: ImageArray
    overlay: ImageArray
    color_scale_max: float
    seconds: float

    @property
    def is_anomaly(self) -> bool:
        return self.status == STATUS_ANOMALY


def describe_score(score: float, calibration: CalibrationRange, threshold: float) -> str:
    """One sentence placing a score relative to the normal calibration scores."""
    ratio = score / threshold if threshold > 0 else float("nan")
    where = (
        f"{ratio:.2f}x the decision threshold ({threshold:.3g}). The held-out normal images "
        f"scored between {calibration.minimum:.3g} and {calibration.maximum:.3g} "
        f"(median {calibration.median:.3g})."
    )
    if score > calibration.maximum:
        position = "This score is higher than every normal calibration image."
    elif score > threshold:
        position = "This score is above the threshold but within the range of normal scores."
    elif score > calibration.median:
        position = "This score is within the normal range, above the typical (median) normal score."
    else:
        position = "This score is at or below the typical (median) normal score."
    return f"{position} It is {where}"


def inspect_image(predictor: Predictor, image: ImageArray, alpha: float = 0.5) -> Inspection:
    """Run the model on an RGB image and build the decision, heatmap, and overlay.

    The decision is `score > stored threshold`. Heatmap colors use a fixed scale anchored at the
    stored pixel threshold (blue = 0, the threshold in the middle, red = twice the threshold), so
    a normal image is not stretched to look alarming.
    """
    start = time.perf_counter()
    prediction = predictor.model.predict(image)
    seconds = time.perf_counter() - start
    if prediction.anomaly_map.shape != image.shape[:2]:
        raise ArtifactError("The model returned a map of the wrong size")
    is_anomaly = bool(classify([prediction.score], predictor.threshold)[0])
    vmax = 2.0 * predictor.pixel_threshold
    return Inspection(
        status=STATUS_ANOMALY if is_anomaly else STATUS_NORMAL,
        score=prediction.score,
        threshold=predictor.threshold,
        threshold_policy=predictor.threshold_policy,
        calibration=predictor.calibration,
        score_vs_normal=describe_score(
            prediction.score, predictor.calibration, predictor.threshold
        ),
        anomaly_map=prediction.anomaly_map,
        heatmap=anomaly_map_to_heatmap(prediction.anomaly_map, 0.0, vmax),
        overlay=overlay_heatmap(image, prediction.anomaly_map, alpha, 0.0, vmax),
        color_scale_max=vmax,
        seconds=seconds,
    )


def panel_png(image: ImageArray, inspection: Inspection) -> bytes:
    """A downloadable PNG: the original, the heatmap, and the overlay side by side."""
    panel = prediction_panel(
        image, inspection.anomaly_map, None, vmin=0.0, vmax=inspection.color_scale_max
    )
    ok, encoded = cv2.imencode(".png", panel[:, :, ::-1])
    if not ok:
        raise ArtifactError("The visualization could not be encoded")
    return encoded.tobytes()


def details_dict(
    predictor: Predictor, inspection: Inspection, image_shape: tuple[int, int], source: str
) -> dict[str, object]:
    """The downloadable prediction details. Contains no file system paths."""
    cal = inspection.calibration
    return {
        "tool": "InspectAI",
        "model": predictor.option.label,
        "model_type": predictor.option.model,
        "profile": predictor.option.profile,
        "category": predictor.category,
        "status": inspection.status,
        "anomaly_score": inspection.score,
        "threshold": inspection.threshold,
        "threshold_policy": inspection.threshold_policy,
        "decision_rule": "anomaly if score > threshold",
        "calibration_normal_scores": {
            "n_images": cal.n,
            "min": cal.minimum,
            "median": cal.median,
            "max": cal.maximum,
        },
        "score_is_probability": False,
        "notes": [NOT_A_PROBABILITY, WRONG_CATEGORY_WARNING],
        "image": {"source": source, "height": image_shape[0], "width": image_shape[1]},
        "heatmap_color_scale": {"min": 0.0, "max": inspection.color_scale_max},
        "inference_seconds": round(inspection.seconds, 4),
        "model_created_utc": predictor.created_utc,
        "pretrained_weights": predictor.weights_identifier,
        "pretrained_weights_sha256": predictor.weights_sha256,
    }


def details_json(
    predictor: Predictor, inspection: Inspection, image_shape: tuple[int, int], source: str
) -> str:
    return json.dumps(details_dict(predictor, inspection, image_shape, source), indent=1) + "\n"
