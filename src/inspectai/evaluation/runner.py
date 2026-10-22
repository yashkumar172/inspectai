"""The evaluation harness: fit on `fit`, set thresholds, score `test`, compute metrics.

Data use follows SPEC.md Section 3.5. The model sees only the `fit` split. Thresholds come from
`calibration_normal` (policy `normal_percentile`) or from `test_calibration`
(policy `labeled_calibration`). With `labeled_calibration`, every reported metric uses
`test_eval` only.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
import sklearn
from numpy.typing import NDArray

from inspectai.data.dataset import Sample, VisADataset
from inspectai.data.splits import SplitResult
from inspectai.errors import EvaluationError
from inspectai.evaluation.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EVAL_RESOLUTION,
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    classification_metrics,
    classify,
    pixel_auroc,
    pixel_average_precision,
    resize_map_for_eval,
    resize_mask_for_eval,
    segmentation_metrics,
)
from inspectai.evaluation.report import PredictionRow
from inspectai.evaluation.thresholds import (
    DEFAULT_QUANTILE,
    LABELED_CALIBRATION,
    NORMAL_PERCENTILE,
    POLICIES,
    Threshold,
    max_f1_threshold,
    normal_percentile_threshold,
    split_labeled_calibration,
)
from inspectai.models.base import AnomalyModel, Prediction
from inspectai.preprocessing.images import ImageArray
from inspectai.visualization.heatmaps import prediction_panel, save_rgb_png

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalOptions:
    """Settings of one evaluation run."""

    threshold_policy: str = NORMAL_PERCENTILE
    quantile: float = DEFAULT_QUANTILE
    eval_resolution: int = DEFAULT_EVAL_RESOLUTION
    split_seed: int = 42
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        if self.threshold_policy not in POLICIES:
            raise EvaluationError(
                f"Unknown threshold policy {self.threshold_policy!r}; expected one of {POLICIES}"
            )
        if isinstance(self.quantile, bool) or not 0.0 < float(self.quantile) <= 1.0:
            raise EvaluationError(f"quantile must be in (0, 1], got {self.quantile!r}")
        for name in ("eval_resolution", "bootstrap_resamples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EvaluationError(f"{name} must be a positive integer, got {value!r}")
        if self.eval_resolution < 8:
            raise EvaluationError(f"eval_resolution must be at least 8, got {self.eval_resolution}")
        for name in ("split_seed", "bootstrap_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise EvaluationError(f"{name} must be an integer, got {value!r}")


@dataclass
class _Scored:
    sample: Sample
    score: float
    eval_map: NDArray[np.float32]
    eval_mask: NDArray[np.uint8]
    seconds: float


@dataclass
class EvaluationResult:
    """Everything an evaluation run produced, ready to be written by `report`."""

    rows: list[PredictionRow]
    summary: dict[str, object]
    example_names: dict[str, str]


def peak_memory_bytes() -> int:
    """Peak working set of this process (Windows), or the current RSS on other systems."""
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def power_state() -> dict[str, object]:
    """Windows power plan and AC/battery state during the run (SPEC.md Section 5.5).

    Values that cannot be determined are reported as "unknown" / None, never guessed.
    """
    plan = "unknown"
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = None  # plan stays "unknown"
        if result is not None:
            match = re.search(r"\(([^)]+)\)\s*$", result.stdout.strip())
            if match:
                plan = match.group(1)
    battery = psutil.sensors_battery()
    return {"power_plan": plan, "plugged_in": None if battery is None else battery.power_plugged}


def environment_info() -> dict[str, object]:
    """Library versions and hardware, recorded with every experiment (SPEC.md Section 9)."""
    from PIL import Image  # local import keeps the module header light

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "opencv_threads": cv2.getNumThreads(),
        **power_state(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": Image.__version__,
        "psutil": psutil.__version__,
    }


def _score_samples(
    model: AnomalyModel,
    dataset: VisADataset,
    samples: Sequence[Sample],
    resolution: int,
    keep_masks: bool,
) -> list[_Scored]:
    scored: list[_Scored] = []
    for index, sample in enumerate(samples, start=1):
        image = dataset.load_image(sample)
        start = time.perf_counter()
        prediction = model.predict(image)
        seconds = time.perf_counter() - start
        _check_prediction(prediction, image.shape[:2], sample)
        mask = dataset.load_mask(sample, image.shape[:2]) if keep_masks else None
        eval_mask = (
            resize_mask_for_eval(mask, resolution)
            if mask is not None
            else np.zeros((resolution, resolution), dtype=np.uint8)
        )
        scored.append(
            _Scored(
                sample,
                prediction.score,
                resize_map_for_eval(prediction.anomaly_map, resolution),
                eval_mask,
                seconds,
            )
        )
        if index % 50 == 0:
            logger.info("Scored %d / %d images", index, len(samples))
    return scored


def _check_prediction(prediction: Prediction, image_hw: tuple[int, int], sample: Sample) -> None:
    if prediction.anomaly_map.shape != image_hw:
        raise EvaluationError(
            f"Anomaly map {prediction.anomaly_map.shape} does not match image {image_hw}: "
            f"{sample.image_rel}"
        )


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _image_metrics(
    scores: NDArray, labels: NDArray, threshold: float, options: EvalOptions
) -> dict[str, object]:
    cls = classification_metrics(scores, labels, threshold)
    interval = bootstrap_auroc_ci(
        scores, labels, options.bootstrap_resamples, options.bootstrap_seed
    )
    return {
        "n_normal": int(np.sum(labels == 0)),
        "n_anomalous": int(np.sum(labels == 1)),
        "auroc": _round(auroc(scores, labels)),
        "auroc_ci95": None if interval is None else [_round(interval[0]), _round(interval[1])],
        "average_precision": _round(average_precision(scores, labels)),
        "threshold": threshold,
        "confusion_matrix": {"tp": cls.tp, "fp": cls.fp, "tn": cls.tn, "fn": cls.fn},
        "precision": _round(cls.precision),
        "recall": _round(cls.recall),
        "f1": _round(cls.f1),
        "false_positive_rate": _round(cls.false_positive_rate),
        "false_negative_rate": _round(cls.false_negative_rate),
    }


def _pixel_metrics(scored: Sequence[_Scored], threshold: float) -> dict[str, object]:
    maps = [s.eval_map for s in scored]
    masks = [s.eval_mask for s in scored]
    seg = segmentation_metrics(maps, masks, threshold)
    return {
        "auroc": _round(pixel_auroc(maps, masks)),
        "average_precision": _round(pixel_average_precision(maps, masks)),
        "threshold": threshold,
        "iou": _round(seg.iou),
        "dice": _round(seg.dice),
    }


def _pick_examples(scored: Sequence[_Scored]) -> dict[str, str]:
    """Representative cases chosen by score and label only (no tuning on them)."""
    anomalous = [s for s in scored if s.sample.is_anomalous]
    normal = [s for s in scored if not s.sample.is_anomalous]
    picks: dict[str, str] = {}
    if anomalous:
        picks["anomaly_highest_score"] = max(anomalous, key=lambda s: s.score).sample.image_rel
        picks["anomaly_lowest_score"] = min(anomalous, key=lambda s: s.score).sample.image_rel
    if normal:
        picks["normal_highest_score"] = max(normal, key=lambda s: s.score).sample.image_rel
        picks["normal_lowest_score"] = min(normal, key=lambda s: s.score).sample.image_rel
    return picks


def save_examples(
    model: AnomalyModel,
    dataset: VisADataset,
    samples: dict[str, Sample],
    names: dict[str, str],
    out_dir: Path,
    vmax: float,
) -> None:
    """Save a panel (image, ground truth, heatmap, overlay) for each representative case.

    All panels use one color scale, from 0 to `vmax`, so heat colors are comparable.
    """
    for tag, rel in names.items():
        sample = samples[rel]
        image = dataset.load_image(sample)
        prediction = model.predict(image)
        mask = dataset.load_mask(sample, image.shape[:2]) if sample.is_anomalous else None
        panel = prediction_panel(image, prediction.anomaly_map, mask, vmin=0.0, vmax=vmax)
        save_rgb_png(out_dir / f"{tag}.png", panel)


@dataclass(frozen=True)
class FitStats:
    """Measured cost of fitting (SPEC.md Section 7.4)."""

    n_fit_images: int
    fit_seconds: float
    peak_memory_gb: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n_fit_images": self.n_fit_images,
            "fit_seconds": round(self.fit_seconds, 3),
            "peak_memory_gb_after_fit": round(self.peak_memory_gb, 2),
        }


@dataclass(frozen=True)
class Calibration:
    """Thresholds and score range from the `calibration_normal` split."""

    image_threshold: Threshold
    pixel_threshold: Threshold
    scores: dict[str, float]  # n, min, median, max of the calibration image scores

    def to_dict(self) -> dict[str, object]:
        return {
            "image": self.image_threshold.to_dict(),
            "pixel": self.pixel_threshold.to_dict(),
            "scores": dict(self.scores),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Calibration:
        """Rebuild from `to_dict` output (as stored in an artifact manifest)."""
        try:
            image, pixel, scores = data["image"], data["pixel"], data["scores"]
            return cls(
                Threshold(image["policy"], float(image["value"]), dict(image["params"])),  # type: ignore[index,call-overload]
                Threshold(pixel["policy"], float(pixel["value"]), dict(pixel["params"])),  # type: ignore[index,call-overload]
                {k: float(v) for k, v in scores.items()},  # type: ignore[attr-defined]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"Invalid calibration record: {exc}") from exc


def _split_samples(
    dataset: VisADataset, splits: SplitResult
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    by_rel = {s.image_rel: s for s in dataset.get_samples(splits.object_name)}
    missing = [
        n for n in (*splits.fit_files, *splits.calibration_normal, *splits.test) if n not in by_rel
    ]
    if missing:
        raise EvaluationError(
            f"{len(missing)} files of the split are not in the split file, for example {missing[0]}"
        )
    return (
        [by_rel[n] for n in splits.fit_files],
        [by_rel[n] for n in splits.calibration_normal],
        [by_rel[n] for n in splits.test],
    )


class _LazyImages:
    """Loads images one at a time but reports its length, so a model can check memory first."""

    def __init__(self, dataset: VisADataset, samples: Sequence[Sample]) -> None:
        self._dataset, self._samples = dataset, samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[ImageArray]:
        for sample in self._samples:
            yield self._dataset.load_image(sample)


def fit_model(model: AnomalyModel, dataset: VisADataset, splits: SplitResult) -> FitStats:
    """Fit on the `fit` split only (or its recorded `max_fit_images` subset), measuring cost."""
    fit_samples, _, _ = _split_samples(dataset, splits)
    logger.info("Fitting on %d images of %s", len(fit_samples), splits.object_name)
    start = time.perf_counter()
    model.fit(_LazyImages(dataset, fit_samples))
    seconds = time.perf_counter() - start
    return FitStats(len(fit_samples), seconds, peak_memory_bytes() / 1024**3)


def calibrate(
    model: AnomalyModel, dataset: VisADataset, splits: SplitResult, options: EvalOptions
) -> Calibration:
    """Score `calibration_normal` and derive the image and pixel thresholds (normal_percentile)."""
    _, calibration_samples, _ = _split_samples(dataset, splits)
    scored = _score_samples(
        model, dataset, calibration_samples, options.eval_resolution, keep_masks=False
    )
    scores = np.array([s.score for s in scored])
    pixel_values = np.concatenate([s.eval_map.ravel() for s in scored])
    return Calibration(
        image_threshold=normal_percentile_threshold(scores, options.quantile),
        pixel_threshold=normal_percentile_threshold(pixel_values, options.quantile),
        scores={
            "n": int(scores.size),
            "min": float(scores.min()),
            "median": float(np.median(scores)),
            "max": float(scores.max()),
        },
    )


def evaluate_fitted(
    model: AnomalyModel,
    model_name: str,
    model_config: dict[str, object],
    dataset: VisADataset,
    splits: SplitResult,
    options: EvalOptions,
    calibration: Calibration,
    fit_stats: Mapping[str, object] | None = None,
) -> EvaluationResult:
    """Score the test split with an already fitted model and compute all metrics.

    `calibration` comes from `calibrate` (or from a saved artifact). Under `normal_percentile`
    its image threshold is used as is. Under `labeled_calibration` the image threshold is
    chosen on `test_calibration` and metrics are reported on `test_eval` only.
    """
    category = splits.object_name
    _, _, test_samples = _split_samples(dataset, splits)
    resolution = options.eval_resolution

    test = _score_samples(model, dataset, test_samples, resolution, keep_masks=True)
    test_split_name = "test"
    evaluated = test
    image_threshold = calibration.image_threshold
    threshold_extra: dict[str, object] = {}
    if options.threshold_policy == LABELED_CALIBRATION:
        names = [s.sample.image_rel for s in test]
        labels = [int(s.sample.is_anomalous) for s in test]
        calib_names, eval_names = split_labeled_calibration(names, labels, options.split_seed)
        calib_set, eval_set = set(calib_names), set(eval_names)
        calib_part = [s for s in test if s.sample.image_rel in calib_set]
        evaluated = [s for s in test if s.sample.image_rel in eval_set]
        image_threshold = max_f1_threshold(
            [s.score for s in calib_part], [int(s.sample.is_anomalous) for s in calib_part]
        )
        test_split_name = "test_eval"
        threshold_extra = {
            "test_calibration_images": len(calib_part),
            "test_eval_images": len(evaluated),
        }
    peak_after_scoring = peak_memory_bytes()

    scores = np.array([s.score for s in evaluated])
    labels = np.array([int(s.sample.is_anomalous) for s in evaluated])
    metrics = {
        "evaluation_split": test_split_name,
        "evaluation_resolution": resolution,
        "image": _image_metrics(scores, labels, image_threshold.value, options),
        "pixel": _pixel_metrics(evaluated, calibration.pixel_threshold.value),
    }

    predicted = classify(scores, image_threshold.value)
    rows = [
        PredictionRow(
            object_name=category,
            model=model_name,
            split=test_split_name,
            image=s.sample.image_rel,
            label=int(s.sample.is_anomalous),
            score=s.score,
            predicted=int(flag),
            threshold=image_threshold.value,
            has_mask=s.sample.mask_path is not None,
        )
        for s, flag in zip(evaluated, predicted, strict=True)
    ]

    seconds = np.array([s.seconds for s in test])
    performance: dict[str, object] = dict(fit_stats or {})
    performance.update(
        {
            "n_timed_images": int(seconds.size),
            "latency_seconds_mean": round(float(seconds.mean()), 4),
            "latency_seconds_median": round(float(np.median(seconds)), 4),
            "latency_seconds_max": round(float(seconds.max()), 4),
            "images_per_second": round(float(1.0 / seconds.mean()), 2),
            "latency_note": "model.predict only; image decoding and metrics are excluded",
            "peak_memory_gb_after_scoring": round(peak_after_scoring / 1024**3, 2),
        }
    )
    summary: dict[str, object] = {
        "model": model_name,
        "category": category,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_config": model_config,
        "options": asdict(options),
        "split": {
            "seed": splits.seed,
            "calibration_fraction": splits.calibration_fraction,
            "counts": splits.to_dict()["counts"],
            "max_fit_images": splits.max_fit_images,
            "file_list_sha256": splits.to_dict()["file_list_sha256"],
        },
        "threshold": {
            "image": image_threshold.to_dict(),
            "pixel": calibration.pixel_threshold.to_dict(),
            **threshold_extra,
        },
        "calibration_scores": {k: v for k, v in calibration.scores.items() if k != "n"},
        "metrics": metrics,
        "performance": performance,
        "environment": environment_info(),
    }
    return EvaluationResult(rows, summary, _pick_examples(evaluated))


def run_evaluation(
    model: AnomalyModel,
    model_name: str,
    model_config: dict[str, object],
    dataset: VisADataset,
    splits: SplitResult,
    options: EvalOptions | None = None,
) -> EvaluationResult:
    """Fit `model` on the fit split, choose thresholds, and evaluate on the test split."""
    options = options or EvalOptions()
    stats = fit_model(model, dataset, splits)
    calibration = calibrate(model, dataset, splits, options)
    return evaluate_fitted(
        model, model_name, model_config, dataset, splits, options, calibration, stats.to_dict()
    )
