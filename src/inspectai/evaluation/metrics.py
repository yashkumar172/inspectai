"""Image-level and pixel-level metrics (SPEC.md Section 7).

A metric that cannot be defined (for example AUROC when only one class is present) is returned
as `None`, never as an invented number. Labels are 1 for anomalous and 0 for normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from inspectai.errors import EvaluationError

DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_EVAL_RESOLUTION = 256


def _scores_and_labels(scores: ArrayLike, labels: ArrayLike) -> tuple[NDArray, NDArray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel()
    if s.size != y.size:
        raise EvaluationError(f"scores and labels differ in length ({s.size} vs {y.size})")
    if s.size == 0:
        raise EvaluationError("scores and labels are empty")
    if not np.isfinite(s).all():
        raise EvaluationError("scores contain NaN or infinity")
    if not np.isin(y, (0, 1)).all():
        raise EvaluationError("labels must be 0 (normal) or 1 (anomalous)")
    return s, y.astype(np.int64)


def _has_both_classes(y: NDArray) -> bool:
    return bool(y.min() == 0 and y.max() == 1)


def auroc(scores: ArrayLike, labels: ArrayLike) -> float | None:
    """Area under the ROC curve, or None if only one class is present."""
    s, y = _scores_and_labels(scores, labels)
    return float(roc_auc_score(y, s)) if _has_both_classes(y) else None


def average_precision(scores: ArrayLike, labels: ArrayLike) -> float | None:
    """Average precision, or None if only one class is present."""
    s, y = _scores_and_labels(scores, labels)
    return float(average_precision_score(y, s)) if _has_both_classes(y) else None


def bootstrap_auroc_ci(
    scores: ArrayLike,
    labels: ArrayLike,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval of the image AUROC.

    Images are resampled with replacement. A resample with a single class has no AUROC and is
    skipped; the interval is None if the data has one class or no valid resample remains.
    """
    s, y = _scores_and_labels(scores, labels)
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise EvaluationError("n_resamples must be positive and confidence in (0, 1)")
    if not _has_both_classes(y):
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        index = rng.integers(0, s.size, size=s.size)
        if _has_both_classes(y[index]):
            values.append(float(roc_auc_score(y[index], s[index])))
    if not values:
        return None
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Confusion matrix and derived rates at one threshold. Undefined rates are None."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def classify(scores: ArrayLike, threshold: float) -> NDArray[np.bool_]:
    """The decision rule: anomalous if and only if score > threshold."""
    return np.asarray(scores, dtype=np.float64) > threshold


def classification_metrics(
    scores: ArrayLike, labels: ArrayLike, threshold: float
) -> ClassificationMetrics:
    """Precision, recall, F1, confusion matrix, FPR, and FNR at `threshold`."""
    s, y = _scores_and_labels(scores, labels)
    predicted = classify(s, threshold)
    positive = y == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    return ClassificationMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        false_positive_rate=_ratio(fp, fp + tn),
        false_negative_rate=_ratio(fn, fn + tp),
    )


def resize_map_for_eval(
    anomaly_map: NDArray, resolution: int = DEFAULT_EVAL_RESOLUTION
) -> NDArray[np.float32]:
    """Bilinear resize of an anomaly map to the common evaluation resolution."""
    if anomaly_map.ndim != 2 or anomaly_map.size == 0:
        raise EvaluationError(f"Anomaly map must be a non-empty 2-D array, got {anomaly_map.shape}")
    return cv2.resize(
        anomaly_map.astype(np.float32), (resolution, resolution), interpolation=cv2.INTER_LINEAR
    )


def resize_mask_for_eval(
    mask: NDArray, resolution: int = DEFAULT_EVAL_RESOLUTION
) -> NDArray[np.uint8]:
    """Nearest-neighbor resize of a binary mask to the common evaluation resolution."""
    if mask.ndim != 2 or mask.size == 0:
        raise EvaluationError(f"Mask must be a non-empty 2-D array, got {mask.shape}")
    if not np.isin(mask, (0, 1)).all():
        raise EvaluationError("Mask must contain only 0 and 1")
    return cv2.resize(
        mask.astype(np.uint8), (resolution, resolution), interpolation=cv2.INTER_NEAREST
    )


def _stack_pixels(
    maps: list[NDArray], masks: list[NDArray]
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    if len(maps) != len(masks):
        raise EvaluationError(f"{len(maps)} maps but {len(masks)} masks")
    if not maps:
        raise EvaluationError("No maps to evaluate")
    shapes = {m.shape for m in maps} | {m.shape for m in masks}
    if len(shapes) != 1:
        raise EvaluationError(f"Maps and masks must share one shape, found {sorted(shapes)}")
    stacked_maps = np.stack(maps).astype(np.float32, copy=False)
    stacked_masks = np.stack(masks).astype(np.uint8, copy=False)
    if not np.isfinite(stacked_maps).all():
        raise EvaluationError("Anomaly maps contain NaN or infinity")
    if not np.isin(stacked_masks, (0, 1)).all():
        raise EvaluationError("Masks must contain only 0 and 1")
    return stacked_maps, stacked_masks


def pixel_auroc(maps: list[NDArray], masks: list[NDArray]) -> float | None:
    """Pixel AUROC pooled over all images (normal images contribute all-normal pixels)."""
    values, truth = _stack_pixels(maps, masks)
    return auroc(values.ravel(), truth.ravel())


def pixel_average_precision(maps: list[NDArray], masks: list[NDArray]) -> float | None:
    """Pixel-level average precision pooled over all images."""
    values, truth = _stack_pixels(maps, masks)
    return average_precision(values.ravel(), truth.ravel())


@dataclass(frozen=True)
class SegmentationMetrics:
    """Pooled IoU and Dice at one pixel threshold. None if there is no positive pixel at all."""

    threshold: float
    iou: float | None
    dice: float | None


def segmentation_metrics(
    maps: list[NDArray], masks: list[NDArray], threshold: float
) -> SegmentationMetrics:
    """IoU and Dice of `map > threshold` against the masks, with pixels pooled over all images.

    Pooling means intersections and unions are summed over images before dividing, so normal
    images (all-zero masks) count against the score through their false-positive pixels. Both
    metrics are undefined (None) if neither the prediction nor the ground truth has a positive
    pixel.
    """
    values, truth = _stack_pixels(maps, masks)
    predicted = values > threshold
    actual = truth == 1
    intersection = int(np.sum(predicted & actual))
    union = int(np.sum(predicted | actual))
    total = int(np.sum(predicted)) + int(np.sum(actual))
    return SegmentationMetrics(
        threshold=float(threshold),
        iou=_ratio(intersection, union),
        dice=_ratio(2 * intersection, total),
    )


def paired_bootstrap_auroc_difference(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    labels: ArrayLike,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float, float] | None:
    """AUROC(a) - AUROC(b) on the same images, with a paired percentile bootstrap interval.

    Both models scored the same images, so each resample draws the same image indices for both
    (a paired comparison, tighter and more appropriate than comparing two separate intervals).
    Returns (difference, low, high), or None if only one class is present or no resample has
    both classes. An interval that excludes 0 indicates a difference beyond sampling noise on
    this test set.
    """
    a, y = _scores_and_labels(scores_a, labels)
    b, _ = _scores_and_labels(scores_b, labels)
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise EvaluationError("n_resamples must be positive and confidence in (0, 1)")
    if not _has_both_classes(y):
        return None
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        index = rng.integers(0, y.size, size=y.size)
        if _has_both_classes(y[index]):
            values.append(
                float(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
            )
    if not values:
        return None
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return point, float(low), float(high)
