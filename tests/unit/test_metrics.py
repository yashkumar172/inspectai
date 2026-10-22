"""Metrics on toy inputs with known answers (SPEC.md Sections 7.1, 7.2, and 12)."""

import numpy as np
import pytest

from inspectai.errors import EvaluationError
from inspectai.evaluation.metrics import (
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

SCORES = [0.1, 0.4, 0.35, 0.8]
LABELS = [0, 0, 1, 1]


def test_auroc_and_average_precision_match_known_values() -> None:
    assert auroc(SCORES, LABELS) == pytest.approx(0.75)
    assert average_precision(SCORES, LABELS) == pytest.approx(5 / 6)


def test_perfect_and_inverted_rankings() -> None:
    assert auroc([1, 2, 3, 4], LABELS) == 1.0
    assert auroc([4, 3, 2, 1], LABELS) == 0.0


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_single_class_metrics_are_undefined(labels: list[int]) -> None:
    assert auroc(SCORES, labels) is None
    assert average_precision(SCORES, labels) is None
    assert bootstrap_auroc_ci(SCORES, labels) is None
    assert pixel_auroc([np.array([[0.1, 0.9]])], [np.array([[labels[0], labels[0]]])]) is None


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(EvaluationError):
        auroc([1, 2, 3], [0, 1])  # length mismatch
    with pytest.raises(EvaluationError):
        auroc([], [])
    with pytest.raises(EvaluationError):
        auroc([1, float("nan")], [0, 1])
    with pytest.raises(EvaluationError):
        auroc([1, 2], [0, 2])  # labels must be 0 or 1


def test_classification_metrics_at_known_thresholds() -> None:
    scores, labels = [1, 2, 3, 4], [0, 0, 1, 1]
    perfect = classification_metrics(scores, labels, 2.5)
    assert (perfect.tp, perfect.fp, perfect.tn, perfect.fn) == (2, 0, 2, 0)
    assert perfect.precision == perfect.recall == perfect.f1 == 1.0
    assert perfect.false_positive_rate == 0.0 and perfect.false_negative_rate == 0.0

    strict = classification_metrics(scores, labels, 3.5)
    assert (strict.tp, strict.fp, strict.tn, strict.fn) == (1, 0, 2, 1)
    assert strict.recall == 0.5 and strict.precision == 1.0
    assert strict.f1 == pytest.approx(2 / 3)
    assert strict.false_negative_rate == 0.5


def test_classification_metrics_report_undefined_rates_as_none() -> None:
    nothing_flagged = classification_metrics([1, 2, 3, 4], [0, 0, 1, 1], 10.0)
    assert nothing_flagged.precision is None  # no positive predictions
    assert nothing_flagged.recall == 0.0 and nothing_flagged.f1 == 0.0
    only_normal = classification_metrics([1, 2], [0, 0], 1.5)
    assert only_normal.recall is None and only_normal.false_negative_rate is None
    assert only_normal.false_positive_rate == 0.5


def test_the_decision_rule_is_strictly_greater_than() -> None:
    assert classify([1.0, 2.0, 3.0], 2.0).tolist() == [False, False, True]


def test_bootstrap_ci_is_reproducible_and_brackets_a_good_estimate() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 50)
    scores = rng.normal(loc=labels * 1.5, scale=1.0)
    first = bootstrap_auroc_ci(scores, labels, n_resamples=200, seed=7)
    second = bootstrap_auroc_ci(scores, labels, n_resamples=200, seed=7)
    other = bootstrap_auroc_ci(scores, labels, n_resamples=200, seed=8)
    assert first == second and first != other
    assert first is not None
    low, high = first
    assert low < auroc(scores, labels) < high  # type: ignore[operator]
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_of_a_perfect_ranking_is_degenerate() -> None:
    assert bootstrap_auroc_ci([1, 2, 3, 4], LABELS, n_resamples=50) == (1.0, 1.0)


def test_bootstrap_rejects_bad_parameters() -> None:
    with pytest.raises(EvaluationError):
        bootstrap_auroc_ci(SCORES, LABELS, n_resamples=0)
    with pytest.raises(EvaluationError):
        bootstrap_auroc_ci(SCORES, LABELS, confidence=1.5)


def _block_mask() -> np.ndarray:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # 4 positive pixels
    return mask


def test_iou_and_dice_match_known_values() -> None:
    prediction = np.zeros((4, 4), dtype=np.float32)
    prediction[0:2, 0:3] = 1.0  # 6 predicted pixels, 4 of them correct
    result = segmentation_metrics([prediction], [_block_mask()], 0.5)
    assert result.iou == pytest.approx(4 / 6)
    assert result.dice == pytest.approx(8 / 10)


def test_iou_and_dice_pool_pixels_so_normal_images_count_against_the_score() -> None:
    hit = np.zeros((4, 4), dtype=np.float32)
    hit[0:2, 0:2] = 1.0
    false_alarm = np.zeros((4, 4), dtype=np.float32)
    false_alarm[3, 3] = 1.0
    result = segmentation_metrics(
        [hit, false_alarm], [_block_mask(), np.zeros((4, 4), dtype=np.uint8)], 0.5
    )
    assert result.iou == pytest.approx(4 / 5)
    assert result.dice == pytest.approx(8 / 9)


def test_iou_and_dice_are_undefined_without_any_positive_pixel() -> None:
    empty = np.zeros((4, 4), dtype=np.uint8)
    result = segmentation_metrics([np.zeros((4, 4), dtype=np.float32)], [empty], 0.5)
    assert result.iou is None and result.dice is None


def test_pixel_auroc_and_ap_for_a_perfect_map() -> None:
    mask = _block_mask()
    perfect = mask.astype(np.float32)
    assert pixel_auroc([perfect], [mask]) == 1.0
    assert pixel_average_precision([perfect], [mask]) == 1.0
    assert pixel_auroc([1.0 - perfect], [mask]) == 0.0


def test_pixel_metrics_reject_inconsistent_inputs() -> None:
    good = np.zeros((4, 4), dtype=np.float32)
    mask = _block_mask()
    with pytest.raises(EvaluationError):
        pixel_auroc([good], [mask, mask])  # count mismatch
    with pytest.raises(EvaluationError):
        pixel_auroc([good], [np.zeros((3, 3), dtype=np.uint8)])  # shape mismatch
    with pytest.raises(EvaluationError):
        pixel_auroc([], [])
    with pytest.raises(EvaluationError):
        pixel_auroc([good * np.nan], [mask])
    with pytest.raises(EvaluationError):
        pixel_auroc([good], [mask * 2])  # masks must be 0/1
    with pytest.raises(EvaluationError):
        segmentation_metrics([good], [mask * 3], 0.5)


def test_eval_resizing_gives_the_common_resolution_and_keeps_masks_binary() -> None:
    anomaly_map = np.random.default_rng(0).random((30, 50)).astype(np.float32)
    assert resize_map_for_eval(anomaly_map, 64).shape == (64, 64)
    mask = np.zeros((30, 50), dtype=np.uint8)
    mask[5:15, 10:30] = 1
    resized = resize_mask_for_eval(mask, 64)
    assert resized.shape == (64, 64) and set(np.unique(resized)) == {0, 1}
    with pytest.raises(EvaluationError):
        resize_map_for_eval(np.zeros((3, 3, 3)), 64)
    with pytest.raises(EvaluationError):
        resize_mask_for_eval(mask * 2, 64)
