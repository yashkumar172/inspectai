"""Threshold policies (SPEC.md Sections 3.5 and 7.3)."""

import inspect

import numpy as np
import pytest

from inspectai.errors import EvaluationError
from inspectai.evaluation.thresholds import (
    LABELED_CALIBRATION,
    NORMAL_PERCENTILE,
    max_f1_threshold,
    normal_percentile_threshold,
    split_labeled_calibration,
)


def test_normal_percentile_is_the_q_quantile_of_calibration_scores() -> None:
    threshold = normal_percentile_threshold(np.arange(101, dtype=float), q=0.99)
    assert threshold.policy == NORMAL_PERCENTILE
    assert threshold.value == pytest.approx(99.0)
    assert threshold.params["q"] == 0.99 and threshold.params["n_calibration_values"] == 101


def test_normal_percentile_has_no_way_to_receive_labels_or_test_scores() -> None:
    parameters = list(inspect.signature(normal_percentile_threshold).parameters)
    assert parameters == ["calibration_normal_scores", "q"]


def test_normal_percentile_ignores_everything_but_its_input() -> None:
    calibration = [1.0, 2.0, 3.0, 4.0]
    first = normal_percentile_threshold(calibration)
    second = normal_percentile_threshold(list(calibration))
    assert first == second


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_normal_percentile_rejects_empty_or_non_finite_input(values: list[float]) -> None:
    with pytest.raises(EvaluationError):
        normal_percentile_threshold(values)


@pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
def test_normal_percentile_rejects_a_bad_quantile(q: float) -> None:
    with pytest.raises(EvaluationError):
        normal_percentile_threshold([1.0, 2.0], q=q)


def _test_rows() -> tuple[list[str], list[int]]:
    names = [f"n{i:02d}" for i in range(20)] + [f"a{i:02d}" for i in range(11)]
    labels = [0] * 20 + [1] * 11
    return names, labels


def test_labeled_split_is_disjoint_complete_and_stratified() -> None:
    names, labels = _test_rows()
    calibration, evaluation = split_labeled_calibration(names, labels, seed=1)
    assert not set(calibration) & set(evaluation)
    assert set(calibration) | set(evaluation) == set(names)
    assert sum(n.startswith("n") for n in calibration) == 10
    assert sum(n.startswith("a") for n in calibration) == 6  # ceil(11 / 2)
    assert sum(n.startswith("a") for n in evaluation) == 5


def test_labeled_split_is_reproducible_seeded_and_order_independent() -> None:
    names, labels = _test_rows()
    first = split_labeled_calibration(names, labels, seed=1)
    assert first == split_labeled_calibration(names, labels, seed=1)
    order = list(reversed(range(len(names))))
    shuffled = split_labeled_calibration([names[i] for i in order], [labels[i] for i in order], 1)
    assert first == shuffled
    assert first != split_labeled_calibration(names, labels, seed=2)


def test_labeled_split_can_stratify_by_defect_type() -> None:
    names = [f"a{i}" for i in range(8)]
    labels = [1] * 8
    strata = ["crack"] * 4 + ["dent"] * 4
    calibration, _ = split_labeled_calibration(names, labels, seed=3, strata=strata)
    assert sum(n in calibration for n in names[:4]) == 2
    assert sum(n in calibration for n in names[4:]) == 2


def test_labeled_split_rejects_inconsistent_inputs() -> None:
    with pytest.raises(EvaluationError):
        split_labeled_calibration(["a", "b"], [0])
    with pytest.raises(EvaluationError):
        split_labeled_calibration(["a", "a"], [0, 1])
    with pytest.raises(EvaluationError):
        split_labeled_calibration(["a", "b"], [0, 1], strata=["x"])


def test_max_f1_threshold_separates_perfectly_separable_scores() -> None:
    threshold = max_f1_threshold([1, 2, 3, 4], [0, 0, 1, 1])
    assert threshold.policy == LABELED_CALIBRATION
    assert threshold.value == pytest.approx(2.5)
    assert threshold.params["calibration_f1"] == 1.0


def test_max_f1_threshold_with_overlapping_scores() -> None:
    # Best F1 is to flag everything above 1.5: tp=2, fp=1, fn=0 -> F1 = 0.8.
    threshold = max_f1_threshold([1, 2, 3, 4], [0, 1, 0, 1])
    assert threshold.value == pytest.approx(1.5)
    assert threshold.params["calibration_f1"] == pytest.approx(0.8)


def test_max_f1_threshold_needs_anomalies() -> None:
    with pytest.raises(EvaluationError):
        max_f1_threshold([1, 2], [0, 0])
    with pytest.raises(EvaluationError):
        max_f1_threshold([], [])
