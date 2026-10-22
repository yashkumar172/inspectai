"""Threshold selection (SPEC.md Sections 3.5 and 7.3).

The default policy, `normal_percentile`, takes only the scores of the held-out normal images.
Its function has no parameter through which test labels or test scores could enter.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from inspectai.errors import EvaluationError
from inspectai.evaluation.metrics import classification_metrics

NORMAL_PERCENTILE = "normal_percentile"
LABELED_CALIBRATION = "labeled_calibration"
POLICIES = (NORMAL_PERCENTILE, LABELED_CALIBRATION)
DEFAULT_QUANTILE = 0.99


@dataclass(frozen=True)
class Threshold:
    """A decision threshold with the policy that produced it (stored per model and category)."""

    policy: str
    value: float
    params: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"policy": self.policy, "value": self.value, "params": dict(self.params)}


def normal_percentile_threshold(
    calibration_normal_scores: ArrayLike, q: float = DEFAULT_QUANTILE
) -> Threshold:
    """The q-quantile (linear interpolation) of values measured on `calibration_normal` images.

    Used for image scores and, with pooled anomaly-map values, for the pixel threshold.
    """
    values = np.asarray(calibration_normal_scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise EvaluationError("Cannot select a threshold without calibration values")
    if not np.isfinite(values).all():
        raise EvaluationError("Calibration values contain NaN or infinity")
    if not 0.0 < q <= 1.0:
        raise EvaluationError(f"q must be in (0, 1], got {q}")
    return Threshold(
        NORMAL_PERCENTILE,
        float(np.quantile(values, q)),
        {"q": q, "n_calibration_values": int(values.size)},
    )


def split_labeled_calibration(
    names: Sequence[str],
    labels: Sequence[int],
    seed: int = 42,
    strata: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the official test rows 50/50 into `test_calibration` and `test_eval`.

    The split is stratified by label (and by `strata`, such as the defect type, when given),
    fixed by `seed`, and independent of row order. In each stratum the first half (rounded up)
    of a seeded shuffle goes to calibration and the rest to evaluation.
    """
    if len(names) != len(labels) or (strata is not None and len(strata) != len(names)):
        raise EvaluationError("names, labels, and strata must have the same length")
    if len(set(names)) != len(names):
        raise EvaluationError("Duplicate names in the test rows")
    groups: dict[tuple[int, str], list[str]] = {}
    for index, name in enumerate(names):
        key = (int(labels[index]), strata[index] if strata is not None else "")
        groups.setdefault(key, []).append(name)
    calibration: list[str] = []
    evaluation: list[str] = []
    for key in sorted(groups):
        members = sorted(groups[key])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(members)
        cut = (len(members) + 1) // 2
        calibration.extend(members[:cut])
        evaluation.extend(members[cut:])
    return tuple(sorted(calibration)), tuple(sorted(evaluation))


def max_f1_threshold(scores: ArrayLike, labels: ArrayLike) -> Threshold:
    """The threshold with the highest F1 on labeled calibration data.

    Candidates lie midway between consecutive distinct scores (plus one below the minimum), so
    the rule `score > threshold` can separate any ordering. Ties go to the larger threshold.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel()
    if s.size == 0 or s.size != y.size:
        raise EvaluationError("scores and labels must be non-empty and of equal length")
    if not np.isfinite(s).all():
        raise EvaluationError("scores contain NaN or infinity")
    if not (y == 1).any():
        raise EvaluationError("labeled_calibration needs at least one anomalous image")
    unique = np.unique(s)
    candidates = np.concatenate(([unique[0] - 1.0], (unique[:-1] + unique[1:]) / 2.0))
    best_value, best_f1 = float(candidates[0]), -1.0
    for candidate in candidates:
        f1 = classification_metrics(s, y, float(candidate)).f1 or 0.0
        if f1 >= best_f1:
            best_value, best_f1 = float(candidate), f1
    return Threshold(
        LABELED_CALIBRATION,
        best_value,
        {"criterion": "max_f1", "calibration_f1": best_f1, "n_calibration_images": int(s.size)},
    )
