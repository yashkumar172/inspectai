"""Machine-readable evaluation output: per-image CSV and JSON summary (SPEC.md Section 7.5)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inspectai.errors import EvaluationError

SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "object",
    "model",
    "split",
    "image",
    "label",
    "score",
    "predicted",
    "threshold",
    "has_mask",
)


@dataclass(frozen=True)
class PredictionRow:
    """One scored image. `label` is 1 for anomalous, from the split file (never predicted)."""

    object_name: str
    model: str
    split: str
    image: str
    label: int
    score: float
    predicted: int
    threshold: float
    has_mask: bool


def write_predictions_csv(path: Path, rows: Sequence[PredictionRow]) -> None:
    """Write one row per image."""
    if not rows:
        raise EvaluationError("No predictions to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.object_name,
                    row.model,
                    row.split,
                    row.image,
                    row.label,
                    repr(row.score),
                    row.predicted,
                    repr(row.threshold),
                    int(row.has_mask),
                ]
            )


def find_undefined(metrics: Mapping[str, object], prefix: str = "") -> list[str]:
    """Dotted names of all metrics whose value is None (undefined)."""
    found: list[str] = []
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if value is None:
            found.append(name)
        elif isinstance(value, Mapping):
            found.extend(find_undefined(value, f"{name}."))
    return found


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationError("Summary contains NaN or infinity")
    return value


def write_summary_json(path: Path, summary: Mapping[str, object]) -> None:
    """Write the experiment summary; undefined metrics are stored as null and listed."""
    body = dict(summary)
    body["schema_version"] = SCHEMA_VERSION
    metrics = body.get("metrics")
    if isinstance(metrics, Mapping):
        body["undefined_metrics"] = find_undefined(metrics)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(body), indent=1, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
