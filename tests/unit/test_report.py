"""CSV and JSON report writers."""

import csv
import json
from pathlib import Path

import pytest

from inspectai.errors import EvaluationError
from inspectai.evaluation.report import (
    CSV_COLUMNS,
    PredictionRow,
    find_undefined,
    write_predictions_csv,
    write_summary_json,
)


def _row(name: str = "cashew/a.JPG", label: int = 1) -> PredictionRow:
    return PredictionRow("cashew", "classical", "test", name, label, 1.25, 1, 1.0, True)


def test_predictions_csv_has_one_row_per_image_with_the_documented_columns(tmp_path: Path) -> None:
    path = tmp_path / "out dir é" / "predictions.csv"
    write_predictions_csv(path, [_row("cashew/a.JPG", 1), _row("cashew/b.JPG", 0)])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert [r["image"] for r in rows] == ["cashew/a.JPG", "cashew/b.JPG"]
    assert [r["label"] for r in rows] == ["1", "0"]
    assert float(rows[0]["score"]) == 1.25 and rows[0]["has_mask"] == "1"


def test_an_empty_prediction_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        write_predictions_csv(tmp_path / "p.csv", [])


def test_undefined_metrics_are_found_at_any_depth() -> None:
    metrics = {"image": {"auroc": None, "f1": 0.5, "ci": None}, "pixel": {"iou": None}}
    assert find_undefined(metrics) == ["image.auroc", "image.ci", "pixel.iou"]


def test_summary_json_lists_undefined_metrics_and_marks_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    write_summary_json(path, {"model": "classical", "metrics": {"image": {"auroc": None}}})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["metrics"]["image"]["auroc"] is None
    assert data["undefined_metrics"] == ["image.auroc"]


def test_summary_json_rejects_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        write_summary_json(tmp_path / "s.json", {"metrics": {"x": float("nan")}})
    assert not (tmp_path / "s.json").exists()
