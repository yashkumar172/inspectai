"""Dashboard data: real report files, validated before anything is shown."""

import csv
import json
import logging
import shutil
from pathlib import Path

import pytest
from make_world import World

from inspectai.evaluation.dashboard import (
    ExampleImage,
    confusion_matrix,
    load_category,
    load_results,
    metric_rows,
    performance_rows,
)


@pytest.fixture
def reports(world: World, tmp_path: Path) -> Path:
    """A private copy of the synthetic evaluation reports that a test may damage."""
    target = tmp_path / "reports copy"
    shutil.copytree(world.reports, target)
    return target


def test_real_reports_load_validated_with_everything_the_page_needs(
    world: World, reports: Path
) -> None:
    results = load_results(reports, world.artifacts)
    assert list(results) == ["cashew", "pipe_fryum"]
    cashew = results["cashew"]
    assert cashew.valid and cashew.problem is None and cashew.detail is None
    assert [r["label"] for r in cashew.records] == [
        "A classical",
        "B Gaussian dev",
        "C PatchCore-inspired dev",
    ]
    assert cashew.counts == (5, 5)
    assert "official `test` split" in cashew.split_label
    assert cashew.not_measured == ("B Gaussian full", "C PatchCore-inspired full")
    assert cashew.paired and all("auroc_difference" in d for d in cashew.paired)
    assert all(r["weights"] is None or "identifier" in r["weights"] for r in cashew.records)


def test_the_metric_rows_come_from_the_saved_summaries(world: World, reports: Path) -> None:
    cashew = load_results(reports, world.artifacts)["cashew"]
    rows = metric_rows(cashew)
    assert [r["Model"] for r in rows] == [r["label"] for r in cashew.records]
    summary = json.loads((reports / "gaussian" / "cashew" / "summary.json").read_text("utf-8"))
    image = summary["metrics"]["image"]
    gaussian = next(r for r in rows if r["Model"] == "B Gaussian dev")
    assert gaussian["Image AUROC"] == f"{image['auroc']:.3f}"
    low, high = image["auroc_ci95"]
    assert gaussian["95% CI"] == f"{low:.3f} - {high:.3f}"
    assert gaussian["Recall"] == f"{image['recall']:.3f}"
    assert {"Pixel AUROC", "Pixel IoU", "Pixel Dice", "Precision", "FPR", "F1"} <= set(gaussian)


def test_performance_and_confusion_matrix_use_the_measured_values(
    world: World, reports: Path
) -> None:
    cashew = load_results(reports, world.artifacts)["cashew"]
    rows = performance_rows(cashew)
    assert all(r["Fit time"].endswith(" s") and r["Latency (mean)"].endswith(" ms") for r in rows)
    record = cashew.records[1]
    cm = confusion_matrix(record)
    raw = record["metrics"]["image"]["confusion_matrix"]
    assert cm["Truly normal"] == {"Predicted normal": raw["tn"], "Predicted anomaly": raw["fp"]}
    assert cm["Truly anomalous"] == {"Predicted normal": raw["fn"], "Predicted anomaly": raw["tp"]}
    assert sum(sum(row.values()) for row in cm.values()) == 10  # all test images are counted


def test_no_report_folder_or_no_results_gives_an_empty_dashboard(
    world: World, tmp_path: Path
) -> None:
    assert load_results(tmp_path / "does not exist", world.artifacts) == {}
    (tmp_path / "empty").mkdir()
    assert load_results(tmp_path / "empty") == {}
    assert load_category(tmp_path / "empty", None, "cashew") is None


def test_a_category_with_a_single_run_shows_it_without_a_comparison(
    world: World, reports: Path
) -> None:
    for folder in ("gaussian", "patchcore"):
        shutil.rmtree(reports / folder / "pipe_fryum")
    result = load_results(reports, world.artifacts)["pipe_fryum"]
    assert result.valid and [r["label"] for r in result.records] == ["A classical"]
    assert result.paired == () and len(result.not_measured) == 4


def test_a_summary_that_does_not_follow_from_its_predictions_is_rejected(
    world: World, reports: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = reports / "gaussian" / "cashew" / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1][5] = "-1000.0"  # rewrite one score
    rows[6][5] = "9999.0"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    with caplog.at_level(logging.WARNING):
        results = load_results(reports, world.artifacts)
    bad = results["cashew"]
    assert not bad.valid and bad.records == () and bad.runs == ()
    assert "do not match" in bad.problem and "AUROC" in (bad.detail or "")
    assert results["pipe_fryum"].valid  # the other category is unaffected
    assert any("rejected" in r.getMessage() for r in caplog.records)  # logged for the operator


def test_runs_that_were_not_evaluated_identically_are_rejected(world: World, reports: Path) -> None:
    path = reports / "patchcore" / "cashew" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["metrics"]["evaluation_resolution"] = 512
    path.write_text(json.dumps(summary), encoding="utf-8")
    bad = load_results(reports, world.artifacts)["cashew"]
    assert not bad.valid and "identical splits" in bad.problem


@pytest.mark.parametrize("damage", ["json", "wrong-folder", "missing-key", "missing-predictions"])
def test_unreadable_or_malformed_files_are_rejected_without_exposing_paths(
    world: World, reports: Path, damage: str
) -> None:
    folder = reports / "classical" / "cashew"
    if damage == "json":
        (folder / "summary.json").write_text("{not json", encoding="utf-8")
    elif damage == "wrong-folder":
        shutil.copytree(
            reports / "gaussian" / "cashew",
            reports / "patchcore" / "pipe_fryum",
            dirs_exist_ok=True,
        )
    elif damage == "missing-key":
        summary = json.loads((folder / "summary.json").read_text("utf-8"))
        del summary["metrics"]["image"]
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    else:
        (folder / "predictions.csv").unlink()
    results = load_results(reports, world.artifacts)
    bad = results["pipe_fryum"] if damage == "wrong-folder" else results["cashew"]
    assert not bad.valid and bad.records == ()
    for forbidden in ("C:\\", "\\", "/", str(reports), "Traceback"):
        assert forbidden not in bad.problem, forbidden


def test_saved_example_figures_are_listed_with_captions(world: World, reports: Path) -> None:
    (reports / "comparison").mkdir()
    (reports / "comparison" / "cashew_examples.png").write_bytes(b"png")
    examples = reports / "gaussian" / "cashew" / "examples"
    examples.mkdir()
    (examples / "anomaly_highest_score.png").write_bytes(b"png")
    result = load_results(reports, world.artifacts)["cashew"]
    assert [e.caption for e in result.examples] == [
        "All models on the same test images",
        "B Gaussian dev: anomaly highest score",
    ]
    assert all(isinstance(e, ExampleImage) and e.path.is_file() for e in result.examples)
    assert load_results(reports, world.artifacts)["pipe_fryum"].examples == ()
