"""Data for the evaluation dashboard: real report files, validated before anything is shown.

The dashboard never computes or invents metrics. It loads the `summary.json` and
`predictions.csv` files written by `scripts/evaluate_model.py`, re-derives each run's image-level
metrics from the saved predictions, checks that the runs of a category are comparable, and only
then exposes them. A category that fails any check gets a short, path-free `problem` and no
metrics at all; the technical detail is kept in `detail` for the server log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from inspectai.data.dataset import SUPPORTED_OBJECTS
from inspectai.errors import EvaluationError
from inspectai.evaluation.comparison import (
    RUN_SPECS,
    Run,
    check_comparable,
    load_run,
    paired_auroc_differences,
    run_record,
    verify_run,
)

logger = logging.getLogger(__name__)

_SPLIT_LABELS = {
    "test": "the official `test` split (never used for fitting or thresholds)",
    "test_eval": "the `test_eval` half of the test split (the other half chose the threshold)",
}


@dataclass(frozen=True)
class ExampleImage:
    """A saved representative-prediction figure."""

    caption: str
    path: Path  # internal; the UI must never display it


@dataclass(frozen=True)
class CategoryResults:
    """Validated results of one category, or why they are not shown."""

    category: str
    runs: tuple[Run, ...] = ()
    records: tuple[dict, ...] = ()  # one `run_record` per run, same order
    paired: tuple[dict, ...] = ()
    not_measured: tuple[str, ...] = ()
    examples: tuple[ExampleImage, ...] = ()
    problem: str | None = None  # safe to show: contains no paths
    detail: str | None = field(default=None, repr=False)  # for logs only

    @property
    def valid(self) -> bool:
        return self.problem is None and bool(self.runs)

    @property
    def split_label(self) -> str:
        """Which data the metrics were computed on."""
        if not self.runs:
            return ""
        split = self.runs[0].summary["metrics"]["evaluation_split"]
        return _SPLIT_LABELS.get(split, f"the `{split}` split")

    @property
    def counts(self) -> tuple[int, int]:
        """(normal, anomalous) test images."""
        first = self.runs[0].image_metrics
        return first["n_normal"], first["n_anomalous"]


def _examples(report_root: Path, category: str, runs: tuple[Run, ...]) -> tuple[ExampleImage, ...]:
    found: list[ExampleImage] = []
    figure = Path(report_root) / "comparison" / f"{category}_examples.png"
    if figure.is_file():
        found.append(ExampleImage("All models on the same test images", figure))
    for run in runs:
        folder = Path(report_root) / run.spec.report_folder / category / "examples"
        for path in sorted(folder.glob("*.png")) if folder.is_dir() else []:
            caption = f"{run.spec.label}: {path.stem.replace('_', ' ')}"
            found.append(ExampleImage(caption, path))
    return tuple(found)


def _rejected(category: str, problem: str, detail: str) -> CategoryResults:
    logger.warning("Results for %s rejected: %s", category, detail)
    return CategoryResults(category, problem=problem, detail=detail)


def load_category(
    report_root: Path, artifact_root: Path | None, category: str
) -> CategoryResults | None:
    """Load and validate the results of one category; None if there are no results at all."""
    runs: list[Run] = []
    missing: list[str] = []
    try:
        for spec in RUN_SPECS:
            run = load_run(report_root, artifact_root, spec, category)
            (runs if run is not None else missing).append(run or spec.label)  # type: ignore[arg-type]
    except EvaluationError as exc:
        return _rejected(
            category, "The result files could not be read or are inconsistent.", str(exc)
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _rejected(category, "The result files are malformed.", repr(exc))
    if not runs:
        return None
    run_tuple = tuple(r for r in runs if isinstance(r, Run))
    try:
        problems = [p for run in run_tuple for p in verify_run(run)]
        if problems:
            return _rejected(
                category,
                "The saved summaries do not match the saved per-image predictions.",
                "; ".join(problems),
            )
        if len(run_tuple) >= 2:
            check_comparable(run_tuple)
    except EvaluationError as exc:
        return _rejected(
            category, "The models were not evaluated on identical splits and settings.", str(exc)
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _rejected(category, "The result files are malformed.", repr(exc))
    paired = paired_auroc_differences(run_tuple) if len(run_tuple) >= 2 else []
    return CategoryResults(
        category=category,
        runs=run_tuple,
        records=tuple(run_record(r) for r in run_tuple),
        paired=tuple(paired),
        not_measured=tuple(str(m) for m in missing),
        examples=_examples(report_root, category, run_tuple),
    )


def load_results(
    report_root: Path, artifact_root: Path | None = None
) -> dict[str, CategoryResults]:
    """All categories that have any result files, validated. Empty if there are none."""
    root = Path(report_root)
    results: dict[str, CategoryResults] = {}
    if not root.is_dir():
        return results
    for category in SUPPORTED_OBJECTS:
        loaded = load_category(root, artifact_root, category)
        if loaded is not None:
            results[category] = loaded
    return results


def metric_rows(results: CategoryResults) -> list[dict[str, object]]:
    """Display rows (one per model) for the metrics table; undefined values are the text 'undefined'."""

    def fmt(value: float | None, digits: int = 3) -> str:
        return "undefined" if value is None else f"{value:.{digits}f}"

    rows: list[dict[str, object]] = []
    for record in results.records:
        image, pixel = record["metrics"]["image"], record["metrics"]["pixel"]
        ci = image["auroc_ci95"]
        rows.append(
            {
                "Model": record["label"],
                "Image AUROC": fmt(image["auroc"]),
                "95% CI": "undefined" if ci is None else f"{ci[0]:.3f} - {ci[1]:.3f}",
                "Image AP": fmt(image["average_precision"]),
                "Pixel AUROC": fmt(pixel["auroc"]),
                "Pixel AP": fmt(pixel["average_precision"]),
                "Pixel IoU": fmt(pixel["iou"], 2),
                "Pixel Dice": fmt(pixel["dice"], 2),
                "Precision": fmt(image["precision"]),
                "Recall": fmt(image["recall"]),
                "FPR": fmt(image["false_positive_rate"]),
                "F1": fmt(image["f1"]),
            }
        )
    return rows


def _perf_text(perf: dict, key: str, scale: float = 1.0, unit: str = "", digits: int = 1) -> str:
    value = perf.get(key)
    return "-" if value is None else f"{value * scale:.{digits}f}{unit}"


def performance_rows(results: CategoryResults) -> list[dict[str, object]]:
    """Display rows for fit time, latency, and memory as measured by the evaluation runs."""
    rows: list[dict[str, object]] = []
    for record in results.records:
        perf = record["performance"]
        rows.append(
            {
                "Model": record["label"],
                "Fit images": perf.get("n_fit_images", "-"),
                "Fit time": _perf_text(perf, "fit_seconds", 1.0, " s"),
                "Latency (mean)": _perf_text(perf, "latency_seconds_mean", 1000.0, " ms", 0),
                "Latency (max)": _perf_text(perf, "latency_seconds_max", 1000.0, " ms", 0),
                "Images / second": _perf_text(perf, "images_per_second"),
                "Peak memory, fit": _perf_text(perf, "peak_memory_gb_after_fit", 1.0, " GB", 2),
                "Peak memory, scoring": _perf_text(
                    perf, "peak_memory_gb_after_scoring", 1.0, " GB", 2
                ),
            }
        )
    return rows


def confusion_matrix(record: dict) -> dict[str, dict[str, int]]:
    """The 2x2 confusion matrix of one run: rows are the true class, columns the decision."""
    cm = record["metrics"]["image"]["confusion_matrix"]
    return {
        "Truly normal": {"Predicted normal": cm["tn"], "Predicted anomaly": cm["fp"]},
        "Truly anomalous": {"Predicted normal": cm["fn"], "Predicted anomaly": cm["tp"]},
    }
