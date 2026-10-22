"""Cross-model comparison reports (SPEC.md Sections 7.5 and 16, Milestone 5).

The functions here read the per-model `summary.json` and `predictions.csv` files that the
evaluation harness wrote, refuse to compare runs that did not use identical splits and
evaluation settings, re-derive every image-level metric from the saved predictions, and build
the comparison tables. Nothing is computed from model code: a metric in the report is either
re-derived from the stored per-image scores and labels, or (for the pixel metrics, which need
the full anomaly maps) copied from the harness summary and labeled as such.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from inspectai.config import PROFILES
from inspectai.errors import EvaluationError
from inspectai.evaluation.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    classification_metrics,
    paired_bootstrap_auroc_difference,
)

SCHEMA_VERSION = 1
_TOLERANCE = 1.5e-6  # summaries store metrics rounded to 6 decimals


@dataclass(frozen=True)
class RunSpec:
    """Where the results of one model configuration live (report and artifact sub-folders)."""

    label: str
    model: str
    profile: str | None
    report_folder: str
    artifact_folder: str | None


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec("A classical", "classical", None, "classical", "classical"),
    RunSpec("B Gaussian dev", "gaussian", "dev", "gaussian", "gaussian"),
    RunSpec("B Gaussian full", "gaussian", "full", "gaussian-full", "gaussian-full"),
    RunSpec("C PatchCore-inspired dev", "patchcore", "dev", "patchcore", "patchcore"),
    RunSpec("C PatchCore-inspired full", "patchcore", "full", "patchcore-full", "patchcore-full"),
)

# Pairs (a, b) for paired AUROC differences: each deep model against the baseline, and
# Model C against Model B of the same profile.
PAIRS: tuple[tuple[str, str], ...] = (
    ("B Gaussian dev", "A classical"),
    ("C PatchCore-inspired dev", "A classical"),
    ("C PatchCore-inspired dev", "B Gaussian dev"),
    ("B Gaussian full", "A classical"),
    ("C PatchCore-inspired full", "A classical"),
    ("C PatchCore-inspired full", "B Gaussian full"),
)

_MACRO_METRICS = (
    "image_auroc",
    "image_average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "pixel_iou",
    "pixel_dice",
    "f1",
    "recall",
    "false_positive_rate",
)

_ENVIRONMENT_KEYS = (
    "python",
    "platform",
    "processor",
    "physical_cores",
    "logical_cores",
    "ram_total_gb",
    "power_plan",
    "plugged_in",
    "numpy",
    "opencv",
    "scikit_learn",
)

_PERFORMANCE_KEYS = (
    "n_fit_images",
    "fit_seconds",
    "peak_memory_gb_after_fit",
    "peak_memory_gb_after_scoring",
    "latency_seconds_mean",
    "latency_seconds_median",
    "latency_seconds_max",
    "images_per_second",
)


@dataclass(frozen=True)
class Run:
    """One model configuration evaluated on one category."""

    spec: RunSpec
    category: str
    summary: dict
    predictions: list[dict]
    manifest: dict | None

    @property
    def image_metrics(self) -> dict:
        return self.summary["metrics"]["image"]

    @property
    def pixel_metrics(self) -> dict:
        return self.summary["metrics"]["pixel"]


def read_predictions(path: Path) -> list[dict]:
    """Read a `predictions.csv` written by `write_predictions_csv`."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvaluationError(f"Predictions could not be read: {path}") from exc
    parsed: list[dict] = []
    try:
        for row in rows:
            parsed.append(
                {
                    "image": row["image"],
                    "split": row["split"],
                    "label": int(row["label"]),
                    "score": float(row["score"]),
                    "predicted": int(row["predicted"]),
                    "threshold": float(row["threshold"]),
                }
            )
    except (KeyError, ValueError) as exc:
        raise EvaluationError(f"Malformed predictions file: {path}") from exc
    if not parsed:
        raise EvaluationError(f"Predictions file is empty: {path}")
    return parsed


def load_run(
    report_root: Path, artifact_root: Path | None, spec: RunSpec, category: str
) -> Run | None:
    """Load one run, or return None if it was not run (a planned or skipped experiment)."""
    folder = Path(report_root) / spec.report_folder / category
    summary_path = folder / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Summary could not be read: {summary_path}") from exc
    if summary.get("model") != spec.model or summary.get("category") != category:
        raise EvaluationError(
            f"{summary_path} is for {summary.get('model')}/{summary.get('category')}, "
            f"expected {spec.model}/{category}"
        )
    if spec.profile is not None:
        expected = PROFILES[spec.profile]["backbone"]
        found = summary.get("model_config", {}).get("backbone")
        if found != expected:
            raise EvaluationError(
                f"{summary_path} used backbone {found!r}, but '{spec.label}' needs {expected!r}"
            )
    predictions = read_predictions(folder / "predictions.csv")
    manifest = None
    if artifact_root is not None and spec.artifact_folder is not None:
        manifest_path = Path(artifact_root) / spec.artifact_folder / category / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Run(spec, category, summary, predictions, manifest)


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= _TOLERANCE


def verify_run(run: Run) -> list[str]:
    """Re-derive the image-level metrics from the saved predictions; list any disagreement.

    A non-empty result means the summary does not follow from the per-image scores and labels.
    """
    problems: list[str] = []
    who = f"{run.spec.label}/{run.category}"
    rows = run.predictions
    scores = np.array([r["score"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    stored = run.image_metrics
    options = run.summary["options"]

    if len({r["image"] for r in rows}) != len(rows):
        problems.append(f"{who}: duplicate images in predictions")
    n_normal, n_anomalous = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    if (n_normal, n_anomalous) != (stored["n_normal"], stored["n_anomalous"]):
        problems.append(f"{who}: label counts {n_normal}/{n_anomalous} differ from the summary")
    thresholds = {r["threshold"] for r in rows}
    threshold = float(stored["threshold"])
    if thresholds != {threshold}:
        problems.append(f"{who}: per-image thresholds differ from the summary threshold")
    if any(int(r["score"] > threshold) != r["predicted"] for r in rows):
        problems.append(f"{who}: 'predicted' is not score > threshold for every image")

    if not _close(stored["auroc"], _round(auroc(scores, labels))):
        problems.append(f"{who}: AUROC does not match the predictions")
    if not _close(stored["average_precision"], _round(average_precision(scores, labels))):
        problems.append(f"{who}: average precision does not match the predictions")
    interval = bootstrap_auroc_ci(
        scores, labels, options["bootstrap_resamples"], options["bootstrap_seed"]
    )
    expected_ci = None if interval is None else [_round(interval[0]), _round(interval[1])]
    stored_ci = stored["auroc_ci95"]
    if (expected_ci is None) != (stored_ci is None) or (
        expected_ci is not None
        and not all(_close(a, b) for a, b in zip(expected_ci, stored_ci, strict=True))
    ):
        problems.append(f"{who}: AUROC confidence interval does not match the predictions")
    cls = classification_metrics(scores, labels, threshold)
    if stored["confusion_matrix"] != {"tp": cls.tp, "fp": cls.fp, "tn": cls.tn, "fn": cls.fn}:
        problems.append(f"{who}: confusion matrix does not match the predictions")
    for key, value in (
        ("precision", cls.precision),
        ("recall", cls.recall),
        ("f1", cls.f1),
        ("false_positive_rate", cls.false_positive_rate),
        ("false_negative_rate", cls.false_negative_rate),
    ):
        if not _close(stored[key], _round(value)):
            problems.append(f"{who}: {key} does not match the predictions")
    return problems


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


_COMPARED_OPTIONS = (
    "threshold_policy",
    "quantile",
    "eval_resolution",
    "split_seed",
    "bootstrap_resamples",
    "bootstrap_seed",
)


def check_comparable(runs: Sequence[Run]) -> None:
    """Raise EvaluationError unless all runs used identical splits and evaluation settings.

    Compared: the SHA-256 of the fit, calibration, and test file lists; the split seed; the
    evaluated split; the evaluation resolution; the threshold policy and its parameters; the
    bootstrap settings; the sample counts; and the exact set of test images with their labels.
    """
    if len(runs) < 2:
        raise EvaluationError("A comparison needs at least two runs")
    first = runs[0]
    problems: list[str] = []
    ref_images = {r["image"]: r["label"] for r in first.predictions}
    for run in runs[1:]:
        pairs = f"'{first.spec.label}' and '{run.spec.label}' ({run.category})"
        if run.category != first.category:
            problems.append(f"{pairs}: different categories")
        a, b = first.summary, run.summary
        if a["split"]["file_list_sha256"] != b["split"]["file_list_sha256"]:
            problems.append(f"{pairs}: different split files")
        if a["split"]["seed"] != b["split"]["seed"]:
            problems.append(f"{pairs}: different split seeds")
        for key in ("evaluation_split", "evaluation_resolution"):
            if a["metrics"][key] != b["metrics"][key]:
                problems.append(f"{pairs}: different {key}")
        for key in _COMPARED_OPTIONS:
            if a["options"].get(key) != b["options"].get(key):
                problems.append(f"{pairs}: different option {key}")
        if a["threshold"]["image"]["policy"] != b["threshold"]["image"]["policy"]:
            problems.append(f"{pairs}: different threshold policy")
        for key in ("n_normal", "n_anomalous"):
            if first.image_metrics[key] != run.image_metrics[key]:
                problems.append(f"{pairs}: different {key}")
        if {r["image"]: r["label"] for r in run.predictions} != ref_images:
            problems.append(f"{pairs}: different test images or labels")
    if problems:
        raise EvaluationError("Runs are not comparable: " + "; ".join(problems))


def run_record(run: Run) -> dict:
    summary = run.summary
    record: dict = {
        "label": run.spec.label,
        "model": run.spec.model,
        "profile": run.spec.profile,
        "created_utc": summary.get("created_utc"),
        "model_config": summary.get("model_config"),
        "metrics": summary["metrics"],
        "threshold": summary["threshold"],
        "calibration_scores": summary.get("calibration_scores"),
        "performance": {
            k: summary["performance"][k] for k in _PERFORMANCE_KEYS if k in summary["performance"]
        },
        "environment": {
            k: summary["environment"][k] for k in _ENVIRONMENT_KEYS if k in summary["environment"]
        },
        "weights": None,
        "artifact_files_sha256": None,
    }
    if run.manifest is not None:
        record["weights"] = run.manifest.get("weights")
        record["artifact_files_sha256"] = run.manifest.get("files")
    return record


def paired_auroc_differences(runs: Sequence[Run]) -> list[dict]:
    by_label = {r.spec.label: r for r in runs}
    out: list[dict] = []
    for label_a, label_b in PAIRS:
        if label_a not in by_label or label_b not in by_label:
            continue
        a, b = by_label[label_a], by_label[label_b]
        by_image_b = {r["image"]: r["score"] for r in b.predictions}
        images = [r["image"] for r in a.predictions]
        scores_a = [r["score"] for r in a.predictions]
        scores_b = [by_image_b[i] for i in images]
        labels = [r["label"] for r in a.predictions]
        options = a.summary["options"]
        result = paired_bootstrap_auroc_difference(
            scores_a,
            scores_b,
            labels,
            options.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES),
            options.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED),
        )
        out.append(
            {
                "a": label_a,
                "b": label_b,
                "auroc_difference": None if result is None else round(result[0], 6),
                "ci95": None if result is None else [round(result[1], 6), round(result[2], 6)],
                "excludes_zero": None if result is None else bool(result[1] > 0 or result[2] < 0),
            }
        )
    return out


def _flat_metrics(run: Run) -> dict[str, float | None]:
    image, pixel = run.image_metrics, run.pixel_metrics
    return {
        "image_auroc": image["auroc"],
        "image_average_precision": image["average_precision"],
        "pixel_auroc": pixel["auroc"],
        "pixel_average_precision": pixel["average_precision"],
        "pixel_iou": pixel["iou"],
        "pixel_dice": pixel["dice"],
        "f1": image["f1"],
        "recall": image["recall"],
        "false_positive_rate": image["false_positive_rate"],
    }


def build_comparison(
    runs_by_category: Mapping[str, Sequence[Run]],
    not_measured: Mapping[str, Sequence[str]] | None = None,
    verify: bool = True,
) -> dict:
    """Build the comparison of all runs, per category first, then macro averages.

    Every category is checked with `check_comparable`, and (unless `verify` is False) every run
    is re-derived from its predictions with `verify_run`; any problem raises EvaluationError.
    Macro averages are taken per model over the categories where that model was measured, and
    skip undefined values (the number of categories actually used is reported).
    """
    if not runs_by_category:
        raise EvaluationError("No categories to compare")
    categories: dict[str, dict] = {}
    undefined: list[str] = []
    for category, runs in runs_by_category.items():
        check_comparable(runs)
        if verify:
            problems = [p for run in runs for p in verify_run(run)]
            if problems:
                raise EvaluationError(
                    "Summaries do not follow from predictions: " + "; ".join(problems)
                )
        first = runs[0]
        categories[category] = {
            "n_normal": first.image_metrics["n_normal"],
            "n_anomalous": first.image_metrics["n_anomalous"],
            "evaluation_split": first.summary["metrics"]["evaluation_split"],
            "evaluation_resolution": first.summary["metrics"]["evaluation_resolution"],
            "split_file_list_sha256": first.summary["split"]["file_list_sha256"],
            "runs": [run_record(run) for run in runs],
            "paired_auroc_differences": paired_auroc_differences(runs),
            "not_measured": list((not_measured or {}).get(category, [])),
        }
        for run in runs:
            undefined.extend(
                f"{category}/{run.spec.label}/{name}"
                for name, value in _flat_metrics(run).items()
                if value is None
            )

    macro: dict[str, dict] = {}
    labels = [s.label for s in RUN_SPECS]
    for label in labels:
        measured = [
            (cat, run)
            for cat, runs in runs_by_category.items()
            for run in runs
            if run.spec.label == label
        ]
        if not measured:
            continue
        entry: dict = {"n_categories": len(measured), "categories": [c for c, _ in measured]}
        for metric in _MACRO_METRICS:
            values = [v for _, run in measured if (v := _flat_metrics(run)[metric]) is not None]
            entry[metric] = round(float(np.mean(values)), 6) if values else None
            entry[f"{metric}_n_defined"] = len(values)
        macro[label] = entry

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": _protocol(runs_by_category),
        "categories": categories,
        "macro_average": macro,
        "undefined_metrics": undefined,
        "verification": {
            "image_metrics_rederived_from_predictions": bool(verify),
            "pixel_metrics": "copied from the harness summaries (they need the full anomaly maps)",
        },
    }


def _protocol(runs_by_category: Mapping[str, Sequence[Run]]) -> dict:
    first = next(iter(runs_by_category.values()))[0].summary
    options = first["options"]
    return {
        "dataset": "VisA (official one-class split, 1cls.csv)",
        "evaluation_split": first["metrics"]["evaluation_split"],
        "evaluation_resolution": first["metrics"]["evaluation_resolution"],
        "threshold_policy": first["threshold"]["image"]["policy"],
        "threshold_quantile": options["quantile"],
        "split_seed": options["split_seed"],
        "bootstrap": {
            "resamples": options["bootstrap_resamples"],
            "seed": options["bootstrap_seed"],
            "confidence": 0.95,
        },
        "pixel_metrics": "pooled over all test images at the evaluation resolution; the pixel "
        "threshold comes from calibration_normal anomaly-map values, never from test masks",
        "published_results": "not compared: published VisA numbers use other protocols",
    }


# ----- tabular output -------------------------------------------------------------------------

CSV_COLUMNS = (
    "category",
    "label",
    "model",
    "profile",
    "n_normal",
    "n_anomalous",
    "image_auroc",
    "image_auroc_ci_low",
    "image_auroc_ci_high",
    "image_average_precision",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "image_threshold",
    "pixel_auroc",
    "pixel_average_precision",
    "pixel_iou",
    "pixel_dice",
    "pixel_threshold",
    "fit_seconds",
    "peak_memory_gb_after_fit",
    "latency_seconds_mean",
)


def comparison_rows(comparison: Mapping) -> list[dict]:
    """Flat rows (one per category and model) for the CSV; undefined metrics are None."""
    rows: list[dict] = []
    for category, data in comparison["categories"].items():
        for run in data["runs"]:
            image, pixel = run["metrics"]["image"], run["metrics"]["pixel"]
            ci = image["auroc_ci95"] or [None, None]
            cm = image["confusion_matrix"]
            perf = run["performance"]
            rows.append(
                {
                    "category": category,
                    "label": run["label"],
                    "model": run["model"],
                    "profile": run["profile"],
                    "n_normal": image["n_normal"],
                    "n_anomalous": image["n_anomalous"],
                    "image_auroc": image["auroc"],
                    "image_auroc_ci_low": ci[0],
                    "image_auroc_ci_high": ci[1],
                    "image_average_precision": image["average_precision"],
                    "precision": image["precision"],
                    "recall": image["recall"],
                    "f1": image["f1"],
                    "false_positive_rate": image["false_positive_rate"],
                    "false_negative_rate": image["false_negative_rate"],
                    "tp": cm["tp"],
                    "fp": cm["fp"],
                    "tn": cm["tn"],
                    "fn": cm["fn"],
                    "image_threshold": image["threshold"],
                    "pixel_auroc": pixel["auroc"],
                    "pixel_average_precision": pixel["average_precision"],
                    "pixel_iou": pixel["iou"],
                    "pixel_dice": pixel["dice"],
                    "pixel_threshold": pixel["threshold"],
                    "fit_seconds": perf.get("fit_seconds"),
                    "peak_memory_gb_after_fit": perf.get("peak_memory_gb_after_fit"),
                    "latency_seconds_mean": perf.get("latency_seconds_mean"),
                }
            )
    return rows


def write_comparison_csv(path: Path, comparison: Mapping) -> None:
    """Write the flat comparison rows; undefined metrics are written as the word `undefined`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in comparison_rows(comparison):
            writer.writerow({k: ("undefined" if v is None else v) for k, v in row.items()})


def write_comparison_json(path: Path, comparison: Mapping) -> None:
    """Write the full comparison, including the configuration metadata of every run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(comparison, indent=1, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _fmt(value: float | None, digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


REPRODUCE_NOTE = r"""
## How to reproduce

Replace `<category>` by `cashew`, `pipe_fryum`, `chewinggum`, or `fryum` (PowerShell; the data
root and the report and artifact folders come from the `INSPECTAI_*` environment variables).
Fitting and thresholds use only the `fit` and `calibration_normal` splits (seed 42); the test
split is only scored. Every run's exact configuration, split file hashes, weights SHA-256,
and library versions are in `comparison.json`.

```powershell
python scripts/fetch_weights.py --backbone resnet18
python scripts/fetch_weights.py --backbone wide_resnet50_2
python scripts/evaluate_model.py --model classical --category <category>
python scripts/fit_model.py --model gaussian --category <category> --profile dev
python scripts/evaluate_model.py --model gaussian --category <category> --profile dev
python scripts/fit_model.py --model gaussian --category <category> --profile full --output-dir <artifacts>\gaussian-full\<category>
python scripts/evaluate_model.py --model gaussian --category <category> --profile full --artifact-dir <artifacts>\gaussian-full\<category> --output-dir <reports>\gaussian-full\<category>
python scripts/fit_model.py --model patchcore --category <category> --profile dev
python scripts/evaluate_model.py --model patchcore --category <category> --profile dev
python scripts/fit_model.py --model patchcore --category <category> --profile full --output-dir <artifacts>\patchcore-full\<category>
python scripts/evaluate_model.py --model patchcore --category <category> --profile full --artifact-dir <artifacts>\patchcore-full\<category> --output-dir <reports>\patchcore-full\<category>
python scripts/compare_models.py
```
"""


def format_markdown(comparison: Mapping) -> str:
    """The comparison as Markdown: per-category tables first, then macro averages."""
    protocol = comparison["protocol"]
    size = protocol["evaluation_resolution"]
    bootstrap = protocol["bootstrap"]
    intro = (
        "Every number below was measured by running the model; nothing is planned or assumed. "
        f"Split: `{protocol['evaluation_split']}`, evaluation resolution {size}x{size}, "
        f"threshold policy `{protocol['threshold_policy']}` (q = {protocol['threshold_quantile']}), "
        f"split seed {protocol['split_seed']}, {bootstrap['resamples']} bootstrap resamples "
        f"(seed {bootstrap['seed']})."
    )
    lines = [
        "# Model comparison",
        "",
        f"Generated {comparison['generated_utc']} UTC. Dataset: {protocol['dataset']}.",
        intro,
        "",
    ]
    table_header = (
        "| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | "
        "Precision | Recall | FPR | F1 | Fit time | Latency |"
    )
    for category, data in comparison["categories"].items():
        verified = (
            f"{data['n_normal']} normal and {data['n_anomalous']} anomalous test images; every "
            "model used identical splits and evaluation settings (verified)."
        )
        lines += [
            f"## {category}",
            "",
            verified,
            "",
            table_header,
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in data["runs"]:
            lines.append(_run_row(run))
        if data["not_measured"]:
            lines += [
                "",
                "Not measured (no results exist): " + ", ".join(data["not_measured"]) + ".",
            ]
        if data["paired_auroc_differences"]:
            lines += [
                "",
                "Paired AUROC differences (same test images, paired bootstrap 95% interval):",
                "",
                "| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |",
                "|---|---|---|---|---|",
            ]
            lines += [_diff_row(diff) for diff in data["paired_auroc_differences"]]
        lines.append("")

    macro_note = (
        "Unweighted mean over the categories in which a model was measured (the number of "
        "categories is shown); undefined values are skipped."
    )
    lines += [
        "## Macro average over the measured categories",
        "",
        macro_note,
        "",
        "| Model | Categories | Image AUROC | Image AP | Pixel AUROC | Pixel IoU | Pixel Dice | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, entry in comparison["macro_average"].items():
        names = ", ".join(entry["categories"])
        lines.append(
            f"| {label} | {entry['n_categories']} ({names}) | {_fmt(entry['image_auroc'])} | "
            f"{_fmt(entry['image_average_precision'])} | {_fmt(entry['pixel_auroc'])} | "
            f"{_fmt(entry['pixel_iou'], 2)} | {_fmt(entry['pixel_dice'], 2)} | "
            f"{_fmt(entry['f1'])} |"
        )

    derived = (
        "- Image-level metrics in this report were re-derived from each run's saved "
        "`predictions.csv` (scores and split-file labels) and agree with the harness summaries."
    )
    thresholds = (
        "- Thresholds are selected from the held-out normal training images only; test images "
        "and test labels never influence fitting or thresholds."
    )
    small = (
        "- Test sets are small (50 to 100 images per class); read the confidence intervals, and "
        "prefer the paired differences when comparing two models on the same images."
    )
    lines += [
        "",
        "## Protocol and caveats",
        "",
        derived,
        f"- Pixel metrics are copied from the harness summaries; {protocol['pixel_metrics']}.",
        thresholds,
        f"- Published results: {protocol['published_results']}.",
        small,
    ]
    if comparison["undefined_metrics"]:
        undefined = ", ".join(comparison["undefined_metrics"])
        lines.append(f"- Undefined metrics (the data cannot define them): {undefined}.")
    return "\n".join(lines) + "\n" + REPRODUCE_NOTE


def _run_row(run: Mapping) -> str:
    image, pixel, perf = run["metrics"]["image"], run["metrics"]["pixel"], run["performance"]
    ci = image["auroc_ci95"]
    ci_text = "undefined" if ci is None else f"{ci[0]:.3f}-{ci[1]:.3f}"
    fit, latency = perf.get("fit_seconds"), perf.get("latency_seconds_mean")
    fit_text = "-" if fit is None else f"{fit:.1f} s"
    latency_text = "-" if latency is None else f"{latency * 1000:.0f} ms"
    return (
        f"| {run['label']} | {_fmt(image['auroc'])} ({ci_text}) | "
        f"{_fmt(image['average_precision'])} | {_fmt(pixel['auroc'])} | "
        f"{_fmt(pixel['iou'], 2)} / {_fmt(pixel['dice'], 2)} | {_fmt(image['precision'])} | "
        f"{_fmt(image['recall'])} | {_fmt(image['false_positive_rate'])} | "
        f"{_fmt(image['f1'])} | {fit_text} | {latency_text} |"
    )


def _diff_row(diff: Mapping) -> str:
    ci, difference, excludes = diff["ci95"], diff["auroc_difference"], diff["excludes_zero"]
    interval = "undefined" if ci is None else f"{ci[0]:+.3f} to {ci[1]:+.3f}"
    value = "undefined" if difference is None else f"{difference:+.3f}"
    verdict = "undefined" if excludes is None else ("yes" if excludes else "no")
    return f"| {diff['a']} | {diff['b']} | {value} | {interval} | {verdict} |"
