"""Export the small set of precomputed files a hosted, dataset-free run of the app needs.

The hosted app (for example on Streamlit Community Cloud) has no VisA dataset, no pretrained
weights, and little memory. What it can show without them is exactly:

- the evaluation dashboard: `summary.json` and `predictions.csv` of every evaluated run, the
  comparison tables, and one comparison figure per category;
- inspection with the classical baseline (Model A), which needs no pretrained weights: its
  complete artifact (about 1.5 MB per category).

The deep models' artifacts (30 to 190 MB each) and the per-run example figures (about 6 MB each)
are left out on purpose. Nothing is computed or edited here: files are copied byte for byte, and
the copied classical artifacts are loaded through the artifact store (checksums included) before
the export is reported as done.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from inspectai.artifacts.store import MANIFEST
from inspectai.data.dataset import SUPPORTED_OBJECTS
from inspectai.errors import ConfigError
from inspectai.evaluation.comparison import RUN_SPECS
from inspectai.evaluation.dashboard import load_results
from inspectai.inference.predictor import load_predictor, option_by_key

ARTIFACT_FILES = ("manifest.json", "tensors.safetensors", "split.json", "config.snapshot.yaml")
RUN_REPORT_FILES = ("summary.json", "predictions.csv")
COMPARISON_FILES = ("comparison.csv", "comparison.json", "comparison.md")
HOSTED_MODEL_KEY = "classical"  # the only model that runs without pretrained weights


@dataclass(frozen=True)
class BundleSummary:
    """What an export copied."""

    destination: Path
    files: int
    bytes: int
    report_categories: tuple[str, ...]
    artifact_categories: tuple[str, ...]


def _copy(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target.stat().st_size


def export_hosted_bundle(
    artifact_root: Path,
    report_root: Path,
    destination: Path,
    categories: tuple[str, ...] | None = None,
    force: bool = False,
) -> BundleSummary:
    """Copy the dashboard files and the classical artifacts into `destination`.

    `destination` receives `artifacts/` and `reports/`. An existing bundle is only replaced with
    `force`. Raises ConfigError for unusable inputs and ArtifactError (from the artifact store) if
    a copied artifact fails its integrity checks.
    """
    artifact_root, report_root, destination = (
        Path(artifact_root),
        Path(report_root),
        Path(destination),
    )
    if not report_root.is_dir() or not artifact_root.is_dir():
        raise ConfigError("The artifact and report folders must exist.")
    chosen = tuple(categories) if categories else SUPPORTED_OBJECTS
    unknown = [c for c in chosen if c not in SUPPORTED_OBJECTS]
    if unknown:
        raise ConfigError(f"Unsupported categories: {unknown}")
    for source in (artifact_root, report_root):
        if destination.resolve() in (source.resolve(), *source.resolve().parents):
            raise ConfigError("The destination must not contain the source folders.")
    out_artifacts, out_reports = destination / "artifacts", destination / "reports"
    if out_artifacts.exists() or out_reports.exists():
        if not force:
            raise ConfigError("The destination already holds a bundle; use --force to replace it.")
        shutil.rmtree(out_artifacts, ignore_errors=True)
        shutil.rmtree(out_reports, ignore_errors=True)

    files = size = 0
    report_categories: list[str] = []
    artifact_categories: list[str] = []
    for category in chosen:
        copied = False
        for spec in RUN_SPECS:
            folder = report_root / spec.report_folder / category
            if not (folder / RUN_REPORT_FILES[0]).is_file():
                continue
            for name in RUN_REPORT_FILES:
                size += _copy(folder / name, out_reports / spec.report_folder / category / name)
                files += 1
            copied = True
        figure = report_root / "comparison" / f"{category}_examples.png"
        if copied and figure.is_file():
            size += _copy(figure, out_reports / "comparison" / figure.name)
            files += 1
        if copied:
            report_categories.append(category)

        folder = artifact_root / HOSTED_MODEL_KEY / category
        if (folder / MANIFEST).is_file():
            for name in ARTIFACT_FILES:
                if (folder / name).is_file():
                    size += _copy(folder / name, out_artifacts / HOSTED_MODEL_KEY / category / name)
                    files += 1
            artifact_categories.append(category)
    for name in COMPARISON_FILES:
        if (report_root / "comparison" / name).is_file():
            size += _copy(report_root / "comparison" / name, out_reports / "comparison" / name)
            files += 1
    if not report_categories and not artifact_categories:
        raise ConfigError("Nothing to export: no evaluation reports or classical artifacts found.")

    # The copy is only useful if it loads and its results are consistent.
    for category in artifact_categories:
        load_predictor(out_artifacts, option_by_key(HOSTED_MODEL_KEY), category)
    if report_categories:
        valid = {c for c, r in load_results(out_reports, out_artifacts).items() if r.valid}
        invalid = [c for c in report_categories if c not in valid]
        if invalid:
            raise ConfigError(f"The exported results of {invalid} do not pass validation.")
    return BundleSummary(
        destination, files, size, tuple(report_categories), tuple(artifact_categories)
    )
