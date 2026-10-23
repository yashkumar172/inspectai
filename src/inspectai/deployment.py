"""Export the precomputed files a hosted, dataset-free run of the app needs.

The hosted app (for example on Streamlit Community Cloud) has no VisA dataset and little memory.
What it needs to show results and to inspect uploaded images is:

- the evaluation dashboard: `summary.json` and `predictions.csv` of every evaluated run, the
  comparison tables, and one comparison figure per category;
- Model C (PatchCore-inspired, `dev` profile) as the primary inspection model: per category its
  artifact (about 46 MB, mostly the coreset memory bank) plus the ResNet-18 ImageNet weights it
  was fitted with (44.7 MB, shared by all categories);
- optionally Model A (classical) as a baseline: about 1.5 MB per category, no weights needed.

Left out on purpose: the dataset, the `full` profile (Wide ResNet-50-2 weights are 132 MB and its
banks 184 MB per category, above GitHub's 100 MB file limit), Model B, and the per-run example
figures (about 6 MB each). Nothing is computed or edited here: files are copied byte for byte,
the weights are checked against the SHA-256 recorded in the artifact manifests, and every copied
artifact is loaded through the artifact store (checksums included) before the export is reported
as done.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from inspectai.artifacts.store import MANIFEST
from inspectai.data.dataset import SUPPORTED_OBJECTS
from inspectai.errors import ArtifactError, ConfigError
from inspectai.evaluation.comparison import RUN_SPECS
from inspectai.evaluation.dashboard import load_results
from inspectai.inference.predictor import load_predictor, option_by_key
from inspectai.models.features import _sha256_file, load_backbone, weights_cache_path

ARTIFACT_FILES = ("manifest.json", "tensors.safetensors", "split.json", "config.snapshot.yaml")
RUN_REPORT_FILES = ("summary.json", "predictions.csv")
COMPARISON_FILES = ("comparison.csv", "comparison.json", "comparison.md")
# Model keys (report folder names) that can be bundled: Model C `dev` and the classical baseline.
HOSTED_MODEL_KEYS = ("patchcore", "classical")
DEFAULT_MODEL_KEYS = ("classical",)
# Where the bundled weights live below the bundle root; TORCH_HOME points at `torch_cache`.
WEIGHTS_SUBDIR = Path("torch_cache") / "hub"


@dataclass(frozen=True)
class BundleSummary:
    """What an export copied."""

    destination: Path
    files: int
    bytes: int
    report_categories: tuple[str, ...]
    artifact_models: dict[str, tuple[str, ...]]  # model key -> categories
    weights: tuple[str, ...]  # file names of the bundled pretrained weights

    @property
    def artifact_categories(self) -> tuple[str, ...]:
        return tuple(
            c for c in SUPPORTED_OBJECTS if any(c in cats for cats in self.artifact_models.values())
        )


def _copy(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target.stat().st_size


def _read_weights_record(manifest_path: Path) -> tuple[str, str | None]:
    """(backbone, sha256) recorded in an artifact manifest."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["weights"]
        return str(record["backbone"]), record["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"The manifest lacks a weights record: {manifest_path.name}") from exc


def export_hosted_bundle(
    artifact_root: Path,
    report_root: Path,
    destination: Path,
    categories: tuple[str, ...] | None = None,
    force: bool = False,
    models: tuple[str, ...] = DEFAULT_MODEL_KEYS,
    weights_hub_dir: Path | None = None,
) -> BundleSummary:
    """Copy the dashboard files, the chosen models' artifacts, and their weights.

    `destination` receives `artifacts/`, `reports/`, and (for Model C) `torch_cache/`. An existing
    bundle is only replaced with `force`. `models` selects which artifacts to copy
    (`HOSTED_MODEL_KEYS`); `weights_hub_dir` is the torchvision hub folder the weights are read
    from (default: the current `TORCH_HOME`). Raises ConfigError for unusable inputs, and
    ArtifactError (from the artifact store) if a copied artifact fails its integrity checks or the
    weights do not match the manifest.
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
    bad_models = [m for m in models if m not in HOSTED_MODEL_KEYS]
    if bad_models or not models:
        raise ConfigError(f"Models to bundle must be a non-empty subset of {HOSTED_MODEL_KEYS}")
    for source in (artifact_root, report_root):
        if destination.resolve() in (source.resolve(), *source.resolve().parents):
            raise ConfigError("The destination must not contain the source folders.")
    out_artifacts, out_reports = destination / "artifacts", destination / "reports"
    out_torch = destination / "torch_cache"
    if out_artifacts.exists() or out_reports.exists() or out_torch.exists():
        if not force:
            raise ConfigError("The destination already holds a bundle; use --force to replace it.")
        for old in (out_artifacts, out_reports, out_torch):
            shutil.rmtree(old, ignore_errors=True)
    source_hub = Path(torch.hub.get_dir()) if weights_hub_dir is None else Path(weights_hub_dir)
    bundle_hub = destination / WEIGHTS_SUBDIR

    files = size = 0
    report_categories: list[str] = []
    artifact_models: dict[str, list[str]] = {m: [] for m in models}
    weights_copied: dict[str, str] = {}  # backbone -> file name
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

        for key in models:
            folder = artifact_root / key / category
            if not (folder / MANIFEST).is_file():
                continue
            for name in ARTIFACT_FILES:
                if (folder / name).is_file():
                    size += _copy(folder / name, out_artifacts / key / category / name)
                    files += 1
            artifact_models[key].append(category)
            if key == "classical":
                continue
            backbone, sha256 = _read_weights_record(folder / MANIFEST)
            if sha256 is None or backbone in weights_copied:
                continue  # injected (test) backbone, or this backbone is already bundled
            cached = weights_cache_path(backbone, source_hub)
            if not cached.is_file():
                raise ConfigError(
                    f"The pretrained weights for {backbone} are not cached; fetch them first with "
                    f"python scripts/fetch_weights.py --backbone {backbone}"
                )
            if _sha256_file(cached) != sha256:
                raise ArtifactError(
                    f"The cached {backbone} weights differ from the ones {key}/{category} was "
                    "fitted with."
                )
            target = weights_cache_path(backbone, bundle_hub)
            size += _copy(cached, target)
            files += 1
            weights_copied[backbone] = target.name
    for name in COMPARISON_FILES:
        if (report_root / "comparison" / name).is_file():
            size += _copy(report_root / "comparison" / name, out_reports / "comparison" / name)
            files += 1
    if not report_categories and not any(artifact_models.values()):
        raise ConfigError("Nothing to export: no evaluation reports or model artifacts found.")

    # The copy is only useful if it loads and its results are consistent.
    for key, cats in artifact_models.items():
        for category in cats:
            backbone_module = None
            if key != "classical":
                backbone, sha256 = _read_weights_record(out_artifacts / key / category / MANIFEST)
                backbone_module, info = load_backbone(backbone, False, bundle_hub)
                if info.sha256 != sha256:
                    raise ArtifactError("The bundled weights differ from the manifest's record.")
            load_predictor(out_artifacts, option_by_key(key), category, backbone_module)
    if report_categories:
        valid = {c for c, r in load_results(out_reports, out_artifacts).items() if r.valid}
        invalid = [c for c in report_categories if c not in valid]
        if invalid:
            raise ConfigError(f"The exported results of {invalid} do not pass validation.")
    return BundleSummary(
        destination,
        files,
        size,
        tuple(report_categories),
        {k: tuple(v) for k, v in artifact_models.items()},
        tuple(weights_copied.values()),
    )
