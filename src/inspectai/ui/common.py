"""Shared helpers for the Streamlit pages: environment, cached loading, and safe messages.

Everything shown to the user passes through fixed message templates here. Exceptions are logged
for the operator, and the user only sees a short explanation that never contains a file system
path or a stack trace (SPEC.md Section 8, UI requirements).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from inspectai.config import PROFILES, Paths, load_app_paths, use_bundled_weights
from inspectai.data.dataset import Sample, VisADataset
from inspectai.errors import (
    ArtifactError,
    ConfigError,
    InspectAIError,
    ModelUnavailableError,
    WeightsNotFoundError,
)
from inspectai.evaluation.dashboard import CategoryResults, load_results
from inspectai.inference.predictor import ModelOption, Predictor, load_predictor, option_by_key

logger = logging.getLogger("inspectai.app")

MSG_NOT_CONFIGURED = (
    "The data location is not configured. Set the INSPECTAI_ARTIFACT_DIR and INSPECTAI_REPORT_DIR "
    "environment variables (and INSPECTAI_DATA_ROOT to use the local dataset; see the README), or "
    "provide the bundled `hosted` folder, and restart the app."
)
MSG_INTEGRITY = (
    "The saved model failed its integrity checks and was not loaded. Fit it again with "
    "`scripts/fit_model.py`."
)
MSG_LOAD_FAILED = "The model could not be loaded."
MSG_INFERENCE_FAILED = "The model could not process this image. No prediction was produced."
VISA_ATTRIBUTION = (
    "Dataset images: VisA (Zou et al., ECCV 2022), licensed CC BY 4.0. Heatmaps and overlays are "
    "modified material derived from the original images."
)


@dataclass(frozen=True)
class Environment:
    """The configured locations, or the reason they are missing."""

    paths: Paths | None
    message: str | None


def environment() -> Environment:
    """Read the INSPECTAI_* environment variables."""
    try:
        paths = load_app_paths()
    except ConfigError:
        return Environment(None, MSG_NOT_CONFIGURED)
    use_bundled_weights(paths)
    return Environment(paths, None)


def fit_hint(option: ModelOption, category: str) -> str:
    """The command that creates the missing model (no paths)."""
    base = f"python scripts/fit_model.py --model {option.model} --category {category}"
    if option.profile is None:
        return base
    hint = f"{base} --profile {option.profile}"
    if option.profile != "dev":
        hint += f" --output-dir <INSPECTAI_ARTIFACT_DIR>\\{option.artifact_folder}\\{category}"
    return hint


def weights_hint(option: ModelOption) -> str:
    backbone = PROFILES[option.profile]["backbone"] if option.profile else "the backbone"
    return (
        f"The pretrained weights for this model ({backbone}) are not available on this computer. "
        f"Fetch them once with: `python scripts/fetch_weights.py --backbone {backbone}`."
    )


@dataclass(frozen=True)
class LoadOutcome:
    """A loaded model or a user-safe reason why there is none. Failures are cached too."""

    predictor: Predictor | None
    message: str | None


@st.cache_resource(max_entries=3, show_spinner=False)
def _cached_load(artifact_root: str, option_key: str, category: str) -> LoadOutcome:
    option = option_by_key(option_key)
    try:
        return LoadOutcome(load_predictor(Path(artifact_root), option, category), None)
    except ModelUnavailableError:
        return LoadOutcome(
            None, f"No fitted model exists for this selection. {fit_hint(option, category)}"
        )
    except WeightsNotFoundError:
        logger.warning("Pretrained weights missing for %s", option_key)
        return LoadOutcome(None, weights_hint(option))
    except ArtifactError:
        logger.exception("Artifact for %s/%s failed its integrity checks", option_key, category)
        return LoadOutcome(None, MSG_INTEGRITY)
    except (InspectAIError, OSError, ValueError, RuntimeError):
        logger.exception("Loading %s/%s failed", option_key, category)
        return LoadOutcome(None, MSG_LOAD_FAILED)


def load_model(paths: Paths, option: ModelOption, category: str) -> LoadOutcome:
    """Load (and cache) a fitted model. The model is never refitted by the app."""
    return _cached_load(str(paths.artifact_dir), option.key, category)


def clear_model_cache() -> None:
    """Forget cached models and failures, e.g. after fitting a model while the app runs."""
    _cached_load.clear()


@st.cache_resource(show_spinner=False)
def _cached_dataset(split_csv: str, images_root: str) -> VisADataset | None:
    try:
        return VisADataset(Path(split_csv), Path(images_root))
    except InspectAIError:
        logger.warning("The local dataset is not available")
        return None


def get_dataset(paths: Paths) -> VisADataset | None:
    """The local VisA dataset if its split file and image folder exist, else None."""
    if not paths.has_dataset:
        return None
    if not paths.split_csv.is_file() or not paths.visa_images_root.is_dir():
        return None
    return _cached_dataset(str(paths.split_csv), str(paths.visa_images_root))


def dataset_test_images(dataset: VisADataset, category: str) -> list[Sample]:
    """The official test images of a category (sorted), or an empty list."""
    try:
        return sorted(dataset.get_samples(category, split="test"), key=lambda s: s.image_rel)
    except InspectAIError:
        return []


def _signature(report_root: Path) -> tuple[tuple[str, int], ...]:
    """Cheap fingerprint of the result files, so new evaluation results appear on reload."""
    files = sorted(Path(report_root).glob("*/*/summary.json")) if Path(report_root).is_dir() else []
    return tuple((f.parent.parent.name + "/" + f.parent.name, f.stat().st_mtime_ns) for f in files)


@st.cache_resource(max_entries=2, show_spinner=False)
def _cached_results(
    report_root: str, artifact_root: str, signature: tuple[tuple[str, int], ...]
) -> dict[str, CategoryResults]:
    return load_results(Path(report_root), Path(artifact_root))


def dashboard_results(paths: Paths) -> dict[str, CategoryResults]:
    """Validated evaluation results (cached until a result file changes)."""
    return _cached_results(
        str(paths.report_dir), str(paths.artifact_dir), _signature(paths.report_dir)
    )
