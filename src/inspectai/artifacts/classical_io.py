"""Saving and loading the fitted classical baseline through the artifact store.

The baseline stores only a per-pixel mean and standard deviation (two float32 arrays), so the
app can run it without refitting. `scripts/evaluate_model.py` still refits the baseline from the
`fit` split (about 8 seconds); the artifact is what `scripts/fit_model.py --model classical`
writes for the app.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from inspectai.artifacts.store import LoadedArtifact, load_artifact, save_artifact
from inspectai.data.splits import SplitResult
from inspectai.errors import ArtifactError, ModelError
from inspectai.models.classical import ClassicalConfig, ClassicalModel

MODEL_TYPE = "classical"


def save_classical(
    directory: Path,
    model: ClassicalModel,
    category: str,
    split: SplitResult,
    calibration: Mapping[str, object],
    fit_stats: Mapping[str, object],
    eval_resolution: int,
) -> Path:
    """Save a fitted baseline with its thresholds and measured fitting cost."""
    cfg = model.config
    info: dict[str, object] = {
        "backbone": None,
        "weights": None,
        "layers": [],
        "image_size": cfg.image_size,
        "feature_dim": 3,
        "resize_mode": cfg.resize_mode,
        "reference_aspect_ratio": model.reference_aspect_ratio,
        "seed": split.seed,
        "n_fit_images": model.n_fit_images,
        "smoothing_sigma": cfg.smoothing_sigma,
        "model_config": asdict(cfg),
        "threshold": {
            "policy": calibration["image"]["policy"],  # type: ignore[index]
            "value": calibration["image"]["value"],  # type: ignore[index]
        },
        "calibration": dict(calibration),
        "eval_resolution": eval_resolution,
        "fit_stats": dict(fit_stats),
    }
    return save_artifact(directory, MODEL_TYPE, category, model.tensors(), split, asdict(cfg), info)


def load_classical(
    directory: Path,
    category: str | None = None,
    expected_config: ClassicalConfig | None = None,
) -> tuple[ClassicalModel, LoadedArtifact]:
    """Load the baseline after full integrity checks.

    `expected_config` makes an artifact fitted with other settings an error (stale).
    """
    artifact = load_artifact(
        directory,
        MODEL_TYPE,
        category,
        asdict(expected_config) if expected_config is not None else None,
    )
    manifest = artifact.manifest
    try:
        config = ClassicalConfig(**artifact.config)  # type: ignore[arg-type]
        aspect = manifest["reference_aspect_ratio"]
        n_fit = int(manifest["n_fit_images"])  # type: ignore[call-overload]
        model = ClassicalModel.from_tensors(
            config,
            artifact.tensors,
            n_fit,
            None if aspect is None else float(aspect),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError, ModelError) as exc:
        raise ArtifactError(f"Manifest or tensors are invalid for the baseline: {exc}") from exc
    return model, artifact
