"""Saving and loading fitted PatchCore-inspired models through the artifact store.

Loading needs only the artifact and the backbone weights; the data and the fitting procedure are
not required (SPEC.md Section 11).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import nn

from inspectai.artifacts.store import RUNTIME_KEY, LoadedArtifact, load_artifact, save_artifact
from inspectai.data.splits import SplitResult
from inspectai.errors import ArtifactError, ModelError
from inspectai.models.features import FeatureExtractor, WeightsInfo, load_backbone
from inspectai.models.patchcore import PatchCoreConfig, PatchCoreModel

MODEL_TYPE = "patchcore"


def save_patchcore(
    directory: Path,
    model: PatchCoreModel,
    category: str,
    split: SplitResult,
    calibration: Mapping[str, object],
    fit_stats: Mapping[str, object],
    eval_resolution: int,
) -> Path:
    """Save a fitted model with its thresholds, coreset statistics, and measured fitting cost."""
    cfg = model.config
    if model.feature_map_shape is None or model.feature_dim is None:
        raise ModelError("The model has not been fitted")
    tensors = model.tensors()
    info: dict[str, object] = {
        "backbone": cfg.backbone,
        "weights": model.weights.to_dict(),
        "layers": list(cfg.layers),
        "image_size": cfg.image_size,
        "feature_dim": model.feature_dim,
        "feature_map_shape": list(model.feature_map_shape),
        "resize_mode": cfg.resize_mode,
        "reference_aspect_ratio": model.reference_aspect_ratio,
        "seed": cfg.seed,
        "n_fit_images": model.n_fit_images,
        "neighborhood": cfg.neighborhood,
        "smoothing_sigma": cfg.smoothing_sigma,
        "model_config": cfg.to_dict(),
        "runtime": cfg.runtime_dict(),
        "memory_bank": {
            "n_candidates": model.n_candidates,
            "n_vectors": model.bank_size,
            "coreset_ratio": cfg.coreset_ratio,
            "projection_dim": cfg.projection_dim,
            "estimated_unreduced_gb": round(model.estimated_bank_gb, 4),
            "bank_bytes": int(sum(a.nbytes for a in tensors.values())),
            "coreset_seconds": round(model.coreset_seconds, 3),
        },
        "threshold": {
            "policy": calibration["image"]["policy"],  # type: ignore[index]
            "value": calibration["image"]["value"],  # type: ignore[index]
        },
        "calibration": dict(calibration),
        "eval_resolution": eval_resolution,
        "fit_stats": dict(fit_stats),
    }
    return save_artifact(directory, MODEL_TYPE, category, tensors, split, cfg.snapshot(), info)


def load_patchcore(
    directory: Path,
    category: str | None = None,
    expected_config: PatchCoreConfig | None = None,
    backbone: nn.Module | None = None,
    runtime: Mapping[str, object] | None = None,
) -> tuple[PatchCoreModel, LoadedArtifact]:
    """Load a PatchCore-inspired model after full integrity checks.

    `expected_config` makes a stale artifact (fitted with different settings) an error.
    `backbone` injects a backbone (tests); otherwise the pretrained weights named in the
    manifest are loaded from the cache and their SHA-256 must match the manifest. `runtime`
    overrides resource settings such as `chunk_size` and `num_threads`.
    """
    artifact = load_artifact(
        directory,
        MODEL_TYPE,
        category,
        expected_config.to_dict() if expected_config is not None else None,
    )
    manifest = artifact.manifest
    try:
        snapshot = dict(artifact.config)
        stored_runtime = snapshot.pop(RUNTIME_KEY, {})
        if not isinstance(stored_runtime, dict):
            raise ArtifactError("The configuration snapshot has an invalid runtime section")
        config = PatchCoreConfig.from_dict(snapshot, {**stored_runtime, **(runtime or {})})
        weights_record = manifest["weights"]
        shape = manifest["feature_map_shape"]
        feature_map_shape = (int(shape[0]), int(shape[1]))  # type: ignore[index]
        weights = WeightsInfo(
            str(weights_record["backbone"]),  # type: ignore[index]
            str(weights_record["identifier"]),  # type: ignore[index]
            weights_record["sha256"],  # type: ignore[index]
        )
        aspect = manifest["reference_aspect_ratio"]
        n_fit = int(manifest["n_fit_images"])  # type: ignore[call-overload]
        n_candidates = int(manifest["memory_bank"]["n_candidates"])  # type: ignore[index,call-overload]
    except (KeyError, TypeError, ValueError, ModelError) as exc:
        raise ArtifactError(f"Manifest is missing or has invalid model fields: {exc}") from exc
    if manifest.get("backbone") != config.backbone or manifest.get("layers") != list(config.layers):
        raise ArtifactError("Manifest and config snapshot disagree about backbone or layers")

    if backbone is None:
        backbone, loaded = load_backbone(config.backbone)
        if loaded.sha256 != weights.sha256 or loaded.identifier != weights.identifier:
            raise ArtifactError(
                "The cached pretrained weights differ from the ones the model was fitted with"
            )
    try:
        model = PatchCoreModel.from_tensors(
            config,
            artifact.tensors,
            FeatureExtractor(backbone, config.layers),
            weights,
            feature_map_shape,
            None if aspect is None else float(aspect),  # type: ignore[arg-type]
            n_fit,
            n_candidates,
        )
    except ModelError as exc:
        raise ArtifactError(f"Stored tensors do not fit the model: {exc}") from exc
    if model.feature_dim != manifest.get("feature_dim"):
        raise ArtifactError("Stored bank dimension does not match the manifest")
    return model, artifact
