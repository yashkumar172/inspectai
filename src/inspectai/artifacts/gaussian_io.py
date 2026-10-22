"""Saving and loading fitted Gaussian models through the artifact store (SPEC.md Section 9.1).

Loading needs only the artifact and the backbone weights; the data and the fitting procedure are
not required (SPEC.md Section 11).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch import nn

from inspectai.artifacts.store import LoadedArtifact, load_artifact, save_artifact
from inspectai.data.splits import SplitResult
from inspectai.errors import ArtifactError, ModelError
from inspectai.models.features import FeatureExtractor, WeightsInfo, load_backbone
from inspectai.models.gaussian import GaussianConfig, GaussianModel

MODEL_TYPE = "gaussian"


def save_gaussian(
    directory: Path,
    model: GaussianModel,
    category: str,
    split: SplitResult,
    calibration: Mapping[str, object],
    fit_stats: Mapping[str, object],
    eval_resolution: int,
) -> Path:
    """Save a fitted model together with its thresholds and measured fitting cost."""
    cfg = model.config
    if model.feature_map_shape is None or model.total_channels is None:
        raise ModelError("The model has not been fitted")
    info: dict[str, object] = {
        "backbone": cfg.backbone,
        "weights": model.weights.to_dict(),
        "layers": list(cfg.layers),
        "image_size": cfg.image_size,
        "feature_dim": cfg.d,
        "epsilon": cfg.epsilon,
        "smoothing_sigma": cfg.smoothing_sigma,
        "model_config": cfg.to_dict(),
        "parameter_bytes": int(sum(a.nbytes for a in model.tensors().values())),
        "total_channels": model.total_channels,
        "feature_map_shape": list(model.feature_map_shape),
        "resize_mode": cfg.resize_mode,
        "reference_aspect_ratio": model.reference_aspect_ratio,
        "seed": cfg.seed,
        "n_fit_images": model.n_fit_images,
        "threshold": {
            "policy": calibration["image"]["policy"],  # type: ignore[index]
            "value": calibration["image"]["value"],  # type: ignore[index]
        },
        "calibration": dict(calibration),
        "eval_resolution": eval_resolution,
        "fit_stats": dict(fit_stats),
    }
    return save_artifact(
        directory, MODEL_TYPE, category, model.tensors(), split, cfg.to_dict(), info
    )


def load_gaussian(
    directory: Path,
    category: str | None = None,
    expected_config: GaussianConfig | None = None,
    backbone: nn.Module | None = None,
) -> tuple[GaussianModel, LoadedArtifact]:
    """Load a Gaussian model after full integrity checks.

    `expected_config` makes a stale artifact (fitted with different settings) an error.
    `backbone` injects a backbone (tests); otherwise the pretrained weights named in the
    manifest are loaded from the cache and their SHA-256 must match the manifest.
    """
    artifact = load_artifact(
        directory,
        MODEL_TYPE,
        category,
        expected_config.to_dict() if expected_config is not None else None,
    )
    manifest = artifact.manifest
    try:
        config = GaussianConfig.from_dict(artifact.config)
        weights_record = manifest["weights"]
        shape = manifest["feature_map_shape"]
        feature_map_shape = (int(shape[0]), int(shape[1]))  # type: ignore[index]
        weights = WeightsInfo(
            str(weights_record["backbone"]),  # type: ignore[index]
            str(weights_record["identifier"]),  # type: ignore[index]
            weights_record["sha256"],  # type: ignore[index]
        )
        total_channels = int(manifest["total_channels"])  # type: ignore[call-overload]
        aspect = manifest["reference_aspect_ratio"]
        n_fit = int(manifest["n_fit_images"])  # type: ignore[call-overload]
    except (KeyError, TypeError, ValueError, ModelError) as exc:
        raise ArtifactError(f"Manifest is missing or has invalid model fields: {exc}") from exc
    if manifest.get("backbone") != config.backbone or manifest.get("layers") != list(config.layers):
        raise ArtifactError("Manifest and config snapshot disagree about backbone or layers")

    if backbone is None:
        backbone, loaded_weights = load_backbone(config.backbone)
        if (
            loaded_weights.sha256 != weights.sha256
            or loaded_weights.identifier != weights.identifier
        ):
            raise ArtifactError(
                "The cached pretrained weights differ from the ones the model was fitted with"
            )
    try:
        model = GaussianModel.from_tensors(
            config,
            artifact.tensors,
            FeatureExtractor(backbone, config.layers),
            weights,
            feature_map_shape,
            total_channels,
            None if aspect is None else float(aspect),  # type: ignore[arg-type]
            n_fit,
        )
    except ModelError as exc:
        raise ArtifactError(f"Stored tensors do not fit the model: {exc}") from exc
    return model, artifact
