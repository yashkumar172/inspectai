"""Pretrained CNN backbone and spatial feature extraction (SPEC.md Sections 5.2 and 5.5).

The backbone is frozen and used only for inference. Weights come from the torchvision cache
(`TORCH_HOME`); they are never downloaded implicitly and never replaced by random weights.
`FeatureExtractor` accepts any `torch.nn.Module`, so tests inject a small fake backbone.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from numpy.typing import NDArray
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from inspectai.config import RESIZE_MODES
from inspectai.errors import ConfigError, ImageError, ModelError, WeightsNotFoundError
from inspectai.preprocessing.images import (
    ImageArray,
    ResizeInfo,
    normalize_imagenet,
    resize_to_square,
)

ENV_ALLOW_DOWNLOAD = "INSPECTAI_ALLOW_WEIGHT_DOWNLOAD"

# Supported backbones: torchvision constructor name -> (weights enum name, member name).
SUPPORTED_BACKBONES: dict[str, tuple[str, str]] = {
    "resnet18": ("ResNet18_Weights", "IMAGENET1K_V1"),
    "wide_resnet50_2": ("Wide_ResNet50_2_Weights", "IMAGENET1K_V1"),
}


@dataclass(frozen=True)
class WeightsInfo:
    """Which pretrained weights a backbone uses, for the artifact manifest."""

    backbone: str
    identifier: str
    sha256: str | None  # None for injected (test) backbones

    def to_dict(self) -> dict[str, object]:
        return {"backbone": self.backbone, "identifier": self.identifier, "sha256": self.sha256}


def _weights_enum(backbone: str):  # type: ignore[no-untyped-def]
    if backbone not in SUPPORTED_BACKBONES:
        raise ConfigError(
            f"Unsupported backbone {backbone!r}; supported: {', '.join(SUPPORTED_BACKBONES)}"
        )
    enum_name, member = SUPPORTED_BACKBONES[backbone]
    return getattr(torchvision.models, enum_name)[member]


def weights_cache_path(backbone: str, hub_dir: Path | None = None) -> Path:
    """Where torchvision caches the weights file of `backbone` (under `hub_dir` if given)."""
    weights = _weights_enum(backbone)
    filename = Path(weights.url).name
    base = Path(torch.hub.get_dir()) if hub_dir is None else Path(hub_dir)
    return base / "checkpoints" / filename


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(backbone: str) -> Path:
    """Download the weights into the torchvision cache (an explicit action) and return the path.

    The download is verified against the hash prefix in the file name.
    """
    weights = _weights_enum(backbone)
    weights.get_state_dict(progress=True, check_hash=True)
    return weights_cache_path(backbone)


def load_backbone(
    backbone: str, allow_download: bool | None = None, hub_dir: Path | None = None
) -> tuple[nn.Module, WeightsInfo]:
    """Load a frozen, pretrained torchvision backbone from the local cache.

    Raises WeightsNotFoundError if the weights are not cached (unless downloading is allowed
    through `allow_download=True` or `INSPECTAI_ALLOW_WEIGHT_DOWNLOAD=1`) or the cached file
    fails its hash check. There is no fallback to random weights. `hub_dir` reads the cache from
    another torchvision hub folder (one holding `checkpoints/`) instead of `TORCH_HOME`.
    """
    weights = _weights_enum(backbone)
    path = weights_cache_path(backbone, hub_dir)
    if allow_download is None:
        allow_download = os.environ.get(ENV_ALLOW_DOWNLOAD) == "1"
    if not path.is_file():
        if not allow_download:
            raise WeightsNotFoundError(
                f"Pretrained weights for {backbone} are not cached at {path}. Fetch them once "
                f"with: python scripts/fetch_weights.py --backbone {backbone}  "
                f"(or set {ENV_ALLOW_DOWNLOAD}=1 to allow a download at load time)."
            )
        fetch_weights(backbone)
    sha256 = _sha256_file(path)
    match = re.search(r"-([0-9a-f]{8})\.pth$", path.name)
    if match and not sha256.startswith(match.group(1)):
        raise WeightsNotFoundError(
            f"The cached weights file {path} is corrupted (hash mismatch). Delete it and fetch "
            f"it again with: python scripts/fetch_weights.py --backbone {backbone}"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = getattr(torchvision.models, backbone)(weights=None)
    model.load_state_dict(state)
    return freeze(model), WeightsInfo(backbone, weights.name, sha256)


def freeze(model: nn.Module) -> nn.Module:
    """Put a model in eval mode and disable gradients for all of its parameters."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class FeatureExtractor:
    """Intermediate spatial features of a frozen backbone.

    `embed` returns the requested layers resized to the resolution of the first layer and
    concatenated along channels, shape (batch, total_channels, h, w).
    """

    def __init__(self, backbone: nn.Module, layers: Sequence[str]) -> None:
        self.layers = tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ModelError(f"layers must be a non-empty list of distinct names, got {layers!r}")
        freeze(backbone)
        try:
            self._net = create_feature_extractor(backbone, {name: name for name in self.layers})
        except (ValueError, RuntimeError, AttributeError) as exc:
            raise ModelError(
                f"Cannot extract layers {self.layers} from the backbone: {exc}"
            ) from exc
        self._net.eval()

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Features of a (batch, 3, H, W) tensor, upsampled to the first layer and concatenated."""
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ModelError(f"Expected a (batch, 3, H, W) tensor, got {tuple(batch.shape)}")
        outputs = self._net(batch)
        maps = [outputs[name] for name in self.layers]
        size = maps[0].shape[-2:]
        resized = [
            m
            if m.shape[-2:] == size
            else F.interpolate(m, size=size, mode="bilinear", align_corners=False)
            for m in maps
        ]
        return torch.cat(resized, dim=1)


def images_to_batch(
    images: Iterable[ImageArray], size: int, mode: str
) -> tuple[torch.Tensor, list[ResizeInfo]]:
    """Resize, normalize with ImageNet statistics, and stack RGB images into a float tensor."""
    if mode not in RESIZE_MODES:
        raise ImageError(f"Unknown resize mode {mode!r}; expected one of {RESIZE_MODES}")
    arrays: list[NDArray[np.float32]] = []
    infos: list[ResizeInfo] = []
    for image in images:
        resized, info = resize_to_square(image, size, mode)
        arrays.append(normalize_imagenet(resized))
        infos.append(info)
    if not arrays:
        raise ImageError("No images to convert")
    return torch.from_numpy(np.stack(arrays)), infos
