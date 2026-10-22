"""Model B: Gaussian feature model, PaDiM-inspired (SPEC.md Section 5.2).

For every spatial position of the intermediate feature map, a multivariate Gaussian (mean and
regularized covariance) is fitted over the `fit` images, using a fixed random subset of `d`
feature channels. The anomaly score at a position is the Mahalanobis distance. The anomaly map
is the per-position score map, upsampled to the input size, smoothed with a Gaussian filter, and
mapped back to the original image size. The image score is the maximum of the map.

Deviations from the PaDiM paper (Defard et al., 2020):
- Layers are upsampled to the first layer's resolution with bilinear interpolation (the paper
  uses nearest-neighbor style embedding concatenation).
- Images are resized with the category's resize mode (`letterbox` or `stretch`) instead of the
  paper's resize-and-center-crop, so border defects are not cropped away.
- The backbone and layers come from the active profile (`resnet18` for `dev`,
  `wide_resnet50_2` for `full`) with d = 100 channels in both, not the paper's per-backbone d.
- Only the running mean and covariance are stored, and the Gaussian maps use float32 inverses
  of float64 covariances.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from inspectai.config import RESIZE_MODES, profile_settings, resize_mode_for
from inspectai.errors import ModelError
from inspectai.models.base import Prediction
from inspectai.models.features import FeatureExtractor, WeightsInfo, images_to_batch
from inspectai.preprocessing.images import ImageArray, map_back

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaussianConfig:
    """Settings of Model B. All values are recorded in the artifact."""

    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 224
    resize_mode: str = "letterbox"
    d: int = 100
    epsilon: float = 0.01
    smoothing_sigma: float = 4.0
    seed: int = 42
    batch_size: int = 8
    num_threads: int = 6

    def __post_init__(self) -> None:
        if self.image_size < 8:
            raise ModelError(f"image_size must be at least 8, got {self.image_size}")
        if self.resize_mode not in RESIZE_MODES:
            raise ModelError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")
        if self.d < 1:
            raise ModelError(f"d must be at least 1, got {self.d}")
        if self.epsilon <= 0:
            raise ModelError(f"epsilon must be positive, got {self.epsilon}")
        if self.smoothing_sigma < 0:
            raise ModelError(f"smoothing_sigma must not be negative, got {self.smoothing_sigma}")
        if self.batch_size < 1 or self.num_threads < 1:
            raise ModelError("batch_size and num_threads must be at least 1")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["layers"] = list(self.layers)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GaussianConfig:
        values = dict(data)
        values["layers"] = tuple(values.get("layers", ()))  # type: ignore[arg-type]
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ModelError(f"Invalid Gaussian model configuration: {exc}") from exc


def gaussian_config_for(
    category: str, profile: str = "dev", d: int = 100, seed: int = 42
) -> GaussianConfig:
    """The configuration of a category under a profile (`dev` or `full`).

    Fit and evaluation both build their configuration here, so an artifact fitted with other
    settings is recognized as stale.
    """
    settings = profile_settings(profile)
    return GaussianConfig(
        backbone=str(settings["backbone"]),
        layers=tuple(settings["layers"]),  # type: ignore[arg-type]
        image_size=int(settings["image_size"]),  # type: ignore[call-overload]
        resize_mode=resize_mode_for(category),
        d=d,
        seed=seed,
    )


def _batches(images: Iterable[ImageArray], size: int) -> Iterator[list[ImageArray]]:
    batch: list[ImageArray] = []
    for image in images:
        batch.append(image)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class GaussianModel:
    """Per-position Gaussian over frozen CNN features, scored by Mahalanobis distance."""

    def __init__(
        self, config: GaussianConfig, extractor: FeatureExtractor, weights: WeightsInfo
    ) -> None:
        self.config = config
        self.weights = weights
        self._extractor = extractor
        self.n_fit_images = 0
        self.reference_aspect_ratio: float | None = None
        self.feature_map_shape: tuple[int, int] | None = None
        self.total_channels: int | None = None
        self._channel_index: torch.Tensor | None = None
        self._mean: torch.Tensor | None = None
        self._inv_cov: torch.Tensor | None = None

    # ----- fitting -------------------------------------------------------------------------

    @torch.inference_mode()
    def fit(self, images: Iterable[ImageArray]) -> None:
        """Fit the per-position Gaussians on normal images (streaming, constant memory)."""
        cfg = self.config
        torch.set_num_threads(cfg.num_threads)
        total: torch.Tensor | None = None
        outer: torch.Tensor | None = None
        count = 0
        aspects: list[float] = []

        def counted() -> Iterator[ImageArray]:
            for image in images:
                aspects.append(image.shape[1] / image.shape[0])
                yield image

        for batch_images in _batches(counted(), cfg.batch_size):
            batch, _ = images_to_batch(batch_images, cfg.image_size, cfg.resize_mode)
            embedding = self._extractor.embed(batch)
            _, channels, height, width = embedding.shape
            if total is None:
                if cfg.d > channels:
                    raise ModelError(f"d={cfg.d} exceeds the {channels} available feature channels")
                rng = np.random.default_rng(cfg.seed)
                index = np.sort(rng.choice(channels, size=cfg.d, replace=False))
                self._channel_index = torch.from_numpy(index.astype(np.int64))
                self.feature_map_shape = (height, width)
                self.total_channels = channels
                total = torch.zeros(height * width, cfg.d, dtype=torch.float64)
                outer = torch.zeros(height * width, cfg.d, cfg.d, dtype=torch.float64)
                logger.info(
                    "Feature map %dx%d, %d channels -> d=%d; covariance memory about %.0f MB",
                    height,
                    width,
                    channels,
                    cfg.d,
                    outer.numel() * 8 / 1e6,
                )
            assert self._channel_index is not None and outer is not None
            x = self._positions(embedding).to(torch.float64)  # (B, P, d)
            total += x.sum(dim=0)
            outer += torch.einsum("bpi,bpj->pij", x, x)
            count += x.shape[0]
        if count < 2 or total is None or outer is None:
            raise ModelError(f"Fitting needs at least 2 images, got {count}")

        mean = total / count
        covariance = (outer - count * torch.einsum("pi,pj->pij", mean, mean)) / (count - 1)
        covariance += cfg.epsilon * torch.eye(cfg.d, dtype=torch.float64)
        self._mean = mean.to(torch.float32).contiguous()
        # contiguous: inv() returns transposed strides, which would change summation order
        self._inv_cov = torch.linalg.inv(covariance).to(torch.float32).contiguous()
        self.n_fit_images = count
        self.reference_aspect_ratio = float(statistics.median(aspects))
        if not torch.isfinite(self._inv_cov).all():
            raise ModelError("The fitted covariance could not be inverted (non-finite values)")

    def _positions(self, embedding: torch.Tensor) -> torch.Tensor:
        """Select the channel subset and reshape (B, C, h, w) to (B, h*w, d)."""
        assert self._channel_index is not None
        chosen = embedding[:, self._channel_index]
        batch, channels, height, width = chosen.shape
        return chosen.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

    # ----- inference -----------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, image: ImageArray) -> Prediction:
        """Score one image; the map has the image's own height and width."""
        if self._mean is None or self._inv_cov is None or self.feature_map_shape is None:
            raise ModelError("The model has not been fitted")
        cfg = self.config
        torch.set_num_threads(cfg.num_threads)
        batch, infos = images_to_batch([image], cfg.image_size, cfg.resize_mode)
        x = self._positions(self._extractor.embed(batch))[0]  # (P, d)
        delta = x - self._mean
        squared = torch.einsum("pi,pij,pj->p", delta, self._inv_cov, delta)
        distance = torch.sqrt(torch.clamp(squared, min=0.0))
        height, width = self.feature_map_shape
        grid = distance.reshape(1, 1, height, width)
        upsampled = F.interpolate(
            grid, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        if cfg.smoothing_sigma > 0:
            upsampled = cv2.GaussianBlur(
                upsampled, (0, 0), cfg.smoothing_sigma, borderType=cv2.BORDER_REFLECT
            )
        anomaly_map = map_back(upsampled.astype(np.float32), infos[0])
        return Prediction(score=float(anomaly_map.max()), anomaly_map=anomaly_map)

    # ----- persistence ---------------------------------------------------------------------

    def tensors(self) -> dict[str, NDArray]:
        """The fitted parameters: means, inverse covariances, and selected channel indices."""
        if self._mean is None or self._inv_cov is None or self._channel_index is None:
            raise ModelError("The model has not been fitted")
        return {
            "mean": self._mean.numpy().copy(),
            "inv_cov": self._inv_cov.numpy().copy(),
            "channel_index": self._channel_index.numpy().copy(),
        }

    @classmethod
    def from_tensors(
        cls,
        config: GaussianConfig,
        tensors: dict[str, NDArray],
        extractor: FeatureExtractor,
        weights: WeightsInfo,
        feature_map_shape: tuple[int, int],
        total_channels: int,
        reference_aspect_ratio: float | None,
        n_fit_images: int,
    ) -> GaussianModel:
        """Rebuild a fitted model from stored tensors, checking that their shapes agree."""
        model = cls(config, extractor, weights)
        positions = feature_map_shape[0] * feature_map_shape[1]
        expected = {
            "mean": (positions, config.d),
            "inv_cov": (positions, config.d, config.d),
            "channel_index": (config.d,),
        }
        for name, shape in expected.items():
            if name not in tensors or tuple(tensors[name].shape) != shape:
                found = tuple(tensors[name].shape) if name in tensors else "missing"
                raise ModelError(f"Tensor {name!r} has shape {found}, expected {shape}")
        index = tensors["channel_index"]
        if index.min() < 0 or index.max() >= total_channels:
            raise ModelError("channel_index contains out-of-range channels")
        model._mean = torch.from_numpy(tensors["mean"].astype(np.float32))
        model._inv_cov = torch.from_numpy(tensors["inv_cov"].astype(np.float32))
        model._channel_index = torch.from_numpy(index.astype(np.int64))
        model.feature_map_shape = feature_map_shape
        model.total_channels = total_channels
        model.reference_aspect_ratio = reference_aspect_ratio
        model.n_fit_images = n_fit_images
        return model
