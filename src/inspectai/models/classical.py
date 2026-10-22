"""Model A: classical OpenCV baseline (SPEC.md Section 5.1).

For every pixel position the model stores the mean and standard deviation of the color of the
normal (`fit`) images after resizing. The anomaly map is the per-pixel root-mean-square z-score
of a new image, smoothed with a Gaussian filter. The image score is the maximum of the map.

Limitations: the statistics are per pixel *position*, so the method only works when objects sit
at nearly the same place and orientation in every image, and when lighting is stable. It has no
notion of object structure, so objects that move, rotate, or vary in shape (multiple-instance
objects, food items) produce high scores on normal images. It is a reference point, not a
competitive model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from inspectai.config import RESIZE_MODES
from inspectai.errors import ModelError
from inspectai.models.base import Prediction
from inspectai.preprocessing.images import ImageArray, map_back, resize_to_square


@dataclass(frozen=True)
class ClassicalConfig:
    """Settings of the classical baseline."""

    image_size: int = 256
    resize_mode: str = "letterbox"
    smoothing_sigma: float = 4.0
    std_floor: float = 5.0  # in 0-255 color units; keeps flat regions from exploding the z-score

    def __post_init__(self) -> None:
        if self.image_size < 8:
            raise ModelError(f"image_size must be at least 8, got {self.image_size}")
        if self.resize_mode not in RESIZE_MODES:
            raise ModelError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")
        if self.smoothing_sigma < 0:
            raise ModelError(f"smoothing_sigma must not be negative, got {self.smoothing_sigma}")
        if self.std_floor <= 0:
            raise ModelError(f"std_floor must be positive, got {self.std_floor}")


class ClassicalModel:
    """Per-pixel mean / standard-deviation baseline."""

    def __init__(self, config: ClassicalConfig | None = None) -> None:
        self.config = config or ClassicalConfig()
        self.n_fit_images = 0
        self.reference_aspect_ratio: float | None = None
        self._mean: NDArray[np.float32] | None = None
        self._std: NDArray[np.float32] | None = None

    def fit(self, images: Iterable[ImageArray]) -> None:
        """Compute the per-pixel statistics from normal images (streaming, constant memory)."""
        size = self.config.image_size
        total = np.zeros((size, size, 3), dtype=np.float64)
        total_sq = np.zeros((size, size, 3), dtype=np.float64)
        count = 0
        aspects: list[float] = []
        for image in images:
            aspects.append(image.shape[1] / image.shape[0])
            resized = self._resize(image).astype(np.float64)
            total += resized
            total_sq += resized * resized
            count += 1
        if count < 2:
            raise ModelError(f"Fitting needs at least 2 images, got {count}")
        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 0.0)
        self._mean = mean.astype(np.float32)
        self._std = np.maximum(np.sqrt(variance), self.config.std_floor).astype(np.float32)
        self.n_fit_images = count
        self.reference_aspect_ratio = float(np.median(aspects))

    def predict(self, image: ImageArray) -> Prediction:
        """Score one image; the map has the image's own height and width."""
        if self._mean is None or self._std is None:
            raise ModelError("The model has not been fitted")
        _, info = resize_to_square(image, self.config.image_size, self.config.resize_mode)
        z = (self._resize(image) - self._mean) / self._std
        square = np.sqrt(np.mean(z * z, axis=2)).astype(np.float32)
        if self.config.smoothing_sigma > 0:
            square = cv2.GaussianBlur(
                square, (0, 0), self.config.smoothing_sigma, borderType=cv2.BORDER_REFLECT
            )
        anomaly_map = map_back(square, info)
        return Prediction(score=float(anomaly_map.max()), anomaly_map=anomaly_map)

    def _resize(self, image: ImageArray) -> NDArray[np.float32]:
        resized, _ = resize_to_square(image, self.config.image_size, self.config.resize_mode)
        return resized.astype(np.float32)

    def tensors(self) -> dict[str, NDArray[np.float32]]:
        """The fitted parameters: the per-pixel mean and standard deviation."""
        if self._mean is None or self._std is None:
            raise ModelError("The model has not been fitted")
        return {"mean": self._mean.copy(), "std": self._std.copy()}

    @classmethod
    def from_tensors(
        cls,
        config: ClassicalConfig,
        tensors: dict[str, NDArray],
        n_fit_images: int,
        reference_aspect_ratio: float | None,
    ) -> ClassicalModel:
        """Rebuild a fitted model from stored tensors, checking shapes and values."""
        size = config.image_size
        for name in ("mean", "std"):
            array = tensors.get(name)
            if array is None or tuple(array.shape) != (size, size, 3):
                found = "missing" if array is None else tuple(array.shape)
                raise ModelError(f"Tensor {name!r} has shape {found}, expected {(size, size, 3)}")
        if not (tensors["std"] > 0).all():
            raise ModelError("Tensor 'std' must be positive everywhere")
        model = cls(config)
        model._mean = tensors["mean"].astype(np.float32)
        model._std = tensors["std"].astype(np.float32)
        model.n_fit_images = n_fit_images
        model.reference_aspect_ratio = reference_aspect_ratio
        return model
