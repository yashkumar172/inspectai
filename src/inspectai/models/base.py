"""The interface every anomaly model implements (SPEC.md Section 2)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from inspectai.errors import ModelError
from inspectai.preprocessing.images import ImageArray


@dataclass(frozen=True)
class Prediction:
    """An image-level anomaly score and a pixel-level anomaly map.

    `anomaly_map` has the height and width of the input image. The score is not a probability
    (SPEC.md Section 2).
    """

    score: float
    anomaly_map: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.anomaly_map.ndim != 2 or self.anomaly_map.size == 0:
            raise ModelError(
                f"Anomaly map must be a non-empty 2-D array, got {self.anomaly_map.shape}"
            )
        if not np.isfinite(self.score) or not np.isfinite(self.anomaly_map).all():
            raise ModelError("Anomaly score and map must be finite")


class AnomalyModel(Protocol):
    """Fit on normal images only, then score unseen images."""

    def fit(self, images: Iterable[ImageArray]) -> None: ...

    def predict(self, image: ImageArray) -> Prediction: ...
