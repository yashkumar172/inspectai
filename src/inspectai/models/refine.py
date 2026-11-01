"""Anomaly-map refinement: Gaussian smoothing (default) or an edge-aware guided filter.

Models 'upsample' a coarse feature-grid score map to the model input size and then smooth it. The
default Gaussian filter is isotropic and blurs a small defect's boundary into its surroundings.
The guided filter (He, Sun, Tang, "Guided Image Filtering", TPAMI 2013) smooths the map while
following the edges of a guide image, here the luminance of the preprocessed input, so the map's
boundaries can snap to image edges. It is a local linear model q = a * I + b fitted in each
window, evaluated with box filters, so it costs O(pixels) at any radius.

Both filters are applied at the model input resolution, before the map is mapped back to the
original image. The guided filter has no free test-time knowledge: its radius and regularization
are fixed settings stored in the artifact's configuration.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from inspectai.config import IMAGENET_MEAN, IMAGENET_STD
from inspectai.errors import ModelError

REFINEMENT_FILTERS = ("gaussian", "guided")
DEFAULT_REFINEMENT = "gaussian"
DEFAULT_GUIDED_RADIUS = 8
DEFAULT_GUIDED_EPS = 0.01  # on a guide in [0, 1]; the standard order of magnitude (0.1 squared)

FloatMap = NDArray[np.float32]
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def validate_refinement(kind: str, radius: int, eps: float) -> None:
    """Raise ModelError for an unknown filter or unusable guided-filter settings."""
    if kind not in REFINEMENT_FILTERS:
        raise ModelError(f"refinement_filter must be one of {REFINEMENT_FILTERS}, got {kind!r}")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ModelError(f"guided_radius must be a positive integer, got {radius!r}")
    if not eps > 0:
        raise ModelError(f"guided_eps must be positive, got {eps!r}")


def luminance_guide(batch_image: torch.Tensor) -> FloatMap:
    """Luminance in [0, 1] of one normalized (3, H, W) model input (padding stays black)."""
    if batch_image.ndim != 3 or batch_image.shape[0] != 3:
        raise ModelError(f"Expected a (3, H, W) tensor, got {tuple(batch_image.shape)}")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (batch_image * std + mean).clamp_(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb @ _LUMA, dtype=np.float32)


def guided_filter(guide: FloatMap, source: FloatMap, radius: int, eps: float) -> FloatMap:
    """Guided filter of `source` with a single-channel `guide` of the same shape.

    Per window of side 2 * radius + 1: a = cov(guide, source) / (var(guide) + eps) and
    b = mean(source) - a * mean(guide); the output is mean(a) * guide + mean(b). Where the guide
    is flat (variance well below `eps`) the output is a plain box average of the source; where it
    has an edge (variance well above `eps`) the output follows the guide's edge.
    """
    if guide.shape != source.shape or guide.ndim != 2:
        raise ModelError(
            f"guide {guide.shape} and source {source.shape} must be the same 2-D shape"
        )
    validate_refinement("guided", radius, eps)
    size = (2 * radius + 1, 2 * radius + 1)

    def box(image: FloatMap) -> FloatMap:
        return cv2.boxFilter(image, cv2.CV_32F, size, borderType=cv2.BORDER_REFLECT)

    guide = guide.astype(np.float32, copy=False)
    source = source.astype(np.float32, copy=False)
    mean_g, mean_s = box(guide), box(source)
    covariance = box(guide * source) - mean_g * mean_s
    variance = np.maximum(box(guide * guide) - mean_g * mean_g, 0.0)
    a = covariance / (variance + eps)
    b = mean_s - a * mean_g
    return (box(a) * guide + box(b)).astype(np.float32, copy=False)


def refine_map(
    upsampled: FloatMap,
    guide: FloatMap | None,
    kind: str,
    sigma: float,
    radius: int = DEFAULT_GUIDED_RADIUS,
    eps: float = DEFAULT_GUIDED_EPS,
) -> FloatMap:
    """Smooth an upsampled score map with the chosen filter.

    `gaussian` is the original behavior (a reflect-bordered Gaussian of std `sigma`, skipped when
    `sigma` is 0). `guided` needs the luminance `guide` of the same input.
    """
    validate_refinement(kind, radius, eps)
    if kind == "gaussian":
        if sigma <= 0:
            return upsampled
        return cv2.GaussianBlur(upsampled, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    if guide is None:
        raise ModelError("The guided filter needs a guide image")
    return guided_filter(guide, upsampled, radius, eps)
