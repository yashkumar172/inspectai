"""Heatmaps, overlays, and prediction panels (SPEC.md Sections 6 and 7.5).

All images are RGB uint8 arrays. OpenCV's colormaps produce BGR, so the result is converted
back to RGB here; input arrays are never modified.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from inspectai.errors import ImageError
from inspectai.preprocessing.images import ImageArray, ensure_three_channels


def _check_pair(image: ImageArray, anomaly_map: NDArray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"Expected an (H, W, 3) image, got shape {image.shape}")
    if anomaly_map.ndim != 2 or anomaly_map.shape != image.shape[:2]:
        raise ImageError(
            f"Anomaly map shape {anomaly_map.shape} does not match image shape {image.shape[:2]}"
        )
    if not np.isfinite(anomaly_map).all():
        raise ImageError("Anomaly map contains NaN or infinity")


def anomaly_map_to_heatmap(
    anomaly_map: NDArray, vmin: float | None = None, vmax: float | None = None
) -> ImageArray:
    """Color an anomaly map (JET, blue = low, red = high) as an RGB image.

    Values are scaled linearly from `vmin` to `vmax` (default: the map's own range) and clipped.
    Pass fixed bounds to compare several maps on one scale.
    """
    if anomaly_map.ndim != 2 or anomaly_map.size == 0:
        raise ImageError(f"Anomaly map must be a non-empty 2-D array, got {anomaly_map.shape}")
    if not np.isfinite(anomaly_map).all():
        raise ImageError("Anomaly map contains NaN or infinity")
    low = float(anomaly_map.min()) if vmin is None else vmin
    high = float(anomaly_map.max()) if vmax is None else vmax
    if high < low:
        raise ImageError(f"vmax ({high}) must not be below vmin ({low})")
    span = high - low
    if span <= 0:
        scaled = np.zeros(anomaly_map.shape, dtype=np.float32)
    else:
        scaled = np.clip((anomaly_map.astype(np.float32) - low) / span, 0.0, 1.0)
    colored = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cv2.COLORMAP_JET)
    return np.ascontiguousarray(colored[:, :, ::-1])


def overlay_heatmap(
    image: ImageArray,
    anomaly_map: NDArray,
    alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
) -> ImageArray:
    """Blend the colored anomaly map onto the image (same size as the image)."""
    if not 0.0 <= alpha <= 1.0:
        raise ImageError(f"alpha must be between 0 and 1, got {alpha}")
    _check_pair(image, anomaly_map)
    heat = anomaly_map_to_heatmap(anomaly_map, vmin, vmax)
    blended = cv2.addWeighted(image, 1.0 - alpha, heat, alpha, 0.0)
    return np.ascontiguousarray(blended)


def mask_to_rgb(mask: NDArray) -> ImageArray:
    """A binary mask (0/1) as a white-on-black RGB image."""
    if mask.ndim != 2:
        raise ImageError(f"Mask must be 2-D, got shape {mask.shape}")
    return ensure_three_channels((mask != 0).astype(np.uint8) * 255)


def _label(panel: ImageArray, text: str) -> ImageArray:
    out = panel.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def prediction_panel(
    image: ImageArray,
    anomaly_map: NDArray,
    mask: NDArray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.5,
) -> ImageArray:
    """Side-by-side panel: original, ground-truth mask (if any), heatmap, overlay."""
    _check_pair(image, anomaly_map)
    parts = [_label(image, "image")]
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ImageError(f"Mask shape {mask.shape} does not match image {image.shape[:2]}")
        parts.append(_label(mask_to_rgb(mask), "ground truth"))
    parts.append(_label(anomaly_map_to_heatmap(anomaly_map, vmin, vmax), "heatmap"))
    parts.append(_label(overlay_heatmap(image, anomaly_map, alpha, vmin, vmax), "overlay"))
    return np.ascontiguousarray(np.hstack(parts))


def save_rgb_png(path: Path, image: ImageArray) -> None:
    """Write an RGB image as PNG (through bytes, so any Windows path works)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"Expected an (H, W, 3) image, got shape {image.shape}")
    ok, encoded = cv2.imencode(".png", image[:, :, ::-1])
    if not ok:
        raise ImageError(f"Image could not be encoded as PNG: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
