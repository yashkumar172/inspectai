"""Side-by-side comparison figures: one test image, every model (SPEC.md Section 7.5).

Each row shows the original image, the ground-truth mask (a black tile for normal images, which
have no mask), and one heatmap overlay per model. A model's tiles share one color scale
(`vmax`, with 0 as the lower bound) so heat colors are comparable across the images of a figure.
All arrays are RGB uint8; inputs are never modified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from inspectai.errors import ImageError
from inspectai.preprocessing.images import ImageArray
from inspectai.visualization.heatmaps import mask_to_rgb, overlay_heatmap

_TILE_HEIGHT = 256
_HEADER_HEIGHT = 22
_FOOTER_HEIGHT = 22


@dataclass(frozen=True)
class ModelPanel:
    """One model's prediction for one image."""

    label: str
    anomaly_map: NDArray[np.float32]
    score: float
    threshold: float
    vmax: float


def _fit_height(tile: ImageArray, height: int) -> ImageArray:
    scale = height / tile.shape[0]
    width = max(1, round(tile.shape[1] * scale))
    return cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)


def _text(
    canvas: ImageArray, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _tile(body: ImageArray, header: str, footer: str = "", flagged: bool = False) -> ImageArray:
    body = _fit_height(body, _TILE_HEIGHT)
    canvas = np.zeros((_HEADER_HEIGHT + _TILE_HEIGHT + _FOOTER_HEIGHT, body.shape[1], 3), np.uint8)
    canvas[_HEADER_HEIGHT : _HEADER_HEIGHT + _TILE_HEIGHT] = body
    _text(canvas, header, (4, 16), (255, 255, 255))
    if footer:
        color = (255, 90, 90) if flagged else (120, 255, 120)
        _text(canvas, footer, (4, _HEADER_HEIGHT + _TILE_HEIGHT + 16), color)
    return canvas


def comparison_row(
    image: ImageArray,
    mask: NDArray | None,
    panels: Sequence[ModelPanel],
    name: str,
    alpha: float = 0.5,
) -> ImageArray:
    """One figure row for one image. `mask` is None for normal images (shown as black)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"Expected an (H, W, 3) image, got shape {image.shape}")
    if not panels:
        raise ImageError("At least one model panel is required")
    if mask is not None and mask.shape != image.shape[:2]:
        raise ImageError(f"Mask shape {mask.shape} does not match image {image.shape[:2]}")
    truth = np.zeros(image.shape[:2], dtype=np.uint8) if mask is None else mask
    tiles = [
        _tile(image, name),
        _tile(mask_to_rgb(truth), "ground truth" if mask is not None else "ground truth: none"),
    ]
    for panel in panels:
        if panel.anomaly_map.shape != image.shape[:2]:
            raise ImageError(
                f"Anomaly map of '{panel.label}' has shape {panel.anomaly_map.shape}, "
                f"expected {image.shape[:2]}"
            )
        overlay = overlay_heatmap(image, panel.anomaly_map, alpha, 0.0, panel.vmax)
        flagged = panel.score > panel.threshold
        tiles.append(
            _tile(
                overlay,
                panel.label,
                f"score {panel.score:.2f} / thr {panel.threshold:.2f}"
                f"  {'ANOMALY' if flagged else 'normal'}",
                flagged,
            )
        )
    return np.ascontiguousarray(np.hstack(tiles))


def stack_rows(rows: Sequence[ImageArray]) -> ImageArray:
    """Stack figure rows vertically, padding narrower rows with black on the right."""
    if not rows:
        raise ImageError("No rows to stack")
    width = max(row.shape[1] for row in rows)
    padded = [
        np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))) if row.shape[1] < width else row
        for row in rows
    ]
    return np.ascontiguousarray(np.vstack(padded))
