"""Heatmap, overlay, and panel generation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from inspectai.errors import ImageError
from inspectai.visualization.heatmaps import (
    anomaly_map_to_heatmap,
    mask_to_rgb,
    overlay_heatmap,
    prediction_panel,
    save_rgb_png,
)


def _image(height: int = 20, width: int = 30) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _ramp(height: int = 20, width: int = 30) -> np.ndarray:
    return np.tile(np.linspace(0, 1, width, dtype=np.float32), (height, 1))


def test_heatmap_has_the_map_size_and_rgb_order() -> None:
    heat = anomaly_map_to_heatmap(_ramp())
    assert heat.shape == (20, 30, 3) and heat.dtype == np.uint8
    low, high = heat[0, 0], heat[0, -1]
    assert low[2] > low[0]  # coldest pixel is blue: blue channel above red in RGB order
    assert high[0] > high[2]  # hottest pixel is red


def test_heatmap_uses_fixed_bounds_when_given() -> None:
    same = np.full((4, 4), 5.0, dtype=np.float32)
    assert np.array_equal(
        anomaly_map_to_heatmap(same, 0.0, 10.0), anomaly_map_to_heatmap(same, 0.0, 10.0)
    )
    assert not np.array_equal(
        anomaly_map_to_heatmap(same, 0.0, 10.0), anomaly_map_to_heatmap(same, 0.0, 5.0)
    )


def test_a_constant_map_does_not_divide_by_zero() -> None:
    assert anomaly_map_to_heatmap(np.ones((4, 4), dtype=np.float32)).shape == (4, 4, 3)


def test_overlay_matches_the_image_size_and_does_not_modify_inputs() -> None:
    image, anomaly_map = _image(), _ramp()
    image_before, map_before = image.copy(), anomaly_map.copy()
    overlay = overlay_heatmap(image, anomaly_map, alpha=0.4)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8
    assert np.array_equal(image, image_before) and np.array_equal(anomaly_map, map_before)


def test_overlay_alpha_extremes() -> None:
    image, anomaly_map = _image(), _ramp()
    assert np.array_equal(overlay_heatmap(image, anomaly_map, alpha=0.0), image)
    assert np.array_equal(
        overlay_heatmap(image, anomaly_map, alpha=1.0), anomaly_map_to_heatmap(anomaly_map)
    )


def test_visualization_rejects_invalid_inputs() -> None:
    with pytest.raises(ImageError):
        overlay_heatmap(_image(), np.zeros((5, 5), dtype=np.float32))  # shape mismatch
    with pytest.raises(ImageError):
        overlay_heatmap(_image(), _ramp(), alpha=1.5)
    with pytest.raises(ImageError):
        anomaly_map_to_heatmap(np.full((4, 4), np.nan, dtype=np.float32))
    with pytest.raises(ImageError):
        anomaly_map_to_heatmap(np.zeros((4, 4, 3)))
    with pytest.raises(ImageError):
        anomaly_map_to_heatmap(_ramp(), vmin=2.0, vmax=1.0)
    with pytest.raises(ImageError):
        overlay_heatmap(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.float32))


def test_prediction_panel_width_depends_on_the_mask() -> None:
    image, anomaly_map = _image(), _ramp()
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    assert prediction_panel(image, anomaly_map).shape == (20, 3 * 30, 3)
    assert prediction_panel(image, anomaly_map, mask).shape == (20, 4 * 30, 3)
    with pytest.raises(ImageError):
        prediction_panel(image, anomaly_map, np.zeros((3, 3), dtype=np.uint8))


def test_mask_to_rgb_is_white_on_black() -> None:
    rgb = mask_to_rgb(np.array([[0, 1]], dtype=np.uint8))
    assert rgb.tolist() == [[[0, 0, 0], [255, 255, 255]]]


def test_save_rgb_png_round_trips_through_a_non_ascii_path(tmp_path: Path) -> None:
    image = _image()
    target = tmp_path / "dossier é 漢字" / "out ü.png"
    save_rgb_png(target, image)
    decoded = cv2.imdecode(np.frombuffer(target.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded[:, :, ::-1], image)
    with pytest.raises(ImageError):
        save_rgb_png(target, np.zeros((4, 4), dtype=np.uint8))
