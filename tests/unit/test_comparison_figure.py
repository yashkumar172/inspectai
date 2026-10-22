"""Comparison figures: one test image, every model."""

import numpy as np
import pytest

from inspectai.errors import ImageError
from inspectai.visualization.comparison import ModelPanel, comparison_row, stack_rows


def _image(height: int = 40, width: int = 60) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _panel(
    label: str = "B", score: float = 2.0, threshold: float = 1.0, shape=(40, 60)
) -> ModelPanel:
    anomaly_map = np.random.default_rng(1).random(shape).astype(np.float32) * 3
    return ModelPanel(label, anomaly_map, score, threshold, vmax=3.0)


def test_a_row_has_the_image_ground_truth_and_one_tile_per_model() -> None:
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 10:30] = 1
    row = comparison_row(_image(), mask, [_panel("A"), _panel("B")], "000.JPG (anomaly)")
    tile_height = 22 + 256 + 22
    scaled_width = round(60 * 256 / 40)  # tiles are scaled to a common height
    assert row.shape == (tile_height, 4 * scaled_width, 3) and row.dtype == np.uint8


def test_a_normal_image_gets_a_black_ground_truth_tile() -> None:
    row = comparison_row(_image(), None, [_panel()], "001.JPG (normal)")
    scaled_width = round(60 * 256 / 40)
    truth = row[22 : 22 + 256, scaled_width : 2 * scaled_width]
    assert truth[40:].max() == 0  # below the label text, the tile is all black


def test_the_decision_color_follows_the_threshold() -> None:
    above = comparison_row(_image(), None, [_panel(score=2.0, threshold=1.0)], "x")
    below = comparison_row(_image(), None, [_panel(score=0.5, threshold=1.0)], "x")
    assert not np.array_equal(above, below)  # ANOMALY (red text) versus normal (green text)


def test_inputs_are_not_modified() -> None:
    image, panel = _image(), _panel()
    image_before, map_before = image.copy(), panel.anomaly_map.copy()
    comparison_row(image, None, [panel], "x")
    assert np.array_equal(image, image_before) and np.array_equal(panel.anomaly_map, map_before)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ImageError, match="At least one"):
        comparison_row(_image(), None, [], "x")
    with pytest.raises(ImageError, match="does not match"):
        comparison_row(_image(), np.zeros((3, 3), dtype=np.uint8), [_panel()], "x")
    with pytest.raises(ImageError, match="Anomaly map"):
        comparison_row(_image(), None, [_panel(shape=(10, 10))], "x")
    with pytest.raises(ImageError, match="Expected an"):
        comparison_row(np.zeros((4, 4), dtype=np.uint8), None, [_panel()], "x")


def test_stacking_pads_narrower_rows_and_rejects_nothing() -> None:
    narrow = np.full((10, 20, 3), 7, dtype=np.uint8)
    wide = np.full((10, 30, 3), 9, dtype=np.uint8)
    stacked = stack_rows([narrow, wide])
    assert stacked.shape == (20, 30, 3)
    assert stacked[0, 25].tolist() == [0, 0, 0] and stacked[10, 25].tolist() == [9, 9, 9]
    with pytest.raises(ImageError):
        stack_rows([])
