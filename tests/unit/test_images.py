"""Image loading, color conversion, resizing, and mask reading."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from inspectai.errors import ImageError
from inspectai.preprocessing.images import (
    bgr_to_rgb,
    check_aspect_ratio,
    decode_image,
    ensure_three_channels,
    map_back,
    normalize_imagenet,
    read_image,
    read_mask,
    resize_to_square,
)


def _png_bytes(array: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


def _rgb_gradient(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_bgr_to_rgb_swaps_channels_and_is_its_own_inverse() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # blue
    bgr[..., 2] = 200  # red
    rgb = bgr_to_rgb(bgr)
    assert rgb[0, 0].tolist() == [200, 0, 10]
    assert np.array_equal(bgr_to_rgb(rgb), bgr)


def test_bgr_to_rgb_does_not_modify_its_input() -> None:
    bgr = _rgb_gradient(4, 5)
    before = bgr.copy()
    bgr_to_rgb(bgr)
    assert np.array_equal(bgr, before)


def test_bgr_to_rgb_rejects_wrong_shape() -> None:
    with pytest.raises(ImageError):
        bgr_to_rgb(np.zeros((4, 4), dtype=np.uint8))


def test_ensure_three_channels_handles_grayscale_and_color() -> None:
    gray = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = ensure_three_channels(gray)
    assert out.shape == (2, 3, 3)
    assert all(np.array_equal(out[..., c], gray) for c in range(3))
    assert ensure_three_channels(gray[:, :, None]).shape == (2, 3, 3)
    color = _rgb_gradient(2, 3)
    copy = ensure_three_channels(color)
    assert np.array_equal(copy, color) and copy is not color
    with pytest.raises(ImageError):
        ensure_three_channels(np.zeros((2, 3, 4), dtype=np.uint8))


def test_decode_image_returns_rgb_in_the_original_order() -> None:
    rgb = _rgb_gradient(6, 8)
    decoded = decode_image(_png_bytes(np.ascontiguousarray(rgb[:, :, ::-1])), "test.png")
    assert decoded.dtype == np.uint8 and decoded.shape == (6, 8, 3)
    assert np.array_equal(decoded, rgb)


def test_decode_image_replicates_grayscale() -> None:
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    decoded = decode_image(_png_bytes(gray), "gray.png")
    assert decoded.shape == (3, 4, 3)
    assert all(np.array_equal(decoded[..., c], gray) for c in range(3))


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_decode_image_rejects_empty_and_corrupt_data(data: bytes) -> None:
    with pytest.raises(ImageError):
        decode_image(data, "bad")


def test_decode_image_rejects_images_over_the_pixel_limit() -> None:
    data = _png_bytes(_rgb_gradient(10, 10))
    with pytest.raises(ImageError, match="too large"):
        decode_image(data, "big.png", max_pixels=50)
    assert decode_image(data, "ok.png", max_pixels=100).shape == (10, 10, 3)


def test_read_image_works_with_spaces_and_non_ascii_paths(tmp_path: Path) -> None:
    folder = tmp_path / "dossier é 漢字 with space"
    folder.mkdir()
    rgb = _rgb_gradient(5, 7)
    target = folder / "image ü.png"
    target.write_bytes(_png_bytes(np.ascontiguousarray(rgb[:, :, ::-1])))
    assert np.array_equal(read_image(target), rgb)


def test_read_image_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ImageError, match="not found"):
        read_image(tmp_path / "missing.png")


def test_read_mask_makes_any_nonzero_pixel_anomalous(tmp_path: Path) -> None:
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1, 2] = 7
    mask[2, 3] = 255
    path = tmp_path / "m.png"
    path.write_bytes(_png_bytes(mask))
    result = read_mask(path, (4, 6))
    assert result.dtype == np.uint8
    assert set(np.unique(result)) == {0, 1}
    assert result.sum() == 2


def test_read_mask_accepts_16_bit_masks(tmp_path: Path) -> None:
    mask = np.zeros((3, 3), dtype=np.uint16)
    mask[0, 0] = 1000
    path = tmp_path / "m16.png"
    path.write_bytes(_png_bytes(mask))
    assert read_mask(path, (3, 3)).sum() == 1


def test_read_mask_rejects_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "m.png"
    path.write_bytes(_png_bytes(np.zeros((4, 6), dtype=np.uint8)))
    with pytest.raises(ImageError, match="does not match image size"):
        read_mask(path, (5, 6))


def test_read_mask_rejects_multichannel_masks(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    path.write_bytes(_png_bytes(np.zeros((4, 6, 3), dtype=np.uint8)))
    with pytest.raises(ImageError, match="single-channel"):
        read_mask(path, (4, 6))


def test_read_mask_rejects_missing_and_corrupt_files(tmp_path: Path) -> None:
    with pytest.raises(ImageError, match="not found"):
        read_mask(tmp_path / "none.png", (4, 6))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"garbage")
    with pytest.raises(ImageError, match="readable"):
        read_mask(bad, (4, 6))


def test_stretch_resize_gives_a_square_of_the_requested_size() -> None:
    image = _rgb_gradient(48, 64)
    before = image.copy()
    resized, info = resize_to_square(image, 32, "stretch")
    assert resized.shape == (32, 32, 3) and resized.dtype == np.uint8
    assert info.original_hw == (48, 64) and info.pad_top == 0 and info.pad_left == 0
    assert np.array_equal(image, before)


def test_letterbox_keeps_aspect_ratio_and_pads_with_black() -> None:
    image = np.full((50, 100, 3), 200, dtype=np.uint8)  # height 50, width 100
    resized, info = resize_to_square(image, 64, "letterbox")
    assert resized.shape == (64, 64, 3)
    assert info.scaled_hw == (32, 64) and info.pad_top == 16 and info.pad_left == 0
    assert resized[:16].max() == 0 and resized[48:].max() == 0
    assert resized[16:48].min() == 200


def test_resize_rejects_unknown_mode_and_bad_input() -> None:
    image = _rgb_gradient(8, 8)
    with pytest.raises(ImageError, match="Unknown resize mode"):
        resize_to_square(image, 16, "crop")
    with pytest.raises(ImageError):
        resize_to_square(np.zeros((8, 8), dtype=np.uint8), 16)
    with pytest.raises(ImageError):
        resize_to_square(image, 0)


@pytest.mark.parametrize("mode", ["stretch", "letterbox"])
@pytest.mark.parametrize("shape", [(48, 64), (64, 48), (30, 30)])
def test_map_back_returns_the_original_image_size(mode: str, shape: tuple[int, int]) -> None:
    image = _rgb_gradient(*shape)
    _, info = resize_to_square(image, 32, mode)
    restored = map_back(np.random.default_rng(0).random((32, 32), dtype=np.float32), info)
    assert restored.shape == shape and restored.dtype == np.float32


def test_map_back_drops_letterbox_padding() -> None:
    image = np.full((50, 100, 3), 200, dtype=np.uint8)
    _, info = resize_to_square(image, 64, "letterbox")
    anomaly_map = np.full((64, 64), -1.0, dtype=np.float32)
    anomaly_map[info.pad_top : info.pad_top + 32, :] = 1.0
    restored = map_back(anomaly_map, info)
    assert restored.shape == (50, 100)
    assert np.allclose(restored, 1.0)


def test_map_back_rejects_a_map_of_the_wrong_shape() -> None:
    _, info = resize_to_square(_rgb_gradient(8, 8), 16, "stretch")
    with pytest.raises(ImageError):
        map_back(np.zeros((8, 8), dtype=np.float32), info)


def test_normalize_imagenet_returns_chw_float32() -> None:
    white = np.full((4, 5, 3), 255, dtype=np.uint8)
    black = np.zeros((4, 5, 3), dtype=np.uint8)
    out_white, out_black = normalize_imagenet(white), normalize_imagenet(black)
    assert out_white.shape == (3, 4, 5) and out_white.dtype == np.float32
    assert out_white[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, rel=1e-5)
    assert out_black[2, 0, 0] == pytest.approx(-0.406 / 0.225, rel=1e-5)


def test_check_aspect_ratio_accepts_within_tolerance_and_rejects_beyond() -> None:
    check_aspect_ratio(100, 100, reference=1.0)
    check_aspect_ratio(125, 100, reference=1.0, tolerance=0.25)
    with pytest.raises(ImageError, match="aspect ratio"):
        check_aspect_ratio(130, 100, reference=1.0, tolerance=0.25)
    with pytest.raises(ImageError):
        check_aspect_ratio(100, 100, reference=0.0)


def test_decode_image_rejects_a_side_over_the_limit_even_under_the_pixel_cap() -> None:
    # 5000 x 2 = 10,000 pixels is far below the total-pixel cap but exceeds 4096 on one side.
    data = _png_bytes(np.zeros((2, 5000, 3), dtype=np.uint8))
    with pytest.raises(ImageError, match="too large"):
        decode_image(data, "wide.png")
    with pytest.raises(ImageError, match="too large"):
        decode_image(_png_bytes(np.zeros((5000, 2, 3), dtype=np.uint8)), "tall.png")


def test_decode_image_accepts_a_side_exactly_at_the_limit() -> None:
    data = _png_bytes(np.zeros((2, 4096, 3), dtype=np.uint8))
    assert decode_image(data, "edge.png").shape == (2, 4096, 3)
    with pytest.raises(ImageError, match="too large"):
        decode_image(data, "edge.png", max_side=4095)
