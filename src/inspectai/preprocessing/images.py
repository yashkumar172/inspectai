"""Safe image and mask loading, color handling, and resizing (SPEC.md Section 6).

Images are handled as RGB `uint8` arrays of shape (height, width, 3). Input arrays are never
modified; every function returns new arrays.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from inspectai.config import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MAX_SIDE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_MODES,
)
from inspectai.errors import ImageError

# Pillow warns or raises above this size; we also reject such images explicitly before decoding.
Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_PIXELS

ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]

_SINGLE_CHANNEL_MODES = frozenset({"1", "L", "P", "I", "I;16", "F"})


def probe_image(
    data: bytes,
    source: str,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[int, int]:
    """Return (width, height) from the image header without decoding the pixels.

    Raises ImageError if the data is not a readable image, if either side exceeds `max_side`,
    or if the total pixel count exceeds `max_pixels`.
    """
    return _probe(io.BytesIO(data), source, max_pixels, max_side)


def probe_image_file(
    path: Path, max_pixels: int = DEFAULT_MAX_PIXELS, max_side: int = DEFAULT_MAX_SIDE
) -> tuple[int, int]:
    """Like `probe_image`, but reads only the header of a file on disk."""
    return _probe(Path(path), str(path), max_pixels, max_side)


def _probe(
    target: io.BytesIO | Path, source: str, max_pixels: int, max_side: int
) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(target) as handle:
                width, height = handle.size
    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageError(f"Image is too large to load safely: {source}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError(f"Not a readable image: {source}") from exc
    if width < 1 or height < 1:
        raise ImageError(f"Image has an empty size ({width}x{height}): {source}")
    if width > max_side or height > max_side:
        raise ImageError(
            f"Image is too large ({width}x{height}; limit {max_side}x{max_side}): {source}"
        )
    if width * height > max_pixels:
        raise ImageError(
            f"Image is too large ({width}x{height} pixels; limit {max_pixels} pixels): {source}"
        )
    return width, height


def decode_image(
    data: bytes,
    source: str,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> ImageArray:
    """Decode image bytes to an RGB uint8 array.

    The size limit is checked on the header before decoding. EXIF orientation is ignored so the
    array matches the stored pixel layout (and therefore the stored masks). Grayscale inputs are
    replicated to three channels and any alpha channel is dropped.
    """
    if not data:
        raise ImageError(f"Image file is empty: {source}")
    probe_image(data, source, max_pixels, max_side)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if bgr is None:
        raise ImageError(f"Image could not be decoded: {source}")
    return bgr_to_rgb(bgr)


def read_image(
    path: Path, max_pixels: int = DEFAULT_MAX_PIXELS, max_side: int = DEFAULT_MAX_SIDE
) -> ImageArray:
    """Read an image file as an RGB uint8 array.

    Reading goes through bytes rather than `cv2.imread`, so paths with spaces or non-ASCII
    characters work on Windows. A failed read raises ImageError and never yields an empty image.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"Image file not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageError(f"Image file could not be read: {path}") from exc
    return decode_image(data, str(path), max_pixels, max_side)


def read_mask(path: Path, expected_hw: tuple[int, int]) -> NDArray[np.uint8]:
    """Read a ground-truth mask as a binary uint8 array of 0 and 1.

    Any nonzero pixel is anomalous (the official VisA script does the same). The mask must be a
    single-channel image with the same height and width as its image.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"Mask file not found: {path}")
    try:
        with Image.open(path) as handle:
            if handle.mode not in _SINGLE_CHANNEL_MODES:
                raise ImageError(
                    f"Mask must be a single-channel image, found mode {handle.mode}: {path}"
                )
            values = np.asarray(handle)
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageError(f"Not a readable mask: {path}") from exc
    if values.ndim != 2:
        raise ImageError(f"Mask must be a single-channel image, found {values.ndim} dimensions")
    if values.shape != tuple(expected_hw):
        raise ImageError(
            f"Mask size {values.shape[1]}x{values.shape[0]} does not match image size "
            f"{expected_hw[1]}x{expected_hw[0]}: {path}"
        )
    return (values != 0).astype(np.uint8)


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    """Convert between OpenCV's BGR order and RGB (the conversion is its own inverse)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"Expected an image with 3 channels, got shape {image.shape}")
    return np.ascontiguousarray(image[:, :, ::-1])


def ensure_three_channels(image: ImageArray) -> ImageArray:
    """Return an (H, W, 3) array from a grayscale (H, W), (H, W, 1), or color array."""
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ImageError(f"Unsupported image shape {image.shape}")


@dataclass(frozen=True)
class ResizeInfo:
    """What `resize_to_square` did, so an anomaly map can be mapped back to the original."""

    mode: str
    original_hw: tuple[int, int]
    size: int
    scaled_hw: tuple[int, int]
    pad_top: int
    pad_left: int


def resize_to_square(
    image: ImageArray, size: int, mode: str = "stretch"
) -> tuple[ImageArray, ResizeInfo]:
    """Resize an RGB image to `size` x `size`.

    `stretch` resizes directly (no cropping, so border defects stay in view). `letterbox` keeps
    the aspect ratio and pads the remainder with black, centered.
    """
    if mode not in RESIZE_MODES:
        raise ImageError(f"Unknown resize mode {mode!r}; expected one of {RESIZE_MODES}")
    if size < 1:
        raise ImageError(f"Target size must be positive, got {size}")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ImageError(f"Expected a non-empty (H, W, 3) image, got shape {image.shape}")
    height, width = image.shape[:2]

    if mode == "stretch":
        resized = cv2.resize(image, (size, size), interpolation=_interpolation(height, width, size))
        info = ResizeInfo(mode, (height, width), size, (size, size), 0, 0)
        return resized, info

    scale = size / max(height, width)
    new_h = max(1, min(size, round(height * scale)))
    new_w = max(1, min(size, round(width * scale)))
    scaled = cv2.resize(image, (new_w, new_h), interpolation=_interpolation(height, width, size))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = scaled
    return canvas, ResizeInfo(mode, (height, width), size, (new_h, new_w), pad_top, pad_left)


def map_back(anomaly_map: FloatArray, info: ResizeInfo) -> FloatArray:
    """Map a square anomaly map back to the original image size.

    For letterboxed input the padding is removed before resizing.
    """
    if anomaly_map.shape != (info.size, info.size):
        raise ImageError(
            f"Anomaly map has shape {anomaly_map.shape}, expected {(info.size, info.size)}"
        )
    values = anomaly_map.astype(np.float32, copy=False)
    if info.mode == "letterbox":
        new_h, new_w = info.scaled_hw
        values = values[info.pad_top : info.pad_top + new_h, info.pad_left : info.pad_left + new_w]
    height, width = info.original_hw
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_imagenet(image: ImageArray) -> FloatArray:
    """Scale an RGB uint8 image to [0, 1], normalize with ImageNet statistics, return CHW."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"Expected an (H, W, 3) image, got shape {image.shape}")
    scaled = image.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(((scaled - mean) / std).transpose(2, 0, 1))


def aspect_ratio(width: int, height: int) -> float:
    """Width divided by height."""
    if width < 1 or height < 1:
        raise ImageError(f"Image size must be positive, got {width}x{height}")
    return width / height


def check_aspect_ratio(width: int, height: int, reference: float, tolerance: float = 0.25) -> None:
    """Raise ImageError if the aspect ratio differs from `reference` by more than `tolerance`."""
    if reference <= 0:
        raise ImageError(f"Reference aspect ratio must be positive, got {reference}")
    deviation = abs(aspect_ratio(width, height) - reference) / reference
    if deviation > tolerance:
        raise ImageError(
            f"Image aspect ratio {width / height:.2f} differs by {deviation:.0%} from the "
            f"expected {reference:.2f} (allowed {tolerance:.0%})."
        )


def _interpolation(height: int, width: int, size: int) -> int:
    """Area interpolation for shrinking, linear for enlarging."""
    return cv2.INTER_AREA if max(height, width) > size else cv2.INTER_LINEAR
