"""Upload validation: size, type, header, dimensions, aspect ratio, decoding (SPEC.md 6 and 8)."""

import cv2
import numpy as np
import pytest

from inspectai.errors import InspectAIError, UploadError
from inspectai.inference.uploads import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    MIN_SIDE,
    validate_upload,
)


def _encode(array: np.ndarray, ext: str = ".png") -> bytes:
    ok, encoded = cv2.imencode(ext, array)
    assert ok
    return encoded.tobytes()


def _rgb(height: int = 48, width: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_the_limits_follow_the_spec() -> None:
    assert MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    assert set(ALLOWED_EXTENSIONS) == {".jpg", ".jpeg", ".png"}


@pytest.mark.parametrize(
    ("ext", "name"), [(".png", "a.png"), (".jpg", "a.jpg"), (".jpg", "A.JPEG")]
)
def test_valid_images_are_returned_as_rgb_arrays(ext: str, name: str) -> None:
    rgb = _rgb()
    image = validate_upload(_encode(rgb[:, :, ::-1], ext), name)
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8
    if ext == ".png":
        assert np.array_equal(image, rgb)  # lossless: BGR on disk, RGB in memory


def test_grayscale_and_alpha_images_become_three_channel_rgb() -> None:
    gray = np.arange(48 * 64, dtype=np.uint8).reshape(48, 64)
    image = validate_upload(_encode(gray), "g.png")
    assert image.shape == (48, 64, 3) and np.array_equal(image[..., 0], gray)
    rgba = np.dstack([_rgb(), np.full((48, 64), 128, np.uint8)])
    assert validate_upload(_encode(rgba), "a.png").shape == (48, 64, 3)


def test_an_empty_file_is_rejected() -> None:
    with pytest.raises(UploadError, match="empty"):
        validate_upload(b"", "a.png")


def test_an_oversized_file_is_rejected_before_anything_else_is_looked_at() -> None:
    data = b"\x00" * (MAX_UPLOAD_BYTES + 1)  # not even an image: the size check comes first
    with pytest.raises(UploadError, match=r"maximum is 10 MB"):
        validate_upload(data, "a.png")


def test_the_size_limit_is_exact() -> None:
    small = _encode(_rgb())
    assert validate_upload(small, "a.png", max_bytes=len(small)).shape == (48, 64, 3)
    with pytest.raises(UploadError, match="maximum"):
        validate_upload(small, "a.png", max_bytes=len(small) - 1)


@pytest.mark.parametrize("name", ["a.gif", "a.bmp", "a.tiff", "a", "", "a.png.exe", "a.jpg.txt"])
def test_unsupported_file_types_are_rejected(name: str) -> None:
    with pytest.raises(UploadError, match="Unsupported file type"):
        validate_upload(_encode(_rgb()), name)


def test_the_file_content_must_really_be_jpeg_or_png() -> None:
    bmp = _encode(_rgb(), ".bmp")
    with pytest.raises(UploadError, match="not JPEG or PNG"):
        validate_upload(bmp, "disguised.png")


@pytest.mark.parametrize(
    "data", [b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 30, b"\xff\xd8"]
)
def test_unreadable_files_are_rejected(data: bytes) -> None:
    with pytest.raises(UploadError, match="could not be read as an image"):
        validate_upload(data, "a.png")


def test_a_truncated_image_with_a_valid_header_is_rejected_at_decoding() -> None:
    data = _encode(_rgb(200, 300))
    with pytest.raises(UploadError, match="could not be decoded"):
        validate_upload(data[: len(data) // 3], "a.png")


def test_oversized_dimensions_are_rejected_before_decoding() -> None:
    wide = _encode(np.zeros((2, 5000, 3), dtype=np.uint8))  # small file, 5000 pixels wide
    with pytest.raises(UploadError, match="too large") as caught:
        validate_upload(wide, "a.png")
    assert "4096" in str(caught.value)
    with pytest.raises(UploadError, match="too large"):
        validate_upload(_encode(np.zeros((5000, 2, 3), dtype=np.uint8)), "a.png")


def test_tiny_images_are_rejected() -> None:
    with pytest.raises(UploadError, match=f"at least {MIN_SIDE}x{MIN_SIDE}"):
        validate_upload(_encode(_rgb(16, 16)), "a.png")
    assert validate_upload(_encode(_rgb(MIN_SIDE, MIN_SIDE)), "a.png").shape[:2] == (
        MIN_SIDE,
        MIN_SIDE,
    )


def test_the_aspect_ratio_is_checked_against_the_category_reference() -> None:
    square_ish = _encode(_rgb(100, 120))  # 1.20 against a reference of 1.0: 20% off, allowed
    assert validate_upload(square_ish, "a.png", reference_aspect_ratio=1.0).shape == (100, 120, 3)
    exactly = _encode(_rgb(100, 125))  # 1.25: exactly at the limit, allowed
    assert validate_upload(exactly, "a.png", reference_aspect_ratio=1.0).shape == (100, 125, 3)
    with pytest.raises(UploadError, match="aspect ratio") as caught:
        validate_upload(_encode(_rgb(100, 126)), "a.png", reference_aspect_ratio=1.0)
    assert "similar proportions" in str(caught.value)
    with pytest.raises(UploadError, match="aspect ratio"):
        validate_upload(_encode(_rgb(40, 120)), "a.png", reference_aspect_ratio=1.5)  # 3.0 vs 1.5
    assert validate_upload(_encode(_rgb(40, 120)), "a.png").shape == (40, 120, 3)  # no reference


@pytest.mark.parametrize(
    ("data", "name"),
    [
        (b"", "secret_name.png"),
        (b"junk", "secret_name.png"),
        (_encode(_rgb(), ".bmp"), "secret_name.png"),
        (_encode(_rgb(16, 16)), "secret_name.png"),
        (_encode(_rgb()), "secret_name.gif"),
        (_encode(np.zeros((2, 5000, 3), dtype=np.uint8)), "secret_name.png"),
        (_encode(_rgb(10, 400)), "secret_name.png"),
    ],
    ids=["empty", "junk", "bmp-content", "tiny", "gif-name", "too-wide", "aspect"],
)
def test_messages_never_contain_the_file_name_a_path_or_a_traceback(data: bytes, name: str) -> None:
    with pytest.raises(UploadError) as caught:
        validate_upload(data, name, reference_aspect_ratio=1.0)
    message = str(caught.value)
    assert "secret_name" not in message and "Traceback" not in message
    assert "\\" not in message and ":/" not in message and "C:" not in message


def test_random_bytes_can_only_ever_raise_upload_errors() -> None:
    rng = np.random.default_rng(1)
    for size in (1, 7, 100, 5000):
        for name in ("a.png", "a.jpg"):
            data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
            try:
                validate_upload(data, name)
            except UploadError as exc:
                assert isinstance(exc, InspectAIError)
            else:
                pytest.fail("random bytes were accepted as an image")
    with pytest.raises(UploadError):  # PNG signature followed by garbage
        validate_upload(
            b"\x89PNG\r\n\x1a\n" + rng.integers(0, 256, 500, dtype=np.uint8).tobytes(), "a.png"
        )
