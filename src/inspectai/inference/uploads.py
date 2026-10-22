"""Validation of images uploaded to the app (SPEC.md Sections 6 and 8).

The checks run from cheapest to most expensive and stop at the first failure, so a rejected file
never reaches model code: size, file type, header (readable, real format, dimensions and pixel
count), minimum size, aspect ratio against the category, and only then full decoding.

Every message raised here is written for the user: it contains no file system path, no stack
trace, and not even the uploaded file name.
"""

from __future__ import annotations

import io
from pathlib import PurePath

from PIL import Image

from inspectai.errors import ImageError, UploadError
from inspectai.preprocessing.images import (
    ImageArray,
    check_aspect_ratio,
    decode_image,
    probe_image,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # SPEC.md Section 8: server.maxUploadSize = 10 MB
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
ALLOWED_FORMATS = ("JPEG", "PNG")
MIN_SIDE = 32
ASPECT_TOLERANCE = 0.25  # SPEC.md Section 6
_LABEL = "the uploaded image"


def validate_upload(
    data: bytes,
    filename: str,
    reference_aspect_ratio: float | None = None,
    aspect_tolerance: float = ASPECT_TOLERANCE,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> ImageArray:
    """Validate an uploaded file and return it as an RGB uint8 array.

    `reference_aspect_ratio` is the category's median fit-image aspect ratio from the model
    manifest; an image that deviates by more than `aspect_tolerance` is rejected. Raises
    UploadError with a user-facing message on any problem.
    """
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadError(
            f"The file is {len(data) / 1024**2:.1f} MB; the maximum is {max_bytes / 1024**2:.0f} MB."
        )
    if PurePath(filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file type. Please upload a JPG, JPEG, or PNG image.")

    try:
        width, height = probe_image(data, _LABEL)
    except ImageError as exc:
        text = str(exc).removesuffix(f": {_LABEL}")
        if text.startswith("Not a readable image"):
            raise UploadError(
                "The file could not be read as an image. It may be corrupted or not an image."
            ) from exc
        raise UploadError(text + ".") from exc
    with Image.open(io.BytesIO(data)) as handle:
        content_format = handle.format
    if content_format not in ALLOWED_FORMATS:
        raise UploadError(
            f"The file content is {content_format}, not JPEG or PNG. Please upload a JPG or PNG."
        )
    if min(width, height) < MIN_SIDE:
        raise UploadError(
            f"The image is only {width}x{height} pixels; at least {MIN_SIDE}x{MIN_SIDE} is required."
        )
    if reference_aspect_ratio is not None:
        try:
            check_aspect_ratio(width, height, reference_aspect_ratio, aspect_tolerance)
        except ImageError as exc:
            raise UploadError(
                f"{exc} Use an image of the selected category with similar proportions."
            ) from exc
    try:
        return decode_image(data, _LABEL)
    except ImageError as exc:
        raise UploadError("The image could not be decoded. It may be corrupted.") from exc
