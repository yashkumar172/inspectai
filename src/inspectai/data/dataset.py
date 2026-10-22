"""VisA split-file parsing and sample loading (SPEC.md Section 3.3).

The loader reads the official one-class split file (`1cls.csv`) directly. It does not copy or
reorganize the images. Structural problems in the split file are rejected with a line number;
problems with individual files (missing, unreadable, wrong size) are found by
`inspectai.data.validation` and also raised when a sample is loaded.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from inspectai.config import DEFAULT_MAX_PIXELS, DEFAULT_MAX_SIDE, Paths
from inspectai.errors import DatasetError
from inspectai.preprocessing.images import ImageArray, read_image, read_mask

logger = logging.getLogger(__name__)

SUPPORTED_OBJECTS = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
SPLITS = ("train", "test")
LABELS = ("normal", "anomaly")
EXPECTED_HEADER = ["object", "split", "label", "image", "mask"]


@dataclass(frozen=True)
class Sample:
    """One row of the split file."""

    object_name: str
    split: str
    label: str
    image_rel: str
    mask_rel: str | None
    image_path: Path
    mask_path: Path | None

    @property
    def is_anomalous(self) -> bool:
        return self.label == "anomaly"


class DatasetAdapter(Protocol):
    """What the rest of the package needs from a dataset, so another dataset can be added."""

    def objects(self) -> list[str]: ...

    def get_samples(
        self, object_name: str, split: str | None = None, label: str | None = None
    ) -> list[Sample]: ...

    def load_image(self, sample: Sample) -> ImageArray: ...

    def load_mask(self, sample: Sample, image_hw: tuple[int, int]) -> NDArray[np.uint8]: ...


def parse_split_file(csv_path: Path, images_root: Path) -> list[Sample]:
    """Parse `1cls.csv` into samples.

    Raises DatasetError (with the line number) for a missing file, a wrong header, wrong column
    count, unsupported object, invalid split or label, an anomalous training row, or an image or
    mask path that is absolute or escapes the image folder. Missing files on disk are not
    checked here.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetError(
            f"Split file not found: {csv_path}. Place 1cls.csv at "
            "<INSPECTAI_DATA_ROOT>\\visa\\split_csv\\ (see SPEC.md Section 3.1)."
        )
    try:
        handle = csv_path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetError(f"Split file could not be opened: {csv_path}") from exc

    samples: list[Sample] = []
    with handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise DatasetError(
                f"Unexpected header in {csv_path.name}: {header!r}; expected {EXPECTED_HEADER!r}"
            )
        for row in reader:
            line = reader.line_num
            if not row:
                continue
            if len(row) != len(EXPECTED_HEADER):
                raise DatasetError(
                    f"{csv_path.name} line {line}: expected {len(EXPECTED_HEADER)} columns, "
                    f"found {len(row)}"
                )
            samples.append(_make_sample(row, line, csv_path.name, images_root))
    if not samples:
        raise DatasetError(f"Split file has no data rows: {csv_path}")
    logger.info("Parsed %d rows from %s", len(samples), csv_path.name)
    return samples


def _make_sample(row: list[str], line: int, file_name: str, images_root: Path) -> Sample:
    object_name, split, label, image_rel, mask_rel = (field.strip() for field in row)
    where = f"{file_name} line {line}"
    if object_name not in SUPPORTED_OBJECTS:
        raise DatasetError(f"{where}: unsupported object {object_name!r}")
    if split not in SPLITS:
        raise DatasetError(f"{where}: invalid split {split!r}; expected one of {SPLITS}")
    if label not in LABELS:
        raise DatasetError(f"{where}: invalid label {label!r}; expected one of {LABELS}")
    if split == "train" and label != "normal":
        raise DatasetError(f"{where}: training rows must be normal, found {label!r}")
    image_path = _resolve_inside(images_root, image_rel, "image", where)
    mask_path = _resolve_inside(images_root, mask_rel, "mask", where) if mask_rel else None
    return Sample(
        object_name=object_name,
        split=split,
        label=label,
        image_rel=image_rel,
        mask_rel=mask_rel or None,
        image_path=image_path,
        mask_path=mask_path,
    )


def _resolve_inside(root: Path, relative: str, kind: str, where: str) -> Path:
    """Join a relative POSIX path onto `root`, rejecting absolute paths and '..'."""
    if not relative:
        raise DatasetError(f"{where}: empty {kind} path")
    pure = PurePosixPath(relative.replace("\\", "/"))
    first = pure.parts[0] if pure.parts else ""
    if pure.is_absolute() or ":" in first or ".." in pure.parts:
        raise DatasetError(
            f"{where}: {kind} path must stay inside the dataset folder: {relative!r}"
        )
    return root.joinpath(*pure.parts)


class VisADataset:
    """The VisA dataset described by a split file."""

    def __init__(
        self,
        split_csv: Path,
        images_root: Path,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_side: int = DEFAULT_MAX_SIDE,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.images_root = Path(images_root)
        self.max_pixels = max_pixels
        self.max_side = max_side
        self.samples: tuple[Sample, ...] = tuple(parse_split_file(self.split_csv, self.images_root))

    @classmethod
    def from_paths(cls, paths: Paths) -> VisADataset:
        return cls(paths.split_csv, paths.visa_images_root)

    def objects(self) -> list[str]:
        """Objects that appear in the split file, in sorted order."""
        return sorted({sample.object_name for sample in self.samples})

    def get_samples(
        self, object_name: str, split: str | None = None, label: str | None = None
    ) -> list[Sample]:
        """Samples of one object, optionally filtered by split and label."""
        if object_name not in SUPPORTED_OBJECTS:
            raise DatasetError(
                f"Unsupported object {object_name!r}; supported: {', '.join(SUPPORTED_OBJECTS)}"
            )
        if split is not None and split not in SPLITS:
            raise DatasetError(f"Invalid split {split!r}; expected one of {SPLITS}")
        if label is not None and label not in LABELS:
            raise DatasetError(f"Invalid label {label!r}; expected one of {LABELS}")
        selected = [
            sample
            for sample in self.samples
            if sample.object_name == object_name
            and (split is None or sample.split == split)
            and (label is None or sample.label == label)
        ]
        if not selected:
            raise DatasetError(
                f"No samples for object {object_name!r}"
                f"{f', split {split!r}' if split else ''}{f', label {label!r}' if label else ''}"
            )
        return selected

    def load_image(self, sample: Sample) -> ImageArray:
        """Read the sample's image as an RGB uint8 array."""
        return read_image(sample.image_path, self.max_pixels, self.max_side)

    def load_mask(self, sample: Sample, image_hw: tuple[int, int]) -> NDArray[np.uint8]:
        """Return a binary mask (0 or 1) with the given (height, width).

        Normal samples have no stored mask and get an all-zero mask. An anomalous sample
        without a mask is an error.
        """
        if sample.mask_path is None:
            if sample.is_anomalous:
                raise DatasetError(f"Anomalous sample has no mask: {sample.image_rel}")
            return np.zeros(image_hw, dtype=np.uint8)
        return read_mask(sample.mask_path, image_hw)


def count_samples(samples: Sequence[Sample]) -> dict[tuple[str, str, str], int]:
    """Count samples by (object, split, label)."""
    counts: dict[tuple[str, str, str], int] = {}
    for sample in samples:
        key = (sample.object_name, sample.split, sample.label)
        counts[key] = counts.get(key, 0) + 1
    return counts
