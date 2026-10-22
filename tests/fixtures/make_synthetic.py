"""Generate a small VisA-style dataset for tests. No real dataset images are used."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticVisA:
    data_root: Path
    images_root: Path
    split_csv: Path
    objects: tuple[str, ...]
    size: tuple[int, int]  # (width, height)


def write_jpg(path: Path, rgb: NDArray[np.uint8]) -> None:
    """Write an RGB array as JPEG (through bytes, so any path works on Windows)."""
    ok, encoded = cv2.imencode(".jpg", rgb[:, :, ::-1])
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def write_png(path: Path, array: NDArray[np.uint8]) -> None:
    """Write a 2-D (grayscale) or 3-D array as PNG."""
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _texture(rng: np.random.Generator, width: int, height: int) -> NDArray[np.uint8]:
    noise = rng.integers(70, 130, size=(height, width, 3)).astype(np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def build_synthetic_visa(
    data_root: Path,
    objects: tuple[str, ...] = ("cashew", "pipe_fryum"),
    n_train: int = 24,
    n_test_normal: int = 5,
    n_test_anomaly: int = 5,
    size: tuple[int, int] = (64, 48),
    seed: int = 0,
) -> SyntheticVisA:
    """Create `<data_root>/visa/VisA/...` and `<data_root>/visa/split_csv/1cls.csv`.

    Normal images are textured noise; anomalous images have a bright rectangle, and the mask is
    255 inside that rectangle. Normal rows have an empty mask column, as in the real file.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    images_root = data_root / "visa" / "VisA"
    split_csv = data_root / "visa" / "split_csv" / "1cls.csv"
    rows: list[list[str]] = []

    for name in objects:
        images = images_root / name / "Data" / "Images"
        masks = images_root / name / "Data" / "Masks" / "Anomaly"
        for index in range(n_train + n_test_normal):
            rel = f"{name}/Data/Images/Normal/{index:04d}.JPG"
            write_jpg(images_root / rel, _texture(rng, width, height))
            split = "train" if index < n_train else "test"
            rows.append([name, split, "normal", rel, ""])
        for index in range(n_test_anomaly):
            image = _texture(rng, width, height)
            x0, y0 = width // 4 + index, height // 4
            image[y0 : y0 + height // 4, x0 : x0 + width // 4] = 250
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y0 : y0 + height // 4, x0 : x0 + width // 4] = 255
            rel_image = f"{name}/Data/Images/Anomaly/{index:03d}.JPG"
            rel_mask = f"{name}/Data/Masks/Anomaly/{index:03d}.png"
            write_jpg(images_root / rel_image, image)
            write_png(images_root / rel_mask, mask)
            rows.append([name, "test", "anomaly", rel_image, rel_mask])
        assert images.is_dir() and masks.is_dir()

    split_csv.parent.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows(rows)
    return SyntheticVisA(data_root, images_root, split_csv, objects, size)
