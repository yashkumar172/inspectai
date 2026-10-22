"""Validated settings and environment-driven paths (SPEC.md Sections 3.5, 6, and 9.2)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from inspectai.errors import ConfigError

ENV_DATA_ROOT = "INSPECTAI_DATA_ROOT"
ENV_ARTIFACT_DIR = "INSPECTAI_ARTIFACT_DIR"
ENV_REPORT_DIR = "INSPECTAI_REPORT_DIR"

RESIZE_MODES = ("stretch", "letterbox")
DEFAULT_MAX_SIDE = 4096
DEFAULT_MAX_PIXELS = DEFAULT_MAX_SIDE * DEFAULT_MAX_SIDE
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# A category is resized with "stretch" only if its median aspect ratio is this close to 1.
STRETCH_MAX_ASPECT_DEVIATION = 0.05


# Resize mode of each VisA object, chosen from the measured median aspect ratios (SPEC.md
# Sections 6 and 16, Milestone 1) with `recommend_resize_mode`: "stretch" only if the median is
# within STRETCH_MAX_ASPECT_DEVIATION of square. Measured with scripts/validate_dataset.py on
# the official archive (VisA_20220922.tar); a needs_dataset test checks this table against the
# data. Median width / height: candle 1.10, capsules 1.50, cashew 1.08, chewinggum 1.20,
# fryum 1.50, macaroni1 1.50, macaroni2 1.50, pcb1 1.31, pcb2 1.31, pcb3 1.63, pcb4 1.23,
# pipe_fryum 1.13. The table is recorded in the artifact manifest from Milestone 3 on.
CATEGORY_RESIZE_MODES: dict[str, str] = {
    "candle": "letterbox",
    "capsules": "letterbox",
    "cashew": "letterbox",
    "chewinggum": "letterbox",
    "fryum": "letterbox",
    "macaroni1": "letterbox",
    "macaroni2": "letterbox",
    "pcb1": "letterbox",
    "pcb2": "letterbox",
    "pcb3": "letterbox",
    "pcb4": "letterbox",
    "pipe_fryum": "letterbox",
}


def resize_mode_for(object_name: str, overrides: Mapping[str, str] | None = None) -> str:
    """The resize mode of one category; `overrides` makes it configurable per category."""
    mode = (overrides or {}).get(object_name) or CATEGORY_RESIZE_MODES.get(object_name)
    if mode is None:
        raise ConfigError(
            f"No resize mode is defined for {object_name!r}; "
            f"known categories: {', '.join(sorted(CATEGORY_RESIZE_MODES))}"
        )
    if mode not in RESIZE_MODES:
        raise ConfigError(f"resize_mode must be one of {RESIZE_MODES}, got {mode!r}")
    return mode


@dataclass(frozen=True)
class Paths:
    """Machine-specific locations, supplied through environment variables."""

    data_root: Path
    artifact_dir: Path
    report_dir: Path

    @property
    def visa_root(self) -> Path:
        return self.data_root / "visa"

    @property
    def visa_images_root(self) -> Path:
        return self.visa_root / "VisA"

    @property
    def split_csv(self) -> Path:
        return self.visa_root / "split_csv" / "1cls.csv"


def load_paths(env: Mapping[str, str] | None = None) -> Paths:
    """Read the dataset, artifact, and report locations from environment variables.

    `INSPECTAI_DATA_ROOT` is required. The artifact and report directories default to
    subfolders of the data root when their variables are not set.
    """
    source = os.environ if env is None else env
    raw_root = source.get(ENV_DATA_ROOT, "").strip()
    if not raw_root:
        raise ConfigError(
            f"{ENV_DATA_ROOT} is not set. Set it to the folder that contains 'visa' "
            "(for example C:\\dev\\inspectai-data); see README.md, section 'Environment variables'."
        )
    data_root = Path(raw_root).expanduser()
    artifact_raw = source.get(ENV_ARTIFACT_DIR, "").strip()
    report_raw = source.get(ENV_REPORT_DIR, "").strip()
    return Paths(
        data_root=data_root,
        artifact_dir=Path(artifact_raw).expanduser() if artifact_raw else data_root / "artifacts",
        report_dir=Path(report_raw).expanduser() if report_raw else data_root / "reports",
    )


@dataclass(frozen=True)
class SplitConfig:
    """Parameters of the fit / calibration_normal split (SPEC.md Section 3.5)."""

    seed: int = 42
    calibration_fraction: float = 0.10
    min_calibration: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigError(f"seed must be an integer, got {self.seed!r}")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ConfigError(
                f"calibration_fraction must be between 0 and 1 (exclusive), "
                f"got {self.calibration_fraction!r}"
            )
        if isinstance(self.min_calibration, bool) or self.min_calibration < 1:
            raise ConfigError(f"min_calibration must be at least 1, got {self.min_calibration!r}")


@dataclass(frozen=True)
class PreprocessConfig:
    """Image preprocessing settings (SPEC.md Section 6)."""

    image_size: int = 224
    resize_mode: str = "stretch"
    aspect_tolerance: float = 0.25
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_side: int = DEFAULT_MAX_SIDE

    @classmethod
    def for_object(
        cls, object_name: str, overrides: Mapping[str, str] | None = None, **kwargs: float
    ) -> PreprocessConfig:
        """Settings for one category, with the resize mode chosen for it."""
        return cls(resize_mode=resize_mode_for(object_name, overrides), **kwargs)

    def __post_init__(self) -> None:
        if isinstance(self.image_size, bool) or self.image_size < 8:
            raise ConfigError(f"image_size must be at least 8, got {self.image_size!r}")
        if self.resize_mode not in RESIZE_MODES:
            raise ConfigError(
                f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}"
            )
        if not 0.0 < self.aspect_tolerance < 1.0:
            raise ConfigError(
                f"aspect_tolerance must be between 0 and 1 (exclusive), "
                f"got {self.aspect_tolerance!r}"
            )
        if isinstance(self.max_pixels, bool) or self.max_pixels < 1:
            raise ConfigError(f"max_pixels must be positive, got {self.max_pixels!r}")
        if isinstance(self.max_side, bool) or self.max_side < 1:
            raise ConfigError(f"max_side must be positive, got {self.max_side!r}")
