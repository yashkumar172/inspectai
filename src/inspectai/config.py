"""Validated settings and environment-driven paths (SPEC.md Sections 3.5, 6, and 9.2)."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from inspectai.errors import ConfigError

ENV_DATA_ROOT = "INSPECTAI_DATA_ROOT"
ENV_ARTIFACT_DIR = "INSPECTAI_ARTIFACT_DIR"
ENV_REPORT_DIR = "INSPECTAI_REPORT_DIR"

# Precomputed artifacts and reports that ship with the repository for a hosted, dataset-free run
# (built by scripts/export_hosted_bundle.py). The app falls back to them when no location is set.


def find_bundle_root(package_file: Path | str = __file__, cwd: Path | None = None) -> Path:
    """The `hosted/` folder next to `src/` (a source checkout), else the working directory's.

    A regular `pip install .` copies the package into site-packages, where the folder next to
    `src/` does not exist; a hosted run starts in the repository root, so `./hosted` is used then.
    """
    beside_src = Path(package_file).resolve().parents[2] / "hosted"
    in_cwd = (Path.cwd() if cwd is None else Path(cwd)) / "hosted"
    return in_cwd if not beside_src.is_dir() and in_cwd.is_dir() else beside_src


BUNDLE_ROOT = find_bundle_root()

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


# Backbone profiles (SPEC.md Section 5.5).
PROFILES: dict[str, dict[str, object]] = {
    "dev": {"backbone": "resnet18", "layers": ("layer2", "layer3"), "image_size": 224},
    "full": {"backbone": "wide_resnet50_2", "layers": ("layer2", "layer3"), "image_size": 224},
}


def profile_settings(name: str) -> dict[str, object]:
    """The backbone, feature layers, and image size of a profile (`dev` or `full`)."""
    if name not in PROFILES:
        raise ConfigError(f"Unknown profile {name!r}; expected one of {tuple(PROFILES)}")
    return dict(PROFILES[name])


@dataclass(frozen=True)
class Paths:
    """Machine-specific locations, supplied through environment variables.

    `data_root` is None for a hosted run that only has precomputed artifacts and reports; the
    dataset locations are then unavailable (`has_dataset` is False).
    """

    data_root: Path | None
    artifact_dir: Path
    report_dir: Path

    @property
    def has_dataset(self) -> bool:
        return self.data_root is not None

    @property
    def visa_root(self) -> Path:
        if self.data_root is None:
            raise ConfigError("No dataset location is configured (INSPECTAI_DATA_ROOT is not set).")
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


def load_app_paths(env: Mapping[str, str] | None = None, bundle_root: Path | None = None) -> Paths:
    """The locations the Streamlit app uses; the dataset is optional.

    With `INSPECTAI_DATA_ROOT` set this is exactly `load_paths`. Without it, the app still runs
    from precomputed files: an explicitly set artifact or report directory is used as given, and
    a directory that is not set falls back to the bundled `hosted/` folder if it exists. If
    neither is available, a `ConfigError` is raised.
    """
    source = os.environ if env is None else env
    if source.get(ENV_DATA_ROOT, "").strip():
        return load_paths(source)
    bundle = BUNDLE_ROOT if bundle_root is None else Path(bundle_root)
    artifact_raw = source.get(ENV_ARTIFACT_DIR, "").strip()
    report_raw = source.get(ENV_REPORT_DIR, "").strip()
    artifact_dir = Path(artifact_raw).expanduser() if artifact_raw else bundle / "artifacts"
    report_dir = Path(report_raw).expanduser() if report_raw else bundle / "reports"
    if not (artifact_raw or report_raw) and not (artifact_dir.is_dir() or report_dir.is_dir()):
        raise ConfigError(
            f"None of {ENV_DATA_ROOT}, {ENV_ARTIFACT_DIR}, or {ENV_REPORT_DIR} is set and there "
            "is no bundled 'hosted' folder; see README.md, section 'Environment variables'."
        )
    return Paths(data_root=None, artifact_dir=artifact_dir, report_dir=report_dir)


def use_bundled_weights(
    paths: Paths, env: MutableMapping[str, str] | None = None, bundle_root: Path | None = None
) -> Path | None:
    """Point `TORCH_HOME` at the bundled pretrained weights for a dataset-free hosted run.

    Only applies without a dataset (`paths.data_root` is None), when `TORCH_HOME` is not set, and
    when the bundle holds weights (`hosted/torch_cache/hub/checkpoints`). Returns the folder it
    set, else None. The weights are still hash-checked against the model manifest when loaded.
    """
    target = os.environ if env is None else env
    home = (BUNDLE_ROOT if bundle_root is None else Path(bundle_root)) / "torch_cache"
    if paths.data_root is not None or target.get("TORCH_HOME", "").strip():
        return None
    if not (home / "hub" / "checkpoints").is_dir():
        return None
    target["TORCH_HOME"] = str(home)
    return home


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
