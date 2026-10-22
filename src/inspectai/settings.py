"""The documented default settings file `configs/default.yaml` (SPEC.md Sections 9 and 9.2).

`load_default_config` reads the file, expands `${NAME}` references from the environment (so no
machine-specific path is ever written into the repository), and validates every section by
building the real settings objects from it. An invalid value therefore fails with a clear
`ConfigError`, and only CPU execution is accepted as the device.

The file documents the defaults; it does not replace the defaults in the code. A test keeps the
two identical.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from inspectai.config import PROFILES, RESIZE_MODES, PreprocessConfig, SplitConfig
from inspectai.data.dataset import SUPPORTED_OBJECTS
from inspectai.errors import ConfigError, EvaluationError, ModelError
from inspectai.evaluation.runner import EvalOptions
from inspectai.models.classical import ClassicalConfig
from inspectai.models.gaussian import GaussianConfig
from inspectai.models.patchcore import PatchCoreConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
SUPPORTED_DEVICES = ("cpu",)
_SECTIONS = (
    "paths",
    "device",
    "seed",
    "categories",
    "splits",
    "preprocessing",
    "resize_modes",
    "profiles",
    "thresholds",
    "evaluation",
    "classical",
    "gaussian",
    "patchcore",
    "app",
)
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class DefaultConfig:
    """The validated contents of the default settings file."""

    paths: dict[str, str | None]
    unresolved: tuple[str, ...]  # environment variables the file refers to that are not set
    device: str
    seed: int
    categories: tuple[str, ...]
    splits: SplitConfig
    preprocessing: PreprocessConfig
    resize_modes: dict[str, str]
    evaluation: EvalOptions
    classical: ClassicalConfig
    gaussian: GaussianConfig
    patchcore: PatchCoreConfig
    max_fit_images: int | None
    app: dict[str, float]


def expand_environment(
    value: str, env: Mapping[str, str] | None = None
) -> tuple[str | None, list[str]]:
    """Replace `${NAME}` with the environment value; None if any referenced name is unset."""
    source = os.environ if env is None else env
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        found = source.get(name, "").strip()
        if not found:
            missing.append(name)
        return found

    expanded = _REFERENCE.sub(replace, value)
    return (None if missing else expanded), missing


def _section(data: Mapping[str, object], name: str) -> dict[str, object]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"configs/default.yaml: section '{name}' must be a mapping")
    return dict(section)


def _build(name: str, factory, **values: object):  # type: ignore[no-untyped-def]
    try:
        return factory(**values)
    except (TypeError, ConfigError, ModelError, EvaluationError) as exc:
        raise ConfigError(f"configs/default.yaml: invalid '{name}' settings: {exc}") from exc


def load_default_config(
    path: Path | None = None, env: Mapping[str, str] | None = None
) -> DefaultConfig:
    """Load and validate the default settings file."""
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"The settings file could not be read: {target.name}") from exc
    if not isinstance(data, dict):
        raise ConfigError("configs/default.yaml must contain a mapping")
    unknown = sorted(set(data) - set(_SECTIONS))
    missing_sections = sorted(set(_SECTIONS) - set(data))
    if unknown or missing_sections:
        raise ConfigError(
            f"configs/default.yaml has unknown sections {unknown} or lacks sections {missing_sections}"
        )

    device = data["device"]
    if device not in SUPPORTED_DEVICES:
        raise ConfigError(
            f"Unsupported device {device!r}: only CPU execution is supported "
            f"({', '.join(SUPPORTED_DEVICES)})."
        )
    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("configs/default.yaml: 'seed' must be an integer")

    paths: dict[str, str | None] = {}
    unresolved: list[str] = []
    for key, raw in _section(data, "paths").items():
        if not isinstance(raw, str):
            raise ConfigError(f"configs/default.yaml: paths.{key} must be a string")
        expanded, missing = expand_environment(raw, env)
        paths[key] = expanded
        unresolved += [m for m in missing if m not in unresolved]

    categories = data["categories"]
    if not isinstance(categories, list) or not categories:
        raise ConfigError("configs/default.yaml: 'categories' must be a non-empty list")
    bad_categories = [c for c in categories if c not in SUPPORTED_OBJECTS]
    if bad_categories:
        raise ConfigError(f"configs/default.yaml: unsupported categories {bad_categories}")

    resize_modes = _section(data, "resize_modes")
    if set(resize_modes) != set(SUPPORTED_OBJECTS) or not set(resize_modes.values()) <= set(
        RESIZE_MODES
    ):
        raise ConfigError(
            "configs/default.yaml: 'resize_modes' needs a valid mode for every supported object"
        )

    profiles = _section(data, "profiles")
    if set(profiles) != set(PROFILES):
        raise ConfigError(
            f"configs/default.yaml: 'profiles' must define exactly {sorted(PROFILES)}"
        )
    dev = profiles["dev"]
    if not isinstance(dev, dict):
        raise ConfigError("configs/default.yaml: profile 'dev' must be a mapping")
    profile_values = {
        "backbone": dev.get("backbone"),
        "layers": tuple(dev.get("layers", ())),
        "image_size": dev.get("image_size"),
    }

    thresholds = _section(data, "thresholds")
    evaluation = _section(data, "evaluation")
    splits = _build("splits", SplitConfig, **_section(data, "splits"))
    options = _build(
        "thresholds/evaluation",
        EvalOptions,
        threshold_policy=thresholds.get("policy"),
        quantile=thresholds.get("quantile"),
        split_seed=splits.seed,
        **evaluation,
    )

    patchcore_values = _section(data, "patchcore")
    max_fit_images = patchcore_values.pop("max_fit_images", None)
    if max_fit_images is not None and (
        isinstance(max_fit_images, bool)
        or not isinstance(max_fit_images, int)
        or max_fit_images < 1
    ):
        raise ConfigError("configs/default.yaml: patchcore.max_fit_images must be null or >= 1")

    app = _section(data, "app")
    for key in ("max_upload_mb", "min_side", "aspect_tolerance"):
        if not isinstance(app.get(key), int | float) or isinstance(app.get(key), bool):
            raise ConfigError(f"configs/default.yaml: app.{key} must be a number")

    return DefaultConfig(
        paths=paths,
        unresolved=tuple(unresolved),
        device=device,
        seed=seed,
        categories=tuple(categories),
        splits=splits,
        preprocessing=_build("preprocessing", PreprocessConfig, **_section(data, "preprocessing")),
        resize_modes={str(k): str(v) for k, v in resize_modes.items()},
        evaluation=options,
        classical=_build("classical", ClassicalConfig, **_section(data, "classical")),
        gaussian=_build("gaussian", GaussianConfig, **profile_values, **_section(data, "gaussian")),
        patchcore=_build("patchcore", PatchCoreConfig, **profile_values, **patchcore_values),
        max_fit_images=max_fit_images,
        app={k: float(v) for k, v in app.items()},
    )
