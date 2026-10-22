"""Model artifact store (SPEC.md Section 9.1).

An artifact is a directory `<artifact_root>/<model>/<category>/` with four files:

- `manifest.json`: schema version, model type, category, model settings, tensor shapes and
  dtypes, threshold, library versions, creation time, and the SHA-256 of each data file.
- `tensors.safetensors`: the fitted parameters.
- `split.json`: the data split used (SPEC.md Section 3.5).
- `config.snapshot.yaml`: the exact configuration used.

Nothing is read with pickle. `load_artifact` verifies everything before returning and raises
`ArtifactError` on any mismatch. The manifest is written last, so an interrupted save leaves an
artifact that fails to load instead of a half-valid one.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from safetensors import SafetensorError
from safetensors import __version__ as safetensors_version
from safetensors.numpy import load_file, save_file

from inspectai.data.splits import SplitResult, load_split, save_split
from inspectai.errors import ArtifactError, SplitError

SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
TENSORS = "tensors.safetensors"
SPLIT = "split.json"
CONFIG = "config.snapshot.yaml"
DATA_FILES = (TENSORS, SPLIT, CONFIG)
ALLOWED_DTYPES = ("float32", "float16", "int64")
_FORBIDDEN_SUFFIXES = frozenset({".pkl", ".pickle", ".pt", ".pth", ".joblib", ".npy", ".npz"})
_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "model_type",
        "category",
        "created_utc",
        "library_versions",
        "tensors",
        "files",
    }
)


@dataclass(frozen=True)
class LoadedArtifact:
    """A verified artifact."""

    directory: Path
    manifest: dict[str, object]
    tensors: dict[str, NDArray]
    split: SplitResult
    config: dict[str, object]


def artifact_dir(root: Path, model: str, category: str) -> Path:
    """The directory of one fitted model: `<root>/<model>/<category>`."""
    return Path(root) / model / category


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "safetensors": safetensors_version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pyyaml": yaml.__version__,
    }


def save_artifact(
    directory: Path,
    model_type: str,
    category: str,
    tensors: dict[str, NDArray],
    split: SplitResult,
    config: dict[str, object],
    info: dict[str, object],
) -> Path:
    """Write the four artifact files. `info` entries are added to the top level of the manifest."""
    clash = _RESERVED_KEYS & set(info)
    if clash:
        raise ArtifactError(f"info uses reserved manifest keys: {sorted(clash)}")
    if split.object_name != category:
        raise ArtifactError(f"The split is for {split.object_name!r}, not {category!r}")
    if not tensors:
        raise ArtifactError("An artifact needs at least one tensor")
    for name, array in tensors.items():
        if str(array.dtype) not in ALLOWED_DTYPES:
            raise ArtifactError(f"Tensor {name!r} has unsupported dtype {array.dtype}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ArtifactError(f"Tensor {name!r} contains NaN or infinity")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST
    if manifest_path.exists():
        manifest_path.unlink()  # an artifact being rewritten is invalid until the new one is done
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(directory / TENSORS))
    save_split(split, directory / SPLIT)
    (directory / CONFIG).write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8", newline="\n"
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "category": category,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "library_versions": _library_versions(),
        "tensors": {
            name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in tensors.items()
        },
        "files": {name: _sha256(directory / name) for name in DATA_FILES},
        **info,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return directory


def load_artifact(
    directory: Path,
    expected_model_type: str | None = None,
    expected_category: str | None = None,
    expected_config: dict[str, object] | None = None,
) -> LoadedArtifact:
    """Load and verify an artifact.

    Checks: manifest readable and `schema_version` known; no pickle-style files; the data files
    listed and matching their SHA-256; tensors loadable as safetensors with the shapes and
    dtypes of the manifest and no NaN or infinity; `split.json` valid with no overlap between
    parts; and, if `expected_config` is given, a config snapshot equal to it (a different
    snapshot means the artifact is stale for the requested configuration).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ArtifactError(f"Artifact folder not found: {directory}")
    manifest = _read_manifest(directory)
    forbidden = sorted(
        p.name for p in directory.iterdir() if p.suffix.lower() in _FORBIDDEN_SUFFIXES
    )
    if forbidden:
        raise ArtifactError(f"Artifact contains files that are not allowed: {forbidden}")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(DATA_FILES):
        raise ArtifactError(f"Manifest must list exactly these data files: {list(DATA_FILES)}")
    for name in DATA_FILES:
        path = directory / name
        if not path.is_file():
            raise ArtifactError(f"Artifact file is missing: {name}")
        if _sha256(path) != files[name]:
            raise ArtifactError(f"Checksum mismatch for {name}; the artifact is corrupted")

    if expected_model_type is not None and manifest.get("model_type") != expected_model_type:
        raise ArtifactError(
            f"Artifact is for model {manifest.get('model_type')!r}, expected {expected_model_type!r}"
        )
    if expected_category is not None and manifest.get("category") != expected_category:
        raise ArtifactError(
            f"Artifact is for category {manifest.get('category')!r}, expected {expected_category!r}"
        )

    tensors = _load_tensors(directory / TENSORS, manifest)
    split = _load_split(directory / SPLIT, manifest)
    config = _load_config(directory / CONFIG)
    if expected_config is not None and config != expected_config:
        raise ArtifactError(
            "The artifact was fitted with a different configuration than requested (stale); "
            "fit the model again."
        )
    return LoadedArtifact(directory, manifest, tensors, split, config)


def _read_manifest(directory: Path) -> dict[str, object]:
    path = directory / MANIFEST
    if not path.is_file():
        raise ArtifactError(f"Artifact manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Artifact manifest is unreadable: {path}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("Artifact manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported artifact schema_version {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return manifest


def _load_tensors(path: Path, manifest: dict[str, object]) -> dict[str, NDArray]:
    try:
        tensors = load_file(str(path))
    except (SafetensorError, OSError) as exc:
        raise ArtifactError(f"{TENSORS} is not a valid safetensors file") from exc
    declared = manifest.get("tensors")
    if not isinstance(declared, dict) or set(declared) != set(tensors):
        raise ArtifactError("Tensor names in the file do not match the manifest")
    for name, array in tensors.items():
        entry = declared[name]
        if not isinstance(entry, dict) or list(array.shape) != entry.get("shape"):
            raise ArtifactError(f"Tensor {name!r} has shape {list(array.shape)}, manifest differs")
        if str(array.dtype) != entry.get("dtype") or str(array.dtype) not in ALLOWED_DTYPES:
            raise ArtifactError(f"Tensor {name!r} has dtype {array.dtype}, manifest differs")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ArtifactError(f"Tensor {name!r} contains NaN or infinity")
    return tensors


def _load_split(path: Path, manifest: dict[str, object]) -> SplitResult:
    try:
        split = load_split(path)  # also checks hashes and that the parts do not overlap
    except SplitError as exc:
        raise ArtifactError(f"{SPLIT} is invalid: {exc}") from exc
    if split.object_name != manifest.get("category"):
        raise ArtifactError(f"{SPLIT} is for {split.object_name!r}, manifest says otherwise")
    return split


def _load_config(path: Path) -> dict[str, object]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"{CONFIG} is unreadable") from exc
    if not isinstance(config, dict):
        raise ArtifactError(f"{CONFIG} must contain a mapping")
    return config
