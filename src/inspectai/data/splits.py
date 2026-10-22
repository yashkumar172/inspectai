"""Reproducible fit / calibration_normal / test splits (SPEC.md Section 3.5).

Splits hold the relative image paths from the split file. The official `train` rows (normal
images only) are divided into `fit` and `calibration_normal`; every official `test` row is in
`test`. The three parts are disjoint, and that is enforced here.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from inspectai.config import SplitConfig
from inspectai.data.dataset import Sample
from inspectai.errors import SplitError

SCHEMA_VERSION = 1
_PARTS = ("fit", "calibration_normal", "test")


@dataclass(frozen=True)
class SplitResult:
    """The three splits of one object."""

    object_name: str
    seed: int
    calibration_fraction: float
    min_calibration: int
    fit: tuple[str, ...]
    calibration_normal: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        """Raise SplitError unless the parts are non-empty (fit, test), unique, and disjoint."""
        parts = {
            "fit": self.fit,
            "calibration_normal": self.calibration_normal,
            "test": self.test,
        }
        for name, files in parts.items():
            if len(set(files)) != len(files):
                raise SplitError(f"{self.object_name}: duplicate files in split '{name}'")
        if not self.fit:
            raise SplitError(f"{self.object_name}: the fit split is empty")
        if not self.test:
            raise SplitError(f"{self.object_name}: the test split is empty")
        fit, cal, test = set(self.fit), set(self.calibration_normal), set(self.test)
        for first, second, label in (
            (fit, cal, "fit and calibration_normal"),
            (fit, test, "fit and test"),
            (cal, test, "calibration_normal and test"),
        ):
            overlap = first & second
            if overlap:
                example = min(overlap)
                raise SplitError(
                    f"{self.object_name}: splits {label} overlap ({len(overlap)} files, "
                    f"for example {example})"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "object": self.object_name,
            "seed": self.seed,
            "calibration_fraction": self.calibration_fraction,
            "min_calibration": self.min_calibration,
            "counts": {
                "fit": len(self.fit),
                "calibration_normal": len(self.calibration_normal),
                "test": len(self.test),
            },
            "file_list_sha256": {
                "fit": file_list_sha256(self.fit),
                "calibration_normal": file_list_sha256(self.calibration_normal),
                "test": file_list_sha256(self.test),
            },
            "fit": list(self.fit),
            "calibration_normal": list(self.calibration_normal),
            "test": list(self.test),
        }


def file_list_sha256(files: Sequence[str]) -> str:
    """SHA-256 of the sorted file names joined by newlines."""
    joined = "\n".join(sorted(files))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def make_splits(
    samples: Sequence[Sample], object_name: str, config: SplitConfig | None = None
) -> SplitResult:
    """Build the splits of one object.

    The result depends only on the sample list, the object name, and `config`: the same inputs
    always give the same splits, and different seeds give different calibration sets. Objects
    are independent, so adding another object never changes this object's split.
    """
    config = config or SplitConfig()
    train = [s for s in samples if s.object_name == object_name and s.split == "train"]
    test = [s for s in samples if s.object_name == object_name and s.split == "test"]
    if not train:
        raise SplitError(f"{object_name}: no training rows in the split file")
    if not test:
        raise SplitError(f"{object_name}: no test rows in the split file")
    anomalous = [s.image_rel for s in train if s.is_anomalous]
    if anomalous:
        raise SplitError(f"{object_name}: training rows must be normal ({anomalous[0]})")

    train_files = sorted(s.image_rel for s in train)
    test_files = sorted(s.image_rel for s in test)
    if len(set(train_files)) != len(train_files) or len(set(test_files)) != len(test_files):
        raise SplitError(f"{object_name}: duplicate image paths in the split file")

    n_calibration = max(config.min_calibration, round(config.calibration_fraction * len(train)))
    if n_calibration >= len(train):
        raise SplitError(
            f"{object_name}: {len(train)} training images are too few to hold out "
            f"{n_calibration} for calibration"
        )
    rng = random.Random(f"{config.seed}:{object_name}")
    calibration = sorted(rng.sample(train_files, n_calibration))
    held_out = set(calibration)
    fit = [name for name in train_files if name not in held_out]

    result = SplitResult(
        object_name=object_name,
        seed=config.seed,
        calibration_fraction=config.calibration_fraction,
        min_calibration=config.min_calibration,
        fit=tuple(fit),
        calibration_normal=tuple(calibration),
        test=tuple(test_files),
    )
    result.validate()
    return result


def split_to_text(result: SplitResult) -> str:
    """The exact text that `save_split` writes."""
    result.validate()
    return json.dumps(result.to_dict(), indent=1) + "\n"


def save_split(result: SplitResult, path: Path) -> None:
    """Write `split.json`."""
    text = split_to_text(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_split(path: Path) -> SplitResult:
    """Read and verify `split.json`.

    Raises SplitError if the file is unreadable, has an unknown schema version, is missing
    fields, has file lists whose hashes do not match, or has overlapping parts.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SplitError(f"Split file could not be read: {path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise SplitError(f"Unsupported split file (schema_version must be {SCHEMA_VERSION})")
    try:
        parts = {name: tuple(data[name]) for name in _PARTS}
        hashes = data["file_list_sha256"]
        result = SplitResult(
            object_name=str(data["object"]),
            seed=int(data["seed"]),
            calibration_fraction=float(data["calibration_fraction"]),
            min_calibration=int(data["min_calibration"]),
            **parts,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"Split file is missing fields or has invalid values: {path}") from exc
    if not all(isinstance(name, str) for files in parts.values() for name in files):
        raise SplitError("Split file lists must contain file names (strings)")
    for name, files in parts.items():
        if not isinstance(hashes, dict) or hashes.get(name) != file_list_sha256(files):
            raise SplitError(f"Split file list '{name}' does not match its recorded SHA-256")
    result.validate()
    return result
