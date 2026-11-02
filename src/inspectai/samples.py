"""Sample images bundled with the hosted app, so a visitor can try it without uploading anything.

`hosted/samples/` holds, for every category, 2 normal and 2 anomalous images of the official test
split, byte for byte copies of the original VisA files (never resized or re-encoded, so a model
scores them exactly as in the saved evaluation), the ground-truth masks of the anomalous ones, and
a `manifest.json` that records where every file came from, its SHA-256, and how it was chosen.

How the images are chosen (`scripts/export_samples.py`), without looking at any model output:

- normal images: a seeded random choice (seed 42) among the category's test normals;
- anomalous images: the test anomalies ranked by ground-truth mask area, taking the ones at the
  25th and the 75th percentile;
- one extra image is chosen by hand, the example shown in the README. The manifest says so.

Only images listed in the `test` list of the fitted artifacts are eligible: never a `fit` or
`calibration_normal` image. A file is only used after its SHA-256 has been checked against the
manifest, and file names from the manifest are never allowed to leave the samples folder.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from inspectai import config
from inspectai.data.dataset import SUPPORTED_OBJECTS, Sample, VisADataset
from inspectai.errors import InspectAIError
from inspectai.preprocessing.images import read_mask

SAMPLES_DIRNAME = "samples"
MANIFEST_NAME = "manifest.json"
ATTRIBUTION_NAME = "ATTRIBUTION.md"
SCHEMA_VERSION = 1
SEED = 42
NORMALS_PER_CATEGORY = 2
ANOMALY_QUANTILES = (0.25, 0.75)
LABEL_NORMAL = "normal"
LABEL_ANOMALY = "anomaly"
SPLIT = "test"

# The one image that is not chosen by the blind rule: the example shown at the top of the README.
FEATURED_CATEGORY = "candle"
FEATURED_SOURCE = "candle/Data/Images/Anomaly/001.JPG"
FEATURED_REASON = (
    "Chosen manually (not by the blind rule): the example shown at the top of the README."
)

SELECTION_RULE = (
    f"Per category, from the test list of the fitted artifacts only: {NORMALS_PER_CATEGORY} normal "
    f"images by seeded random choice (seed {SEED}), and the anomalous images at the 25th and 75th "
    "percentile of ground-truth mask area. No model output was used. One featured example was "
    "added by hand."
)

_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
_SHA = re.compile(r"^[0-9a-f]{64}$")

ATTRIBUTION_TEXT = """# Attribution and license of the sample images

The images in this folder are taken from the **VisA** dataset (Visual Anomaly), published by Amazon.

> Zou, Y., Jeong, J., Pemula, L., Zhang, D., and Dabeer, O. "SPot-the-Difference Self-Supervised
> Pre-training for Anomaly Detection and Segmentation." ECCV 2022 (arXiv:2207.14315).

- Source: <https://github.com/amazon-science/spot-diff>
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/>

The images and the ground-truth masks here are **unmodified copies** of the original files (not
resized or re-encoded); `manifest.json` records the original path and a SHA-256 for each. Heatmaps,
overlays and outlines that the app draws on top of them are **modified material** derived from the
original images. The project's own license does not cover these images.

How the images were chosen is described in `manifest.json` and in `scripts/export_samples.py`.
"""


class SampleError(InspectAIError):
    """The bundled samples are missing, malformed, or fail their integrity check."""


def samples_root(bundle_root: Path | None = None) -> Path:
    """The `samples` folder of the hosted bundle (looked up when called, so tests can redirect it)."""
    return (config.BUNDLE_ROOT if bundle_root is None else Path(bundle_root)) / SAMPLES_DIRNAME


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class SampleEntry:
    """One bundled image, as recorded in the manifest."""

    category: str
    label: str  # "normal" or "anomaly"
    file: str  # relative to the samples folder, for example "candle/anomaly_001.JPG"
    source: str  # the original's path relative to the VisA images folder
    split: str
    sha256: str
    mask_file: str | None
    mask_source: str | None
    mask_sha256: str | None
    featured: bool
    selection: Mapping[str, object]

    @property
    def is_anomalous(self) -> bool:
        return self.label == LABEL_ANOMALY

    @property
    def truth(self) -> str:
        return "anomalous" if self.is_anomalous else "normal"

    @property
    def title(self) -> str:
        """What the picker shows."""
        if self.featured:
            return f"Featured example ({self.truth})"
        method = self.selection.get("method")
        if method == "mask_area_quantile":
            size = "smaller" if float(self.selection.get("quantile", 0.5)) < 0.5 else "larger"  # type: ignore[arg-type]
            return f"Anomalous: {size} defect"
        return f"Normal image {int(self.selection.get('rank', 0)) + 1}"  # type: ignore[call-overload]

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "label": self.label,
            "file": self.file,
            "source": self.source,
            "split": self.split,
            "sha256": self.sha256,
            "mask_file": self.mask_file,
            "mask_source": self.mask_source,
            "mask_sha256": self.mask_sha256,
            "featured": self.featured,
            "selection": dict(self.selection),
        }


def _entry_from_dict(raw: object) -> SampleEntry:
    if not isinstance(raw, dict):
        raise SampleError("A sample entry is not an object.")
    try:
        entry = SampleEntry(
            category=str(raw["category"]),
            label=str(raw["label"]),
            file=str(raw["file"]),
            source=str(raw["source"]),
            split=str(raw["split"]),
            sha256=str(raw["sha256"]),
            mask_file=None if raw.get("mask_file") is None else str(raw["mask_file"]),
            mask_source=None if raw.get("mask_source") is None else str(raw["mask_source"]),
            mask_sha256=None if raw.get("mask_sha256") is None else str(raw["mask_sha256"]),
            featured=raw["featured"] is True,
            selection=dict(raw["selection"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SampleError("A sample entry is incomplete.") from exc
    _validate_entry(entry)
    return entry


def _safe_relative(value: str, category: str) -> None:
    parts = value.split("/")
    if (
        len(parts) != 2
        or parts[0] != category
        or not _NAME.match(parts[1])
        or parts[1] in {".", ".."}
    ):
        raise SampleError("A sample file name is not allowed.")


def _validate_entry(entry: SampleEntry) -> None:
    if entry.category not in SUPPORTED_OBJECTS:
        raise SampleError("A sample has an unknown category.")
    if entry.label not in (LABEL_NORMAL, LABEL_ANOMALY) or entry.split != SPLIT:
        raise SampleError("A sample has an invalid label or is not from the test split.")
    _safe_relative(entry.file, entry.category)
    if not _SHA.match(entry.sha256):
        raise SampleError("A sample has an invalid checksum.")
    has_mask = entry.mask_file is not None
    if has_mask != entry.is_anomalous:
        raise SampleError("Exactly the anomalous samples must have a ground-truth mask.")
    if has_mask:
        assert entry.mask_file is not None
        _safe_relative(entry.mask_file, entry.category)
        if entry.mask_sha256 is None or not _SHA.match(entry.mask_sha256):
            raise SampleError("A sample mask has an invalid checksum.")


@dataclass(frozen=True)
class SampleSet:
    """The bundled samples: the manifest, plus verified access to the files."""

    root: Path
    entries: tuple[SampleEntry, ...]
    seed: int

    def categories(self) -> list[str]:
        return [c for c in SUPPORTED_OBJECTS if any(e.category == c for e in self.entries)]

    def for_category(self, category: str) -> list[SampleEntry]:
        """The category's samples: the featured one first, then normals, then anomalies."""
        chosen = [e for e in self.entries if e.category == category]
        return sorted(chosen, key=lambda e: (not e.featured, e.is_anomalous, e.file))

    def featured(self) -> SampleEntry | None:
        return next((e for e in self.entries if e.featured), None)

    def find(self, category: str, source: str) -> SampleEntry | None:
        return next(
            (e for e in self.entries if e.category == category and e.source == source), None
        )

    def _read_verified(self, relative: str, expected: str) -> bytes:
        root = self.root.resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise SampleError("A sample file is missing.")
        data = path.read_bytes()
        if sha256_bytes(data) != expected:
            raise SampleError("A sample file failed its integrity check.")
        return data

    def read_image_bytes(self, entry: SampleEntry) -> bytes:
        """The original file's bytes, after the checksum has been verified."""
        return self._read_verified(entry.file, entry.sha256)

    def read_mask(self, entry: SampleEntry, image_hw: tuple[int, int]) -> NDArray[np.uint8] | None:
        """The binary ground-truth mask (0/1) of an anomalous sample, else None."""
        if entry.mask_file is None or entry.mask_sha256 is None:
            return None
        self._read_verified(entry.mask_file, entry.mask_sha256)
        return read_mask(self.root / entry.mask_file, image_hw)


def load_samples(root: Path) -> SampleSet | None:
    """The bundled samples, or None if there is no manifest (for example a local checkout).

    Raises SampleError if a manifest exists but is malformed. File checksums are verified when the
    files are read, not here.
    """
    manifest = Path(root) / MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        version, seed, items = raw["schema_version"], raw["seed"], raw["samples"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SampleError("The sample manifest could not be read.") from exc
    if version != SCHEMA_VERSION or not isinstance(items, list):
        raise SampleError("The sample manifest has an unsupported format.")
    entries = tuple(_entry_from_dict(item) for item in items)
    files = [e.file for e in entries] + [e.mask_file for e in entries if e.mask_file]
    if len(set(files)) != len(files):
        raise SampleError("The sample manifest lists a file twice.")
    return SampleSet(Path(root), entries, int(seed))


# --------------------------------------------------------------------------------- selection


def select_normals(
    candidates: Sequence[str], count: int = NORMALS_PER_CATEGORY, seed: int = SEED
) -> list[str]:
    """`count` of the candidates, by seeded random choice over the sorted list (sorted result)."""
    ordered = sorted(candidates)
    if len(ordered) < count:
        raise SampleError(f"Only {len(ordered)} normal test images; {count} are needed.")
    return sorted(random.Random(seed).sample(ordered, count))


def select_by_mask_area(
    areas: Mapping[str, int], quantiles: Sequence[float] = ANOMALY_QUANTILES
) -> list[tuple[str, float, int]]:
    """The images at the given quantiles of mask area: (image, quantile, rank among the sorted).

    Images are ranked by (area, path), and the one at rank `round(q * (n - 1))` is taken.
    """
    ranked = sorted(areas, key=lambda image: (areas[image], image))
    picks = [
        (ranked[round(q * (len(ranked) - 1))], q, round(q * (len(ranked) - 1))) for q in quantiles
    ]
    if len({image for image, _, _ in picks}) != len(picks):
        raise SampleError("Too few anomalous test images to pick distinct ones.")
    return picks


# --------------------------------------------------------------------------------- building


@dataclass(frozen=True)
class BuildSummary:
    destination: Path
    entries: tuple[SampleEntry, ...]
    bytes: int

    @property
    def files(self) -> int:
        return len(self.entries) + sum(1 for e in self.entries if e.mask_file)


def _copy(source: Path, target: Path) -> str:
    """Copy byte for byte and return the SHA-256 of the copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return sha256_file(target)


def build_samples(
    dataset: VisADataset,
    test_files: Mapping[str, Sequence[str]],
    destination: Path,
    *,
    test_list_sha256: Mapping[str, str] | None = None,
    split_file_sha256: str | None = None,
    force: bool = False,
) -> BuildSummary:
    """Choose and copy the samples of every category in `test_files` and write the manifest.

    `test_files` maps a category to the `test` list of its fitted artifacts (dataset-relative
    image paths); only those images are eligible. Nothing here reads a model output.
    """
    destination = Path(destination)
    if (destination / MANIFEST_NAME).exists():
        if not force:
            raise SampleError("The destination already holds samples; use --force to replace them.")
        shutil.rmtree(destination, ignore_errors=True)
    entries: list[SampleEntry] = []
    for category in SUPPORTED_OBJECTS:
        if category not in test_files:
            continue
        by_rel: dict[str, Sample] = {
            s.image_rel: s for s in dataset.get_samples(category, split=SPLIT)
        }
        listed = list(test_files[category])
        missing = [rel for rel in listed if rel not in by_rel]
        if missing:
            raise SampleError(f"{category}: {len(missing)} listed test images are not test images.")
        normals = [rel for rel in listed if not by_rel[rel].is_anomalous]
        anomalies = [rel for rel in listed if by_rel[rel].is_anomalous]
        areas: dict[str, int] = {}
        for rel in anomalies:
            image = dataset.load_image(by_rel[rel])
            areas[rel] = int(dataset.load_mask(by_rel[rel], image.shape[:2]).sum())

        chosen: dict[str, dict[str, object]] = {}
        for rank, rel in enumerate(select_normals(normals)):
            chosen[rel] = {
                "method": "seeded_random",
                "seed": SEED,
                "rank": rank,
                "pool": len(normals),
            }
        for rel, quantile, rank in select_by_mask_area(areas):
            chosen[rel] = {
                "method": "mask_area_quantile",
                "quantile": quantile,
                "rank_in_area_order": rank,
                "pool": len(areas),
                "mask_area_pixels": areas[rel],
            }
        featured: str | None = None
        if category == FEATURED_CATEGORY and FEATURED_SOURCE in by_rel:
            if FEATURED_SOURCE not in listed:
                raise SampleError("The featured example is not in the test list of the artifacts.")
            featured = FEATURED_SOURCE
            blind = chosen.get(featured)
            chosen[featured] = {
                "method": "manual",
                "reason": FEATURED_REASON,
                "also_chosen_by_blind_rule": blind is not None,
                **({"blind_rule": blind} if blind else {}),
            }
        for rel in sorted(chosen, key=lambda r: (by_rel[r].is_anomalous, r)):
            entries.append(
                _copy_entry(by_rel[rel], rel == featured, chosen[rel], destination, dataset)
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "selection_rule": SELECTION_RULE,
        "source": "VisA (Zou et al., ECCV 2022), CC BY 4.0; see ATTRIBUTION.md",
        "split_file_sha256": split_file_sha256,
        "test_list_sha256": dict(test_list_sha256 or {}),
        "samples": [e.to_dict() for e in entries],
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (destination / ATTRIBUTION_NAME).write_text(ATTRIBUTION_TEXT, encoding="utf-8")
    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    return BuildSummary(destination, tuple(entries), total)


def _copy_entry(
    sample: Sample,
    featured: bool,
    selection: Mapping[str, object],
    destination: Path,
    dataset: VisADataset,
) -> SampleEntry:
    label = LABEL_ANOMALY if sample.is_anomalous else LABEL_NORMAL
    stem = Path(sample.image_rel).stem
    suffix = Path(sample.image_rel).suffix
    file = f"{sample.object_name}/{label}_{stem}{suffix}"
    digest = _copy(sample.image_path, destination / file)
    mask_file = mask_source = mask_digest = None
    if sample.is_anomalous:
        if sample.mask_path is None or sample.mask_rel is None:
            raise SampleError(f"{sample.image_rel} has no mask.")
        if sample.mask_path.suffix.lower() != ".png":
            raise SampleError(f"The mask of {sample.image_rel} is not a PNG file.")
        mask_file = f"{sample.object_name}/{label}_{stem}_mask.png"
        mask_source = sample.mask_rel
        mask_digest = _copy(sample.mask_path, destination / mask_file)
        image = dataset.load_image(sample)
        read_mask(destination / mask_file, image.shape[:2])  # decodes, same size as the image
    return SampleEntry(
        category=sample.object_name,
        label=label,
        file=file,
        source=sample.image_rel,
        split=SPLIT,
        sha256=digest,
        mask_file=mask_file,
        mask_source=mask_source,
        mask_sha256=mask_digest,
        featured=featured,
        selection=dict(selection),
    )
