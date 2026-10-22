"""Dataset validation and per-object image statistics (SPEC.md Sections 3.3 and 6).

`validate_dataset` checks every row of the split file against the files on disk. It reports
problems grouped by kind instead of stopping at the first one, and it measures image sizes and
aspect ratios so each category's resize mode can be chosen from data.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from inspectai.config import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MAX_SIDE,
    STRETCH_MAX_ASPECT_DEVIATION,
)
from inspectai.data.dataset import Sample
from inspectai.errors import DatasetError, ImageError
from inspectai.preprocessing.images import probe_image_file

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
_MAX_EXAMPLES = 5
_SINGLE_CHANNEL_MODES = frozenset({"1", "L", "P", "I", "I;16", "F"})


@dataclass
class IssueGroup:
    """All problems of one kind."""

    severity: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, example: str) -> None:
        self.count += 1
        if len(self.examples) < _MAX_EXAMPLES:
            self.examples.append(example)


@dataclass
class ObjectReport:
    """Counts and image-size statistics of one object."""

    object_name: str
    counts: dict[str, int] = field(default_factory=dict)
    measured_images: int = 0
    width_min: int = 0
    width_median: float = 0.0
    width_max: int = 0
    height_min: int = 0
    height_median: float = 0.0
    height_max: int = 0
    aspect_min: float = 0.0
    aspect_median: float = 0.0
    aspect_max: float = 0.0
    top_sizes: list[tuple[int, int, int]] = field(default_factory=list)
    recommended_resize_mode: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": self.counts,
            "measured_images": self.measured_images,
            "width": {"min": self.width_min, "median": self.width_median, "max": self.width_max},
            "height": {
                "min": self.height_min,
                "median": self.height_median,
                "max": self.height_max,
            },
            "aspect_ratio": {
                "min": round(self.aspect_min, 4),
                "median": round(self.aspect_median, 4),
                "max": round(self.aspect_max, 4),
            },
            "top_sizes": [{"width": w, "height": h, "count": n} for w, h, n in self.top_sizes],
            "recommended_resize_mode": self.recommended_resize_mode,
        }


@dataclass
class ValidationReport:
    """The result of `validate_dataset`."""

    rows_checked: int = 0
    objects: dict[str, ObjectReport] = field(default_factory=dict)
    issues: dict[str, IssueGroup] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(g.count for g in self.issues.values() if g.severity == ERROR)

    @property
    def warning_count(self) -> int:
        return sum(g.count for g in self.issues.values() if g.severity == WARNING)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def add_issue(self, code: str, severity: str, example: str) -> None:
        self.issues.setdefault(code, IssueGroup(severity)).add(example)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_checked": self.rows_checked,
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "objects": {name: rep.to_dict() for name, rep in sorted(self.objects.items())},
            "issues": {
                code: {"severity": g.severity, "count": g.count, "examples": g.examples}
                for code, g in sorted(self.issues.items())
            },
        }


def recommend_resize_mode(median_aspect: float) -> str:
    """`stretch` if the median aspect ratio is within 5% of square, otherwise `letterbox`."""
    # The small epsilon keeps values exactly at the limit (such as 1.05) on the "stretch" side
    # despite floating-point rounding.
    if abs(median_aspect - 1.0) <= STRETCH_MAX_ASPECT_DEVIATION + 1e-9:
        return "stretch"
    return "letterbox"


def validate_dataset(
    samples: Sequence[Sample],
    images_root: Path,
    objects: Sequence[str] | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> ValidationReport:
    """Check every row against the files on disk and measure image sizes.

    Checks: the image folder exists; each object's folder exists; every image exists, is
    readable, and is within the size limits (`max_side` per side, `max_pixels` in total); no image appears twice; anomalous rows have a
    readable single-channel mask with the image's height and width and at least one nonzero
    pixel (an empty mask is only a warning); normal rows have no mask (SPEC.md Section 3.3). Counts per object, split, and label are reported without
    assuming any expected numbers.
    """
    images_root = Path(images_root)
    if not images_root.is_dir():
        raise DatasetError(
            f"Image folder not found: {images_root}. Extract the VisA archive so that the object "
            "folders are under <INSPECTAI_DATA_ROOT>\\visa\\VisA\\ (see SPEC.md Section 3.1)."
        )
    wanted = set(objects) if objects else None
    chosen = [s for s in samples if wanted is None or s.object_name in wanted]
    if wanted is not None:
        unknown = wanted - {s.object_name for s in samples}
        if unknown:
            raise DatasetError(f"Objects not in the split file: {', '.join(sorted(unknown))}")

    report = ValidationReport(rows_checked=len(chosen))
    sizes: dict[str, list[tuple[int, int]]] = {}
    missing_folders: set[str] = set()
    seen: Counter[str] = Counter()

    for index, sample in enumerate(chosen, start=1):
        obj = report.objects.setdefault(sample.object_name, ObjectReport(sample.object_name))
        key = f"{sample.split}/{sample.label}"
        obj.counts[key] = obj.counts.get(key, 0) + 1
        seen[sample.image_rel] += 1

        if (
            sample.object_name not in missing_folders
            and not (images_root / sample.object_name).is_dir()
        ):
            missing_folders.add(sample.object_name)
            report.add_issue("missing_object_folder", ERROR, sample.object_name)
        if sample.object_name in missing_folders:
            continue

        size = _check_image(sample, max_pixels, max_side, report)
        if size is not None:
            sizes.setdefault(sample.object_name, []).append(size)
        _check_mask(sample, size, report)
        if index % 2000 == 0:
            logger.info("Checked %d / %d rows", index, len(chosen))

    for rel, times in seen.items():
        if times > 1:
            report.add_issue("duplicate_image", ERROR, f"{rel} (x{times})")
    for name, measured in sizes.items():
        _fill_size_stats(report.objects[name], measured)
    return report


def _check_image(
    sample: Sample, max_pixels: int, max_side: int, report: ValidationReport
) -> tuple[int, int] | None:
    if not sample.image_path.is_file():
        report.add_issue("missing_image", ERROR, sample.image_rel)
        return None
    try:
        return probe_image_file(sample.image_path, max_pixels, max_side)
    except ImageError as exc:
        report.add_issue("unreadable_image", ERROR, f"{sample.image_rel}: {exc}")
        return None


def _check_mask(sample: Sample, size: tuple[int, int] | None, report: ValidationReport) -> None:
    if sample.mask_path is None:
        if sample.is_anomalous:
            report.add_issue("anomaly_without_mask", ERROR, sample.image_rel)
        return
    if not sample.is_anomalous:
        report.add_issue("normal_row_has_mask", ERROR, sample.image_rel)
    if sample.mask_rel is None or not sample.mask_path.is_file():
        report.add_issue("missing_mask", ERROR, sample.mask_rel or sample.image_rel)
        return
    try:
        with Image.open(sample.mask_path) as handle:
            if handle.mode not in _SINGLE_CHANNEL_MODES:
                report.add_issue("multichannel_mask", ERROR, f"{sample.mask_rel} ({handle.mode})")
                return
            values = np.asarray(handle)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        report.add_issue("unreadable_mask", ERROR, sample.mask_rel or sample.image_rel)
        return
    if size is not None and (values.shape[1], values.shape[0]) != size:
        report.add_issue(
            "mask_size_mismatch",
            ERROR,
            f"{sample.mask_rel}: mask {values.shape[1]}x{values.shape[0]}, image {size[0]}x{size[1]}",
        )
    if sample.is_anomalous and not values.any():
        report.add_issue("empty_mask", WARNING, sample.mask_rel or sample.image_rel)


def _fill_size_stats(obj: ObjectReport, measured: list[tuple[int, int]]) -> None:
    widths = [w for w, _ in measured]
    heights = [h for _, h in measured]
    aspects = [w / h for w, h in measured]
    obj.measured_images = len(measured)
    obj.width_min, obj.width_max = min(widths), max(widths)
    obj.height_min, obj.height_max = min(heights), max(heights)
    obj.width_median = float(statistics.median(widths))
    obj.height_median = float(statistics.median(heights))
    obj.aspect_min, obj.aspect_max = min(aspects), max(aspects)
    obj.aspect_median = float(statistics.median(aspects))
    obj.top_sizes = [(w, h, n) for (w, h), n in Counter(measured).most_common(3)]
    obj.recommended_resize_mode = recommend_resize_mode(obj.aspect_median)


def format_report(report: ValidationReport) -> str:
    """Plain-text summary for the command line."""
    lines = [f"Rows checked: {report.rows_checked}", ""]
    header = f"{'object':<12}{'train/normal':>13}{'test/normal':>12}{'test/anomaly':>13}"
    lines.append(header + "   median WxH      aspect (min-med-max)      resize")
    for name, obj in sorted(report.objects.items()):
        c = obj.counts
        row = (
            f"{name:<12}{c.get('train/normal', 0):>13}{c.get('test/normal', 0):>12}"
            f"{c.get('test/anomaly', 0):>13}"
        )
        if obj.measured_images:
            row += (
                f"   {obj.width_median:.0f}x{obj.height_median:.0f}"
                f"   {obj.aspect_min:.2f}-{obj.aspect_median:.2f}-{obj.aspect_max:.2f}"
                f"   {obj.recommended_resize_mode}"
            )
        lines.append(row)
    lines.append("")
    if not report.issues:
        lines.append("No problems found.")
    for code, group in sorted(report.issues.items()):
        lines.append(f"[{group.severity.upper()}] {code}: {group.count}")
        lines.extend(f"    {example}" for example in group.examples)
    lines.append("")
    lines.append(f"Errors: {report.error_count}  Warnings: {report.warning_count}")
    return "\n".join(lines)
