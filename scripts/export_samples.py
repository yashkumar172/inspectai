"""Copy the sample images of the hosted app from the local VisA dataset into `hosted/samples/`.

Usage (PowerShell):
    python scripts/export_samples.py
    python scripts/export_samples.py --categories candle cashew --force

Needs INSPECTAI_DATA_ROOT (the dataset) and the fitted Model C `dev` and classical artifacts
(INSPECTAI_ARTIFACT_DIR, or the folder under the data root): the images are chosen from the
`test` list stored in their `split.json`, so a `fit` or `calibration_normal` image can never be
picked. Per category: 2 normal images by seeded random choice (seed 42) and the anomalous images at
the 25th and 75th percentile of ground-truth mask area, chosen without looking at any model output,
plus the README's featured candle image (chosen by hand; the manifest says so). Files are copied
byte for byte. Writes `hosted/samples/<category>/...`, `manifest.json` and `ATTRIBUTION.md`.

Exit codes: 0 success; 2 configuration or data error; 1 any other failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from inspectai.config import load_paths
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.errors import ConfigError, InspectAIError
from inspectai.samples import SampleError, build_samples, samples_root, sha256_file

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2
SPLIT_SOURCES = ("patchcore", "classical")  # the models the hosted app serves

logger = logging.getLogger("export_samples")


def _test_list(artifact_dir: Path, category: str) -> tuple[list[str], str]:
    """The `test` list of the artifacts of every hosted model, which must agree."""
    lists: list[list[str]] = []
    hashes: list[str] = []
    for model in SPLIT_SOURCES:
        path = artifact_dir / model / category / "split.json"
        if not path.is_file():
            raise ConfigError(f"No fitted {model} artifact for {category}; fit it first.")
        split = json.loads(path.read_text(encoding="utf-8"))
        lists.append(list(split["test"]))
        hashes.append(str(split["file_list_sha256"]["test"]))
    if any(other != lists[0] for other in lists[1:]) or len(set(hashes)) != 1:
        raise ConfigError(f"The models of {category} were not fitted on the same test split.")
    return lists[0], hashes[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--categories", nargs="+", help="categories (default: all with artifacts)")
    parser.add_argument("--output", type=Path, default=None, help="samples folder")
    parser.add_argument("--force", action="store_true", help="replace existing samples")
    args = parser.parse_args(argv)
    try:
        paths = load_paths()
        chosen = tuple(args.categories) if args.categories else SUPPORTED_OBJECTS
        unknown = [c for c in chosen if c not in SUPPORTED_OBJECTS]
        if unknown:
            raise ConfigError(f"Unsupported categories: {unknown}")
        test_files: dict[str, list[str]] = {}
        test_hashes: dict[str, str] = {}
        for category in chosen:
            if not (paths.artifact_dir / SPLIT_SOURCES[0] / category).is_dir():
                if args.categories:
                    raise ConfigError(f"No fitted artifacts for {category}.")
                continue
            test_files[category], test_hashes[category] = _test_list(paths.artifact_dir, category)
        if not test_files:
            raise ConfigError("No fitted artifacts were found.")
        dataset = VisADataset.from_paths(paths)
        summary = build_samples(
            dataset,
            test_files,
            args.output or samples_root(),
            test_list_sha256=test_hashes,
            split_file_sha256=sha256_file(paths.split_csv),
            force=args.force,
        )
    except (ConfigError, SampleError) as exc:
        print(f"Configuration or data error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except InspectAIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(
        f"Wrote {len(summary.entries)} samples ({summary.files} files, "
        f"{summary.bytes / 1e6:.2f} MB) to {summary.destination}"
    )
    for category in SUPPORTED_OBJECTS:
        mine = [e for e in summary.entries if e.category == category]
        if mine:
            print(f"  {category}: " + ", ".join(Path(e.file).name for e in mine))
    return EXIT_OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
