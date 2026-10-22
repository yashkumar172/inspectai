"""Validate the VisA dataset and optionally write the per-object splits.

Usage (PowerShell):
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --objects cashew pipe_fryum --write-splits

Exit codes: 0 success; 2 configuration or data error; 1 any other failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from inspectai.config import Paths, SplitConfig, load_paths
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.data.splits import make_splits, save_split, split_to_text
from inspectai.data.validation import format_report, validate_dataset
from inspectai.errors import ConfigError, DatasetError, InspectAIError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DATA_ERROR = 2

logger = logging.getLogger("validate_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the VisA dataset (SPEC.md Section 3.3).")
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=SUPPORTED_OBJECTS,
        metavar="OBJECT",
        help="Validate only these objects (default: all objects in the split file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the JSON report (default: <INSPECTAI_REPORT_DIR>/visa_validation.json).",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="Write split.json for each validated object to <INSPECTAI_ARTIFACT_DIR>/splits/.",
    )
    parser.add_argument("--seed", type=int, default=SplitConfig().seed, help="Split seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = load_paths()
        return _run(args, paths)
    except (ConfigError, DatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except InspectAIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _run(args: argparse.Namespace, paths: Paths) -> int:
    dataset = VisADataset.from_paths(paths)
    report = validate_dataset(dataset.samples, paths.visa_images_root, objects=args.objects)
    print(format_report(report))

    output = args.output or paths.report_dir / "visa_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Report written to {output}")

    if not report.ok:
        print("Validation failed; splits were not written.", file=sys.stderr)
        return EXIT_DATA_ERROR
    if args.write_splits:
        _write_splits(
            dataset, args.objects or dataset.objects(), paths, SplitConfig(seed=args.seed)
        )
    return EXIT_OK


def _write_splits(
    dataset: VisADataset, objects: Sequence[str], paths: Paths, config: SplitConfig
) -> None:
    for name in objects:
        result = make_splits(dataset.samples, name, config)
        target = paths.artifact_dir / "splits" / name / "split.json"
        if target.exists():
            if target.read_text(encoding="utf-8") != split_to_text(result):
                raise DatasetError(
                    f"{target} exists and differs from the split for seed {config.seed}; "
                    "delete it first if you really want a new split."
                )
            print(f"Split for {name} unchanged: {target}")
            continue
        save_split(result, target)
        print(
            f"Split for {name}: fit={len(result.fit)} calibration={len(result.calibration_normal)} "
            f"test={len(result.test)} -> {target}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
