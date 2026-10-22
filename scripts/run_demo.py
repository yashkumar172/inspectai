"""End-to-end smoke test on synthetic data. No dataset, no pretrained weights, no network.

Usage (PowerShell, from a repository checkout):
    python scripts/run_demo.py
    python scripts/run_demo.py --output-dir C:\\temp\\inspectai-demo

The script walks the whole pipeline once:

1. generates a small VisA-style synthetic dataset (textured normal images, anomalous images with
   a drawn defect, masks, and a `1cls.csv` split file);
2. fits the classical baseline, Model B, and Model C (with a deterministic fake backbone instead
   of pretrained weights), chooses thresholds on held-out normal images, evaluates on the test
   split, and writes the CSV predictions, JSON summaries, and saved artifacts;
3. reloads every artifact through the integrity checks, predicts every test image the way the
   app does, and checks that each score and decision equals the saved evaluation report;
4. builds the comparison report and writes a visualization and prediction details.

The metrics printed here come from random synthetic images and say nothing about real
performance. The real results are in the README.

Exit codes: 0 success; 2 the repository's test fixtures are missing; 1 any other failure.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_NO_FIXTURES = 2

logger = logging.getLogger("run_demo")


class DemoError(Exception):
    """A step of the end-to-end path did not produce what it should."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end smoke test on synthetic data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep everything in this (new or empty) folder. Default: a temporary folder that "
        "is deleted at the end.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the temporary folder.")
    return parser


def _verify_saved_models(world, category: str) -> int:  # type: ignore[no-untyped-def]
    """Reload each artifact and check every test image against the saved evaluation report."""
    from fake_backbone import FakeBackbone

    from inspectai.data.dataset import VisADataset
    from inspectai.inference.predictor import (
        STATUS_ANOMALY,
        STATUS_NORMAL,
        available_options,
        inspect_image,
        load_predictor,
    )

    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    by_rel = {s.image_rel: s for s in dataset.get_samples(category)}
    verified = 0
    options = available_options(world.artifacts)
    if len(options) != 3:
        raise DemoError(f"Expected 3 fitted models, found {len(options)}")
    for option in options:
        predictor = load_predictor(world.artifacts, option, category, backbone=FakeBackbone())
        path = world.reports / option.key / category / "predictions.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            result = inspect_image(predictor, dataset.load_image(by_rel[row["image"]]))
            expected = STATUS_ANOMALY if int(row["predicted"]) else STATUS_NORMAL
            if abs(result.score - float(row["score"])) > 1e-6 or result.status != expected:
                raise DemoError(f"{option.label}: {row['image']} differs from the saved report")
            verified += 1
    return verified


def _write_outputs(world, category: str, out_dir: Path) -> dict[str, float | None]:  # type: ignore[no-untyped-def]
    """Comparison report, a visualization PNG, and prediction details."""
    from fake_backbone import FakeBackbone

    from inspectai.data.dataset import VisADataset
    from inspectai.evaluation.comparison import (
        RUN_SPECS,
        build_comparison,
        format_markdown,
        load_run,
        write_comparison_csv,
        write_comparison_json,
    )
    from inspectai.inference.predictor import (
        details_json,
        inspect_image,
        load_predictor,
        model_options,
        panel_png,
    )

    runs = [load_run(world.reports, world.artifacts, spec, category) for spec in RUN_SPECS]
    found = [run for run in runs if run is not None]
    comparison = build_comparison({category: found}, {category: []})
    write_comparison_json(out_dir / "comparison.json", comparison)
    write_comparison_csv(out_dir / "comparison.csv", comparison)
    (out_dir / "comparison.md").write_text(format_markdown(comparison), encoding="utf-8")

    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    sample = dataset.get_samples(category, "test", "anomaly")[0]
    image = dataset.load_image(sample)
    predictor = load_predictor(
        world.artifacts, next(o for o in model_options() if o.key == "gaussian"), category,
        backbone=FakeBackbone(),
    )  # fmt: skip
    inspection = inspect_image(predictor, image)
    (out_dir / "visualization.png").write_bytes(panel_png(image, inspection))
    (out_dir / "prediction_details.json").write_text(
        details_json(predictor, inspection, image.shape[:2], "synthetic demo image"),
        encoding="utf-8",
    )
    return {
        run["label"]: run["metrics"]["image"]["auroc"]
        for run in comparison["categories"][category]["runs"]
    }


def run(out_dir: Path) -> None:
    sys.path.insert(0, str(FIXTURES))
    from make_world import build_world

    category = "cashew"
    print("1/4 generating synthetic data and fitting classical, Model B, and Model C ...")
    world = build_world(out_dir / "world", categories=(category,))
    print("2/4 reloading the saved artifacts and predicting every test image ...")
    verified = _verify_saved_models(world, category)
    print(f"    {verified} predictions equal the saved evaluation reports")
    print("3/4 writing the comparison report, visualization, and prediction details ...")
    aurocs = _write_outputs(world, category, out_dir)
    print("4/4 done. Image AUROC on the SYNTHETIC test images (meaningless for real performance):")
    for label, value in aurocs.items():
        print(f"    {label:28s} {'undefined' if value is None else f'{value:.3f}'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not FIXTURES.is_dir():
        print(
            "ERROR: tests/fixtures was not found. Run this script from a repository checkout.",
            file=sys.stderr,
        )
        return EXIT_NO_FIXTURES
    explicit = args.output_dir is not None
    if explicit and args.output_dir.exists() and any(args.output_dir.iterdir()):
        print("ERROR: --output-dir must be new or empty.", file=sys.stderr)
        return EXIT_FAILURE
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="inspectai-demo-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(out_dir)
    except (DemoError, OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("The demo failed")
        print(f"ERROR: the demo failed: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        if not explicit and not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)
    if explicit or args.keep:
        print(f"Outputs kept in {out_dir}")
    return EXIT_OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
