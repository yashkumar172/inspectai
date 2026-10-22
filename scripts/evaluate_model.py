"""Evaluate a model on one VisA category with the shared harness.

Usage (PowerShell):
    python scripts/evaluate_model.py --model classical --category cashew
    python scripts/evaluate_model.py --model classical --category cashew --threshold-policy labeled_calibration

Until the artifact store exists (Milestone 3), the classical model is fitted inside this command
from the `fit` split only; test images are used only for scoring. Output goes to
<INSPECTAI_REPORT_DIR>/<model>/<category>/: predictions.csv, summary.json, and examples/*.png.

Exit codes: 0 success; 2 configuration or data error; 1 any other failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from inspectai.config import SplitConfig, load_paths, resize_mode_for
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.data.splits import make_splits
from inspectai.errors import (
    ConfigError,
    DatasetError,
    EvaluationError,
    ImageError,
    InspectAIError,
    ModelError,
    SplitError,
)
from inspectai.evaluation.report import write_predictions_csv, write_summary_json
from inspectai.evaluation.runner import EvalOptions, run_evaluation, save_examples
from inspectai.evaluation.thresholds import DEFAULT_QUANTILE, POLICIES
from inspectai.models.base import AnomalyModel
from inspectai.models.classical import ClassicalConfig, ClassicalModel

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DATA_ERROR = 2

MODELS = ("classical",)
logger = logging.getLogger("evaluate_model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model on one VisA category.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--category", choices=SUPPORTED_OBJECTS, required=True)
    parser.add_argument("--threshold-policy", choices=POLICIES, default=POLICIES[0])
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--seed", type=int, default=SplitConfig().seed, help="Split seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <INSPECTAI_REPORT_DIR>/<model>/<category>.",
    )
    return parser


def build_model(name: str, category: str) -> tuple[AnomalyModel, dict[str, object]]:
    if name == "classical":
        config = ClassicalConfig(resize_mode=resize_mode_for(category))
        return ClassicalModel(config), asdict(config)
    raise ConfigError(f"Unknown model {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ConfigError, DatasetError, SplitError, ImageError, EvaluationError, ModelError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except InspectAIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _run(args: argparse.Namespace) -> int:
    paths = load_paths()
    dataset = VisADataset.from_paths(paths)
    splits = make_splits(dataset.samples, args.category, SplitConfig(seed=args.seed))
    model, model_config = build_model(args.model, args.category)
    options = EvalOptions(
        threshold_policy=args.threshold_policy, quantile=args.quantile, split_seed=args.seed
    )
    result = run_evaluation(model, args.model, model_config, dataset, splits, options)

    out_dir = args.output_dir or paths.report_dir / args.model / args.category
    write_predictions_csv(out_dir / "predictions.csv", result.rows)
    write_summary_json(out_dir / "summary.json", result.summary)
    by_rel = {s.image_rel: s for s in dataset.get_samples(args.category)}
    vmax = max(row.score for row in result.rows)
    save_examples(model, dataset, by_rel, result.example_names, out_dir / "examples", vmax)

    metrics = result.summary["metrics"]
    assert isinstance(metrics, dict)
    image = metrics["image"]
    print(f"{args.model} / {args.category} ({metrics['evaluation_split']}): {image}")
    print(f"Report written to {out_dir}")
    return EXIT_OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
