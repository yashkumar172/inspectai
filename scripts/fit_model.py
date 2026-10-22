"""Fit a model on one VisA category and save it as an artifact.

Usage (PowerShell):
    python scripts/fit_model.py --model gaussian --category cashew --profile dev

The model is fitted on the `fit` split only. Thresholds are then selected from the
`calibration_normal` split (SPEC.md Section 3.5) and stored in the artifact, together with the
split, the configuration, and the measured fitting time and peak memory. The artifact goes to
<INSPECTAI_ARTIFACT_DIR>/<model>/<category>/ and replaces any earlier one.

Exit codes: 0 success; 2 configuration or data error; 3 artifact error;
4 pretrained weights unavailable; 1 any other failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from inspectai.artifacts.gaussian_io import save_gaussian
from inspectai.artifacts.store import artifact_dir
from inspectai.config import PROFILES, SplitConfig, load_paths
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.data.splits import make_splits
from inspectai.errors import (
    ArtifactError,
    ConfigError,
    DatasetError,
    EvaluationError,
    ImageError,
    InspectAIError,
    ModelError,
    SplitError,
    WeightsNotFoundError,
)
from inspectai.evaluation.runner import EvalOptions, calibrate, environment_info, fit_model
from inspectai.evaluation.thresholds import DEFAULT_QUANTILE
from inspectai.models.features import FeatureExtractor, load_backbone
from inspectai.models.gaussian import GaussianModel, gaussian_config_for

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DATA_ERROR = 2
EXIT_ARTIFACT_ERROR = 3
EXIT_WEIGHTS_UNAVAILABLE = 4

MODELS = ("gaussian",)
logger = logging.getLogger("fit_model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a model and save its artifact.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--category", choices=SUPPORTED_OBJECTS, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="dev")
    parser.add_argument("--feature-dim", type=int, default=100, help="Channels kept (d).")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--seed", type=int, default=SplitConfig().seed)
    parser.add_argument(
        "--output-dir", type=Path, help="Default: <ARTIFACT_DIR>/<model>/<category>."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except WeightsNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_WEIGHTS_UNAVAILABLE
    except ArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ARTIFACT_ERROR
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
    config = gaussian_config_for(args.category, args.profile, args.feature_dim, args.seed)
    backbone, weights = load_backbone(config.backbone)
    model = GaussianModel(config, FeatureExtractor(backbone, config.layers), weights)

    stats = fit_model(model, dataset, splits)
    options = EvalOptions(quantile=args.quantile, split_seed=args.seed)
    calibration = calibrate(model, dataset, splits, options)

    target = args.output_dir or artifact_dir(paths.artifact_dir, args.model, args.category)
    if (target / "manifest.json").exists():
        print(f"Replacing the existing artifact in {target}")
    save_gaussian(
        target,
        model,
        args.category,
        splits,
        calibration.to_dict(),
        {**stats.to_dict(), "environment": environment_info()},
        options.eval_resolution,
    )
    print(
        f"Fitted {args.model} / {args.category} ({args.profile}) on {stats.n_fit_images} images "
        f"in {stats.fit_seconds:.1f} s, peak memory {stats.peak_memory_gb:.2f} GB"
    )
    print(
        f"Image threshold {calibration.image_threshold.value:.4f} "
        f"(calibration scores {calibration.scores['min']:.3f} / {calibration.scores['median']:.3f} "
        f"/ {calibration.scores['max']:.3f} min / median / max)"
    )
    print(f"Artifact written to {target}")
    return EXIT_OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
