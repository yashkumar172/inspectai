"""Evaluate a model on one VisA category with the shared harness.

Usage (PowerShell):
    python scripts/evaluate_model.py --model classical --category cashew
    python scripts/evaluate_model.py --model gaussian --category cashew --profile dev
    python scripts/evaluate_model.py --model patchcore --category cashew --profile dev
    python scripts/evaluate_model.py --model gaussian --category cashew --threshold-policy labeled_calibration

`gaussian` and `patchcore` load the artifact written by scripts/fit_model.py (after integrity
checks) and never refit; their split and thresholds come from the artifact, and an artifact
fitted with other settings than requested here is rejected as stale. `classical` has no
artifact yet and is fitted inside this command from the `fit` split only. Test images are
used only for scoring.
Output goes to <INSPECTAI_REPORT_DIR>/<model>/<category>/: predictions.csv, summary.json, and
examples/*.png.

Exit codes: 0 success; 2 configuration or data error; 3 artifact error;
4 pretrained weights unavailable; 1 any other failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from inspectai.artifacts.gaussian_io import load_gaussian
from inspectai.artifacts.patchcore_io import load_patchcore
from inspectai.artifacts.store import artifact_dir
from inspectai.config import PROFILES, SplitConfig, load_paths, resize_mode_for
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.data.splits import make_splits, with_max_fit_images
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
from inspectai.evaluation.report import write_predictions_csv, write_summary_json
from inspectai.evaluation.runner import (
    Calibration,
    EvalOptions,
    EvaluationResult,
    evaluate_fitted,
    run_evaluation,
    save_examples,
)
from inspectai.evaluation.thresholds import DEFAULT_QUANTILE, POLICIES
from inspectai.models.base import AnomalyModel
from inspectai.models.classical import ClassicalConfig, ClassicalModel
from inspectai.models.gaussian import gaussian_config_for
from inspectai.models.patchcore import patchcore_config_for

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DATA_ERROR = 2
EXIT_ARTIFACT_ERROR = 3
EXIT_WEIGHTS_UNAVAILABLE = 4

MODELS = ("classical", "gaussian", "patchcore")
logger = logging.getLogger("evaluate_model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model on one VisA category.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--category", choices=SUPPORTED_OBJECTS, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="dev", help="gaussian only.")
    parser.add_argument(
        "--feature-dim", type=int, default=100, help="gaussian only; must match the fit."
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=0.10, help="patchcore only; must match the fit."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16384, help="patchcore only; search block size."
    )
    parser.add_argument("--threshold-policy", choices=POLICIES, default=POLICIES[0])
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--seed", type=int, default=SplitConfig().seed, help="Split seed.")
    parser.add_argument("--artifact-dir", type=Path, help="gaussian and patchcore.")
    parser.add_argument(
        "--max-fit-images",
        type=int,
        help="classical only: fit on the same seeded subset as the other models of this category "
        "(gaussian and patchcore use the subset recorded in their artifact).",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Default: <INSPECTAI_REPORT_DIR>/<model>/<category>."
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


def _evaluate(
    args: argparse.Namespace, dataset: VisADataset, paths_artifact_dir: Path
) -> tuple[AnomalyModel, EvaluationResult]:
    options = EvalOptions(
        threshold_policy=args.threshold_policy, quantile=args.quantile, split_seed=args.seed
    )
    if args.max_fit_images is not None and args.model != "classical":
        raise ConfigError(
            "--max-fit-images applies to the classical model only; gaussian and patchcore "
            "evaluate the subset recorded in their artifact (set it with fit_model.py)"
        )
    if args.model == "classical":
        splits = make_splits(dataset.samples, args.category, SplitConfig(seed=args.seed))
        if args.max_fit_images is not None:
            splits = with_max_fit_images(splits, args.max_fit_images)
        config = ClassicalConfig(resize_mode=resize_mode_for(args.category))
        model = ClassicalModel(config)
        return model, run_evaluation(model, "classical", asdict(config), dataset, splits, options)

    directory = args.artifact_dir or artifact_dir(paths_artifact_dir, args.model, args.category)
    if args.model == "gaussian":
        requested = gaussian_config_for(args.category, args.profile, args.feature_dim, args.seed)
        model, artifact = load_gaussian(directory, args.category, requested)
    else:
        requested = patchcore_config_for(args.category, args.profile, args.coreset_ratio, args.seed)
        model, artifact = load_patchcore(
            directory, args.category, requested, runtime={"chunk_size": args.chunk_size}
        )
    calibration = Calibration.from_dict(artifact.manifest["calibration"])  # type: ignore[arg-type]
    stored_resolution = artifact.manifest.get("eval_resolution")
    if not isinstance(stored_resolution, int):
        raise ArtifactError("Manifest has no eval_resolution")
    options = replace(options, eval_resolution=stored_resolution)  # must match the calibration
    fit_stats = artifact.manifest.get("fit_stats")
    result = evaluate_fitted(
        model,
        args.model,
        requested.to_dict(),
        dataset,
        artifact.split,
        options,
        calibration,
        fit_stats if isinstance(fit_stats, dict) else None,
    )
    return model, result


def _run(args: argparse.Namespace) -> int:
    paths = load_paths()
    dataset = VisADataset.from_paths(paths)
    model, result = _evaluate(args, dataset, paths.artifact_dir)

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
