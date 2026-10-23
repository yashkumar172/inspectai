"""Compare all evaluated models per category and write the comparison report.

Usage (PowerShell):
    python scripts/compare_models.py
    python scripts/compare_models.py --categories cashew pipe_fryum --no-figures

Reads the results written by scripts/evaluate_model.py under <INSPECTAI_REPORT_DIR>
(classical, gaussian, gaussian-full, patchcore, patchcore-full, each with a sub-folder per
category) and writes to <INSPECTAI_REPORT_DIR>/comparison/:

- comparison.json: the full comparison with the configuration metadata of every run
- comparison.csv: one flat row per category and model
- comparison.md: per-category tables, paired AUROC differences, and macro averages
- <category>_examples.png: the same randomly chosen test images, every model, with ground truth

The comparison is refused (exit code 2) if the runs of a category did not use identical splits
and evaluation settings, or if a summary does not follow from its saved predictions. Models
that were not run for a category are listed as not measured, never filled in.

Exit codes: 0 success; 2 configuration, data, or comparability error; 3 artifact error;
4 pretrained weights unavailable; 1 any other failure.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from inspectai.artifacts.gaussian_io import load_gaussian
from inspectai.artifacts.patchcore_io import load_patchcore
from inspectai.artifacts.store import artifact_dir
from inspectai.config import Paths, SplitConfig, load_paths, resize_mode_for
from inspectai.data.dataset import Sample, VisADataset
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
from inspectai.evaluation.comparison import (
    RUN_SPECS,
    Run,
    RunSpec,
    build_comparison,
    format_markdown,
    load_run,
    write_comparison_csv,
    write_comparison_json,
)
from inspectai.evaluation.runner import fit_model
from inspectai.models.base import AnomalyModel
from inspectai.models.classical import ClassicalConfig, ClassicalModel
from inspectai.models.gaussian import gaussian_config_for
from inspectai.models.patchcore import patchcore_config_for
from inspectai.preprocessing.images import ImageArray, probe_image_file, read_mask
from inspectai.visualization.comparison import ModelPanel, comparison_row, stack_rows
from inspectai.visualization.heatmaps import save_rgb_png

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DATA_ERROR = 2
EXIT_ARTIFACT_ERROR = 3
EXIT_WEIGHTS_UNAVAILABLE = 4

DEFAULT_CATEGORIES = ("cashew", "pipe_fryum", "chewinggum", "fryum", "candle", "pcb1", "pcb2")
N_NORMAL_EXAMPLES = 2
ANOMALY_SIZE_QUANTILES = (0.15, 0.5, 0.85)
SCORE_CHECK_TOLERANCE = 1e-3

FIGURE_NOTE = (
    "\n## Example figures\n\n"
    "{files}: one row per test image, with the original image, the ground-truth mask, and one "
    "heatmap overlay per model. Images were chosen without looking at any model output: two "
    "normal images at random (seed {seed}) and the small, median, and large defect (15th, 50th, and "
    "85th percentile of ground-truth mask area). Each model's colors are anchored at its own "
    "pixel threshold (the 99th percentile of normal calibration pixels): blue is 0, the middle "
    "of the scale is the threshold, and red is twice the threshold. Text under each tile shows "
    "the image score, the image threshold, and the decision.\n"
)

logger = logging.getLogger("compare_models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare all evaluated models per category.")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=SplitConfig().seed, help="Split seed.")
    parser.add_argument(
        "--feature-dim", type=int, default=100, help="Model B d the artifacts were fitted with."
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.10,
        help="Model C coreset ratio the artifacts were fitted with.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip the example figures.")
    parser.add_argument(
        "--output-dir", type=Path, help="Default: <INSPECTAI_REPORT_DIR>/comparison."
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


def _collect(
    paths: Paths, categories: Sequence[str]
) -> tuple[dict[str, list[Run]], dict[str, list[str]], list[str]]:
    """Runs per usable category, the models missing per category, and categories with no runs."""
    runs_by_category: dict[str, list[Run]] = {}
    not_measured: dict[str, list[str]] = {}
    skipped: list[str] = []
    for category in categories:
        found: list[Run] = []
        missing: list[str] = []
        for spec in RUN_SPECS:
            run = load_run(paths.report_dir, paths.artifact_dir, spec, category)
            if run is None:
                missing.append(spec.label)
            else:
                found.append(run)
        if len(found) < 2:
            skipped.append(category)
            continue
        runs_by_category[category] = found
        not_measured[category] = missing
    return runs_by_category, not_measured, skipped


def pick_examples(
    test: Sequence[Sample], seed: int, category: str, area_of: Callable[[Sample], int]
) -> list[Sample]:
    """The images of the comparison figure, chosen without looking at any model output.

    Normal images: a seeded random choice. Anomalous images: the small, median, and large
    defect, i.e. the images at the 15th, 50th, and 85th percentile of ground-truth mask area
    (`area_of`), so the figure covers the range of defect sizes.
    """
    rng = random.Random(f"{seed}:{category}:comparison-figure")
    normal = sorted((s for s in test if not s.is_anomalous), key=lambda s: s.image_rel)
    anomalous = sorted((s for s in test if s.is_anomalous), key=lambda s: (area_of(s), s.image_rel))
    chosen = rng.sample(normal, min(N_NORMAL_EXAMPLES, len(normal)))
    if anomalous:
        last = len(anomalous) - 1
        positions = sorted({round(q * last) for q in ANOMALY_SIZE_QUANTILES})
        chosen += [anomalous[position] for position in positions]
    return chosen


def mask_area(sample: Sample) -> int:
    """Number of anomalous pixels in the ground-truth mask of an anomalous sample."""
    if sample.mask_path is None:
        return 0
    width, height = probe_image_file(sample.image_path)
    return int(read_mask(sample.mask_path, (height, width)).sum())


def _load_model(
    spec: RunSpec,
    category: str,
    paths: Paths,
    dataset: VisADataset,
    seed: int,
    feature_dim: int,
    coreset_ratio: float,
    max_fit_images: int | None = None,
) -> AnomalyModel:
    """The fitted model of a run: loaded from its artifact (classical: refitted, 8 s).

    `max_fit_images` is the cap recorded in the run's summary; the classical refit must use the
    same subset as the evaluated run.
    """
    if spec.model == "classical":
        splits = make_splits(dataset.samples, category, SplitConfig(seed=seed))
        if max_fit_images is not None:
            splits = with_max_fit_images(splits, max_fit_images)
        model = ClassicalModel(ClassicalConfig(resize_mode=resize_mode_for(category)))
        fit_model(model, dataset, splits)
        return model
    assert spec.artifact_folder is not None and spec.profile is not None
    directory = artifact_dir(paths.artifact_dir, spec.artifact_folder, category)
    if spec.model == "gaussian":
        gaussian, _ = load_gaussian(
            directory, category, gaussian_config_for(category, spec.profile, feature_dim, seed)
        )
        return gaussian
    patchcore, _ = load_patchcore(
        directory, category, patchcore_config_for(category, spec.profile, coreset_ratio, seed)
    )
    return patchcore


def build_figure(
    category: str,
    runs: Sequence[Run],
    paths: Paths,
    dataset: VisADataset,
    seed: int,
    feature_dim: int = 100,
    coreset_ratio: float = 0.10,
) -> ImageArray:
    """Rows of the same test images, with every model's overlay, ground truth, and decision."""
    splits = make_splits(dataset.samples, category, SplitConfig(seed=seed))
    by_rel = {s.image_rel: s for s in dataset.get_samples(category)}
    chosen = pick_examples([by_rel[n] for n in splits.test], seed, category, mask_area)
    images = [dataset.load_image(s) for s in chosen]
    masks = [
        dataset.load_mask(s, im.shape[:2]) if s.is_anomalous else None
        for s, im in zip(chosen, images, strict=True)
    ]

    panels: list[list[ModelPanel]] = [[] for _ in chosen]
    for run in runs:
        model = _load_model(
            run.spec,
            category,
            paths,
            dataset,
            seed,
            feature_dim,
            coreset_ratio,
            run.summary["split"].get("max_fit_images"),
        )
        reported = {r["image"]: r["score"] for r in run.predictions}
        threshold = float(run.summary["threshold"]["image"]["value"])
        predictions = [model.predict(image) for image in images]
        for sample, prediction in zip(chosen, predictions, strict=True):
            expected = reported[sample.image_rel]
            if abs(prediction.score - expected) > SCORE_CHECK_TOLERANCE * max(1.0, abs(expected)):
                raise EvaluationError(
                    f"'{run.spec.label}' scores {sample.image_rel} as {prediction.score:.4f} "
                    f"but the saved predictions say {expected:.4f}; the artifact and the "
                    "report do not belong together."
                )
        # blue = 0, the pixel threshold (99th percentile of normal calibration pixels) is the
        # middle of the scale, red = twice the threshold; independent of the images shown
        vmax = 2.0 * float(run.summary["threshold"]["pixel"]["value"])
        for row, prediction in zip(panels, predictions, strict=True):
            row.append(
                ModelPanel(
                    run.spec.label, prediction.anomaly_map, prediction.score, threshold, vmax
                )
            )
        del model

    rows = [
        comparison_row(
            image,
            mask,
            row,
            f"{sample.image_rel.split('/')[-1]} ({'anomaly' if sample.is_anomalous else 'normal'})",
        )
        for image, mask, row, sample in zip(images, masks, panels, chosen, strict=True)
    ]
    return stack_rows(rows)


def _run(args: argparse.Namespace) -> int:
    paths = load_paths()
    runs_by_category, not_measured, skipped = _collect(paths, args.categories)
    if not runs_by_category:
        raise EvaluationError(
            "No category has results for at least two models under "
            f"{paths.report_dir}; run scripts/evaluate_model.py first."
        )
    comparison = build_comparison(runs_by_category, not_measured)
    comparison["categories_without_results"] = skipped
    out_dir = args.output_dir or paths.report_dir / "comparison"
    write_comparison_json(out_dir / "comparison.json", comparison)
    write_comparison_csv(out_dir / "comparison.csv", comparison)

    figures: list[str] = []
    if not args.no_figures:
        dataset = VisADataset.from_paths(paths)
        for category, runs in runs_by_category.items():
            name = f"{category}_examples.png"
            save_rgb_png(
                out_dir / name,
                build_figure(
                    category, runs, paths, dataset, args.seed, args.feature_dim, args.coreset_ratio
                ),
            )
            figures.append(name)

    markdown = format_markdown(comparison)
    if figures:
        markdown += FIGURE_NOTE.format(seed=args.seed, files=", ".join(f"`{f}`" for f in figures))
    if skipped:
        markdown += (
            "\nCategories with no (or only one) evaluated model, so not compared: "
            + ", ".join(skipped)
            + ".\n"
        )
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(markdown)
    print(f"Comparison written to {out_dir}")
    return EXIT_OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unexpected failure")
        sys.exit(EXIT_FAILURE)
