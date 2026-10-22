"""A complete synthetic "world": dataset, evaluation reports, and fitted artifacts for all models.

Everything is produced by the real code (the evaluation harness, the artifact store) on tiny
synthetic images with a deterministic fake backbone, so tests of the comparison report, the
predictor, and the app exercise real behavior without real data or weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from fake_backbone import FakeBackbone
from make_synthetic import SyntheticVisA, build_synthetic_visa

from inspectai.artifacts.classical_io import save_classical
from inspectai.artifacts.gaussian_io import save_gaussian
from inspectai.artifacts.patchcore_io import save_patchcore
from inspectai.config import SplitConfig, resize_mode_for
from inspectai.data.dataset import VisADataset
from inspectai.data.splits import make_splits
from inspectai.evaluation.report import write_predictions_csv, write_summary_json
from inspectai.evaluation.runner import (
    EvalOptions,
    calibrate,
    environment_info,
    evaluate_fitted,
    fit_model,
)
from inspectai.models.classical import ClassicalConfig, ClassicalModel
from inspectai.models.features import FeatureExtractor, WeightsInfo
from inspectai.models.gaussian import GaussianModel, gaussian_config_for
from inspectai.models.patchcore import PatchCoreModel, patchcore_config_for

FAKE_WEIGHTS = WeightsInfo("fake", "injected", None)
OPTIONS = EvalOptions(eval_resolution=32, bootstrap_resamples=60)
CATEGORIES = ("cashew", "pipe_fryum")
FEATURE_DIM = 20
CORESET_RATIO = 0.02


@dataclass(frozen=True)
class World:
    synthetic: SyntheticVisA
    reports: Path
    artifacts: Path


def _write_report(folder: Path, result) -> None:
    write_predictions_csv(folder / "predictions.csv", result.rows)
    write_summary_json(folder / "summary.json", result.summary)


def build_world(root: Path, categories: tuple[str, ...] = CATEGORIES) -> World:
    """Create `<root>/data root`, `<root>/reports out`, and `<root>/artifacts out`."""
    synthetic = build_synthetic_visa(root / "data root", objects=categories)
    reports, artifacts = root / "reports out", root / "artifacts out"
    dataset = VisADataset(synthetic.split_csv, synthetic.images_root)
    for category in categories:
        splits = make_splits(dataset.samples, category, SplitConfig())
        fit_record = {}

        classical_config = ClassicalConfig(resize_mode=resize_mode_for(category))
        classical = ClassicalModel(classical_config)
        stats = fit_model(classical, dataset, splits)
        calibration = calibrate(classical, dataset, splits, OPTIONS)
        result = evaluate_fitted(
            classical, "classical", asdict(classical_config), dataset, splits, OPTIONS,
            calibration, stats.to_dict(),
        )  # fmt: skip
        fit_record = {**stats.to_dict(), "environment": environment_info()}
        save_classical(
            artifacts / "classical" / category, classical, category, splits,
            calibration.to_dict(), fit_record, OPTIONS.eval_resolution,
        )  # fmt: skip
        _write_report(reports / "classical" / category, result)

        gaussian_config = gaussian_config_for(category, "dev", FEATURE_DIM, 42)
        gaussian = GaussianModel(
            gaussian_config, FeatureExtractor(FakeBackbone(), gaussian_config.layers), FAKE_WEIGHTS
        )
        stats = fit_model(gaussian, dataset, splits)
        calibration = calibrate(gaussian, dataset, splits, OPTIONS)
        result = evaluate_fitted(
            gaussian, "gaussian", gaussian_config.to_dict(), dataset, splits, OPTIONS,
            calibration, stats.to_dict(),
        )  # fmt: skip
        save_gaussian(
            artifacts / "gaussian" / category, gaussian, category, splits,
            calibration.to_dict(), {**stats.to_dict(), "environment": environment_info()},
            OPTIONS.eval_resolution,
        )  # fmt: skip
        _write_report(reports / "gaussian" / category, result)

        patchcore_config = patchcore_config_for(category, "dev", CORESET_RATIO, 42)
        patchcore = PatchCoreModel(
            patchcore_config,
            FeatureExtractor(FakeBackbone(), patchcore_config.layers),
            FAKE_WEIGHTS,
        )
        stats = fit_model(patchcore, dataset, splits)
        calibration = calibrate(patchcore, dataset, splits, OPTIONS)
        result = evaluate_fitted(
            patchcore, "patchcore", patchcore_config.to_dict(), dataset, splits, OPTIONS,
            calibration, stats.to_dict(),
        )  # fmt: skip
        save_patchcore(
            artifacts / "patchcore" / category, patchcore, category, splits,
            calibration.to_dict(), {**stats.to_dict(), "environment": environment_info()},
            OPTIONS.eval_resolution,
        )  # fmt: skip
        _write_report(reports / "patchcore" / category, result)
    return World(synthetic, reports, artifacts)
