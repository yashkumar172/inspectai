"""Comparison report end to end: real harness runs on two synthetic categories, then the script."""

import csv
import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
from fake_backbone import FakeBackbone
from make_synthetic import SyntheticVisA, build_synthetic_visa

from inspectai.artifacts.gaussian_io import save_gaussian
from inspectai.artifacts.patchcore_io import save_patchcore
from inspectai.config import SplitConfig, resize_mode_for
from inspectai.data.dataset import VisADataset
from inspectai.data.splits import make_splits
from inspectai.evaluation.report import write_predictions_csv, write_summary_json
from inspectai.evaluation.runner import (
    EvalOptions,
    calibrate,
    evaluate_fitted,
    fit_model,
    run_evaluation,
)
from inspectai.models.classical import ClassicalConfig, ClassicalModel
from inspectai.models.features import FeatureExtractor, WeightsInfo
from inspectai.models.gaussian import GaussianModel, gaussian_config_for
from inspectai.models.patchcore import PatchCoreModel, patchcore_config_for

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_models.py"
FAKE_WEIGHTS = WeightsInfo("fake", "injected", None)
OPTIONS = EvalOptions(eval_resolution=32, bootstrap_resamples=60)
ARGS = ["--feature-dim", "20", "--coreset-ratio", "0.02"]
CATEGORIES = ("cashew", "pipe_fryum")


@dataclass(frozen=True)
class World:
    synthetic: SyntheticVisA
    reports: Path
    artifacts: Path


def _write(folder: Path, result) -> None:
    write_predictions_csv(folder / "predictions.csv", result.rows)
    write_summary_json(folder / "summary.json", result.summary)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    root = tmp_path_factory.mktemp("comparison world")
    synthetic = build_synthetic_visa(root / "data root", objects=CATEGORIES)
    reports, artifacts = root / "reports out", root / "artifacts out"
    dataset = VisADataset(synthetic.split_csv, synthetic.images_root)
    for category in CATEGORIES:
        splits = make_splits(dataset.samples, category, SplitConfig())

        classical_config = ClassicalConfig(resize_mode=resize_mode_for(category))
        classical = run_evaluation(
            ClassicalModel(classical_config),
            "classical",
            asdict(classical_config),
            dataset,
            splits,
            OPTIONS,
        )
        _write(reports / "classical" / category, classical)

        gaussian_config = gaussian_config_for(category, "dev", 20, 42)
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
            calibration.to_dict(), stats.to_dict(), OPTIONS.eval_resolution,
        )  # fmt: skip
        _write(reports / "gaussian" / category, result)

        patchcore_config = patchcore_config_for(category, "dev", 0.02, 42)
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
            calibration.to_dict(), stats.to_dict(), OPTIONS.eval_resolution,
        )  # fmt: skip
        _write(reports / "patchcore" / category, result)
    return World(synthetic, reports, artifacts)


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the environment at a private copy of the reports; return that report folder."""
    reports = tmp_path / "reports copy"
    shutil.copytree(world.reports, reports)
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(world.synthetic.data_root))
    monkeypatch.setenv("INSPECTAI_ARTIFACT_DIR", str(world.artifacts))
    monkeypatch.setenv("INSPECTAI_REPORT_DIR", str(reports))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty torch home"))
    fake = lambda name, allow_download=None: (FakeBackbone(), FAKE_WEIGHTS)
    monkeypatch.setattr("inspectai.artifacts.gaussian_io.load_backbone", fake)
    monkeypatch.setattr("inspectai.artifacts.patchcore_io.load_backbone", fake)
    return reports


def _load_json(reports: Path) -> dict:
    return json.loads((reports / "comparison" / "comparison.json").read_text(encoding="utf-8"))


def test_a_comparison_of_two_categories_and_three_models_is_written(
    cli: ModuleType, env: Path
) -> None:
    assert cli.main([*ARGS, "--categories", *CATEGORIES]) == 0
    out = env / "comparison"
    comparison = _load_json(env)
    assert list(comparison["categories"]) == list(CATEGORIES)
    for category in CATEGORIES:
        data = comparison["categories"][category]
        assert [r["model"] for r in data["runs"]] == ["classical", "gaussian", "patchcore"]
        assert (data["n_normal"], data["n_anomalous"]) == (5, 5)
        assert data["evaluation_resolution"] == 32 and data["evaluation_split"] == "test"
        assert data["not_measured"] == ["B Gaussian full", "C PatchCore-inspired full"]
        assert {len(r["metrics"]["image"]["confusion_matrix"]) for r in data["runs"]} == {4}
        assert all(r["model_config"] and r["environment"]["python"] for r in data["runs"])
        assert data["paired_auroc_differences"]
    assert comparison["verification"]["image_metrics_rederived_from_predictions"] is True
    assert set(comparison["macro_average"]) == {
        "A classical",
        "B Gaussian dev",
        "C PatchCore-inspired dev",
    }
    assert comparison["macro_average"]["A classical"]["n_categories"] == 2
    assert comparison["categories_without_results"] == []

    with (out / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and {r["category"] for r in rows} == set(CATEGORIES)
    markdown = (out / "comparison.md").read_text(encoding="utf-8")
    assert "## cashew" in markdown and "## pipe_fryum" in markdown
    assert "Not measured (no results exist)" in markdown
    assert "## Example figures" in markdown and "cashew_examples.png" in markdown
    assert "without looking at any model output" in markdown


def test_the_figures_show_the_same_images_for_every_model(cli: ModuleType, env: Path) -> None:
    assert cli.main([*ARGS, "--categories", *CATEGORIES]) == 0
    for category in CATEGORIES:
        figure = cv2.imdecode(
            np.frombuffer((env / "comparison" / f"{category}_examples.png").read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert figure is not None
        rows = 2 + 3  # 2 normal images and the small, median, and large defect
        assert figure.shape[0] == rows * (22 + 256 + 22)
        # image + ground truth + 3 models, each tile 256 high and 341 wide (4:3 images)
        assert figure.shape[1] == 5 * round(64 * 256 / 48)


def _test_samples(world: World):
    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    by_rel = {s.image_rel: s for s in dataset.get_samples("cashew")}
    return [by_rel[n] for n in splits.test], splits


def test_figure_images_are_chosen_without_any_model_output(cli: ModuleType, world: World) -> None:
    test, splits = _test_samples(world)
    # Give the five anomalous images distinct areas that run opposite to their names.
    areas = {s.image_rel: -int(s.image_rel[-7:-4]) for s in test if s.is_anomalous}
    chosen = cli.pick_examples(test, 42, "cashew", lambda s: areas[s.image_rel])
    normal, anomalous = chosen[:2], chosen[2:]
    assert [s.is_anomalous for s in chosen] == [False, False, True, True, True]
    assert {s.image_rel for s in chosen} <= set(splits.test)  # test images only
    # Anomalies are the small, median, and large defect (positions 1, 2, 3 of 5 by area).
    by_area = sorted((s for s in test if s.is_anomalous), key=lambda s: areas[s.image_rel])
    assert anomalous == [by_area[1], by_area[2], by_area[3]]
    assert [areas[s.image_rel] for s in anomalous] == sorted(areas[s.image_rel] for s in anomalous)
    # Normal images are seeded: reproducible, independent of input order, different per seed.
    again = cli.pick_examples(list(reversed(test)), 42, "cashew", lambda s: areas[s.image_rel])
    assert again == chosen
    other = cli.pick_examples(test, 7, "cashew", lambda s: areas[s.image_rel])
    assert {s.image_rel for s in other[:2]} != {s.image_rel for s in normal}
    assert other[2:] == anomalous  # the defect-size choice does not depend on the seed


def test_few_anomalous_images_give_fewer_figure_rows(cli: ModuleType, world: World) -> None:
    test, _ = _test_samples(world)
    anomalous = [s for s in test if s.is_anomalous]
    chosen = cli.pick_examples(
        [s for s in test if not s.is_anomalous] + anomalous[:1], 42, "cashew", lambda s: 1
    )
    assert [s.is_anomalous for s in chosen] == [False, False, True]
    assert cli.pick_examples([s for s in test if s.is_anomalous], 42, "cashew", lambda s: 1)[
        0
    ].is_anomalous


def test_the_defect_area_is_read_from_the_ground_truth_mask(cli: ModuleType, world: World) -> None:
    test, _ = _test_samples(world)
    anomalous = next(s for s in test if s.is_anomalous)
    assert cli.mask_area(anomalous) == (64 // 4) * (48 // 4)  # the synthetic rectangle
    normal = next(s for s in test if not s.is_anomalous)
    assert cli.mask_area(normal) == 0


def test_the_comparison_is_reproducible(cli: ModuleType, env: Path) -> None:
    assert cli.main([*ARGS, "--categories", *CATEGORIES, "--no-figures"]) == 0
    first = _load_json(env)
    assert cli.main([*ARGS, "--categories", *CATEGORIES, "--no-figures"]) == 0
    second = _load_json(env)
    first.pop("generated_utc"), second.pop("generated_utc")
    assert first == second
    assert not list((env / "comparison").glob("*.png"))  # --no-figures wrote none


def test_categories_without_enough_results_are_listed_and_not_compared(
    cli: ModuleType, env: Path
) -> None:
    for folder in ("gaussian", "patchcore"):  # leave pipe_fryum with the classical run only
        shutil.rmtree(env / folder / "pipe_fryum")
    assert (
        cli.main([*ARGS, "--categories", "cashew", "pipe_fryum", "chewinggum", "--no-figures"]) == 0
    )
    comparison = _load_json(env)
    assert list(comparison["categories"]) == ["cashew"]
    assert comparison["categories_without_results"] == ["pipe_fryum", "chewinggum"]
    markdown = (env / "comparison" / "comparison.md").read_text(encoding="utf-8")
    assert "so not compared: pipe_fryum, chewinggum" in markdown


def test_no_results_at_all_exits_two(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([*ARGS, "--categories", "chewinggum", "--no-figures"]) == 2
    assert "at least two models" in capsys.readouterr().err


def test_a_summary_that_does_not_follow_from_its_predictions_exits_two(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = env / "gaussian" / "cashew" / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    scores = [float(r[5]) for r in rows[1:]]
    top = scores.index(max(scores)) + 1
    rows[top][5] = "-1000.0"  # the most anomalous-looking image is rewritten as normal
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    assert cli.main([*ARGS, "--categories", *CATEGORIES, "--no-figures"]) == 2
    assert "do not follow from predictions" in capsys.readouterr().err


def test_runs_with_different_evaluation_settings_are_refused(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = env / "patchcore" / "cashew" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["metrics"]["evaluation_resolution"] = 256
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert cli.main([*ARGS, "--categories", *CATEGORIES, "--no-figures"]) == 2
    assert "not comparable" in capsys.readouterr().err


def test_runs_with_a_different_split_are_refused(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = env / "classical" / "pipe_fryum" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["split"]["file_list_sha256"]["test"] = "0" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert cli.main([*ARGS, "--categories", *CATEGORIES, "--no-figures"]) == 2
    assert "different split files" in capsys.readouterr().err


def test_an_artifact_that_does_not_match_the_report_is_caught_by_the_figure_check(
    cli: ModuleType, env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = GaussianModel.predict

    def shifted(self, image):  # type: ignore[no-untyped-def]
        prediction = original(self, image)
        return type(prediction)(prediction.score + 5.0, prediction.anomaly_map + 5.0)

    monkeypatch.setattr(GaussianModel, "predict", shifted)
    assert cli.main([*ARGS, "--categories", "cashew"]) == 2
    assert "do not belong together" in capsys.readouterr().err


def test_a_stale_artifact_exits_three(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(["--feature-dim", "10", "--coreset-ratio", "0.02", "--categories", "cashew"]) == 3
    )
    assert "stale" in capsys.readouterr().err
