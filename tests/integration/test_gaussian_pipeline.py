"""Model B end to end on a synthetic dataset: fit, save, reload, evaluate, and the scripts."""

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fake_backbone import FakeBackbone
from make_synthetic import SyntheticVisA, write_jpg

from inspectai.artifacts.gaussian_io import load_gaussian, save_gaussian
from inspectai.artifacts.store import MANIFEST, TENSORS, load_artifact
from inspectai.config import SplitConfig
from inspectai.data.dataset import VisADataset
from inspectai.data.splits import make_splits
from inspectai.errors import ArtifactError
from inspectai.evaluation.runner import (
    EvalOptions,
    calibrate,
    environment_info,
    evaluate_fitted,
    fit_model,
)
from inspectai.models.features import FeatureExtractor, WeightsInfo
from inspectai.models.gaussian import GaussianConfig, GaussianModel

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
FAKE_WEIGHTS = WeightsInfo("fake", "injected", None)
CONFIG = GaussianConfig(layers=("layer2", "layer3"), image_size=64, d=20, resize_mode="stretch")
OPTIONS = EvalOptions(eval_resolution=32, bootstrap_resamples=100)


def _setup(synthetic: SyntheticVisA):
    dataset = VisADataset(synthetic.split_csv, synthetic.images_root)
    return dataset, make_splits(dataset.samples, "cashew", SplitConfig())


def _fit(dataset: VisADataset, splits):
    model = GaussianModel(CONFIG, FeatureExtractor(FakeBackbone(), CONFIG.layers), FAKE_WEIGHTS)
    stats = fit_model(model, dataset, splits)
    return model, stats, calibrate(model, dataset, splits, OPTIONS)


def _save(directory: Path, model, splits, calibration, stats) -> Path:
    return save_gaussian(
        directory,
        model,
        "cashew",
        splits,
        calibration.to_dict(),
        {**stats.to_dict(), "environment": environment_info()},
        32,
    )


def test_a_fitted_model_is_saved_reloaded_and_gives_identical_predictions(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art é", model, splits, calibration, stats)
    loaded, artifact = load_gaussian(directory, "cashew", CONFIG, backbone=FakeBackbone())
    assert artifact.split == splits
    image = dataset.load_image(dataset.get_samples("cashew", "test", "anomaly")[0])
    original, reloaded = model.predict(image), loaded.predict(image)
    assert original.score == reloaded.score
    assert np.array_equal(original.anomaly_map, reloaded.anomaly_map)


def test_the_loaded_model_runs_without_the_dataset(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    image = dataset.load_image(dataset.get_samples("cashew", "test", "normal")[0])
    expected = model.predict(image).score
    loaded, _ = load_gaussian(directory, backbone=FakeBackbone())
    assert loaded.predict(image).score == expected  # no data root or split file is touched


def test_the_manifest_records_the_model_thresholds_and_measured_costs(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["model_type"] == "gaussian" and manifest["category"] == "cashew"
    assert manifest["backbone"] == "resnet18" and manifest["layers"] == ["layer2", "layer3"]
    assert manifest["weights"] == {"backbone": "fake", "identifier": "injected", "sha256": None}
    assert manifest["image_size"] == 64 and manifest["feature_dim"] == 20
    assert manifest["resize_mode"] == "stretch" and manifest["seed"] == 42
    assert manifest["reference_aspect_ratio"] == pytest.approx(64 / 48)
    assert manifest["threshold"] == {
        "policy": "normal_percentile",
        "value": calibration.image_threshold.value,
    }
    assert manifest["fit_stats"]["n_fit_images"] == len(splits.fit)
    assert manifest["fit_stats"]["fit_seconds"] >= 0
    assert manifest["tensors"]["inv_cov"]["shape"] == [256, 20, 20]
    assert manifest["epsilon"] == 0.01 and manifest["smoothing_sigma"] == 4.0
    assert manifest["model_config"] == CONFIG.to_dict()
    expected_bytes = 4 * (256 * 20 + 256 * 20 * 20) + 8 * 20  # mean + inv_cov (f32), index (i64)
    assert manifest["parameter_bytes"] == expected_bytes
    assert "power_plan" in manifest["fit_stats"]["environment"]


def test_evaluating_the_loaded_model_gives_the_same_metrics_as_the_fitted_one(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    live = evaluate_fitted(model, "gaussian", {}, dataset, splits, OPTIONS, calibration)
    loaded, artifact = load_gaussian(directory, backbone=FakeBackbone())
    from inspectai.evaluation.runner import Calibration

    stored = Calibration.from_dict(artifact.manifest["calibration"])
    again = evaluate_fitted(loaded, "gaussian", {}, dataset, artifact.split, OPTIONS, stored)
    assert live.summary["metrics"] == again.summary["metrics"]
    assert [r.score for r in live.rows] == [r.score for r in again.rows]
    image = live.summary["metrics"]["image"]
    assert image["auroc"] > 0.9 and (image["n_normal"], image["n_anomalous"]) == (5, 5)
    assert live.summary["metrics"]["pixel"]["auroc"] > 0.85


def test_a_stale_artifact_is_rejected(synthetic_visa: SyntheticVisA, tmp_path: Path) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    other = GaussianConfig(layers=("layer2", "layer3"), image_size=64, d=10, resize_mode="stretch")
    with pytest.raises(ArtifactError, match="stale"):
        load_gaussian(directory, "cashew", other, backbone=FakeBackbone())
    with pytest.raises(ArtifactError, match="category"):
        load_gaussian(directory, "pipe_fryum", CONFIG, backbone=FakeBackbone())


def test_a_tampered_artifact_is_rejected_before_use(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    with (directory / TENSORS).open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"\x00\x00\xc0\x7f")  # a float32 NaN
    with pytest.raises(ArtifactError, match="Checksum"):
        load_gaussian(directory, backbone=FakeBackbone())
    # Rewriting the manifest checksum cannot hide NaN values either.
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    import hashlib

    manifest["files"][TENSORS] = hashlib.sha256((directory / TENSORS).read_bytes()).hexdigest()
    (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="NaN"):
        load_artifact(directory)


def test_the_fitted_parameters_do_not_depend_on_the_test_images(
    synthetic_visa: SyntheticVisA,
) -> None:
    """Leakage check: replacing every test image changes neither the model nor the thresholds."""
    dataset, splits = _setup(synthetic_visa)
    before_model, _, before = _fit(dataset, splits)
    for rel in splits.test:
        sample = next(s for s in dataset.samples if s.image_rel == rel)
        write_jpg(sample.image_path, dataset.load_image(sample)[::-1, ::-1].copy() // 2)
    after_model, _, after = _fit(dataset, splits)
    for name, array in before_model.tensors().items():
        assert np.array_equal(array, after_model.tensors()[name])
    assert before.to_dict() == after.to_dict()


def test_the_model_is_fitted_on_the_fit_split_only(synthetic_visa: SyntheticVisA) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, _ = _fit(dataset, splits)
    assert model.n_fit_images == stats.n_fit_images == len(splits.fit)
    assert len(splits.fit) < len(splits.fit) + len(splits.calibration_normal)


# ----- scripts --------------------------------------------------------------------------------


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(synthetic_visa: SyntheticVisA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(synthetic_visa.data_root))
    monkeypatch.setenv("INSPECTAI_ARTIFACT_DIR", str(tmp_path / "artifacts out"))
    monkeypatch.setenv("INSPECTAI_REPORT_DIR", str(tmp_path / "reports out"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty torch home"))
    monkeypatch.delenv("INSPECTAI_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    return tmp_path


@pytest.fixture
def fit_cli(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _script("fit_model")
    fake = lambda name, allow_download=None: (FakeBackbone(), FAKE_WEIGHTS)
    monkeypatch.setattr(module, "load_backbone", fake)
    monkeypatch.setattr("inspectai.artifacts.gaussian_io.load_backbone", fake)
    return module


@pytest.fixture
def evaluate_cli(fit_cli: ModuleType) -> ModuleType:  # fit_cli installs the fake backbone loader
    return _script("evaluate_model")


FIT = ["--model", "gaussian", "--category", "cashew", "--feature-dim", "20"]


def test_fit_then_evaluate_through_the_scripts(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path
) -> None:
    assert fit_cli.main(FIT) == 0
    artifact = env / "artifacts out" / "gaussian" / "cashew"
    assert {p.name for p in artifact.iterdir()} == {
        "manifest.json",
        "tensors.safetensors",
        "split.json",
        "config.snapshot.yaml",
    }
    assert evaluate_cli.main(FIT) == 0
    out = env / "reports out" / "gaussian" / "cashew"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "gaussian" and summary["metrics"]["image"]["n_anomalous"] == 5
    performance = summary["performance"]
    assert performance["fit_seconds"] >= 0 and performance["n_fit_images"] > 0
    assert performance["latency_seconds_mean"] > 0 and "peak_memory_gb_after_fit" in performance
    assert summary["threshold"]["image"]["policy"] == "normal_percentile"
    with (out / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 10
    assert len(list((out / "examples").glob("*.png"))) == 4


def test_evaluation_uses_the_artifact_split_and_threshold_without_refitting(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert fit_cli.main(FIT) == 0

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluation must not fit")

    monkeypatch.setattr(GaussianModel, "fit", forbidden)
    assert evaluate_cli.main(FIT) == 0
    manifest = json.loads(
        (env / "artifacts out" / "gaussian" / "cashew" / MANIFEST).read_text(encoding="utf-8")
    )
    summary = json.loads(
        (env / "reports out" / "gaussian" / "cashew" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["threshold"]["image"]["value"] == manifest["threshold"]["value"]


def test_evaluation_supports_labeled_calibration(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path
) -> None:
    assert fit_cli.main(FIT) == 0
    assert evaluate_cli.main([*FIT, "--threshold-policy", "labeled_calibration"]) == 0
    summary = json.loads(
        (env / "reports out" / "gaussian" / "cashew" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["metrics"]["evaluation_split"] == "test_eval"


def test_evaluating_a_stale_artifact_exits_three(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fit_cli.main(FIT) == 0
    stale = ["--model", "gaussian", "--category", "cashew", "--feature-dim", "10"]
    assert evaluate_cli.main(stale) == 3
    assert "stale" in capsys.readouterr().err


def test_evaluating_without_an_artifact_exits_three(
    evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert evaluate_cli.main(FIT) == 3
    assert "folder not found" in capsys.readouterr().err


def test_a_corrupted_artifact_makes_evaluation_exit_three(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path
) -> None:
    assert fit_cli.main(FIT) == 0
    tensors = env / "artifacts out" / "gaussian" / "cashew" / TENSORS
    tensors.write_bytes(tensors.read_bytes()[:-8])
    assert evaluate_cli.main(FIT) == 3


def test_fitting_without_cached_weights_exits_four_with_instructions(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script("fit_model")  # the real loader: the cache under TORCH_HOME is empty
    assert module.main(FIT) == 4
    err = capsys.readouterr().err
    assert "fetch_weights.py --backbone resnet18" in err


def test_the_scripts_report_configuration_errors_with_exit_two(
    fit_cli: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("INSPECTAI_DATA_ROOT", raising=False)
    assert fit_cli.main(FIT) == 2
    assert "INSPECTAI_DATA_ROOT" in capsys.readouterr().err


def test_fitting_with_too_large_a_feature_dim_exits_two(
    fit_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--model", "gaussian", "--category", "cashew", "--feature-dim", "500"]
    assert fit_cli.main(args) == 2
    assert "exceeds" in capsys.readouterr().err


def test_fetch_weights_exits_four_when_the_download_fails(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script("fetch_weights")

    def failing(name: str) -> Path:
        raise OSError("network unreachable")

    monkeypatch.setattr(module, "fetch_weights", failing)
    assert module.main(["--backbone", "resnet18"]) == 4
    assert "network unreachable" in capsys.readouterr().err


def test_fetch_weights_rejects_unsupported_backbones() -> None:
    with pytest.raises(SystemExit) as caught:
        _script("fetch_weights").main(["--backbone", "vgg16"])
    assert caught.value.code == 2


def test_a_different_set_of_cached_weights_than_the_fit_is_rejected(
    synthetic_visa: SyntheticVisA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    other = WeightsInfo("fake", "injected", "f" * 64)
    monkeypatch.setattr(
        "inspectai.artifacts.gaussian_io.load_backbone", lambda name: (FakeBackbone(), other)
    )
    with pytest.raises(ArtifactError, match="differ from the ones"):
        load_gaussian(directory, "cashew", CONFIG)  # no injected backbone: weights are checked


def test_evaluating_without_cached_weights_exits_four(
    fit_cli: ModuleType,
    evaluate_cli: ModuleType,
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from inspectai.models.features import load_backbone as real_loader

    assert fit_cli.main(FIT) == 0
    monkeypatch.setattr("inspectai.artifacts.gaussian_io.load_backbone", real_loader)
    assert evaluate_cli.main(FIT) == 4  # the artifact is fine; the cache under TORCH_HOME is empty
    assert "fetch_weights.py --backbone resnet18" in capsys.readouterr().err


def test_max_fit_images_also_caps_a_gaussian_fit_and_is_recorded(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inspectai.data.splits import load_split

    assert fit_cli.main([*FIT, "--max-fit-images", "6"]) == 0
    assert "subset of 6 of 14" in capsys.readouterr().out
    directory = env / "artifacts out" / "gaussian" / "cashew"
    saved = load_split(directory / "split.json")
    assert saved.max_fit_images == 6 and len(saved.fit_used or ()) == 6
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["n_fit_images"] == 6 and manifest["fit_stats"]["n_fit_images"] == 6
    assert evaluate_cli.main(FIT) == 0
    summary = json.loads(
        (env / "reports out" / "gaussian" / "cashew" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["split"]["max_fit_images"] == 6
