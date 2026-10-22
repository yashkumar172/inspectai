"""Model C end to end on a synthetic dataset: fit, save, reload, evaluate, compare, scripts."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml
from fake_backbone import FakeBackbone
from make_synthetic import SyntheticVisA, write_jpg

from inspectai.artifacts.patchcore_io import load_patchcore, save_patchcore
from inspectai.artifacts.store import MANIFEST, TENSORS
from inspectai.config import SplitConfig
from inspectai.data.dataset import VisADataset
from inspectai.data.splits import load_split, make_splits, with_max_fit_images
from inspectai.errors import ArtifactError
from inspectai.evaluation.runner import (
    Calibration,
    EvalOptions,
    calibrate,
    environment_info,
    evaluate_fitted,
    fit_model,
    run_evaluation,
)
from inspectai.models.classical import ClassicalConfig, ClassicalModel
from inspectai.models.features import FeatureExtractor, WeightsInfo
from inspectai.models.gaussian import GaussianConfig, GaussianModel
from inspectai.models.patchcore import PatchCoreConfig, PatchCoreModel

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
FAKE_WEIGHTS = WeightsInfo("fake", "injected", None)
CONFIG = PatchCoreConfig(
    layers=("layer2", "layer3"), image_size=64, resize_mode="stretch", coreset_ratio=0.1
)
OPTIONS = EvalOptions(eval_resolution=32, bootstrap_resamples=100)


def _setup(synthetic: SyntheticVisA, max_fit_images: int | None = None):
    dataset = VisADataset(synthetic.split_csv, synthetic.images_root)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    if max_fit_images is not None:
        splits = with_max_fit_images(splits, max_fit_images)
    return dataset, splits


def _fit(dataset: VisADataset, splits, config: PatchCoreConfig = CONFIG):
    model = PatchCoreModel(config, FeatureExtractor(FakeBackbone(), config.layers), FAKE_WEIGHTS)
    stats = fit_model(model, dataset, splits)
    return model, stats, calibrate(model, dataset, splits, OPTIONS)


def _save(directory: Path, model, splits, calibration, stats) -> Path:
    return save_patchcore(
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
    loaded, artifact = load_patchcore(directory, "cashew", CONFIG, backbone=FakeBackbone())
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
    loaded, _ = load_patchcore(directory, backbone=FakeBackbone())
    assert loaded.predict(image).score == expected


def test_the_manifest_records_the_bank_the_settings_and_measured_costs(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["model_type"] == "patchcore" and manifest["category"] == "cashew"
    assert manifest["backbone"] == "resnet18" and manifest["layers"] == ["layer2", "layer3"]
    assert manifest["weights"] == {"backbone": "fake", "identifier": "injected", "sha256": None}
    assert manifest["feature_dim"] == 40 and manifest["neighborhood"] == 3
    assert manifest["resize_mode"] == "stretch" and manifest["seed"] == 42
    assert manifest["reference_aspect_ratio"] == pytest.approx(64 / 48)
    bank = manifest["memory_bank"]
    assert bank["n_candidates"] == 14 * 256 and bank["n_vectors"] == round(0.1 * 14 * 256)
    assert bank["coreset_ratio"] == 0.1 and bank["projection_dim"] == 128
    assert bank["bank_bytes"] == bank["n_vectors"] * 40 * 4
    assert bank["estimated_unreduced_gb"] == pytest.approx(14 * 256 * 40 * 4 / 1024**3, abs=1e-4)
    assert bank["coreset_seconds"] >= 0
    assert manifest["model_config"] == CONFIG.to_dict()
    snapshot = yaml.safe_load((directory / "config.snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot == CONFIG.snapshot()  # model settings and the resource settings used
    assert snapshot["runtime"]["chunk_size"] == 16384 and snapshot["runtime"]["num_threads"] == 6
    assert (
        manifest["runtime"]["max_bank_gb"] == 4.0 and manifest["runtime"]["max_fit_minutes"] == 30.0
    )
    assert manifest["threshold"] == {
        "policy": "normal_percentile",
        "value": calibration.image_threshold.value,
    }
    assert manifest["fit_stats"]["n_fit_images"] == len(splits.fit)
    assert "power_plan" in manifest["fit_stats"]["environment"]
    assert manifest["tensors"]["bank"] == {"shape": [bank["n_vectors"], 40], "dtype": "float32"}


def test_evaluating_the_loaded_model_gives_the_same_metrics_as_the_fitted_one(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    live = evaluate_fitted(model, "patchcore", {}, dataset, splits, OPTIONS, calibration)
    loaded, artifact = load_patchcore(directory, backbone=FakeBackbone())
    stored = Calibration.from_dict(artifact.manifest["calibration"])
    again = evaluate_fitted(loaded, "patchcore", {}, dataset, artifact.split, OPTIONS, stored)
    assert live.summary["metrics"] == again.summary["metrics"]
    image = live.summary["metrics"]["image"]
    assert image["auroc"] > 0.9 and (image["n_normal"], image["n_anomalous"]) == (5, 5)
    assert live.summary["metrics"]["pixel"]["auroc"] > 0.85


def test_a_stale_artifact_is_rejected_but_resource_limits_do_not_make_one_stale(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    other = PatchCoreConfig(
        layers=("layer2", "layer3"), image_size=64, resize_mode="stretch", coreset_ratio=0.2
    )
    with pytest.raises(ArtifactError, match="stale"):
        load_patchcore(directory, "cashew", other, backbone=FakeBackbone())
    relaxed = PatchCoreConfig(
        layers=("layer2", "layer3"),
        image_size=64,
        resize_mode="stretch",
        coreset_ratio=0.1,
        max_bank_gb=99.0,
        chunk_size=11,
    )
    loaded, _ = load_patchcore(
        directory, "cashew", relaxed, backbone=FakeBackbone(), runtime={"chunk_size": 11}
    )
    assert loaded.config.chunk_size == 11
    with pytest.raises(ArtifactError, match="category"):
        load_patchcore(directory, "pipe_fryum", CONFIG, backbone=FakeBackbone())


def test_a_tampered_bank_is_rejected_before_use(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa)
    model, stats, calibration = _fit(dataset, splits)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    with (directory / TENSORS).open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"\x00\x00\xc0\x7f")  # a float32 NaN
    with pytest.raises(ArtifactError, match="Checksum"):
        load_patchcore(directory, backbone=FakeBackbone())
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    manifest["feature_dim"] = 7
    (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_patchcore(directory, backbone=FakeBackbone())


def test_a_gaussian_artifact_is_not_accepted_as_a_patchcore_artifact(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    from inspectai.artifacts.gaussian_io import save_gaussian

    dataset, splits = _setup(synthetic_visa)
    gaussian_config = GaussianConfig(
        layers=("layer2", "layer3"), image_size=64, d=20, resize_mode="stretch"
    )
    model = GaussianModel(
        gaussian_config, FeatureExtractor(FakeBackbone(), gaussian_config.layers), FAKE_WEIGHTS
    )
    stats = fit_model(model, dataset, splits)
    calibration = calibrate(model, dataset, splits, OPTIONS)
    save_gaussian(
        tmp_path / "g", model, "cashew", splits, calibration.to_dict(), stats.to_dict(), 32
    )
    with pytest.raises(ArtifactError, match="model"):
        load_patchcore(tmp_path / "g", backbone=FakeBackbone())


def test_the_bank_and_thresholds_do_not_depend_on_the_test_images(
    synthetic_visa: SyntheticVisA,
) -> None:
    """Leakage check: replacing every test image changes neither the bank nor the thresholds."""
    dataset, splits = _setup(synthetic_visa)
    before_model, _, before = _fit(dataset, splits)
    for rel in splits.test:
        sample = next(s for s in dataset.samples if s.image_rel == rel)
        write_jpg(sample.image_path, dataset.load_image(sample)[::-1, ::-1].copy() // 2)
    after_model, _, after = _fit(dataset, splits)
    assert np.array_equal(before_model.tensors()["bank"], after_model.tensors()["bank"])
    assert before.to_dict() == after.to_dict()


def test_the_model_is_fitted_on_the_capped_subset_and_the_cap_is_recorded(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits = _setup(synthetic_visa, max_fit_images=6)
    seen: list[str] = []
    original = dataset.load_image
    dataset.load_image = lambda sample: (seen.append(sample.image_rel), original(sample))[1]  # type: ignore[method-assign]
    model, stats, calibration = _fit(dataset, splits)
    assert model.n_fit_images == stats.n_fit_images == 6
    assert sorted(seen[:6]) == sorted(splits.fit_used or ())  # the recorded subset, nothing else
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    saved = load_split(directory / "split.json")
    assert saved.max_fit_images == 6 and saved.fit_used == splits.fit_used
    _, artifact = load_patchcore(directory, backbone=FakeBackbone())
    assert artifact.split.fit_files == splits.fit_used


def test_all_models_are_compared_on_identical_splits_and_evaluation_resolution(
    synthetic_visa: SyntheticVisA,
) -> None:
    """Acceptance (SPEC.md Milestone 4): same splits and resolution for every model."""
    dataset, splits = _setup(synthetic_visa)
    options = OPTIONS
    classical = run_evaluation(
        ClassicalModel(ClassicalConfig(image_size=32, resize_mode="stretch")),
        "classical", {}, dataset, splits, options,
    )  # fmt: skip
    gaussian_config = GaussianConfig(
        layers=("layer2", "layer3"), image_size=64, d=20, resize_mode="stretch"
    )
    gaussian = run_evaluation(
        GaussianModel(
            gaussian_config, FeatureExtractor(FakeBackbone(), gaussian_config.layers), FAKE_WEIGHTS
        ),
        "gaussian", {}, dataset, splits, options,
    )  # fmt: skip
    patchcore = run_evaluation(
        PatchCoreModel(CONFIG, FeatureExtractor(FakeBackbone(), CONFIG.layers), FAKE_WEIGHTS),
        "patchcore", {}, dataset, splits, options,
    )  # fmt: skip
    results = [classical, gaussian, patchcore]
    assert len({json.dumps(r.summary["split"], sort_keys=True) for r in results}) == 1
    assert len({r.summary["metrics"]["evaluation_resolution"] for r in results}) == 1
    assert {r.summary["metrics"]["evaluation_split"] for r in results} == {"test"}
    assert len({tuple(sorted(row.image for row in r.rows)) for r in results}) == 1
    assert len({(r.summary["metrics"]["image"]["n_normal"]) for r in results}) == 1
    assert {r.summary["model"] for r in results} == {"classical", "gaussian", "patchcore"}


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
    monkeypatch.setattr("inspectai.artifacts.patchcore_io.load_backbone", fake)
    return module


@pytest.fixture
def evaluate_cli(fit_cli: ModuleType) -> ModuleType:
    return _script("evaluate_model")


FIT = ["--model", "patchcore", "--category", "cashew", "--coreset-ratio", "0.02"]


def test_fit_then_evaluate_through_the_scripts(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fit_cli.main(FIT) == 0
    assert "Memory bank:" in capsys.readouterr().out
    artifact = env / "artifacts out" / "patchcore" / "cashew"
    assert {p.name for p in artifact.iterdir()} == {
        "manifest.json", "tensors.safetensors", "split.json", "config.snapshot.yaml",
    }  # fmt: skip
    assert evaluate_cli.main(FIT) == 0
    out = env / "reports out" / "patchcore" / "cashew"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "patchcore" and summary["metrics"]["image"]["n_anomalous"] == 5
    performance = summary["performance"]
    assert performance["fit_seconds"] >= 0 and performance["latency_seconds_mean"] > 0
    assert len(list((out / "examples").glob("*.png"))) == 4


def test_evaluation_does_not_refit_and_may_override_the_search_chunk_size(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert fit_cli.main(FIT) == 0

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluation must not fit")

    monkeypatch.setattr(PatchCoreModel, "fit", forbidden)
    assert evaluate_cli.main([*FIT, "--chunk-size", "5"]) == 0  # a resource limit, not stale


def test_the_command_records_a_max_fit_images_cap(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fit_cli.main([*FIT, "--max-fit-images", "6"]) == 0
    assert "subset of 6 of 14" in capsys.readouterr().out
    saved = load_split(env / "artifacts out" / "patchcore" / "cashew" / "split.json")
    assert saved.max_fit_images == 6 and len(saved.fit_used or ()) == 6
    assert evaluate_cli.main(FIT) == 0
    summary = json.loads(
        (env / "reports out" / "patchcore" / "cashew" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["split"]["max_fit_images"] == 6
    assert summary["performance"]["n_fit_images"] == 6


def test_evaluating_a_stale_artifact_exits_three(
    fit_cli: ModuleType, evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fit_cli.main(FIT) == 0
    stale = ["--model", "patchcore", "--category", "cashew", "--coreset-ratio", "0.5"]
    assert evaluate_cli.main(stale) == 3
    assert "stale" in capsys.readouterr().err


def test_evaluating_without_an_artifact_exits_three(
    evaluate_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert evaluate_cli.main(FIT) == 3
    assert "folder not found" in capsys.readouterr().err


def test_a_memory_bank_over_the_limit_exits_two_and_writes_no_artifact(
    fit_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fit_cli.main([*FIT, "--max-bank-gb", "0.0000001"]) == 2
    err = capsys.readouterr().err
    assert "max_bank_gb" in err and "max_fit_images" in err
    assert not (env / "artifacts out" / "patchcore" / "cashew").exists()


def test_an_exhausted_fit_time_limit_exits_two_and_writes_no_artifact(
    fit_cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fit_cli.main([*FIT, "--max-fit-minutes", "0.0000001"]) == 2
    assert "random subsampling" in capsys.readouterr().err
    assert not (env / "artifacts out" / "patchcore" / "cashew").exists()


def test_fitting_without_cached_weights_exits_four_with_instructions(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script("fit_model")  # the real loader: the cache under TORCH_HOME is empty
    assert module.main(FIT) == 4
    assert "fetch_weights.py --backbone resnet18" in capsys.readouterr().err
