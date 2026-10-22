"""The classical baseline as a saved artifact (used by the app); no pretrained weights involved."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from make_synthetic import SyntheticVisA

from inspectai.artifacts.classical_io import load_classical, save_classical
from inspectai.artifacts.store import MANIFEST, TENSORS
from inspectai.config import SplitConfig, resize_mode_for
from inspectai.data.dataset import VisADataset
from inspectai.data.splits import make_splits
from inspectai.errors import ArtifactError
from inspectai.evaluation.runner import EvalOptions, calibrate, environment_info, fit_model
from inspectai.models.classical import ClassicalConfig, ClassicalModel

OPTIONS = EvalOptions(eval_resolution=32, bootstrap_resamples=50)
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fit_model.py"


def _fit(synthetic: SyntheticVisA):
    dataset = VisADataset(synthetic.split_csv, synthetic.images_root)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    config = ClassicalConfig(resize_mode=resize_mode_for("cashew"))
    model = ClassicalModel(config)
    stats = fit_model(model, dataset, splits)
    return dataset, splits, model, stats, calibrate(model, dataset, splits, OPTIONS)


def _save(directory: Path, model, splits, calibration, stats) -> Path:
    return save_classical(
        directory,
        model,
        "cashew",
        splits,
        calibration.to_dict(),
        {**stats.to_dict(), "environment": environment_info()},
        32,
    )


def test_a_saved_baseline_reloads_and_predicts_identically(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    dataset, splits, model, stats, calibration = _fit(synthetic_visa)
    directory = _save(tmp_path / "art é", model, splits, calibration, stats)
    loaded, artifact = load_classical(directory, "cashew", model.config)
    assert artifact.split == splits
    for sample in dataset.get_samples("cashew", "test"):
        image = dataset.load_image(sample)
        original, reloaded = model.predict(image), loaded.predict(image)
        assert original.score == reloaded.score
        assert np.array_equal(original.anomaly_map, reloaded.anomaly_map)


def test_the_manifest_records_the_baseline_without_any_weights(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    _, splits, model, stats, calibration = _fit(synthetic_visa)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["model_type"] == "classical" and manifest["category"] == "cashew"
    assert manifest["weights"] is None and manifest["backbone"] is None
    assert manifest["resize_mode"] == "letterbox" and manifest["image_size"] == 256
    assert manifest["reference_aspect_ratio"] == pytest.approx(64 / 48)
    assert manifest["tensors"]["mean"] == {"shape": [256, 256, 3], "dtype": "float32"}
    assert manifest["threshold"]["value"] == calibration.image_threshold.value
    assert manifest["calibration"]["scores"]["max"] == calibration.scores["max"]
    assert manifest["fit_stats"]["n_fit_images"] == len(splits.fit)


def test_a_stale_wrong_type_or_tampered_baseline_is_rejected(
    synthetic_visa: SyntheticVisA, tmp_path: Path
) -> None:
    _, splits, model, stats, calibration = _fit(synthetic_visa)
    directory = _save(tmp_path / "art", model, splits, calibration, stats)
    with pytest.raises(ArtifactError, match="stale"):
        load_classical(directory, "cashew", ClassicalConfig(resize_mode="stretch"))
    with pytest.raises(ArtifactError, match="category"):
        load_classical(directory, "pipe_fryum")
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    manifest["model_type"] = "gaussian"
    (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="model"):
        load_classical(directory)
    manifest["model_type"] = "classical"
    (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with (directory / TENSORS).open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"\x00\x00\xc0\x7f")  # a float32 NaN
    with pytest.raises(ArtifactError, match="Checksum"):
        load_classical(directory)


@pytest.fixture
def fit_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fit_model", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_fit_command_writes_a_baseline_artifact_without_pretrained_weights(
    fit_cli: ModuleType,
    synthetic_visa: SyntheticVisA,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(synthetic_visa.data_root))
    monkeypatch.setenv("INSPECTAI_ARTIFACT_DIR", str(tmp_path / "artifacts out"))
    monkeypatch.setenv("INSPECTAI_REPORT_DIR", str(tmp_path / "reports out"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty torch home"))  # no weights anywhere
    assert fit_cli.main(["--model", "classical", "--category", "cashew"]) == 0
    directory = tmp_path / "artifacts out" / "classical" / "cashew"
    assert {p.name for p in directory.iterdir()} == {
        "manifest.json",
        "tensors.safetensors",
        "split.json",
        "config.snapshot.yaml",
    }
    assert "Fitted classical / cashew" in capsys.readouterr().out
    model, _ = load_classical(directory, "cashew")
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    refit = ClassicalModel(ClassicalConfig(resize_mode=resize_mode_for("cashew")))
    fit_model(refit, dataset, splits)
    image = dataset.load_image(dataset.get_samples("cashew", "test", "anomaly")[0])
    assert model.predict(image).score == refit.predict(image).score  # same as a fresh fit
