"""Artifact store: round trip and rejection of every kind of corruption (SPEC.md Section 9.1)."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from inspectai.artifacts.store import (
    CONFIG,
    MANIFEST,
    SPLIT,
    TENSORS,
    artifact_dir,
    load_artifact,
    save_artifact,
)
from inspectai.data.splits import SplitResult, file_list_sha256, save_split
from inspectai.errors import ArtifactError

CONFIG_DICT: dict[str, object] = {"d": 3, "layers": ["layer2"], "backbone": "fake"}
INFO: dict[str, object] = {
    "backbone": "fake",
    "threshold": {"policy": "normal_percentile", "value": 1.5},
}


def _split(object_name: str = "cashew") -> SplitResult:
    return SplitResult(
        object_name=object_name,
        seed=42,
        calibration_fraction=0.1,
        min_calibration=1,
        fit=("c/a.jpg", "c/b.jpg", "c/c.jpg"),
        calibration_normal=("c/d.jpg",),
        test=("c/e.jpg", "c/f.jpg"),
    )


def _tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "mean": rng.random((4, 3)).astype(np.float32),
        "inv_cov": rng.random((4, 3, 3)).astype(np.float32),
        "channel_index": np.arange(3, dtype=np.int64),
    }


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    directory = artifact_dir(tmp_path / "artifacts é", "gaussian", "cashew")
    save_artifact(directory, "gaussian", "cashew", _tensors(), _split(), CONFIG_DICT, INFO)
    return directory


def _rewrite_manifest(directory: Path, **changes: object) -> None:
    path = directory / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _refresh_checksum(directory: Path, name: str) -> None:
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    manifest["files"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def test_the_artifact_directory_is_model_then_category(tmp_path: Path) -> None:
    assert artifact_dir(tmp_path, "gaussian", "cashew") == tmp_path / "gaussian" / "cashew"


def test_a_saved_artifact_loads_back_unchanged(saved: Path) -> None:
    artifact = load_artifact(saved, "gaussian", "cashew", CONFIG_DICT)
    expected = _tensors()
    assert set(artifact.tensors) == set(expected)
    for name, array in expected.items():
        assert artifact.tensors[name].dtype == array.dtype
        assert np.array_equal(artifact.tensors[name], array)
    assert artifact.split == _split() and artifact.config == CONFIG_DICT


def test_the_manifest_records_what_the_spec_requires(saved: Path) -> None:
    manifest = json.loads((saved / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model_type"] == "gaussian" and manifest["category"] == "cashew"
    assert manifest["tensors"]["inv_cov"] == {"shape": [4, 3, 3], "dtype": "float32"}
    assert manifest["tensors"]["channel_index"]["dtype"] == "int64"
    assert set(manifest["files"]) == {TENSORS, SPLIT, CONFIG}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((saved / name).read_bytes()).hexdigest() == digest
    assert {"python", "numpy", "safetensors", "torch"} <= set(manifest["library_versions"])
    assert "created_utc" in manifest and manifest["threshold"]["value"] == 1.5
    assert sorted(p.name for p in saved.iterdir()) == sorted([MANIFEST, TENSORS, SPLIT, CONFIG])


def test_a_wrong_checksum_is_rejected(saved: Path) -> None:
    with (saved / TENSORS).open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x01")
    with pytest.raises(ArtifactError, match="Checksum"):
        load_artifact(saved)


def test_a_tampered_split_or_config_is_rejected(saved: Path) -> None:
    (saved / CONFIG).write_text(yaml.safe_dump({"d": 99}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="Checksum"):
        load_artifact(saved)


def test_a_wrong_shape_in_the_manifest_is_rejected(saved: Path) -> None:
    manifest = json.loads((saved / MANIFEST).read_text(encoding="utf-8"))
    manifest["tensors"]["mean"]["shape"] = [4, 5]
    _rewrite_manifest(saved, tensors=manifest["tensors"])
    with pytest.raises(ArtifactError, match="shape"):
        load_artifact(saved)


def test_a_wrong_dtype_in_the_manifest_is_rejected(saved: Path) -> None:
    manifest = json.loads((saved / MANIFEST).read_text(encoding="utf-8"))
    manifest["tensors"]["mean"]["dtype"] = "float16"
    _rewrite_manifest(saved, tensors=manifest["tensors"])
    with pytest.raises(ArtifactError, match="dtype"):
        load_artifact(saved)


def test_missing_or_extra_tensors_are_rejected(saved: Path) -> None:
    tensors = _tensors()
    del tensors["mean"]
    save_file(tensors, str(saved / TENSORS))
    _refresh_checksum(saved, TENSORS)
    with pytest.raises(ArtifactError, match="Tensor names"):
        load_artifact(saved)


def test_nan_values_are_rejected_on_load(saved: Path) -> None:
    tensors = _tensors()
    tensors["mean"][0, 0] = np.nan
    save_file(tensors, str(saved / TENSORS))
    _refresh_checksum(saved, TENSORS)
    with pytest.raises(ArtifactError, match="NaN"):
        load_artifact(saved)


def test_infinite_values_are_rejected_on_load(saved: Path) -> None:
    tensors = _tensors()
    tensors["inv_cov"][1, 1, 1] = np.inf
    save_file(tensors, str(saved / TENSORS))
    _refresh_checksum(saved, TENSORS)
    with pytest.raises(ArtifactError, match="NaN or infinity"):
        load_artifact(saved)


def test_nan_values_are_rejected_on_save(tmp_path: Path) -> None:
    tensors = _tensors()
    tensors["mean"][0, 0] = np.nan
    with pytest.raises(ArtifactError, match="NaN"):
        save_artifact(tmp_path / "a", "gaussian", "cashew", tensors, _split(), CONFIG_DICT, INFO)
    assert not (tmp_path / "a" / MANIFEST).exists()


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_an_unknown_schema_version_is_rejected(saved: Path, version: object) -> None:
    _rewrite_manifest(saved, schema_version=version)
    with pytest.raises(ArtifactError, match="schema_version"):
        load_artifact(saved)


def test_a_pickle_file_in_place_of_the_tensors_is_rejected(saved: Path) -> None:
    (saved / TENSORS).write_bytes(pickle.dumps(_tensors()))
    _refresh_checksum(saved, TENSORS)  # the checksum matches, so only the format can fail
    with pytest.raises(ArtifactError, match="safetensors"):
        load_artifact(saved)


@pytest.mark.parametrize("name", ["extra.pkl", "model.pickle", "weights.pt", "state.pth"])
def test_stray_pickle_style_files_are_rejected(saved: Path, name: str) -> None:
    (saved / name).write_bytes(pickle.dumps({"x": 1}))
    with pytest.raises(ArtifactError, match="not allowed"):
        load_artifact(saved)


def test_a_missing_file_manifest_or_folder_is_rejected(saved: Path, tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="folder not found"):
        load_artifact(tmp_path / "nowhere")
    (saved / SPLIT).unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(saved)
    (saved / MANIFEST).unlink()
    with pytest.raises(ArtifactError, match="manifest is missing"):
        load_artifact(saved)


def test_an_unreadable_manifest_is_rejected(saved: Path) -> None:
    (saved / MANIFEST).write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="unreadable"):
        load_artifact(saved)
    (saved / MANIFEST).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ArtifactError, match="JSON object"):
        load_artifact(saved)


def test_a_manifest_that_lists_the_wrong_files_is_rejected(saved: Path) -> None:
    _rewrite_manifest(saved, files={TENSORS: "0" * 64})
    with pytest.raises(ArtifactError, match="exactly these data files"):
        load_artifact(saved)


def _overlapping_split_json(directory: Path) -> None:
    """Write a split.json with a test file inside `fit`, with all hashes made consistent."""
    data = json.loads((directory / SPLIT).read_text(encoding="utf-8"))
    data["fit"] = [*data["fit"], data["test"][0]]
    data["file_list_sha256"]["fit"] = file_list_sha256(data["fit"])
    (directory / SPLIT).write_text(json.dumps(data), encoding="utf-8")
    _refresh_checksum(directory, SPLIT)


def test_a_split_listing_a_test_file_in_fit_is_rejected(saved: Path) -> None:
    _overlapping_split_json(saved)
    with pytest.raises(ArtifactError, match="overlap"):
        load_artifact(saved)


def test_a_split_for_another_category_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="split is for"):
        save_artifact(
            tmp_path / "a",
            "gaussian",
            "cashew",
            _tensors(),
            _split("pipe_fryum"),
            CONFIG_DICT,
            INFO,
        )
    directory = tmp_path / "b"
    save_artifact(directory, "gaussian", "cashew", _tensors(), _split(), CONFIG_DICT, INFO)
    save_split(_split("pipe_fryum"), directory / SPLIT)
    _refresh_checksum(directory, SPLIT)
    with pytest.raises(ArtifactError, match="manifest says otherwise"):
        load_artifact(directory)


def test_a_stale_config_snapshot_is_rejected(saved: Path) -> None:
    assert load_artifact(saved, expected_config=CONFIG_DICT).config == CONFIG_DICT
    with pytest.raises(ArtifactError, match="stale"):
        load_artifact(saved, expected_config={**CONFIG_DICT, "d": 4})


def test_the_expected_model_type_and_category_are_enforced(saved: Path) -> None:
    with pytest.raises(ArtifactError, match="model"):
        load_artifact(saved, expected_model_type="patchcore")
    with pytest.raises(ArtifactError, match="category"):
        load_artifact(saved, expected_category="pipe_fryum")


def test_info_cannot_overwrite_reserved_manifest_keys(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="reserved"):
        save_artifact(
            tmp_path / "a",
            "gaussian",
            "cashew",
            _tensors(),
            _split(),
            CONFIG_DICT,
            {"schema_version": 99},
        )


def test_unsupported_dtypes_are_rejected_on_save(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="dtype"):
        save_artifact(
            tmp_path / "a",
            "gaussian",
            "cashew",
            {"x": np.zeros(3, dtype=np.float64)},
            _split(),
            CONFIG_DICT,
            INFO,
        )
    with pytest.raises(ArtifactError, match="at least one tensor"):
        save_artifact(tmp_path / "b", "gaussian", "cashew", {}, _split(), CONFIG_DICT, INFO)


def test_float16_tensors_are_supported(tmp_path: Path) -> None:
    tensors = {"mean": np.ones((2, 2), dtype=np.float16)}
    directory = tmp_path / "half"
    save_artifact(directory, "gaussian", "cashew", tensors, _split(), CONFIG_DICT, INFO)
    assert load_artifact(directory).tensors["mean"].dtype == np.float16


def test_resaving_replaces_the_artifact_and_an_interrupted_save_does_not_load(saved: Path) -> None:
    changed = {**_tensors(), "mean": np.zeros((4, 3), dtype=np.float32)}
    save_artifact(saved, "gaussian", "cashew", changed, _split(), CONFIG_DICT, INFO)
    assert np.array_equal(load_artifact(saved).tensors["mean"], changed["mean"])
    (saved / MANIFEST).unlink()  # what an interrupted save looks like: no manifest yet
    with pytest.raises(ArtifactError):
        load_artifact(saved)


def test_the_runtime_section_is_stored_but_ignored_by_the_stale_check(tmp_path: Path) -> None:
    config = {**CONFIG_DICT, "runtime": {"chunk_size": 8, "max_bank_gb": 4.0}}
    directory = tmp_path / "rt"
    save_artifact(directory, "gaussian", "cashew", _tensors(), _split(), config, INFO)
    assert load_artifact(directory).config == config  # the exact configuration is kept
    assert load_artifact(directory, expected_config=CONFIG_DICT).config == config
    other_runtime = {**CONFIG_DICT, "runtime": {"chunk_size": 999}}
    assert load_artifact(directory, expected_config=other_runtime).config == config
    with pytest.raises(ArtifactError, match="stale"):
        load_artifact(directory, expected_config={**CONFIG_DICT, "d": 4})
