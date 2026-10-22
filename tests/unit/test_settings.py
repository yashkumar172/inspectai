"""configs/default.yaml: it must be valid, resolve environment variables, and match the code."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from inspectai.config import CATEGORY_RESIZE_MODES, PROFILES, PreprocessConfig, SplitConfig
from inspectai.errors import ConfigError
from inspectai.evaluation.runner import EvalOptions
from inspectai.evaluation.thresholds import DEFAULT_QUANTILE, NORMAL_PERCENTILE
from inspectai.inference.uploads import ASPECT_TOLERANCE, MAX_UPLOAD_BYTES, MIN_SIDE
from inspectai.models.classical import ClassicalConfig
from inspectai.models.gaussian import GaussianConfig
from inspectai.models.patchcore import PatchCoreConfig
from inspectai.settings import (
    DEFAULT_CONFIG_PATH,
    expand_environment,
    load_default_config,
)

ENV = {
    "INSPECTAI_DATA_ROOT": "D:/some data",
    "INSPECTAI_ARTIFACT_DIR": "E:/models",
    "INSPECTAI_REPORT_DIR": "F:/out",
    "TORCH_HOME": "G:/torch",
}


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ----- the shipped file ----------------------------------------------------------------------


def test_the_default_file_exists_loads_and_is_cpu_only() -> None:
    assert DEFAULT_CONFIG_PATH.is_file()
    config = load_default_config(env=ENV)
    assert config.device == "cpu" and config.seed == 42
    assert config.categories == ("cashew", "pipe_fryum", "chewinggum", "fryum")
    assert config.unresolved == ()


def test_environment_variables_are_expanded_and_nothing_machine_specific_is_stored() -> None:
    config = load_default_config(env=ENV)
    assert config.paths == {
        "data_root": "D:/some data",
        "artifact_dir": "E:/models",
        "report_dir": "F:/out",
        "torch_home": "G:/torch",
    }
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "${INSPECTAI_DATA_ROOT}" in text
    for personal in ("C:\\", "C:/", "Users", "yashr", "OneDrive"):
        assert personal not in text, personal  # SPEC.md Section 9.2


def test_unset_variables_are_reported_not_guessed() -> None:
    config = load_default_config(env={"INSPECTAI_DATA_ROOT": "D:/d"})
    assert config.paths["data_root"] == "D:/d" and config.paths["artifact_dir"] is None
    assert config.unresolved == ("INSPECTAI_ARTIFACT_DIR", "INSPECTAI_REPORT_DIR", "TORCH_HOME")
    assert load_default_config(env={}).unresolved == (
        "INSPECTAI_DATA_ROOT",
        "INSPECTAI_ARTIFACT_DIR",
        "INSPECTAI_REPORT_DIR",
        "TORCH_HOME",
    )


def test_expand_environment_handles_partial_and_missing_references() -> None:
    assert expand_environment("${A}/sub", {"A": "x"}) == ("x/sub", [])
    assert expand_environment("${A}/${B}", {"A": "x"}) == (None, ["B"])
    assert expand_environment("plain", {}) == ("plain", [])
    assert expand_environment("${A}", {"A": "   "}) == (None, ["A"])


# ----- the file documents the defaults of the code; it must never drift from them ------------------


def test_every_value_in_the_file_equals_the_default_in_the_code() -> None:
    raw = _raw()
    assert raw["splits"] == asdict(SplitConfig())
    assert raw["preprocessing"] == asdict(PreprocessConfig())
    assert raw["classical"] == asdict(ClassicalConfig())
    assert raw["thresholds"] == {"policy": NORMAL_PERCENTILE, "quantile": DEFAULT_QUANTILE}
    options = asdict(EvalOptions())
    assert raw["evaluation"] == {
        k: options[k] for k in ("eval_resolution", "bootstrap_resamples", "bootstrap_seed")
    }
    assert raw["resize_modes"] == CATEGORY_RESIZE_MODES
    assert {n: {**p, "layers": list(p["layers"])} for n, p in PROFILES.items()} == raw["profiles"]

    gaussian = {**GaussianConfig().to_dict(), "batch_size": 8, "num_threads": 6}
    for key in ("backbone", "layers", "image_size", "resize_mode"):
        gaussian.pop(key)  # these come from the profile and the category
    assert raw["gaussian"] == gaussian

    patchcore = {**PatchCoreConfig().to_dict(), **PatchCoreConfig().runtime_dict()}
    for key in ("backbone", "layers", "image_size", "resize_mode"):
        patchcore.pop(key)
    assert raw["patchcore"] == {**patchcore, "max_fit_images": None}

    assert raw["app"] == {
        "max_upload_mb": MAX_UPLOAD_BYTES / 1024**2,
        "min_side": MIN_SIDE,
        "aspect_tolerance": ASPECT_TOLERANCE,
    }
    seeds = {raw["seed"], raw["splits"]["seed"], raw["gaussian"]["seed"], raw["patchcore"]["seed"]}
    assert seeds == {42} and raw["evaluation"]["bootstrap_seed"] == 42


def test_the_loaded_objects_are_the_code_defaults_with_the_dev_profile() -> None:
    config = load_default_config(env=ENV)
    assert config.splits == SplitConfig() and config.preprocessing == PreprocessConfig()
    assert config.classical == ClassicalConfig() and config.evaluation == EvalOptions()
    assert config.gaussian == GaussianConfig() and config.patchcore == PatchCoreConfig()
    assert config.gaussian.backbone == "resnet18" and config.max_fit_images is None
    assert config.resize_modes == CATEGORY_RESIZE_MODES and config.app["max_upload_mb"] == 10


# ----- validation ----------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda", "gpu", "auto", "mps", None])
def test_any_device_other_than_cpu_is_rejected(tmp_path: Path, device: object) -> None:
    raw = _raw()
    raw["device"] = device
    with pytest.raises(ConfigError, match="only CPU execution is supported"):
        load_default_config(_write(tmp_path, raw), ENV)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("splits", "calibration_fraction", 1.5, "splits"),
        ("preprocessing", "resize_mode", "crop", "preprocessing"),
        ("classical", "std_floor", 0.0, "classical"),
        ("gaussian", "epsilon", -1.0, "gaussian"),
        ("gaussian", "d", 0, "gaussian"),
        ("patchcore", "coreset_ratio", 2.0, "patchcore"),
        ("patchcore", "neighborhood", 2, "patchcore"),
        ("patchcore", "max_bank_gb", 0, "patchcore"),
        ("thresholds", "quantile", 1.5, "thresholds/evaluation"),
        ("thresholds", "policy", "magic", "thresholds/evaluation"),
        ("evaluation", "eval_resolution", "big", "thresholds/evaluation"),
    ],
)
def test_invalid_values_are_rejected_with_the_section_name(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        load_default_config(_write(tmp_path, raw), ENV)


def test_unknown_keys_and_sections_are_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["gaussian"]["colour"] = "red"
    with pytest.raises(ConfigError, match="gaussian"):
        load_default_config(_write(tmp_path, raw), ENV)
    raw = _raw()
    raw["extras"] = {}
    with pytest.raises(ConfigError, match="unknown sections"):
        load_default_config(_write(tmp_path, raw), ENV)
    raw = _raw()
    del raw["app"]
    with pytest.raises(ConfigError, match="lacks sections"):
        load_default_config(_write(tmp_path, raw), ENV)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(categories=["cashew", "not_an_object"]),
        lambda r: r.update(categories=[]),
        lambda r: r["resize_modes"].pop("fryum"),
        lambda r: r["resize_modes"].update(fryum="crop"),
        lambda r: r["profiles"].pop("full"),
        lambda r: r["patchcore"].update(max_fit_images=0),
        lambda r: r.update(seed="forty-two"),
        lambda r: r["app"].update(max_upload_mb="ten"),
        lambda r: r["paths"].update(data_root=5),
        lambda r: r.update(splits="not a mapping"),
    ],
)
def test_structurally_wrong_files_are_rejected(tmp_path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    raw = _raw()
    mutate(raw)
    with pytest.raises(ConfigError):
        load_default_config(_write(tmp_path, raw), ENV)


def test_a_missing_or_unreadable_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not be read"):
        load_default_config(tmp_path / "missing.yaml", ENV)
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not be read"):
        load_default_config(bad, ENV)
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_default_config(bad, ENV)


def test_a_valid_override_is_accepted(tmp_path: Path) -> None:
    raw = _raw()
    raw["patchcore"]["coreset_ratio"] = 0.05  # the hosted-profile ratio of SPEC.md Section 15
    raw["patchcore"]["max_fit_images"] = 200
    config = load_default_config(_write(tmp_path, raw), ENV)
    assert config.patchcore.coreset_ratio == 0.05 and config.max_fit_images == 200
