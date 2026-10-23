"""Configuration validation and environment-driven paths."""

from pathlib import Path

import pytest

from inspectai.config import (
    PreprocessConfig,
    SplitConfig,
    load_app_paths,
    load_paths,
    use_bundled_weights,
)
from inspectai.errors import ConfigError


def test_load_paths_uses_data_root_defaults() -> None:
    paths = load_paths({"INSPECTAI_DATA_ROOT": "D:/data"})
    assert paths.data_root == Path("D:/data")
    assert paths.artifact_dir == Path("D:/data") / "artifacts"
    assert paths.report_dir == Path("D:/data") / "reports"
    assert paths.visa_images_root == Path("D:/data") / "visa" / "VisA"
    assert paths.split_csv == Path("D:/data") / "visa" / "split_csv" / "1cls.csv"


def test_load_paths_reads_every_variable() -> None:
    paths = load_paths(
        {
            "INSPECTAI_DATA_ROOT": "D:/data",
            "INSPECTAI_ARTIFACT_DIR": "E:/models",
            "INSPECTAI_REPORT_DIR": "F:/out",
        }
    )
    assert paths.artifact_dir == Path("E:/models")
    assert paths.report_dir == Path("F:/out")


@pytest.mark.parametrize("env", [{}, {"INSPECTAI_DATA_ROOT": ""}, {"INSPECTAI_DATA_ROOT": "  "}])
def test_load_paths_requires_data_root(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="INSPECTAI_DATA_ROOT"):
        load_paths(env)


def test_load_paths_reads_the_real_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", "D:/from-env")
    assert load_paths().data_root == Path("D:/from-env")


def test_split_config_defaults_match_the_spec() -> None:
    config = SplitConfig()
    assert (config.seed, config.calibration_fraction, config.min_calibration) == (42, 0.10, 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calibration_fraction": 0.0},
        {"calibration_fraction": 1.0},
        {"calibration_fraction": -0.1},
        {"min_calibration": 0},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_split_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        SplitConfig(**kwargs)  # type: ignore[arg-type]


def test_preprocess_config_defaults() -> None:
    config = PreprocessConfig()
    assert config.image_size == 224
    assert config.resize_mode == "stretch"
    assert config.aspect_tolerance == 0.25
    assert config.max_pixels == 4096 * 4096


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 4},
        {"resize_mode": "crop"},
        {"aspect_tolerance": 0.0},
        {"aspect_tolerance": 1.5},
        {"max_pixels": 0},
    ],
)
def test_preprocess_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        PreprocessConfig(**kwargs)  # type: ignore[arg-type]


def test_every_object_has_a_valid_resize_mode() -> None:
    from inspectai.config import CATEGORY_RESIZE_MODES, RESIZE_MODES
    from inspectai.data.dataset import SUPPORTED_OBJECTS

    assert set(CATEGORY_RESIZE_MODES) == set(SUPPORTED_OBJECTS)
    assert set(CATEGORY_RESIZE_MODES.values()) <= set(RESIZE_MODES)


def test_resize_mode_is_chosen_per_category_and_can_be_overridden() -> None:
    from inspectai.config import resize_mode_for

    assert resize_mode_for("cashew") == "letterbox"
    assert resize_mode_for("cashew", {"cashew": "stretch"}) == "stretch"
    assert PreprocessConfig.for_object("pcb3").resize_mode == "letterbox"
    assert (
        PreprocessConfig.for_object("cashew", {"cashew": "stretch"}, image_size=64).image_size == 64
    )
    with pytest.raises(ConfigError):
        resize_mode_for("unknown_object")
    with pytest.raises(ConfigError):
        resize_mode_for("cashew", {"cashew": "crop"})


def test_preprocess_config_rejects_a_bad_max_side() -> None:
    with pytest.raises(ConfigError):
        PreprocessConfig(max_side=0)
    assert PreprocessConfig().max_side == 4096


def test_app_paths_with_a_data_root_match_load_paths() -> None:
    env = {"INSPECTAI_DATA_ROOT": "D:/data", "INSPECTAI_REPORT_DIR": "F:/out"}
    assert load_app_paths(env) == load_paths(env)
    assert load_app_paths(env).has_dataset


def test_app_paths_without_a_data_root_use_the_explicit_directories(tmp_path: Path) -> None:
    paths = load_app_paths(
        {"INSPECTAI_ARTIFACT_DIR": "E:/models", "INSPECTAI_REPORT_DIR": "F:/out"},
        bundle_root=tmp_path / "missing",
    )
    assert paths.data_root is None and not paths.has_dataset
    assert (paths.artifact_dir, paths.report_dir) == (Path("E:/models"), Path("F:/out"))


def test_app_paths_fall_back_to_the_bundled_folder(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    paths = load_app_paths({}, bundle_root=tmp_path)
    assert paths.artifact_dir == tmp_path / "artifacts"
    assert paths.report_dir == tmp_path / "reports"
    mixed = load_app_paths({"INSPECTAI_REPORT_DIR": "F:/out"}, bundle_root=tmp_path)
    assert mixed.artifact_dir == tmp_path / "artifacts" and mixed.report_dir == Path("F:/out")


def test_app_paths_without_any_source_are_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="hosted"):
        load_app_paths({}, bundle_root=tmp_path / "nothing")


def test_dataset_locations_need_a_data_root(tmp_path: Path) -> None:
    paths = load_app_paths({"INSPECTAI_REPORT_DIR": str(tmp_path)}, bundle_root=tmp_path / "x")
    with pytest.raises(ConfigError, match="INSPECTAI_DATA_ROOT"):
        _ = paths.split_csv


def test_bundled_weights_are_used_only_for_a_dataset_free_run_without_torch_home(
    tmp_path: Path,
) -> None:
    (tmp_path / "torch_cache" / "hub" / "checkpoints").mkdir(parents=True)
    hosted = load_app_paths({"INSPECTAI_REPORT_DIR": str(tmp_path)}, bundle_root=tmp_path)
    local = load_app_paths({"INSPECTAI_DATA_ROOT": str(tmp_path)})

    env: dict[str, str] = {}
    assert use_bundled_weights(local, env, tmp_path) is None and env == {}
    assert use_bundled_weights(hosted, env, tmp_path) == tmp_path / "torch_cache"
    assert env == {"TORCH_HOME": str(tmp_path / "torch_cache")}

    own = {"TORCH_HOME": "E:/torch"}
    assert use_bundled_weights(hosted, own, tmp_path) is None and own == {"TORCH_HOME": "E:/torch"}
    assert use_bundled_weights(hosted, {}, tmp_path / "no bundle") is None
