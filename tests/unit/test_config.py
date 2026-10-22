"""Configuration validation and environment-driven paths."""

from pathlib import Path

import pytest

from inspectai.config import PreprocessConfig, SplitConfig, load_paths
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
