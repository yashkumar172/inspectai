"""End-to-end test of scripts/validate_dataset.py on a synthetic dataset."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from make_synthetic import SyntheticVisA

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_dataset.py"


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(synthetic_visa: SyntheticVisA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the three environment variables at temporary folders; return the report folder."""
    reports = tmp_path / "reports out"
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(synthetic_visa.data_root))
    monkeypatch.setenv("INSPECTAI_ARTIFACT_DIR", str(tmp_path / "artifacts out"))
    monkeypatch.setenv("INSPECTAI_REPORT_DIR", str(reports))
    return reports


def test_a_valid_dataset_exits_zero_and_writes_a_json_report(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "cashew" in out and "No problems found" in out
    report = json.loads((env / "visa_validation.json").read_text(encoding="utf-8"))
    assert report["ok"] is True and report["rows_checked"] == 68


def test_a_missing_data_root_exits_two_with_a_clear_message(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("INSPECTAI_DATA_ROOT", raising=False)
    assert cli.main([]) == 2
    assert "INSPECTAI_DATA_ROOT" in capsys.readouterr().err


def test_a_missing_dataset_exits_two_and_says_where_to_put_it(
    cli: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(tmp_path / "empty"))
    assert cli.main([]) == 2
    assert "split_csv" in capsys.readouterr().err


def test_a_dataset_with_errors_exits_two_and_writes_no_splits(
    cli: ModuleType,
    env: Path,
    synthetic_visa: SyntheticVisA,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (synthetic_visa.images_root / "cashew/Data/Images/Normal/0001.JPG").unlink()
    assert cli.main(["--write-splits"]) == 2
    assert "missing_image" in capsys.readouterr().out
    assert not (tmp_path / "artifacts out").exists()
    assert (env / "visa_validation.json").is_file()


def test_write_splits_creates_one_file_per_object_and_is_repeatable(
    cli: ModuleType, env: Path, tmp_path: Path
) -> None:
    assert cli.main(["--write-splits"]) == 0
    first = tmp_path / "artifacts out" / "splits" / "cashew" / "split.json"
    second = tmp_path / "artifacts out" / "splits" / "pipe_fryum" / "split.json"
    assert first.is_file() and second.is_file()
    before = first.read_text(encoding="utf-8")
    assert cli.main(["--write-splits"]) == 0
    assert first.read_text(encoding="utf-8") == before


def test_write_splits_refuses_to_overwrite_a_different_split(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--write-splits", "--seed", "1"]) == 0
    assert cli.main(["--write-splits", "--seed", "2"]) == 2
    assert "differs" in capsys.readouterr().err


def test_objects_can_be_selected(cli: ModuleType, env: Path) -> None:
    assert cli.main(["--objects", "pipe_fryum"]) == 0
    report = json.loads((env / "visa_validation.json").read_text(encoding="utf-8"))
    assert list(report["objects"]) == ["pipe_fryum"]
