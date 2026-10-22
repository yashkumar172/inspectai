"""The end-to-end smoke script (`scripts/run_demo.py`) as a test."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_demo.py"


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_demo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def finished(demo: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("demo run") / "out dir é"
    assert demo.main(["--output-dir", str(out)]) == 0
    return out


def test_the_end_to_end_path_runs_and_writes_every_output(finished: Path) -> None:
    for name in (
        "comparison.json",
        "comparison.csv",
        "comparison.md",
        "visualization.png",
        "prediction_details.json",
    ):
        assert (finished / name).is_file() and (finished / name).stat().st_size > 0, name
    # the fitted artifacts and the evaluation reports of all three models exist
    for model in ("classical", "gaussian", "patchcore"):
        assert (finished / "world" / "artifacts out" / model / "cashew" / "manifest.json").is_file()
        assert (finished / "world" / "reports out" / model / "cashew" / "predictions.csv").is_file()


def test_the_demo_output_is_a_real_comparison_with_verified_metrics(finished: Path) -> None:
    comparison = json.loads((finished / "comparison.json").read_text(encoding="utf-8"))
    runs = comparison["categories"]["cashew"]["runs"]
    assert [r["model"] for r in runs] == ["classical", "gaussian", "patchcore"]
    assert comparison["verification"]["image_metrics_rederived_from_predictions"] is True
    details = json.loads((finished / "prediction_details.json").read_text(encoding="utf-8"))
    assert details["score_is_probability"] is False and details["category"] == "cashew"
    assert details["image"]["source"] == "synthetic demo image"


def test_the_demo_prints_a_warning_that_its_metrics_are_synthetic(
    demo: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo.main(["--output-dir", str(tmp_path / "again")]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "meaningless for real performance" in out
    assert "30 predictions equal the saved evaluation reports" in out


def test_a_temporary_folder_is_removed_unless_kept(
    demo: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made: list[Path] = []
    real = demo.tempfile.mkdtemp

    def tracking(*args: object, **kwargs: object) -> str:
        path = real(*args, dir=str(tmp_path), **kwargs)  # type: ignore[arg-type]
        made.append(Path(path))
        return path

    monkeypatch.setattr(demo.tempfile, "mkdtemp", tracking)
    monkeypatch.setattr(demo, "run", lambda out_dir: None)  # the cleanup logic only
    assert demo.main([]) == 0 and not made[0].exists()
    assert demo.main(["--keep"]) == 0 and made[1].exists()


def test_a_non_empty_output_folder_is_refused(
    demo: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "something.txt").write_text("x", encoding="utf-8")
    assert demo.main(["--output-dir", str(tmp_path)]) == 1
    assert "new or empty" in capsys.readouterr().err


def test_a_failing_step_gives_exit_code_one_and_a_clear_message(
    demo: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(out_dir: Path) -> None:
        raise demo.DemoError("a saved prediction differs")

    monkeypatch.setattr(demo, "run", broken)
    assert demo.main(["--output-dir", str(tmp_path / "x")]) == 1
    assert "the demo failed: a saved prediction differs" in capsys.readouterr().err


def test_missing_fixtures_give_exit_code_two(
    demo: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(demo, "FIXTURES", tmp_path / "nowhere")
    assert demo.main([]) == 2
    assert "repository checkout" in capsys.readouterr().err
