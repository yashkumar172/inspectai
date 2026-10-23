"""The hosted-bundle export: only the needed files, byte for byte, and a loadable result."""

import shutil
from pathlib import Path

import pytest
from make_world import World

from inspectai.deployment import export_hosted_bundle
from inspectai.errors import ArtifactError, ConfigError
from inspectai.evaluation.dashboard import load_results
from inspectai.inference.predictor import available_categories, option_by_key


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_the_bundle_holds_dashboard_files_and_classical_artifacts_only(
    world: World, tmp_path: Path
) -> None:
    summary = export_hosted_bundle(world.artifacts, world.reports, tmp_path / "hosted")
    files = _files(tmp_path / "hosted")
    assert summary.files == len(files) and summary.bytes > 0
    assert summary.report_categories == summary.artifact_categories == ("cashew", "pipe_fryum")
    assert "artifacts/classical/cashew/tensors.safetensors" in files
    assert "reports/gaussian/cashew/summary.json" in files
    assert not any(f.startswith(("artifacts/gaussian", "artifacts/patchcore")) for f in files)
    assert not any("/examples/" in f or f.startswith("artifacts/splits") for f in files)
    assert not any(f.endswith(".png") and "/comparison/" not in f for f in files)


def test_comparison_tables_and_figures_are_copied_but_run_examples_are_not(
    world: World, tmp_path: Path
) -> None:
    reports = tmp_path / "reports"
    shutil.copytree(world.reports, reports)
    (reports / "comparison").mkdir()
    for name in ("comparison.csv", "comparison.json", "comparison.md", "cashew_examples.png"):
        (reports / "comparison" / name).write_bytes(b"x")
    (reports / "classical" / "cashew" / "examples").mkdir()
    (reports / "classical" / "cashew" / "examples" / "normal_lowest_score.png").write_bytes(b"x")
    export_hosted_bundle(world.artifacts, reports, tmp_path / "hosted")
    files = _files(tmp_path / "hosted")
    assert {f for f in files if "/comparison/" in f} == {
        f"reports/comparison/{n}"
        for n in ("comparison.csv", "comparison.json", "comparison.md", "cashew_examples.png")
    }
    assert not any("/examples/" in f for f in files)


def test_copies_are_identical_and_the_results_validate(world: World, tmp_path: Path) -> None:
    export_hosted_bundle(world.artifacts, world.reports, tmp_path / "hosted")
    source = world.reports / "classical" / "cashew" / "predictions.csv"
    copy = tmp_path / "hosted" / "reports" / "classical" / "cashew" / "predictions.csv"
    assert copy.read_bytes() == source.read_bytes()
    results = load_results(tmp_path / "hosted" / "reports", tmp_path / "hosted" / "artifacts")
    assert set(results) == {"cashew", "pipe_fryum"} and all(r.valid for r in results.values())
    classical = option_by_key("classical")
    assert available_categories(tmp_path / "hosted" / "artifacts", classical) == [
        "cashew",
        "pipe_fryum",
    ]


def test_categories_can_be_limited(world: World, tmp_path: Path) -> None:
    summary = export_hosted_bundle(
        world.artifacts, world.reports, tmp_path / "hosted", categories=("cashew",)
    )
    assert summary.report_categories == summary.artifact_categories == ("cashew",)
    assert not any("pipe_fryum" in f for f in _files(tmp_path / "hosted"))


def test_an_existing_bundle_is_only_replaced_with_force(world: World, tmp_path: Path) -> None:
    target = tmp_path / "hosted"
    export_hosted_bundle(world.artifacts, world.reports, target)
    with pytest.raises(ConfigError, match="--force"):
        export_hosted_bundle(world.artifacts, world.reports, target)
    stale = target / "reports" / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    export_hosted_bundle(world.artifacts, world.reports, target, force=True)
    assert not stale.exists()


def test_bad_inputs_are_config_errors(world: World, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unsupported"):
        export_hosted_bundle(world.artifacts, world.reports, tmp_path / "h", categories=("nope",))
    with pytest.raises(ConfigError, match="must exist"):
        export_hosted_bundle(tmp_path / "none", world.reports, tmp_path / "h")
    with pytest.raises(ConfigError, match="must not contain"):
        export_hosted_bundle(world.artifacts, world.reports, world.artifacts.parent)
    empty = tmp_path / "empty"
    (empty / "a").mkdir(parents=True)
    (empty / "r").mkdir()
    with pytest.raises(ConfigError, match="Nothing to export"):
        export_hosted_bundle(empty / "a", empty / "r", tmp_path / "h2")


def test_a_corrupted_classical_artifact_fails_the_export(world: World, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(world.artifacts, artifacts)
    tensors = artifacts / "classical" / "cashew" / "tensors.safetensors"
    tensors.write_bytes(tensors.read_bytes()[:-8] + b"\x00" * 8)
    with pytest.raises(ArtifactError):
        export_hosted_bundle(artifacts, world.reports, tmp_path / "hosted")
