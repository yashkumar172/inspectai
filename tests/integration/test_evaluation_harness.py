"""The evaluation harness and scripts/evaluate_model.py on a synthetic dataset."""

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from make_synthetic import SyntheticVisA, write_jpg

from inspectai.config import SplitConfig
from inspectai.data.dataset import VisADataset
from inspectai.data.splits import make_splits
from inspectai.evaluation.runner import EvalOptions, run_evaluation
from inspectai.models.classical import ClassicalConfig, ClassicalModel

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_model.py"


def _setup(synthetic: SyntheticVisA):
    dataset = VisADataset(synthetic.split_csv, synthetic.images_root)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    return dataset, splits


def _model() -> ClassicalModel:
    return ClassicalModel(ClassicalConfig(image_size=32, resize_mode="stretch"))


def _run(synthetic: SyntheticVisA, options: EvalOptions | None = None, model=None):
    dataset, splits = _setup(synthetic)
    options = options or EvalOptions(eval_resolution=32, bootstrap_resamples=100)
    return run_evaluation(model or _model(), "classical", {}, dataset, splits, options), splits


def test_the_baseline_is_measured_with_real_metrics(synthetic_visa: SyntheticVisA) -> None:
    result, _ = _run(synthetic_visa)
    metrics = result.summary["metrics"]
    image, pixel = metrics["image"], metrics["pixel"]
    assert metrics["evaluation_split"] == "test"
    assert (image["n_normal"], image["n_anomalous"]) == (5, 5)
    assert image["auroc"] > 0.95 and pixel["auroc"] > 0.9
    assert image["auroc_ci95"][0] <= image["auroc"] <= image["auroc_ci95"][1]
    cm = image["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == 5 and cm["tn"] + cm["fp"] == 5


def test_rows_carry_split_file_labels_and_the_stored_threshold(
    synthetic_visa: SyntheticVisA,
) -> None:
    result, splits = _run(synthetic_visa)
    assert sorted(r.image for r in result.rows) == sorted(splits.test)
    threshold = result.summary["threshold"]["image"]["value"]
    for row in result.rows:
        assert row.label == int("/Anomaly/" in row.image)
        assert row.predicted == int(row.score > threshold) and row.threshold == threshold


def test_the_run_is_reproducible(synthetic_visa: SyntheticVisA) -> None:
    first, _ = _run(synthetic_visa)
    second, _ = _run(synthetic_visa)
    assert [r.score for r in first.rows] == [r.score for r in second.rows]
    assert first.summary["metrics"] == second.summary["metrics"]


class _SpyModel(ClassicalModel):
    """Records which images the model is fitted on and the order of predict calls."""

    def __init__(self) -> None:
        super().__init__(ClassicalConfig(image_size=32, resize_mode="stretch"))
        self.fit_count = 0
        self.predict_count = 0

    def fit(self, images):  # type: ignore[no-untyped-def]
        listed = list(images)
        self.fit_count = len(listed)
        super().fit(listed)

    def predict(self, image):  # type: ignore[no-untyped-def]
        self.predict_count += 1
        return super().predict(image)


def test_fitting_uses_only_the_fit_split_and_thresholds_come_before_test_scoring(
    synthetic_visa: SyntheticVisA,
) -> None:
    spy = _SpyModel()
    result, splits = _run(synthetic_visa, model=spy)
    assert spy.fit_count == len(splits.fit)
    assert spy.predict_count == len(splits.calibration_normal) + len(splits.test)
    assert result.summary["split"]["counts"]["fit"] == len(splits.fit)


def test_changing_test_images_does_not_change_the_threshold_or_the_model(
    synthetic_visa: SyntheticVisA,
) -> None:
    """Leakage check: the normal_percentile threshold depends only on fit/calibration data."""
    before, _ = _run(synthetic_visa)
    dataset, splits = _setup(synthetic_visa)
    for rel in splits.test:  # overwrite every test image with something completely different
        sample = next(s for s in dataset.samples if s.image_rel == rel)
        write_jpg(sample.image_path, dataset.load_image(sample)[::-1, ::-1].copy() // 2)
    after, _ = _run(synthetic_visa)
    assert before.summary["threshold"] == after.summary["threshold"]
    assert before.summary["calibration_scores"] == after.summary["calibration_scores"]
    assert [r.score for r in before.rows] != [r.score for r in after.rows]


def test_labeled_calibration_reports_on_test_eval_only(synthetic_visa: SyntheticVisA) -> None:
    options = EvalOptions(
        threshold_policy="labeled_calibration", eval_resolution=32, bootstrap_resamples=50
    )
    result, splits = _run(synthetic_visa, options)
    metrics = result.summary["metrics"]
    assert metrics["evaluation_split"] == "test_eval"
    # 5 normal + 5 anomalous test rows: 3 + 3 go to calibration, 2 + 2 to evaluation.
    assert len(result.rows) == 4 and {r.split for r in result.rows} == {"test_eval"}
    assert metrics["image"]["n_normal"] + metrics["image"]["n_anomalous"] == len(result.rows)
    assert len(result.rows) < len(splits.test)
    extra = result.summary["threshold"]
    assert extra["test_calibration_images"] + extra["test_eval_images"] == len(splits.test)


def test_a_single_class_test_set_gives_undefined_metrics(synthetic_visa: SyntheticVisA) -> None:
    dataset, splits = _setup(synthetic_visa)
    only_normal = [n for n in splits.test if "/Normal/" in n]
    from dataclasses import replace

    narrowed = replace(splits, test=tuple(only_normal))
    result = run_evaluation(
        _model(), "classical", {}, dataset, narrowed, EvalOptions(eval_resolution=32)
    )
    image = result.summary["metrics"]["image"]
    assert image["auroc"] is None and image["auroc_ci95"] is None
    assert image["recall"] is None


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_model", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(synthetic_visa: SyntheticVisA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    reports = tmp_path / "reports out"
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(synthetic_visa.data_root))
    monkeypatch.setenv("INSPECTAI_ARTIFACT_DIR", str(tmp_path / "artifacts out"))
    monkeypatch.setenv("INSPECTAI_REPORT_DIR", str(reports))
    return reports


def test_the_command_writes_csv_json_and_example_figures(cli: ModuleType, env: Path) -> None:
    assert cli.main(["--model", "classical", "--category", "cashew"]) == 0
    out = env / "classical" / "cashew"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "classical" and summary["category"] == "cashew"
    assert summary["metrics"]["image"]["n_anomalous"] == 5
    assert summary["performance"]["fit_seconds"] >= 0 and "environment" in summary
    with (out / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 10
    assert len(list((out / "examples").glob("*.png"))) == 4


def test_the_command_accepts_labeled_calibration(cli: ModuleType, env: Path) -> None:
    args = ["--model", "classical", "--category", "pipe_fryum", "--threshold-policy"]
    assert cli.main([*args, "labeled_calibration"]) == 0
    summary = json.loads(
        (env / "classical" / "pipe_fryum" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["metrics"]["evaluation_split"] == "test_eval"


def test_the_command_exits_two_on_missing_configuration_or_data(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("INSPECTAI_DATA_ROOT", raising=False)
    assert cli.main(["--model", "classical", "--category", "cashew"]) == 2
    assert "INSPECTAI_DATA_ROOT" in capsys.readouterr().err


def test_the_command_exits_two_for_an_object_missing_from_the_split_file(
    cli: ModuleType, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--model", "classical", "--category", "pcb1"]) == 2
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantile": 0.0},
        {"quantile": 1.5},
        {"quantile": True},
        {"eval_resolution": 4},
        {"eval_resolution": "big"},
        {"eval_resolution": 0},
        {"bootstrap_resamples": 0},
        {"split_seed": 1.5},
        {"bootstrap_seed": "x"},
        {"threshold_policy": "magic"},
    ],
)
def test_invalid_evaluation_options_are_rejected_when_created(kwargs: dict[str, object]) -> None:
    from inspectai.errors import EvaluationError

    with pytest.raises(EvaluationError):
        EvalOptions(**kwargs)  # type: ignore[arg-type]


def test_valid_evaluation_options_are_accepted() -> None:
    options = EvalOptions(quantile=1.0, eval_resolution=8, bootstrap_resamples=1, split_seed=-3)
    assert options.quantile == 1.0 and options.split_seed == -3
