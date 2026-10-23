"""Comparison reports: comparability checks, verification against predictions, tables."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from inspectai.errors import EvaluationError
from inspectai.evaluation.comparison import (
    CSV_COLUMNS,
    RUN_SPECS,
    Run,
    build_comparison,
    check_comparable,
    comparison_rows,
    format_markdown,
    load_run,
    read_predictions,
    verify_run,
    write_comparison_csv,
    write_comparison_json,
)
from inspectai.evaluation.report import PredictionRow, write_predictions_csv
from inspectai.evaluation.runner import EvalOptions, _image_metrics

SPECS = {spec.label: spec for spec in RUN_SPECS}
CLASSICAL, GAUSSIAN, PATCHCORE = (
    SPECS["A classical"],
    SPECS["B Gaussian dev"],
    SPECS["C PatchCore-inspired dev"],
)
SPLIT_HASHES = {"fit": "a" * 64, "calibration_normal": "b" * 64, "test": "c" * 64}
OPTIONS = EvalOptions(eval_resolution=256, bootstrap_resamples=40)
LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
IMAGES = [
    f"cashew/Data/Images/{'Normal' if y == 0 else 'Anomaly'}/{i:03d}.JPG"
    for i, y in enumerate(LABELS)
]


def _scores(quality: float, seed: int) -> list[float]:
    """Scores whose ranking quality grows with `quality` (0 = random)."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=len(LABELS)) + quality * np.array(LABELS)).tolist()


def make_run(
    spec=GAUSSIAN,
    category="cashew",
    scores=None,
    labels=LABELS,
    images=IMAGES,
    threshold=0.5,
    pixel=None,
) -> Run:
    scores = _scores(2.0, 1) if scores is None else scores
    image_metrics = _image_metrics(np.array(scores), np.array(labels), threshold, OPTIONS)
    summary = {
        "model": spec.model,
        "category": category,
        "created_utc": "2026-10-23T00:00:00+00:00",
        "model_config": {"backbone": "wide_resnet50_2" if spec.profile == "full" else "resnet18"},
        "options": {
            "threshold_policy": "normal_percentile",
            "quantile": 0.99,
            "eval_resolution": 256,
            "split_seed": 42,
            "bootstrap_resamples": OPTIONS.bootstrap_resamples,
            "bootstrap_seed": OPTIONS.bootstrap_seed,
        },
        "split": {"seed": 42, "file_list_sha256": dict(SPLIT_HASHES)},
        "metrics": {
            "evaluation_split": "test",
            "evaluation_resolution": 256,
            "image": image_metrics,
            "pixel": pixel
            or {"auroc": 0.9, "average_precision": 0.5, "threshold": 1.0, "iou": 0.2, "dice": 0.33},
        },
        "threshold": {"image": {"policy": "normal_percentile", "value": threshold, "params": {}}},
        "calibration_scores": {"min": 0.1, "median": 0.2, "max": 0.4},
        "performance": {"fit_seconds": 12.0, "latency_seconds_mean": 0.05, "n_fit_images": 5},
        "environment": {"python": "3.12", "power_plan": "Balanced", "plugged_in": True},
    }
    predictions = [
        {
            "image": image,
            "split": "test",
            "label": label,
            "score": float(score),
            "predicted": int(score > threshold),
            "threshold": threshold,
        }
        for image, label, score in zip(images, labels, scores, strict=True)
    ]
    return Run(spec, category, summary, predictions, None)


# ----- verification against predictions ------------------------------------------------------


def test_a_consistent_run_verifies() -> None:
    assert verify_run(make_run()) == []


def test_a_changed_score_is_detected() -> None:
    run = make_run()
    run.predictions[7]["score"] += 5.0  # now ranks differently than the stored AUROC says
    run.predictions[0]["score"] = 99.0
    problems = verify_run(run)
    assert any("AUROC" in p for p in problems)


def test_an_inconsistent_decision_flag_is_detected() -> None:
    run = make_run()
    run.predictions[3]["predicted"] = 1 - run.predictions[3]["predicted"]
    assert any("predicted" in p for p in verify_run(run))


def test_wrong_counts_threshold_and_confusion_matrix_are_detected() -> None:
    run = make_run()
    run.summary["metrics"]["image"]["n_normal"] = 4
    assert any("label counts" in p for p in verify_run(run))
    run = make_run()
    run.summary["metrics"]["image"]["confusion_matrix"]["tp"] += 1
    assert any("confusion matrix" in p for p in verify_run(run))
    run = make_run()
    run.predictions[2]["threshold"] = 123.0
    assert any("thresholds" in p for p in verify_run(run))
    run = make_run()
    run.summary["metrics"]["image"]["f1"] = 0.123456
    assert any("f1" in p for p in verify_run(run))
    run = make_run()
    run.summary["metrics"]["image"]["auroc_ci95"] = [0.0, 0.1]
    assert any("confidence interval" in p for p in verify_run(run))


def test_duplicate_images_are_detected() -> None:
    run = make_run()
    run.predictions[1]["image"] = run.predictions[0]["image"]
    assert any("duplicate" in p for p in verify_run(run))


# ----- comparability -------------------------------------------------------------------------


def _pair() -> list[Run]:
    return [make_run(CLASSICAL, scores=_scores(0.5, 2)), make_run(GAUSSIAN, scores=_scores(2.5, 3))]


def test_runs_with_identical_settings_are_comparable() -> None:
    check_comparable(_pair())


def test_a_comparison_needs_at_least_two_runs() -> None:
    with pytest.raises(EvaluationError, match="at least two"):
        check_comparable([make_run()])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["split"]["file_list_sha256"].update(test="d" * 64), "different split files"),
        (lambda s: s["split"].update(seed=7), "different split seeds"),
        (lambda s: s["metrics"].update(evaluation_resolution=128), "evaluation_resolution"),
        (lambda s: s["metrics"].update(evaluation_split="test_eval"), "evaluation_split"),
        (lambda s: s["options"].update(quantile=0.95), "option quantile"),
        (lambda s: s["options"].update(bootstrap_seed=1), "option bootstrap_seed"),
        (lambda s: s["options"].update(threshold_policy="labeled_calibration"), "threshold_policy"),
        (
            lambda s: s["threshold"]["image"].update(policy="labeled_calibration"),
            "threshold policy",
        ),
        (lambda s: s["metrics"]["image"].update(n_anomalous=4), "n_anomalous"),
    ],
)
def test_runs_that_differ_in_splits_or_evaluation_settings_are_refused(mutate, message) -> None:
    first, second = _pair()
    mutate(second.summary)
    with pytest.raises(EvaluationError, match=message):
        check_comparable([first, second])


def test_runs_on_different_test_images_or_labels_are_refused() -> None:
    first = make_run(CLASSICAL)
    other_images = make_run(GAUSSIAN, images=[f"other/{i}.JPG" for i in range(10)])
    with pytest.raises(EvaluationError, match="different test images"):
        check_comparable([first, other_images])
    flipped = make_run(GAUSSIAN, labels=[1, *LABELS[1:]])
    with pytest.raises(EvaluationError, match="different test images or labels"):
        check_comparable([first, flipped])


def test_runs_of_different_categories_are_refused() -> None:
    with pytest.raises(EvaluationError, match="different categories"):
        check_comparable([make_run(CLASSICAL), make_run(GAUSSIAN, category="pipe_fryum")])


# ----- building the comparison ---------------------------------------------------------------


def _runs(category: str) -> list[Run]:
    return [
        make_run(CLASSICAL, category, _scores(0.3, 4)),
        make_run(GAUSSIAN, category, _scores(2.0, 5)),
        make_run(PATCHCORE, category, _scores(3.0, 6)),
    ]


def test_the_comparison_has_per_category_tables_paired_differences_and_macro_averages() -> None:
    comparison = build_comparison(
        {"cashew": _runs("cashew"), "pipe_fryum": _runs("pipe_fryum")},
        {"cashew": ["B Gaussian full"], "pipe_fryum": ["B Gaussian full"]},
    )
    assert list(comparison["categories"]) == ["cashew", "pipe_fryum"]
    cashew = comparison["categories"]["cashew"]
    assert [r["label"] for r in cashew["runs"]] == [
        s.label for s in (CLASSICAL, GAUSSIAN, PATCHCORE)
    ]
    assert (cashew["n_normal"], cashew["n_anomalous"]) == (5, 5)
    assert cashew["not_measured"] == ["B Gaussian full"]
    assert cashew["split_file_list_sha256"] == SPLIT_HASHES
    pairs = {(d["a"], d["b"]): d for d in cashew["paired_auroc_differences"]}
    assert ("B Gaussian dev", "A classical") in pairs
    assert ("C PatchCore-inspired dev", "B Gaussian dev") in pairs
    assert ("B Gaussian full", "A classical") not in pairs  # that model was not run
    assert pairs[("B Gaussian dev", "A classical")]["auroc_difference"] > 0
    macro = comparison["macro_average"]["B Gaussian dev"]
    assert macro["n_categories"] == 2 and macro["categories"] == ["cashew", "pipe_fryum"]
    assert comparison["verification"]["image_metrics_rederived_from_predictions"] is True
    assert comparison["protocol"]["evaluation_resolution"] == 256
    assert "B Gaussian full" not in comparison["macro_average"]


def test_a_capped_fit_set_is_stated_and_an_uncapped_one_is_not() -> None:
    uncapped = build_comparison({"cashew": _runs("cashew")})
    assert uncapped["categories"]["cashew"]["fit_images"]["max_fit_images"] is None
    assert "fit images" not in format_markdown(uncapped)

    runs = _runs("candle")
    for run in runs:
        run.summary["split"].update(
            max_fit_images=400, counts={"fit": 855, "fit_used": 400, "test": 200}
        )
    capped = build_comparison({"candle": runs})
    assert capped["categories"]["candle"]["fit_images"] == {
        "available": 855,
        "used": 400,
        "max_fit_images": 400,
    }
    assert "subset of 400 of the 855 fit images" in format_markdown(capped)


def test_macro_average_is_the_mean_over_categories() -> None:
    runs = {c: _runs(c) for c in ("cashew", "pipe_fryum")}
    runs["pipe_fryum"][1] = make_run(GAUSSIAN, "pipe_fryum", _scores(1.0, 9))
    comparison = build_comparison(runs)
    values = [
        comparison["categories"][c]["runs"][1]["metrics"]["image"]["auroc"]
        for c in ("cashew", "pipe_fryum")
    ]
    assert comparison["macro_average"]["B Gaussian dev"]["image_auroc"] == pytest.approx(
        np.mean(values), abs=1e-6
    )


def test_a_tampered_run_stops_the_comparison_unless_verification_is_off() -> None:
    runs = _runs("cashew")
    runs[1].predictions[0]["score"] = 50.0
    with pytest.raises(EvaluationError, match="do not follow from predictions"):
        build_comparison({"cashew": runs})
    assert (
        build_comparison({"cashew": runs}, verify=False)["verification"][
            "image_metrics_rederived_from_predictions"
        ]
        is False
    )


def test_non_comparable_runs_stop_the_comparison() -> None:
    runs = _runs("cashew")
    runs[2].summary["metrics"]["evaluation_resolution"] = 512
    with pytest.raises(EvaluationError, match="not comparable"):
        build_comparison({"cashew": runs})
    with pytest.raises(EvaluationError, match="No categories"):
        build_comparison({})


# ----- undefined metrics ---------------------------------------------------------------------


def _single_class_runs() -> list[Run]:
    labels = [0] * 10
    return [
        make_run(CLASSICAL, "cashew", _scores(0.0, 1), labels=labels),
        make_run(GAUSSIAN, "cashew", _scores(0.0, 2), labels=labels),
    ]


def test_single_class_data_gives_undefined_metrics_not_invented_numbers() -> None:
    runs = _single_class_runs()
    assert runs[0].image_metrics["auroc"] is None and runs[0].image_metrics["auroc_ci95"] is None
    assert verify_run(runs[0]) == []  # undefined agrees with undefined
    comparison = build_comparison({"cashew": runs})
    assert (
        comparison["categories"]["cashew"]["paired_auroc_differences"][0]["auroc_difference"]
        is None
    )
    assert "cashew/B Gaussian dev/image_auroc" in comparison["undefined_metrics"]
    macro = comparison["macro_average"]["B Gaussian dev"]
    assert macro["image_auroc"] is None and macro["image_auroc_n_defined"] == 0
    text = format_markdown(comparison)
    assert "undefined" in text and "Undefined metrics" in text


def test_macro_average_skips_undefined_categories_and_reports_how_many_were_used() -> None:
    defined = _runs("cashew")[:2]
    undefined = [
        make_run(CLASSICAL, "pipe_fryum", _scores(0.0, 1), labels=[0] * 10),
        make_run(GAUSSIAN, "pipe_fryum", _scores(0.0, 2), labels=[0] * 10),
    ]
    comparison = build_comparison({"cashew": defined, "pipe_fryum": undefined})
    macro = comparison["macro_average"]["B Gaussian dev"]
    assert macro["n_categories"] == 2
    assert macro["image_auroc_n_defined"] == 1
    assert macro["image_auroc"] == defined[1].image_metrics["auroc"]


# ----- output formats ------------------------------------------------------------------------


def test_csv_has_one_row_per_category_and_model_and_marks_undefined(tmp_path: Path) -> None:
    comparison = build_comparison({"cashew": _single_class_runs()})
    rows = comparison_rows(comparison)
    assert len(rows) == 2 and set(rows[0]) == set(CSV_COLUMNS)
    path = tmp_path / "out dir é" / "comparison.csv"
    write_comparison_csv(path, comparison)
    with path.open(newline="", encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert [r["label"] for r in read] == ["A classical", "B Gaussian dev"]
    assert read[0]["image_auroc"] == "undefined" and read[0]["n_normal"] == "10"


def test_json_round_trips_and_keeps_configuration_metadata(tmp_path: Path) -> None:
    comparison = build_comparison({"cashew": _runs("cashew")})
    path = tmp_path / "comparison.json"
    write_comparison_json(path, comparison)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and "generated_utc" in data
    record = data["categories"]["cashew"]["runs"][1]
    assert record["model_config"]["backbone"] == "resnet18"
    assert record["environment"]["power_plan"] == "Balanced"
    assert record["threshold"]["image"]["value"] == 0.5
    with pytest.raises(ValueError, match="Out of range float"):
        write_comparison_json(tmp_path / "bad.json", {"x": float("nan")})


def test_markdown_distinguishes_measured_from_not_measured_results() -> None:
    comparison = build_comparison(
        {"cashew": _runs("cashew")}, {"cashew": ["B Gaussian full", "C PatchCore-inspired full"]}
    )
    text = format_markdown(comparison)
    assert text.startswith("# Model comparison")
    assert "## cashew" in text and "5 normal and 5 anomalous" in text
    assert "Not measured (no results exist): B Gaussian full, C PatchCore-inspired full." in text
    assert "Paired AUROC differences" in text and "## Macro average" in text
    assert "re-derived from each run's saved `predictions.csv`" in text
    assert "Published results: not compared" in text
    assert "## How to reproduce" in text and "scripts/compare_models.py" in text
    assert "--model patchcore --category <category> --profile full" in text
    assert "<artifacts>\\gaussian-full\\<category>" in text
    assert "| A classical |" in text and "| C PatchCore-inspired dev |" in text


# ----- loading runs --------------------------------------------------------------------------


def _write_run(root: Path, spec, category: str, run: Run) -> None:
    folder = root / spec.report_folder / category
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(run.summary), encoding="utf-8")
    write_predictions_csv(
        folder / "predictions.csv",
        [
            PredictionRow(category, spec.model, "test", r["image"], r["label"], r["score"],
                          r["predicted"], r["threshold"], r["label"] == 1)
            for r in run.predictions
        ],
    )  # fmt: skip


def test_a_missing_run_is_none_and_a_saved_run_loads_back(tmp_path: Path) -> None:
    assert load_run(tmp_path, None, GAUSSIAN, "cashew") is None
    original = make_run(GAUSSIAN)
    _write_run(tmp_path, GAUSSIAN, "cashew", original)
    loaded = load_run(tmp_path, tmp_path / "artifacts", GAUSSIAN, "cashew")
    assert loaded is not None and loaded.summary == original.summary
    assert [r["image"] for r in loaded.predictions] == IMAGES
    assert verify_run(loaded) == []
    assert loaded.manifest is None


def test_a_summary_in_the_wrong_folder_or_profile_is_rejected(tmp_path: Path) -> None:
    wrong_model = make_run(GAUSSIAN)
    _write_run(
        tmp_path, PATCHCORE, "cashew", wrong_model
    )  # a gaussian summary in the patchcore folder
    with pytest.raises(EvaluationError, match="expected patchcore/cashew"):
        load_run(tmp_path, None, PATCHCORE, "cashew")
    full = SPECS["B Gaussian full"]
    dev_summary = make_run(GAUSSIAN)  # resnet18 results filed under the full profile
    _write_run(tmp_path, full, "cashew", dev_summary)
    with pytest.raises(EvaluationError, match="needs 'wide_resnet50_2'"):
        load_run(tmp_path, None, full, "cashew")


def test_the_artifact_manifest_adds_weights_and_checksums(tmp_path: Path) -> None:
    _write_run(tmp_path, GAUSSIAN, "cashew", make_run(GAUSSIAN))
    folder = tmp_path / "artifacts" / "gaussian" / "cashew"
    folder.mkdir(parents=True)
    manifest = {"weights": {"identifier": "IMAGENET1K_V1", "sha256": "f" * 64}, "files": {"a": "b"}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run = load_run(tmp_path, tmp_path / "artifacts", GAUSSIAN, "cashew")
    assert run is not None and run.manifest == manifest
    record = build_comparison({"cashew": [run, make_run(CLASSICAL)]})["categories"]["cashew"][
        "runs"
    ]
    gaussian = next(r for r in record if r["model"] == "gaussian")
    assert gaussian["weights"]["identifier"] == "IMAGENET1K_V1"
    assert gaussian["artifact_files_sha256"] == {"a": "b"}


def test_malformed_predictions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text("image,label\nx,1\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Malformed"):
        read_predictions(path)
    path.write_text("image,split,label,score,predicted,threshold\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="empty"):
        read_predictions(path)
    with pytest.raises(EvaluationError, match="could not be read"):
        read_predictions(tmp_path / "missing.csv")


def test_comparison_does_not_modify_the_runs() -> None:
    runs = _runs("cashew")
    before = copy.deepcopy([(r.summary, r.predictions) for r in runs])
    build_comparison({"cashew": runs})
    assert [(r.summary, r.predictions) for r in runs] == before
