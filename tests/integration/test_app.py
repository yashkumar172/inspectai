"""The Streamlit pages, run for real with Streamlit's AppTest against synthetic artifacts."""

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import streamlit as st
from fake_backbone import FakeBackbone
from make_world import FAKE_WEIGHTS, World
from streamlit.testing.v1 import AppTest

from inspectai.artifacts.store import TENSORS
from inspectai.data.dataset import VisADataset
from inspectai.inference.predictor import (
    NOT_A_PROBABILITY,
    STATUS_ANOMALY,
    STATUS_NORMAL,
    STATUS_UNAVAILABLE,
    WRONG_CATEGORY_WARNING,
)
from inspectai.inference.uploads import MAX_UPLOAD_BYTES
from inspectai.samples import build_samples
from inspectai.ui.common import MSG_INFERENCE_FAILED, MSG_INTEGRITY, MSG_NOT_CONFIGURED
from inspectai.ui.dashboard_page import NO_RESULTS
from inspectai.ui.inspect_page import SOURCE_DATASET, SOURCE_SAMPLE, SOURCE_UPLOAD

REPO = Path(__file__).resolve().parents[2]


def _page(name: str) -> AppTest:
    return AppTest.from_string(
        f"from inspectai.ui import {name}\n{name}.render()", default_timeout=120
    )


def _texts(at: AppTest) -> str:
    """All user-visible text of a rendered page."""
    parts: list[str] = []
    for kind in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "error",
        "warning",
        "info",
        "success",
        "text",
    ):
        parts += [str(getattr(el, "value", "")) for el in getattr(at, kind)]
    for metric in at.metric:
        parts += [metric.label, str(metric.value)]
    for frame in at.dataframe:
        parts.append(frame.value.to_string())
    for table in at.table:
        parts.append(table.value.to_string())
    return "\n".join(parts)


def _no_leaks(at: AppTest, *extra: object) -> None:
    text = _texts(at)
    for forbidden in ("Traceback", "C:\\", "Users\\", "/c/", "\\\\", *map(str, extra)):
        assert forbidden not in text, forbidden


def _predictions(world: World, folder: str, category: str) -> dict[str, dict]:
    path = world.reports / folder / category / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image"]: row for row in csv.DictReader(handle)}


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    st.cache_resource.clear()


@pytest.fixture(autouse=True)
def _no_real_samples(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the repository's real `hosted/samples` out of tests that do not ask for samples."""
    monkeypatch.setattr("inspectai.config.BUNDLE_ROOT", tmp_path_factory.mktemp("no bundle"))


def _configure(
    monkeypatch: pytest.MonkeyPatch, data: Path, artifacts: Path, reports: Path, torch_home: Path
) -> None:
    monkeypatch.setenv("INSPECTAI_DATA_ROOT", str(data))
    monkeypatch.setenv("INSPECTAI_ARTIFACT_DIR", str(artifacts))
    monkeypatch.setenv("INSPECTAI_REPORT_DIR", str(reports))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.delenv("INSPECTAI_ALLOW_WEIGHT_DOWNLOAD", raising=False)


@pytest.fixture
def app_env(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """The shared synthetic world, with a fake backbone standing in for the pretrained weights."""
    _configure(
        monkeypatch, world.synthetic.data_root, world.artifacts, world.reports, tmp_path / "torch"
    )
    fake = lambda name, allow_download=None: (FakeBackbone(), FAKE_WEIGHTS)
    monkeypatch.setattr("inspectai.artifacts.gaussian_io.load_backbone", fake)
    monkeypatch.setattr("inspectai.artifacts.patchcore_io.load_backbone", fake)
    return world


@pytest.fixture
def private_env(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """Like `app_env`, but with private copies of artifacts and reports that tests may damage."""
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    shutil.copytree(world.artifacts, artifacts)
    shutil.copytree(world.reports, reports)
    _configure(monkeypatch, world.synthetic.data_root, artifacts, reports, tmp_path / "torch")
    fake = lambda name, allow_download=None: (FakeBackbone(), FAKE_WEIGHTS)
    monkeypatch.setattr("inspectai.artifacts.gaussian_io.load_backbone", fake)
    monkeypatch.setattr("inspectai.artifacts.patchcore_io.load_backbone", fake)
    return World(world.synthetic, reports, artifacts)


def _png(height: int = 48, width: int = 64, seed: int = 0) -> bytes:
    array = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


def _use_dataset(at: AppTest, index: int) -> AppTest:
    at.radio(key="inspect_source").set_value(SOURCE_DATASET).run()
    at.selectbox(key="inspect_sample").select_index(index).run()
    return at


# ----- home and about --------------------------------------------------------------------------


def test_the_home_page_lists_the_fitted_models_and_the_required_sections(app_env: World) -> None:
    at = _page("home").run()
    assert not at.exception and at.title[0].value == "InspectAI"
    text = _texts(at)
    for phrase in (
        "What is anomaly detection?",
        "How to inspect an image",
        "Limitations and intended use",
        "10 MB",
    ):
        assert phrase in text, phrase
    assert WRONG_CATEGORY_WARNING in text and "not a probability" in text
    rows = at.dataframe[0].value
    fitted = dict(zip(rows["Model"], rows["Categories with a fitted model"], strict=True))
    assert fitted["A classical"] == "cashew, pipe_fryum"
    assert (
        fitted["B Gaussian full"] == "none"
        and fitted["C PatchCore-inspired dev"] == "cashew, pipe_fryum"
    )
    _no_leaks(at, app_env.artifacts)


def test_the_home_page_warns_when_no_models_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(
        monkeypatch, tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports", tmp_path / "t"
    )
    at = _page("home").run()
    assert not at.exception and any("No fitted models were found" in w.value for w in at.warning)


def test_the_about_page_covers_every_required_topic() -> None:
    at = _page("about").run()
    assert not at.exception
    text = _texts(at)
    for topic in (
        "Anomaly detection versus supervised classification",
        "How an anomaly map is created",
        "Pretrained weights and algorithms",
        "Known failure cases",
        "Dataset limitations",
        "research and educational prototype",
        "not a certified industrial quality-control system",
        "IMAGENET1K_V1",
        "CC BY 4.0",
    ):
        assert topic in text, topic


# ----- inspect page: model and category selection ---------------------------------------------


def test_the_inspect_page_starts_with_an_available_model_and_the_category_warning(
    app_env: World,
) -> None:
    at = _page("inspect_page").run()
    assert not at.exception
    assert at.selectbox(key="inspect_model").value == "patchcore"
    assert at.selectbox(key="inspect_category").value == "cashew"
    assert any(w.value == WRONG_CATEGORY_WARNING for w in at.warning)
    labels = at.selectbox(key="inspect_model").options
    assert "B Gaussian full (not fitted)" in labels and "A classical" in labels
    assert not at.metric  # nothing is predicted until the user runs an inspection
    assert at.button(key="inspect_run").disabled  # no image chosen yet


def test_an_unfitted_model_or_category_says_model_unavailable_and_shows_no_result(
    app_env: World,
) -> None:
    at = _page("inspect_page").run()
    at.selectbox(key="inspect_model").set_value("gaussian-full").run()
    assert any(STATUS_UNAVAILABLE in e.value for e in at.error)
    assert any("scripts/fit_model.py --model gaussian --category" in i.value for i in at.info)
    assert not at.metric
    at.selectbox(key="inspect_model").set_value("gaussian").run()
    at.selectbox(key="inspect_category").set_value("fryum").run()
    assert any(STATUS_UNAVAILABLE in e.value and "fryum" in e.value for e in at.error)
    assert not at.metric
    _no_leaks(at, app_env.artifacts)


def test_the_inspect_page_without_configuration_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("INSPECTAI_DATA_ROOT", "INSPECTAI_ARTIFACT_DIR", "INSPECTAI_REPORT_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("inspectai.config.BUNDLE_ROOT", tmp_path / "no hosted folder")
    at = _page("inspect_page").run()
    assert not at.exception and any(MSG_NOT_CONFIGURED in e.value for e in at.error)
    assert not at.metric


# ----- inspect page: real predictions ---------------------------------------------------------


@pytest.mark.parametrize("model_key", ["classical", "gaussian", "patchcore"])
@pytest.mark.parametrize("index", [0, 9])  # an anomalous and a normal test image
def test_running_a_test_image_produces_the_real_prediction_with_the_stored_threshold(
    app_env: World, model_key: str, index: int
) -> None:
    at = _page("inspect_page").run()
    at.selectbox(key="inspect_model").set_value(model_key).run()
    _use_dataset(at, index)
    at.button(key="inspect_run").click().run()
    assert not at.exception
    name = at.selectbox(key="inspect_sample").options[index]
    image_rel = next(
        rel for rel in _predictions(app_env, model_key, "cashew") if rel.endswith(name)
    )
    row = _predictions(app_env, model_key, "cashew")[image_rel]
    score, threshold = float(row["score"]), float(row["threshold"])

    values = {m.label.split(" (")[0]: m.value for m in at.metric}
    assert float(values["Anomaly score"]) == pytest.approx(score, abs=5e-4)
    assert float(values["Decision threshold"]) == pytest.approx(threshold, abs=5e-4)
    status = STATUS_ANOMALY if int(row["predicted"]) else STATUS_NORMAL
    shown = (
        [e.value for e in at.error] if status == STATUS_ANOMALY else [s.value for s in at.success]
    )
    assert any(status in text for text in shown), (status, shown)
    assert (status == STATUS_ANOMALY) == (score > threshold)  # the stored threshold decides

    text = _texts(at)
    assert text.count(WRONG_CATEGORY_WARNING) >= 2  # before and within the result
    assert NOT_A_PROBABILITY in text and "probability" in at.metric[0].label
    assert "min / median / max" in at.metric[2].label and at.metric[2].value.count("/") == 2
    assert f"Ground truth from the dataset: **{'anomalous' if index == 0 else 'normal'}**" in text
    assert "CC BY 4.0" in text and len(at.get("image")) >= 3
    assert len(at.get("download_button")) == 2
    _no_leaks(at, app_env.artifacts, app_env.reports)


def test_the_model_is_loaded_once_and_cached_across_runs(
    app_env: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    from inspectai.ui import common

    original = common.load_predictor

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(args[1].key)
        return original(*args, **kwargs)

    monkeypatch.setattr(common, "load_predictor", counting)
    at = _page("inspect_page").run()
    _use_dataset(at, 0)
    at.button(key="inspect_run").click().run()
    at.selectbox(key="inspect_sample").select_index(1).run()
    at.button(key="inspect_run").click().run()
    assert not at.exception and len(at.metric) == 3
    assert calls == ["patchcore"]  # the second run reused the cached model


def test_the_result_survives_a_rerun_so_downloads_do_not_lose_it(app_env: World) -> None:
    at = _page("inspect_page").run()
    _use_dataset(at, 0)
    at.button(key="inspect_run").click().run()
    first = [m.value for m in at.metric]
    at.run()  # what a download click triggers
    assert [m.value for m in at.metric] == first


# ----- inspect page: uploads ------------------------------------------------------------------


def test_a_valid_upload_gets_a_real_prediction(app_env: World) -> None:
    at = _page("inspect_page").run()
    assert at.radio(key="inspect_source").value == SOURCE_UPLOAD
    at.file_uploader(key="inspect_upload").upload("photo.png", _png(), "image/png").run()
    assert not at.button(key="inspect_run").disabled
    at.button(key="inspect_run").click().run()
    assert not at.exception and not any("cannot be used" in e.value for e in at.error)
    assert len(at.metric) == 3 and float(at.metric[0].value) > 0
    assert "Ground truth" not in _texts(at)  # only dataset images have a known label
    _no_leaks(at, "photo.png")


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("broken.png", b"not an image", "could not be read as an image"),
        ("tiny.png", _png(16, 16), "at least 32x32"),
        ("wide.png", _png(40, 400), "aspect ratio"),
        ("empty.png", b"", "empty"),
    ],
    ids=["unreadable", "tiny", "wrong-aspect-ratio", "empty"],
)
def test_bad_uploads_are_rejected_before_any_model_runs(
    app_env: World, monkeypatch: pytest.MonkeyPatch, name: str, content: bytes, expected: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model must not run on a rejected upload")

    monkeypatch.setattr("inspectai.ui.inspect_page.inspect_image", forbidden)
    at = _page("inspect_page").run()
    at.file_uploader(key="inspect_upload").upload(name, content).run()
    at.button(key="inspect_run").click().run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("This image cannot be used" in e and expected in e for e in errors), errors
    assert not at.metric and not at.get("download_button")
    _no_leaks(at, name.split(".")[0] if name != "empty.png" else "zzz")


def test_an_oversized_upload_is_rejected_by_the_size_check(
    app_env: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inspectai.ui.inspect_page.inspect_image", lambda *a, **k: pytest.fail("ran")
    )
    at = _page("inspect_page").run()
    at.file_uploader(key="inspect_upload").upload("big.png", b"\x00" * (MAX_UPLOAD_BYTES + 1)).run()
    at.button(key="inspect_run").click().run()
    assert any("maximum is 10 MB" in e.value for e in at.error) and not at.metric


# ----- inspect page: failures never produce fake results --------------------------------------


def test_a_corrupted_artifact_shows_model_unavailable_without_a_prediction(
    private_env: World,
) -> None:
    tensors = private_env.artifacts / "patchcore" / "cashew" / TENSORS
    data = bytearray(tensors.read_bytes())
    data[-1] ^= 0xFF
    tensors.write_bytes(bytes(data))
    at = _page("inspect_page").run()
    _use_dataset(at, 0)
    at.button(key="inspect_run").click().run()
    assert not at.exception and not at.metric
    errors = " ".join(e.value for e in at.error)
    assert STATUS_UNAVAILABLE in errors and MSG_INTEGRITY in errors
    assert at.button(key="inspect_retry") is not None  # the user can retry after fixing it
    _no_leaks(at, private_env.artifacts)
    # an intact model of the same category still works
    at.selectbox(key="inspect_model").set_value("gaussian").run()
    at.button(key="inspect_run").click().run()
    assert len(at.metric) == 3


def test_missing_pretrained_weights_say_how_to_fetch_them(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(
        monkeypatch, world.synthetic.data_root, world.artifacts, world.reports, tmp_path / "empty"
    )
    at = _page("inspect_page").run()
    at.selectbox(key="inspect_model").set_value("gaussian").run()
    _use_dataset(at, 0)
    at.button(key="inspect_run").click().run()
    errors = " ".join(e.value for e in at.error)
    assert STATUS_UNAVAILABLE in errors and "fetch_weights.py --backbone resnet18" in errors
    assert not at.metric
    _no_leaks(at, world.artifacts, tmp_path)


def test_the_classical_baseline_needs_no_weights(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(
        monkeypatch, world.synthetic.data_root, world.artifacts, world.reports, tmp_path / "empty"
    )
    at = _page("inspect_page").run()
    at.selectbox(key="inspect_model").set_value("classical").run()
    _use_dataset(at, 0)
    at.button(key="inspect_run").click().run()
    assert not at.exception and len(at.metric) == 3


def test_an_inference_failure_is_reported_without_a_prediction_or_a_traceback(
    app_env: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail C:\\internal\\path")

    monkeypatch.setattr("inspectai.ui.inspect_page.inspect_image", broken)
    at = _page("inspect_page").run()
    _use_dataset(at, 0)
    at.button(key="inspect_run").click().run()
    assert not at.exception and not at.metric
    assert any(MSG_INFERENCE_FAILED in e.value for e in at.error)
    _no_leaks(at, "secret internal detail")


# ----- dashboard ------------------------------------------------------------------------------


def test_the_dashboard_shows_validated_metrics_from_the_report_files(app_env: World) -> None:
    at = _page("dashboard_page").run()
    assert not at.exception
    text = _texts(at)
    assert "official `test` split" in text and "5 normal and 5 anomalous" in text
    for heading in (
        "Model comparison",
        "Paired AUROC differences",
        "Confusion matrix",
        "Inference performance",
    ):
        assert heading in text, heading
    table = at.dataframe[0].value
    summary = json.loads(
        (app_env.reports / "gaussian" / "cashew" / "summary.json").read_text("utf-8")
    )
    row = table[table["Model"] == "B Gaussian dev"].iloc[0]
    assert row["Image AUROC"] == f"{summary['metrics']['image']['auroc']:.3f}"
    assert "Not measured for this category: B Gaussian full, C PatchCore-inspired full" in text
    at.selectbox(key="dashboard_category").set_value("pipe_fryum").run()
    assert not at.exception and "5 normal and 5 anomalous" in _texts(at)
    at.selectbox(key="dashboard_confusion_model").set_value("A classical").run()
    assert "Truly anomalous" in _texts(at)
    _no_leaks(at, app_env.reports, app_env.artifacts)


def test_the_dashboard_chart_carries_the_full_model_names(app_env: World) -> None:
    at = _page("dashboard_page").run()
    charts = at.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert spec["encoding"]["y"]["axis"]["labelLimit"] >= 300
    # Streamlit ships the rows separately; the spec still lists every full model name in order
    assert spec["encoding"]["y"]["sort"] == [
        "A classical",
        "B Gaussian dev",
        "C PatchCore-inspired dev",
    ]


def test_tiled_runs_appear_in_the_performance_table_when_their_reports_exist(
    private_env: World,
) -> None:
    summary = json.loads(
        (private_env.reports / "patchcore" / "cashew" / "summary.json").read_text("utf-8")
    )
    summary["performance"] = {**summary["performance"], "fit_seconds": 50.0}
    folder = private_env.reports / "patchcore-tiled" / "cashew"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    at = _page("dashboard_page").run()
    assert not at.exception and not at.error
    performance = next(d.value for d in at.dataframe if "Fit time" in d.value.columns)
    assert list(performance["Model"])[-1] == "C PatchCore-inspired dev (tiled)"
    assert performance.iloc[-1]["Fit time"] == "50.0 s"
    assert "None" not in performance.to_string()
    assert "(tiled)" in _texts(at)  # the caption explains the extra row
    metrics = at.dataframe[0].value
    assert not any("tiled" in model for model in metrics["Model"])
    _no_leaks(at, private_env.reports, private_env.artifacts)


def test_the_dashboard_confusion_matrix_matches_the_saved_summary(app_env: World) -> None:
    at = _page("dashboard_page").run()
    at.selectbox(key="dashboard_confusion_model").set_value("B Gaussian dev").run()
    cm = json.loads((app_env.reports / "gaussian" / "cashew" / "summary.json").read_text("utf-8"))[
        "metrics"
    ]["image"]["confusion_matrix"]
    matrix = at.table[0].value
    assert int(matrix.loc[1, "Predicted anomaly"]) == cm["tp"]
    assert int(matrix.loc[0, "Predicted anomaly"]) == cm["fp"]


ERROR_MODELS = {
    "A classical": "classical",
    "B Gaussian dev": "gaussian",
    "C PatchCore-inspired dev": "patchcore",
}


def _expected_mistakes(world: World, folder: str, category: str = "pipe_fryum") -> dict[str, list]:
    rows = list(_predictions(world, folder, category).values())
    flagged = [float(r["score"]) > float(r["threshold"]) for r in rows]
    pairs = list(zip(rows, flagged, strict=True))
    return {
        "false_positive": [r for r, f in pairs if r["label"] == "0" and f],
        "false_negative": [r for r, f in pairs if r["label"] == "1" and not f],
    }


def _error_table(at: AppTest):  # type: ignore[no-untyped-def]
    tables = [d.value for d in at.dataframe if "Score / threshold" in d.value.columns]
    return tables[0] if tables else None


def test_the_dashboard_error_analysis_lists_the_saved_mistakes_with_their_images(
    app_env: World,
) -> None:
    at = _page("dashboard_page").run()
    at.selectbox(key="dashboard_category").set_value("pipe_fryum").run()
    assert not at.exception and "Error analysis" in _texts(at)
    shown_a_sample = False
    for label, folder in ERROR_MODELS.items():
        at.selectbox(key="dashboard_error_model").set_value(label).run()
        expected = _expected_mistakes(app_env, folder)
        for kind, rows in expected.items():
            at.radio(key="dashboard_error_kind").set_value(kind).run()
            assert not at.exception, (label, kind)
            table = _error_table(at)
            if not rows:
                assert table is None and any("no mistakes" in s.value for s in at.success)
                continue
            assert len(table) == len(rows), (label, kind)
            assert sorted(table["Score"]) == sorted(round(float(r["score"]), 3) for r in rows)
            truth = "normal" if kind == "false_positive" else "anomalous"
            assert set(table["Ground truth"]) == {truth}
            # the first sample: score, threshold and truth side by side with the images
            at.selectbox(key="dashboard_error_sample").select_index(0).run()
            assert not at.exception
            metrics = {m.label: str(m.value) for m in at.metric}
            first = table.iloc[0]
            assert metrics["Anomaly score"] == f"{first['Score']:.3f}"
            assert metrics["Calibrated threshold"] == f"{first['Threshold']:.3f}"
            assert metrics["Ground truth"] == first["Ground truth"]
            decision = "Anomaly detected" if kind == "false_positive" else "Normal"
            assert metrics["Model decision"] == decision
            images = len(at.get("image"))
            assert images >= (3 if kind == "false_negative" else 2)  # + defect mask
            shown_a_sample = True
            _no_leaks(at, app_env.reports, app_env.artifacts, "Data/Images", "Data\\Images")
    assert shown_a_sample  # pipe_fryum / Gaussian has three false positives


def _fake_mistakes(image_rel: str, label: int) -> object:
    entry = {
        "image": image_rel,
        "name": image_rel.split("/", 2)[-1].split("/", 1)[-1],
        "label": label,
        "score": 0.5 if label else 9.0,
        "threshold": 4.0,
        "ratio": 0.125 if label else 2.25,
    }
    return lambda run: {
        "false_positive": [] if label else [entry],
        "false_negative": [entry] if label else [],
    }


def test_a_missed_defect_is_shown_with_its_overlay_and_ground_truth_mask(
    app_env: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = VisADataset(app_env.synthetic.split_csv, app_env.synthetic.images_root)
    anomalous = next(s for s in dataset.get_samples("cashew", split="test") if s.is_anomalous)
    monkeypatch.setattr(
        "inspectai.ui.dashboard_page.misclassified_images", _fake_mistakes(anomalous.image_rel, 1)
    )
    at = _page("dashboard_page").run()
    at.radio(key="dashboard_error_kind").set_value("false_negative").run()
    at.selectbox(key="dashboard_error_sample").select_index(0).run()
    assert not at.exception and not at.error
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics["Ground truth"] == "anomalous" and metrics["Model decision"] == "Normal"
    assert metrics["Anomaly score"] == "0.500" and metrics["Calibrated threshold"] == "4.000"
    assert len(at.get("image")) >= 3  # original, overlay, ground-truth mask
    _no_leaks(at, app_env.reports, app_env.artifacts, "Data/Images")


def test_an_image_the_split_file_does_not_list_is_never_opened(
    app_env: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inspectai.ui.dashboard_page.misclassified_images",
        _fake_mistakes("../../outside/secret.JPG", 1),
    )
    at = _page("dashboard_page").run()
    at.radio(key="dashboard_error_kind").set_value("false_negative").run()
    at.selectbox(key="dashboard_error_sample").select_index(0).run()
    assert not at.exception
    assert any("not listed in the local dataset" in w.value for w in at.warning)
    _no_leaks(at, app_env.reports)


def test_a_tampered_prediction_path_invalidates_the_category_instead_of_being_opened(
    private_env: World,
) -> None:
    path = private_env.reports / "gaussian" / "pipe_fryum" / "predictions.csv"
    original = path.read_text(encoding="utf-8")
    target = _expected_mistakes(private_env, "gaussian")["false_positive"][0]["image"]
    path.write_text(original.replace(target, "../../outside/secret.JPG"), encoding="utf-8")
    at = _page("dashboard_page").run()
    at.selectbox(key="dashboard_category").set_value("pipe_fryum").run()
    assert not at.exception
    assert any("did not pass validation" in e.value for e in at.error)
    assert _error_table(at) is None
    _no_leaks(at, private_env.reports)


def test_the_dashboard_without_results_shows_no_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(
        monkeypatch, tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports", tmp_path / "t"
    )
    at = _page("dashboard_page").run()
    assert not at.exception and any(NO_RESULTS in i.value for i in at.info)
    assert not at.dataframe and not at.get("vega_lite_chart") and not at.table


def test_the_dashboard_rejects_results_that_do_not_match_their_predictions(
    private_env: World,
) -> None:
    path = private_env.reports / "gaussian" / "cashew" / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1][5], rows[6][5] = "-1000.0", "9999.0"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    at = _page("dashboard_page").run()
    assert not at.exception
    assert any("did not pass validation and are not shown" in e.value for e in at.error)
    assert not at.dataframe  # not one metric from the damaged category
    _no_leaks(at, private_env.reports)
    at.selectbox(key="dashboard_category").set_value("pipe_fryum").run()  # the other one is fine
    assert at.dataframe and not at.error


def test_the_whole_app_starts_on_the_home_page(app_env: World) -> None:
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "InspectAI"


def test_the_app_configuration_enforces_the_spec_limits() -> None:
    import tomllib

    config = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] == 10  # MB (SPEC.md Section 8)
    assert config["server"]["address"] == "localhost"  # not exposed to the network by default
    assert config["browser"]["gatherUsageStats"] is False  # no unnecessary external calls
    assert config["client"]["showErrorDetails"] == "none"  # no stack traces for users


def test_the_file_watcher_is_off_so_it_cannot_evict_modules_during_an_import() -> None:
    """Regression: `KeyError: 'inspectai.ui'` on Streamlit Cloud (the watcher removed the package
    from sys.modules while a session was importing one of its pages)."""
    import tomllib

    from streamlit.watcher.path_watcher import NoOpPathWatcher, get_path_watcher_class

    config = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert get_path_watcher_class(config["server"]["fileWatcherType"]) is NoOpPathWatcher


def test_the_app_imports_in_a_fresh_interpreter() -> None:
    import subprocess
    import sys

    source = (REPO / "app.py").read_text(encoding="utf-8").splitlines()
    line = next(x for x in source if x.startswith("from inspectai.ui import"))
    done = subprocess.run(
        [sys.executable, "-I", "-c", line],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-500:]


# ----- hosted run: bundled files, no dataset, no INSPECTAI_* variables -------------------------


@pytest.fixture
def hosted_env(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bundle exported from the shared world, used as the only source of files."""
    from inspectai.deployment import export_hosted_bundle

    bundle = tmp_path / "hosted"
    export_hosted_bundle(world.artifacts, world.reports, bundle)
    for name in ("INSPECTAI_DATA_ROOT", "INSPECTAI_ARTIFACT_DIR", "INSPECTAI_REPORT_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("inspectai.config.BUNDLE_ROOT", bundle)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))  # empty: no weights available
    return bundle


def test_the_hosted_home_page_lists_only_the_bundled_classical_models(hosted_env: Path) -> None:
    at = _page("home").run()
    assert not at.exception and not any(MSG_NOT_CONFIGURED in w.value for w in at.warning)
    rows = at.dataframe[0].value
    fitted = dict(zip(rows["Model"], rows["Categories with a fitted model"], strict=True))
    assert fitted["A classical"] == "cashew, pipe_fryum"
    assert fitted["B Gaussian dev"] == "none" and fitted["C PatchCore-inspired dev"] == "none"
    _no_leaks(at, hosted_env)


def test_the_hosted_dashboard_shows_the_exported_results_without_a_dataset(
    hosted_env: Path,
) -> None:
    at = _page("dashboard_page").run()
    assert not at.exception and not at.error
    text = _texts(at)
    assert "Model comparison" in text and "5 normal and 5 anomalous" in text
    assert len(at.dataframe[0].value) >= 1
    _no_leaks(at, hosted_env)


def test_the_hosted_inspect_page_offers_uploads_but_no_local_dataset(hosted_env: Path) -> None:
    at = _page("inspect_page").run()
    assert not at.exception and not at.error
    assert at.selectbox(key="inspect_model").value == "classical"
    assert all(SOURCE_DATASET not in r.options for r in at.radio)
    _no_leaks(at, hosted_env)


def test_the_hosted_app_inspects_with_model_c_by_default_and_keeps_model_a_optional(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspectai.deployment import export_hosted_bundle

    fake = lambda name, allow_download=None, hub_dir=None: (FakeBackbone(), FAKE_WEIGHTS)
    monkeypatch.setattr("inspectai.deployment.load_backbone", fake)
    bundle = tmp_path / "hosted"
    export_hosted_bundle(world.artifacts, world.reports, bundle, models=("patchcore", "classical"))
    for name in ("INSPECTAI_DATA_ROOT", "INSPECTAI_ARTIFACT_DIR", "INSPECTAI_REPORT_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("inspectai.config.BUNDLE_ROOT", bundle)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr("inspectai.artifacts.patchcore_io.load_backbone", lambda name: fake(name))

    at = _page("inspect_page").run()
    assert not at.exception and not at.error
    assert at.selectbox(key="inspect_model").value == "patchcore"  # Model C is the primary model
    assert "A classical" in at.selectbox(key="inspect_model").options  # Model A stays selectable
    at.file_uploader(key="inspect_upload").upload("photo.png", _png(), "image/png").run()
    at.button(key="inspect_run").click().run()
    assert not at.exception and not any("cannot be used" in e.value for e in at.error)
    assert len(at.metric) == 3 and float(at.metric[0].value) > 0
    assert "C PatchCore-inspired dev" in _texts(at)
    _no_leaks(at, bundle)


def test_the_hosted_error_analysis_shows_scores_but_no_images(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inspectai.ui.dashboard_page.misclassified_images",
        _fake_mistakes("cashew/Data/Images/Normal/0001.JPG", 0),
    )
    at = _page("dashboard_page").run()
    assert not at.exception and "Error analysis" in _texts(at)
    before = len(at.get("image"))
    at.radio(key="dashboard_error_kind").set_value("false_positive").run()
    assert not at.exception and _error_table(at) is not None
    at.selectbox(key="dashboard_error_sample").select_index(0).run()
    assert any("isn't included in the online demo" in i.value for i in at.info)
    assert not any("INSPECTAI_DATA_ROOT" in i.value for i in at.info)
    assert len(at.get("image")) == before  # no dataset image was shown
    _no_leaks(at, hosted_env, "Data/Images")


# ----- bundled sample images ---------------------------------------------------------------


def _captions(at: AppTest) -> list[str]:
    return [im.caption for element in at.get("image") for im in element.proto.imgs]


def _build_samples(world: World, destination: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Samples chosen from the synthetic test split; the featured one is a cashew anomaly."""
    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    lists = {
        c: json.loads((world.artifacts / "patchcore" / c / "split.json").read_text("utf-8"))["test"]
        for c in ("cashew", "pipe_fryum")
    }
    featured = next(rel for rel in lists["cashew"] if "/Anomaly/" in rel)
    monkeypatch.setattr("inspectai.samples.FEATURED_CATEGORY", "cashew")
    monkeypatch.setattr("inspectai.samples.FEATURED_SOURCE", featured)
    build_samples(dataset, lists, destination)


@pytest.fixture
def sample_env(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hosted bundle (Model C and Model A, no dataset) that also holds sample images."""
    from inspectai.deployment import export_hosted_bundle

    fake = lambda name, allow_download=None, hub_dir=None: (FakeBackbone(), FAKE_WEIGHTS)
    monkeypatch.setattr("inspectai.deployment.load_backbone", fake)
    bundle = tmp_path / "hosted"
    export_hosted_bundle(world.artifacts, world.reports, bundle, models=("patchcore", "classical"))
    _build_samples(world, bundle / "samples", monkeypatch)
    for name in ("INSPECTAI_DATA_ROOT", "INSPECTAI_ARTIFACT_DIR", "INSPECTAI_REPORT_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("inspectai.config.BUNDLE_ROOT", bundle)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr("inspectai.artifacts.patchcore_io.load_backbone", lambda name: fake(name))
    return bundle


def test_in_hosted_mode_the_sample_option_is_the_default_and_starts_on_the_featured_example(
    sample_env: Path,
) -> None:
    at = _page("inspect_page").run()
    assert not at.exception and not at.error
    source = at.radio(key="inspect_source")
    assert source.value == SOURCE_SAMPLE and SOURCE_UPLOAD in source.options
    assert at.selectbox(key="inspect_model").value == "patchcore"
    assert at.selectbox(key="inspect_category").value == "cashew"  # the featured example's category
    picker = at.selectbox(key="inspect_bundled_sample_cashew")
    assert picker.index == 0 and picker.options[0].startswith("Featured example")
    assert "chosen by hand" in _texts(at)  # the label stays visible
    assert not at.file_uploader  # nothing to upload for a sample
    assert not at.metric  # nothing runs until the button is clicked


def test_running_the_featured_sample_shows_the_decision_the_ground_truth_and_the_outline(
    sample_env: Path,
) -> None:
    at = _page("inspect_page").run()
    at.button(key="inspect_run").click().run()
    assert not at.exception and not any("cannot be used" in e.value for e in at.error)
    assert len(at.metric) == 3 and float(at.metric[0].value) > 0
    decision = [*at.error, *at.success]
    assert any(STATUS_ANOMALY in d.value or STATUS_NORMAL in d.value for d in decision)
    assert any("Ground truth: anomalous" in i.value for i in at.info)
    assert "Heatmap with the ground-truth defect outline" in _captions(at)  # next to the heatmap
    text = _texts(at)
    assert "VisA" in text and "CC BY 4.0" in text  # the attribution under the image
    assert "Featured example" in text
    _no_leaks(at, sample_env, "Data/Images")


def test_a_normal_sample_shows_its_ground_truth_and_no_outline(sample_env: Path) -> None:
    at = _page("inspect_page").run()
    picker = at.selectbox(key="inspect_bundled_sample_cashew")
    normal = next(i for i, o in enumerate(picker.options) if o.startswith("Normal image"))
    picker.select_index(normal).run()
    at.button(key="inspect_run").click().run()
    assert not at.exception and len(at.metric) == 3
    assert any("Ground truth: normal" in i.value for i in at.info)
    assert "Heatmap with the ground-truth defect outline" not in _captions(at)


def test_a_sample_goes_through_the_same_checks_as_an_upload(
    sample_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, str, object]] = []
    from inspectai.inference.uploads import validate_upload as real

    def spy(data, filename, reference=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((len(data), filename, reference))
        return real(data, filename, reference, *args, **kwargs)

    monkeypatch.setattr("inspectai.ui.inspect_page.validate_upload", spy)
    at = _page("inspect_page").run()
    at.button(key="inspect_run").click().run()
    assert not at.exception and len(seen) == 1
    size, name, reference = seen[0]
    assert size > 0 and name.endswith(".JPG") and reference is not None  # the category's ratio


def test_a_damaged_sample_file_is_refused_and_nothing_is_predicted(sample_env: Path) -> None:
    from inspectai.samples import load_samples

    loaded = load_samples(sample_env / "samples")
    assert loaded is not None and loaded.featured() is not None
    damaged = sample_env / "samples" / loaded.featured().file  # type: ignore[union-attr]
    damaged.write_bytes(damaged.read_bytes() + b"x")
    at = _page("inspect_page").run()  # starts on the featured sample
    at.button(key="inspect_run").click().run()
    assert any("could not be loaded" in e.value for e in at.error)
    assert not at.metric
    _no_leaks(at, sample_env)


def test_uploads_still_work_next_to_the_samples(sample_env: Path) -> None:
    at = _page("inspect_page").run()
    at.radio(key="inspect_source").set_value(SOURCE_UPLOAD).run()
    at.file_uploader(key="inspect_upload").upload("photo.png", _png(), "image/png").run()
    at.button(key="inspect_run").click().run()
    assert not at.exception and len(at.metric) == 3
    assert not any("Ground truth" in i.value for i in at.info)  # an upload has no ground truth


def test_with_a_local_dataset_the_default_stays_upload_and_samples_are_still_offered(
    app_env: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "hosted with samples"
    _build_samples(app_env, bundle / "samples", monkeypatch)
    monkeypatch.setattr("inspectai.config.BUNDLE_ROOT", bundle)
    at = _page("inspect_page").run()
    source = at.radio(key="inspect_source")
    assert source.value == SOURCE_UPLOAD
    assert SOURCE_DATASET in source.options and SOURCE_SAMPLE in source.options


def test_without_samples_the_page_is_as_before(hosted_env: Path) -> None:
    at = _page("inspect_page").run()
    assert at.radio(key="inspect_source").options == [SOURCE_UPLOAD]


@pytest.mark.parametrize(
    ("anomalous", "kind"), [(False, "false_positive"), (True, "false_negative")]
)
def test_error_analysis_shows_the_image_when_it_is_a_bundled_sample(
    sample_env: Path, monkeypatch: pytest.MonkeyPatch, anomalous: bool, kind: str
) -> None:
    from inspectai.samples import load_samples

    loaded = load_samples(sample_env / "samples")
    assert loaded is not None
    entry = next(e for e in loaded.for_category("cashew") if e.is_anomalous == anomalous)
    monkeypatch.setattr(
        "inspectai.ui.dashboard_page.misclassified_images",
        _fake_mistakes(entry.source, int(anomalous)),
    )
    at = _page("dashboard_page").run()
    at.radio(key="dashboard_error_kind").set_value(kind).run()
    at.selectbox(key="dashboard_error_sample").select_index(0).run()
    assert not at.exception and not at.error
    assert not any("isn't included in the online demo" in i.value for i in at.info)
    captions = _captions(at)
    assert "Original image" in captions and "Anomaly overlay" in captions
    outline = "Heatmap with the ground-truth defect outline"
    assert (outline in captions) == anomalous  # only an anomalous sample has a mask to outline
    _no_leaks(at, sample_env, "Data/Images")
