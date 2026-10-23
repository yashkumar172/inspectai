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
from inspectai.inference.predictor import (
    NOT_A_PROBABILITY,
    STATUS_ANOMALY,
    STATUS_NORMAL,
    STATUS_UNAVAILABLE,
    WRONG_CATEGORY_WARNING,
)
from inspectai.inference.uploads import MAX_UPLOAD_BYTES
from inspectai.ui.common import MSG_INFERENCE_FAILED, MSG_INTEGRITY, MSG_NOT_CONFIGURED
from inspectai.ui.dashboard_page import NO_RESULTS
from inspectai.ui.inspect_page import SOURCE_DATASET, SOURCE_UPLOAD

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


def test_the_dashboard_confusion_matrix_matches_the_saved_summary(app_env: World) -> None:
    at = _page("dashboard_page").run()
    at.selectbox(key="dashboard_confusion_model").set_value("B Gaussian dev").run()
    cm = json.loads((app_env.reports / "gaussian" / "cashew" / "summary.json").read_text("utf-8"))[
        "metrics"
    ]["image"]["confusion_matrix"]
    matrix = at.table[0].value
    assert int(matrix.loc[1, "Predicted anomaly"]) == cm["tp"]
    assert int(matrix.loc[0, "Predicted anomaly"]) == cm["fp"]


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
