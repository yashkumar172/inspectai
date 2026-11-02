"""The sample images that ship in `hosted/samples/`, checked against the rest of the hosted bundle.

These use the real committed files: the manifest against the files, the test-split rule against the
bundled artifacts, and the score of every sample against the saved evaluation of the same run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from inspectai.data.dataset import SUPPORTED_OBJECTS as SUPPORTED_ORDER
from inspectai.inference.predictor import inspect_image, load_predictor, option_by_key
from inspectai.inference.uploads import validate_upload
from inspectai.samples import (
    ATTRIBUTION_NAME,
    MANIFEST_NAME,
    NORMALS_PER_CATEGORY,
    SampleSet,
    load_samples,
    sha256_file,
)

BUNDLE = Path(__file__).resolve().parents[2] / "hosted"
SAMPLES = BUNDLE / "samples"
ARTIFACTS = BUNDLE / "artifacts"
REPORTS = BUNDLE / "reports"
# the seven categories that were fitted and evaluated (the other five VisA objects were not run)
EVALUATED = ("cashew", "pipe_fryum", "chewinggum", "fryum", "candle", "pcb1", "pcb2")
MODELS = ("patchcore", "classical")  # Model C dev and Model A, the models the hosted app serves


@pytest.fixture(scope="module")
def sample_set() -> SampleSet:
    loaded = load_samples(SAMPLES)
    assert loaded is not None, "hosted/samples/manifest.json is missing"
    return loaded


def test_the_manifest_matches_the_files_and_lists_every_file(sample_set: SampleSet) -> None:
    listed = set()
    for entry in sample_set.entries:
        assert sha256_file(SAMPLES / entry.file) == entry.sha256
        listed.add(entry.file)
        if entry.mask_file is not None:
            assert sha256_file(SAMPLES / entry.mask_file) == entry.mask_sha256
            listed.add(entry.mask_file)
        sample_set.read_image_bytes(entry)  # the loader's own verification agrees
    on_disk = {p.relative_to(SAMPLES).as_posix() for p in SAMPLES.rglob("*") if p.is_file()}
    assert on_disk - listed == {MANIFEST_NAME, ATTRIBUTION_NAME}  # nothing unlisted is shipped
    assert listed <= on_disk


def test_every_category_has_two_normal_and_two_anomalous_samples_and_one_featured(
    sample_set: SampleSet,
) -> None:
    assert sample_set.categories() == sorted(EVALUATED, key=list(SUPPORTED_ORDER).index)
    for category in EVALUATED:
        mine = [e for e in sample_set.entries if e.category == category and not e.featured]
        by_rule = [e for e in mine if e.selection["method"] != "manual"]
        assert sum(not e.is_anomalous for e in by_rule) == NORMALS_PER_CATEGORY
        assert sum(e.is_anomalous for e in by_rule) == 2
    featured = [e for e in sample_set.entries if e.featured]
    assert len(featured) == 1 and featured[0].category == "candle"
    assert featured[0].selection["method"] == "manual"
    assert featured[0].source == "candle/Data/Images/Anomaly/001.JPG"
    assert all(
        e.selection["method"] in {"seeded_random", "mask_area_quantile"}
        for e in sample_set.entries
        if not e.featured
    )


def test_no_sample_comes_from_the_fit_or_calibration_images(sample_set: SampleSet) -> None:
    for entry in sample_set.entries:
        assert entry.split == "test"
        for model in MODELS:
            split = json.loads(
                (ARTIFACTS / model / entry.category / "split.json").read_text("utf-8")
            )
            assert entry.source in split["test"]
            assert entry.source not in split["fit"]
            assert entry.source not in split["calibration_normal"]
            assert entry.mask_source is None or entry.mask_source not in split["fit"]


def test_the_manifest_has_no_host_paths(sample_set: SampleSet) -> None:
    text = (SAMPLES / MANIFEST_NAME).read_text("utf-8")
    assert "\\" not in text and ":/" not in text and "Users" not in text


def _saved_scores(model: str, category: str) -> dict[str, float]:
    with (REPORTS / model / category / "predictions.csv").open(encoding="utf-8", newline="") as f:
        return {row["image"]: float(row["score"]) for row in csv.DictReader(f)}


@pytest.mark.parametrize("category", EVALUATED)
def test_every_sample_scores_like_the_saved_evaluation_for_model_c_and_model_a(
    category: str, sample_set: SampleSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCH_HOME", str(BUNDLE / "torch_cache"))
    monkeypatch.delenv("INSPECTAI_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    entries = sample_set.for_category(category)
    assert entries
    for model in MODELS:
        predictor = load_predictor(ARTIFACTS, option_by_key(model), category)
        saved = _saved_scores(model, category)
        for entry in entries:
            # the same validation and decoding as an upload, from the verified original bytes
            image = validate_upload(
                sample_set.read_image_bytes(entry),
                Path(entry.file).name,
                predictor.reference_aspect_ratio,
            )
            score = inspect_image(predictor, image).score
            assert score == pytest.approx(saved[entry.source], rel=1e-5), (model, entry.source)
