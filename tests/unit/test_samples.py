"""The bundled sample images: choosing them, copying them, and checking them again on load."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from make_world import World

from inspectai import samples
from inspectai.data.dataset import VisADataset
from inspectai.samples import (
    ATTRIBUTION_NAME,
    MANIFEST_NAME,
    SampleError,
    build_samples,
    load_samples,
    select_by_mask_area,
    select_normals,
    sha256_file,
)


def _test_lists(world: World) -> dict[str, list[str]]:
    out = {}
    for category in ("cashew", "pipe_fryum"):
        split = json.loads(
            (world.artifacts / "patchcore" / category / "split.json").read_text("utf-8")
        )
        out[category] = list(split["test"])
    return out


@pytest.fixture
def built(world: World, tmp_path: Path) -> tuple[Path, samples.BuildSummary, VisADataset]:
    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    destination = tmp_path / "samples"
    summary = build_samples(dataset, _test_lists(world), destination)
    return destination, summary, dataset


# ----- the selection rules -----------------------------------------------------------------


def test_normals_are_a_seeded_random_choice_independent_of_the_input_order() -> None:
    names = [f"cat/Data/Images/Normal/{i:03d}.JPG" for i in range(50)]
    first = select_normals(names)
    assert first == select_normals(list(reversed(names)))  # same pool, same choice
    assert first == sorted(first) and len(set(first)) == 2
    assert first != select_normals(names, seed=7)  # the seed matters
    with pytest.raises(SampleError):
        select_normals(names[:1])


def test_anomalies_are_taken_at_the_25th_and_75th_percentile_of_mask_area() -> None:
    areas = {f"a{i:02d}": (100 - i) * 10 for i in range(100)}  # larger index, smaller area
    (low, q_low, rank_low), (high, q_high, rank_high) = select_by_mask_area(areas)
    assert (q_low, q_high) == (0.25, 0.75)
    ranked = sorted(areas, key=lambda k: (areas[k], k))
    assert low == ranked[round(0.25 * 99)] and high == ranked[round(0.75 * 99)]
    assert (rank_low, rank_high) == (round(0.25 * 99), round(0.75 * 99))
    assert areas[low] < areas[high]


def test_equal_areas_are_ordered_by_path_so_the_choice_is_deterministic() -> None:
    areas = {f"a{i}": 50 for i in range(9)}
    assert select_by_mask_area(areas) == select_by_mask_area(dict(reversed(areas.items())))
    with pytest.raises(SampleError):
        select_by_mask_area({"a": 1})  # both quantiles land on the same image


# ----- building ------------------------------------------------------------------------------


def test_the_files_are_byte_for_byte_copies_of_test_images_only(
    world: World, built: tuple[Path, samples.BuildSummary, VisADataset]
) -> None:
    destination, summary, dataset = built
    assert len(summary.entries) == 8  # 2 categories x (2 normal + 2 anomalous)
    by_source = {
        s.image_rel: s
        for category in ("cashew", "pipe_fryum")
        for s in dataset.get_samples(category)
    }
    for category, test_list in _test_lists(world).items():
        split = json.loads(
            (world.artifacts / "patchcore" / category / "split.json").read_text("utf-8")
        )
        for entry in (e for e in summary.entries if e.category == category):
            assert entry.source in test_list and entry.split == "test"
            assert (
                entry.source not in split["fit"] and entry.source not in split["calibration_normal"]
            )
            original = by_source[entry.source]
            assert (destination / entry.file).read_bytes() == original.image_path.read_bytes()
            assert entry.sha256 == sha256_file(original.image_path)
            if entry.is_anomalous:
                assert entry.mask_file is not None and original.mask_path is not None
                assert (
                    destination / entry.mask_file
                ).read_bytes() == original.mask_path.read_bytes()
                assert entry.mask_sha256 == sha256_file(original.mask_path)
            else:
                assert entry.mask_file is None
    for category in ("cashew", "pipe_fryum"):
        mine = [e for e in summary.entries if e.category == category]
        assert sorted(e.label for e in mine) == ["anomaly", "anomaly", "normal", "normal"]


def test_the_manifest_records_how_each_image_was_chosen(
    built: tuple[Path, samples.BuildSummary, VisADataset],
) -> None:
    destination, summary, _ = built
    manifest = json.loads((destination / MANIFEST_NAME).read_text("utf-8"))
    assert manifest["seed"] == 42 and "No model output" in manifest["selection_rule"]
    assert (destination / ATTRIBUTION_NAME).read_text("utf-8").count("CC BY 4.0") >= 1
    methods = {(e["label"], e["selection"]["method"]) for e in manifest["samples"]}
    assert methods == {("normal", "seeded_random"), ("anomaly", "mask_area_quantile")}
    for raw in manifest["samples"]:
        assert not Path(raw["source"]).is_absolute() and ":" not in raw["source"]  # no host paths
        if raw["label"] == "anomaly":
            assert raw["selection"]["quantile"] in (0.25, 0.75)
            assert raw["selection"]["mask_area_pixels"] > 0
    assert [e["file"] for e in manifest["samples"]] == [e.file for e in summary.entries]


def test_a_featured_example_is_added_and_marked_as_chosen_by_hand(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    test_list = _test_lists(world)
    featured = next(
        rel for rel in test_list["cashew"] if "/Normal/" not in rel and "Anomaly" in rel
    )
    monkeypatch.setattr(samples, "FEATURED_CATEGORY", "cashew")
    monkeypatch.setattr(samples, "FEATURED_SOURCE", featured)
    out = tmp_path / "samples"
    summary = build_samples(dataset, test_list, out)
    entry = next(e for e in summary.entries if e.featured)
    assert entry.source == featured and entry.selection["method"] == "manual"
    assert "manually" in str(entry.selection["reason"])
    assert sum(e.featured for e in summary.entries) == 1
    assert entry.title.startswith("Featured example")
    loaded = load_samples(out)
    assert loaded is not None and loaded.featured() == entry
    assert loaded.for_category("cashew")[0] == entry  # the featured one is listed first


def test_an_image_that_is_not_in_the_test_list_is_refused(world: World, tmp_path: Path) -> None:
    dataset = VisADataset(world.synthetic.split_csv, world.synthetic.images_root)
    lists = _test_lists(world)
    fit_image = next(
        s.image_rel for s in dataset.get_samples("cashew", split="train") if not s.is_anomalous
    )
    lists["cashew"].append(fit_image)
    with pytest.raises(SampleError, match="not test images"):
        build_samples(dataset, lists, tmp_path / "samples")


def test_existing_samples_are_only_replaced_with_force(
    world: World, built: tuple[Path, samples.BuildSummary, VisADataset]
) -> None:
    destination, _, dataset = built
    with pytest.raises(SampleError, match="--force"):
        build_samples(dataset, _test_lists(world), destination)
    again = build_samples(dataset, _test_lists(world), destination, force=True)
    assert len(again.entries) == 8


# ----- loading and verification --------------------------------------------------------------


def test_the_loaded_samples_verify_their_files(
    built: tuple[Path, samples.BuildSummary, VisADataset],
) -> None:
    destination, summary, dataset = built
    loaded = load_samples(destination)
    assert loaded is not None and loaded.entries == summary.entries
    assert loaded.categories() == ["cashew", "pipe_fryum"]
    entry = next(e for e in loaded.entries if e.is_anomalous)
    assert loaded.read_image_bytes(entry) == (destination / entry.file).read_bytes()
    image = dataset.load_image(next(s for s in dataset.samples if s.image_rel == entry.source))
    mask = loaded.read_mask(entry, image.shape[:2])
    assert mask is not None and mask.shape == image.shape[:2] and mask.max() == 1
    normal = next(e for e in loaded.entries if not e.is_anomalous)
    assert loaded.read_mask(normal, image.shape[:2]) is None
    assert loaded.find(entry.category, entry.source) == entry
    assert loaded.find(entry.category, "cashew/Data/Images/Normal/none.JPG") is None


def test_a_changed_file_is_rejected(
    built: tuple[Path, samples.BuildSummary, VisADataset],
) -> None:
    destination, summary, _ = built
    loaded = load_samples(destination)
    assert loaded is not None
    entry = summary.entries[0]
    path = destination / entry.file
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(SampleError, match="integrity"):
        loaded.read_image_bytes(entry)
    path.unlink()
    with pytest.raises(SampleError, match="missing"):
        loaded.read_image_bytes(entry)


def test_a_manifest_cannot_point_outside_the_samples_folder(
    built: tuple[Path, samples.BuildSummary, VisADataset], tmp_path: Path
) -> None:
    destination, _, _ = built
    manifest = json.loads((destination / MANIFEST_NAME).read_text("utf-8"))
    for bad in (
        "../secret.JPG",
        "cashew/../../secret.JPG",
        "/etc/passwd",
        "cashew/a/b.JPG",
        "pipe_fryum/x.JPG",
    ):
        broken = json.loads(json.dumps(manifest))
        broken["samples"][0]["file"] = bad
        folder = tmp_path / f"bad{abs(hash(bad))}"
        shutil.copytree(destination, folder)
        (folder / MANIFEST_NAME).write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(SampleError):
            load_samples(folder)


@pytest.mark.parametrize(
    "damage",
    [
        lambda m: m.update(schema_version=99),
        lambda m: m["samples"][0].update(split="train"),
        lambda m: m["samples"][0].update(category="nothing"),
        lambda m: m["samples"][0].update(sha256="abc"),
        lambda m: m["samples"][0].update(label="maybe"),
        lambda m: m["samples"][0].update(mask_file=None, label="anomaly"),
        lambda m: m["samples"].append(m["samples"][0]),  # a file listed twice
        lambda m: m.update(samples="none"),
    ],
)
def test_a_malformed_manifest_is_rejected(
    built: tuple[Path, samples.BuildSummary, VisADataset], damage
) -> None:
    destination, _, _ = built
    manifest = json.loads((destination / MANIFEST_NAME).read_text("utf-8"))
    # the first entry of the first category is a normal image: make the damage apply to an anomaly too
    manifest["samples"].sort(key=lambda s: s["label"] == "normal")
    damage(manifest)
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SampleError):
        load_samples(destination)


def test_no_manifest_means_no_samples_and_a_broken_file_is_an_error(tmp_path: Path) -> None:
    assert load_samples(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(SampleError):
        load_samples(tmp_path)
