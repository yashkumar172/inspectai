"""Reproducible, disjoint fit / calibration_normal / test splits."""

import dataclasses
import json
import random
from pathlib import Path

import pytest
from make_synthetic import SyntheticVisA, build_synthetic_visa

from inspectai.config import SplitConfig
from inspectai.data.dataset import Sample, parse_split_file
from inspectai.data.splits import (
    SplitResult,
    file_list_sha256,
    load_split,
    make_splits,
    save_split,
)
from inspectai.errors import SplitError


@pytest.fixture
def samples(synthetic_visa: SyntheticVisA) -> list[Sample]:
    return parse_split_file(synthetic_visa.split_csv, synthetic_visa.images_root)


def test_splits_have_the_expected_sizes(samples: list[Sample]) -> None:
    result = make_splits(samples, "cashew")
    # 24 training images; the 10% rule gives 2, so the minimum of 10 applies.
    assert len(result.calibration_normal) == 10
    assert len(result.fit) == 14
    assert len(result.test) == 10  # 5 normal + 5 anomalous


def test_splits_are_disjoint_and_cover_the_training_rows(samples: list[Sample]) -> None:
    result = make_splits(samples, "cashew")
    fit, cal, test = set(result.fit), set(result.calibration_normal), set(result.test)
    assert not (fit & cal) and not (fit & test) and not (cal & test)
    train = {s.image_rel for s in samples if s.object_name == "cashew" and s.split == "train"}
    assert fit | cal == train
    assert test == {s.image_rel for s in samples if s.object_name == "cashew" and s.split == "test"}


def test_same_seed_gives_the_same_splits(samples: list[Sample]) -> None:
    assert make_splits(samples, "cashew") == make_splits(samples, "cashew")
    assert make_splits(samples, "cashew", SplitConfig(seed=7)) == make_splits(
        samples, "cashew", SplitConfig(seed=7)
    )


def test_a_different_seed_gives_a_different_calibration_set(samples: list[Sample]) -> None:
    first = make_splits(samples, "cashew", SplitConfig(seed=1))
    second = make_splits(samples, "cashew", SplitConfig(seed=2))
    assert first.calibration_normal != second.calibration_normal
    assert first.test == second.test


def test_splits_do_not_depend_on_row_order_or_other_objects(samples: list[Sample]) -> None:
    baseline = make_splits(samples, "cashew")
    shuffled = list(samples)
    random.Random(0).shuffle(shuffled)
    assert make_splits(shuffled, "cashew") == baseline
    only_cashew = [s for s in samples if s.object_name == "cashew"]
    assert make_splits(only_cashew, "cashew") == baseline


def test_calibration_fraction_applies_when_it_exceeds_the_minimum(tmp_path: Path) -> None:
    data = build_synthetic_visa(tmp_path / "d", objects=("cashew",), n_train=100)
    rows = parse_split_file(data.split_csv, data.images_root)
    result = make_splits(rows, "cashew", SplitConfig(calibration_fraction=0.2))
    assert len(result.calibration_normal) == 20 and len(result.fit) == 80


def test_too_few_training_images_are_rejected(tmp_path: Path) -> None:
    data = build_synthetic_visa(tmp_path / "d", objects=("cashew",), n_train=10)
    rows = parse_split_file(data.split_csv, data.images_root)
    with pytest.raises(SplitError, match="too few"):
        make_splits(rows, "cashew")


def test_missing_training_or_test_rows_are_rejected(samples: list[Sample]) -> None:
    with pytest.raises(SplitError, match="no test rows"):
        make_splits([s for s in samples if s.split == "train"], "cashew")
    with pytest.raises(SplitError, match="no training rows"):
        make_splits([s for s in samples if s.split == "test"], "cashew")
    with pytest.raises(SplitError, match="no training rows"):
        make_splits(samples, "candle")


def test_anomalous_or_duplicate_training_rows_are_rejected(samples: list[Sample]) -> None:
    train = next(s for s in samples if s.object_name == "cashew" and s.split == "train")
    anomalous = [dataclasses.replace(train, label="anomaly") if s is train else s for s in samples]
    with pytest.raises(SplitError, match="must be normal"):
        make_splits(anomalous, "cashew")
    with pytest.raises(SplitError, match="duplicate"):
        make_splits([*samples, train], "cashew")


def _result(**overrides: tuple[str, ...]) -> SplitResult:
    parts = {"fit": ("a", "b"), "calibration_normal": ("c",), "test": ("d", "e")}
    parts.update(overrides)
    return SplitResult("cashew", 42, 0.1, 10, **parts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fit": ("a", "c")}, "fit and calibration_normal"),
        ({"fit": ("a", "d")}, "fit and test"),
        ({"calibration_normal": ("d",)}, "calibration_normal and test"),
        ({"fit": ()}, "fit split is empty"),
        ({"test": ()}, "test split is empty"),
        ({"fit": ("a", "a")}, "duplicate"),
    ],
)
def test_validate_rejects_overlapping_empty_or_duplicate_parts(
    overrides: dict[str, tuple[str, ...]], message: str
) -> None:
    with pytest.raises(SplitError, match=message):
        _result(**overrides).validate()


def test_a_valid_result_passes_validation() -> None:
    _result().validate()


def test_file_list_hash_ignores_order() -> None:
    assert file_list_sha256(["b", "a"]) == file_list_sha256(["a", "b"])
    assert file_list_sha256(["a"]) != file_list_sha256(["b"])


def test_save_and_load_round_trip(samples: list[Sample], tmp_path: Path) -> None:
    result = make_splits(samples, "cashew")
    target = tmp_path / "out" / "split.json"
    save_split(result, target)
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert load_split(target) == result
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["counts"] == {"fit": 14, "calibration_normal": 10, "test": 10}
    assert data["seed"] == 42 and data["object"] == "cashew"


def _rewrite(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def saved(samples: list[Sample], tmp_path: Path) -> Path:
    target = tmp_path / "split.json"
    save_split(make_splits(samples, "cashew"), target)
    return target


def test_load_rejects_a_list_that_no_longer_matches_its_hash(saved: Path) -> None:
    _rewrite(saved, lambda d: d["fit"].pop())
    with pytest.raises(SplitError, match="SHA-256"):
        load_split(saved)


def test_load_rejects_a_test_file_moved_into_fit_even_with_fixed_hashes(saved: Path) -> None:
    def move(data: dict) -> None:
        data["fit"].append(data["test"][0])
        data["file_list_sha256"]["fit"] = file_list_sha256(data["fit"])

    _rewrite(saved, move)
    with pytest.raises(SplitError, match="overlap"):
        load_split(saved)


def test_load_rejects_unknown_schema_missing_fields_and_unreadable_files(
    saved: Path, tmp_path: Path
) -> None:
    _rewrite(saved, lambda d: d.update(schema_version=99))
    with pytest.raises(SplitError, match="schema_version"):
        load_split(saved)
    _rewrite(saved, lambda d: (d.update(schema_version=1), d.pop("seed")))
    with pytest.raises(SplitError, match="missing fields"):
        load_split(saved)
    with pytest.raises(SplitError, match="could not be read"):
        load_split(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SplitError, match="could not be read"):
        load_split(broken)
