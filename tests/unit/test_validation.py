"""Dataset validation and image-size statistics."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from make_synthetic import SyntheticVisA, build_synthetic_visa, write_jpg, write_png

from inspectai.data.dataset import Sample, parse_split_file
from inspectai.data.validation import (
    ValidationReport,
    format_report,
    recommend_resize_mode,
    validate_dataset,
)
from inspectai.errors import DatasetError


def _validate(data: SyntheticVisA, **kwargs) -> ValidationReport:
    samples = parse_split_file(data.split_csv, data.images_root)
    return validate_dataset(samples, data.images_root, **kwargs)


def _find(data: SyntheticVisA, *, label: str, split: str, obj: str = "cashew") -> Sample:
    samples = parse_split_file(data.split_csv, data.images_root)
    return next(s for s in samples if (s.object_name, s.split, s.label) == (obj, split, label))


def _edit_rows(data: SyntheticVisA, edit) -> None:
    with data.split_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    edit(rows)
    with data.split_csv.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_a_clean_dataset_has_no_issues_and_correct_counts(synthetic_visa: SyntheticVisA) -> None:
    report = _validate(synthetic_visa)
    assert report.ok and not report.issues
    assert report.rows_checked == 68
    assert report.objects["cashew"].counts == {
        "train/normal": 24,
        "test/normal": 5,
        "test/anomaly": 5,
    }


def test_size_statistics_and_the_letterbox_recommendation(synthetic_visa: SyntheticVisA) -> None:
    cashew = _validate(synthetic_visa).objects["cashew"]
    assert cashew.measured_images == 34
    assert (cashew.width_min, cashew.width_max, cashew.width_median) == (64, 64, 64.0)
    assert (cashew.height_min, cashew.height_max, cashew.height_median) == (48, 48, 48.0)
    assert cashew.aspect_median == pytest.approx(64 / 48)
    assert cashew.top_sizes == [(64, 48, 34)]
    assert cashew.recommended_resize_mode == "letterbox"


def test_square_images_get_the_stretch_recommendation(tmp_path: Path) -> None:
    data = build_synthetic_visa(tmp_path / "d", objects=("cashew",), size=(48, 48))
    assert _validate(data).objects["cashew"].recommended_resize_mode == "stretch"


def test_recommend_resize_mode_threshold() -> None:
    assert recommend_resize_mode(1.0) == "stretch"
    assert recommend_resize_mode(1.05) == "stretch"
    assert recommend_resize_mode(0.95) == "stretch"
    assert recommend_resize_mode(1.06) == "letterbox"
    assert recommend_resize_mode(1.5) == "letterbox"


def test_a_missing_image_is_reported(synthetic_visa: SyntheticVisA) -> None:
    _find(synthetic_visa, label="normal", split="train").image_path.unlink()
    report = _validate(synthetic_visa)
    assert not report.ok
    assert report.issues["missing_image"].count == 1


def test_a_corrupt_image_is_reported(synthetic_visa: SyntheticVisA) -> None:
    _find(synthetic_visa, label="normal", split="test").image_path.write_bytes(b"junk")
    report = _validate(synthetic_visa)
    assert report.issues["unreadable_image"].count == 1 and not report.ok


def test_an_image_over_the_pixel_limit_is_reported(synthetic_visa: SyntheticVisA) -> None:
    report = _validate(synthetic_visa, max_pixels=64 * 48 - 1)
    assert report.issues["unreadable_image"].count == 68


def test_an_anomalous_row_without_a_mask_is_reported(synthetic_visa: SyntheticVisA) -> None:
    def drop_mask(rows: list[list[str]]) -> None:
        row = next(r for r in rows if r[2] == "anomaly")
        row[4] = ""

    _edit_rows(synthetic_visa, drop_mask)
    report = _validate(synthetic_visa)
    assert report.issues["anomaly_without_mask"].count == 1 and not report.ok


def test_a_normal_row_with_a_mask_is_an_error(synthetic_visa: SyntheticVisA) -> None:
    def add_mask(rows: list[list[str]]) -> None:
        row = next(r for r in rows[1:] if r[2] == "normal")
        row[4] = "cashew/Data/Masks/Anomaly/000.png"

    _edit_rows(synthetic_visa, add_mask)
    report = _validate(synthetic_visa)
    assert not report.ok
    assert report.issues["normal_row_has_mask"].severity == "error"
    assert report.issues["normal_row_has_mask"].count == 1


def test_a_missing_mask_file_is_reported(synthetic_visa: SyntheticVisA) -> None:
    mask = _find(synthetic_visa, label="anomaly", split="test").mask_path
    assert mask is not None
    mask.unlink()
    assert _validate(synthetic_visa).issues["missing_mask"].count == 1


def test_a_mask_with_the_wrong_size_is_reported(synthetic_visa: SyntheticVisA) -> None:
    mask = _find(synthetic_visa, label="anomaly", split="test").mask_path
    assert mask is not None
    write_png(mask, np.full((10, 10), 255, dtype=np.uint8))
    report = _validate(synthetic_visa)
    assert report.issues["mask_size_mismatch"].count == 1 and not report.ok


def test_a_multichannel_mask_is_reported(synthetic_visa: SyntheticVisA) -> None:
    mask = _find(synthetic_visa, label="anomaly", split="test").mask_path
    assert mask is not None
    write_png(mask, np.zeros((48, 64, 3), dtype=np.uint8))
    assert _validate(synthetic_visa).issues["multichannel_mask"].count == 1


def test_an_unreadable_mask_is_reported(synthetic_visa: SyntheticVisA) -> None:
    mask = _find(synthetic_visa, label="anomaly", split="test").mask_path
    assert mask is not None
    mask.write_bytes(b"junk")
    assert _validate(synthetic_visa).issues["unreadable_mask"].count == 1


def test_an_all_zero_anomalous_mask_is_a_warning(synthetic_visa: SyntheticVisA) -> None:
    mask = _find(synthetic_visa, label="anomaly", split="test").mask_path
    assert mask is not None
    write_png(mask, np.zeros((48, 64), dtype=np.uint8))
    report = _validate(synthetic_visa)
    assert report.ok and report.issues["empty_mask"].severity == "warning"


def test_a_duplicated_row_is_reported(synthetic_visa: SyntheticVisA) -> None:
    _edit_rows(synthetic_visa, lambda rows: rows.append(list(rows[1])))
    report = _validate(synthetic_visa)
    assert report.issues["duplicate_image"].count == 1 and not report.ok


def test_a_missing_object_folder_is_reported_once(synthetic_visa: SyntheticVisA) -> None:
    import shutil

    shutil.rmtree(synthetic_visa.images_root / "cashew")
    report = _validate(synthetic_visa)
    assert report.issues["missing_object_folder"].count == 1
    assert "missing_image" not in report.issues


def test_a_missing_image_folder_explains_where_to_put_the_data(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"Image folder not found.*VisA"):
        validate_dataset([], tmp_path / "absent")


def test_objects_can_be_selected_and_unknown_ones_are_rejected(
    synthetic_visa: SyntheticVisA,
) -> None:
    report = _validate(synthetic_visa, objects=["cashew"])
    assert list(report.objects) == ["cashew"] and report.rows_checked == 34
    with pytest.raises(DatasetError, match="not in the split file"):
        _validate(synthetic_visa, objects=["candle"])


def test_the_report_is_json_serializable_and_formats_as_text(
    synthetic_visa: SyntheticVisA,
) -> None:
    _find(synthetic_visa, label="normal", split="train").image_path.unlink()
    report = _validate(synthetic_visa)
    data = json.loads(json.dumps(report.to_dict()))
    assert data["ok"] is False and data["issues"]["missing_image"]["count"] == 1
    text = format_report(report)
    assert "cashew" in text and "missing_image" in text and "Errors: 1" in text


def test_images_with_different_sizes_widen_the_statistics(tmp_path: Path) -> None:
    data = build_synthetic_visa(tmp_path / "d", objects=("cashew",), size=(40, 40))
    sample = _find(data, label="normal", split="train")
    write_jpg(sample.image_path, np.full((20, 80, 3), 100, dtype=np.uint8))
    cashew = _validate(data).objects["cashew"]
    assert cashew.width_max == 80 and cashew.height_min == 20
    assert cashew.aspect_max == pytest.approx(4.0) and cashew.aspect_min == pytest.approx(1.0)
    assert cashew.recommended_resize_mode == "stretch"  # the median is still square


def test_an_image_with_a_side_over_the_limit_is_reported(synthetic_visa: SyntheticVisA) -> None:
    report = _validate(synthetic_visa, max_side=63)  # the synthetic images are 64 x 48
    assert report.issues["unreadable_image"].count == 68 and not report.ok
