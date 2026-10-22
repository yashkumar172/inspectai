"""Split-file parsing and sample loading."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from make_synthetic import SyntheticVisA, write_jpg, write_png

from inspectai.data.dataset import (
    VisADataset,
    count_samples,
    parse_split_file,
)
from inspectai.errors import DatasetError, ImageError

HEADER = ["object", "split", "label", "image", "mask"]


def _write_csv(path: Path, rows: Sequence[Sequence[str]], header: Sequence[str] = HEADER) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _parse(tmp_path: Path, rows: Sequence[Sequence[str]], header: Sequence[str] = HEADER):
    csv_path = _write_csv(tmp_path / "1cls.csv", rows, header)
    return parse_split_file(csv_path, tmp_path / "VisA")


GOOD_ROW = ["cashew", "train", "normal", "cashew/Data/Images/Normal/0001.JPG", ""]


def test_parse_reads_every_row(synthetic_visa: SyntheticVisA) -> None:
    samples = parse_split_file(synthetic_visa.split_csv, synthetic_visa.images_root)
    assert len(samples) == 2 * (24 + 5 + 5)
    first = samples[0]
    assert (first.object_name, first.split, first.label) == ("cashew", "train", "normal")
    assert first.mask_path is None and first.mask_rel is None
    assert first.image_path == synthetic_visa.images_root / "cashew/Data/Images/Normal/0000.JPG"
    anomalous = next(s for s in samples if s.is_anomalous)
    assert anomalous.mask_path is not None and anomalous.mask_path.name.endswith(".png")


def test_count_samples_groups_by_object_split_and_label(synthetic_visa: SyntheticVisA) -> None:
    samples = parse_split_file(synthetic_visa.split_csv, synthetic_visa.images_root)
    counts = count_samples(samples)
    assert counts[("cashew", "train", "normal")] == 24
    assert counts[("cashew", "test", "normal")] == 5
    assert counts[("pipe_fryum", "test", "anomaly")] == 5


def test_parse_reports_a_missing_split_file_with_guidance(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"1cls\.csv.*split_csv"):
        parse_split_file(tmp_path / "nope.csv", tmp_path)


def test_parse_rejects_a_wrong_header(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Unexpected header"):
        _parse(tmp_path, [GOOD_ROW], header=["object", "set", "label", "image", "mask"])


def test_parse_rejects_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no data rows"):
        _parse(tmp_path, [])


def test_parse_rejects_the_wrong_number_of_columns(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="line 2.*5 columns"):
        _parse(tmp_path, [["cashew", "train", "normal"]])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (["widgets", "train", "normal", "widgets/a.JPG", ""], "unsupported object"),
        (["cashew", "val", "normal", "cashew/a.JPG", ""], "invalid split"),
        (["cashew", "train", "defect", "cashew/a.JPG", ""], "invalid label"),
        (["cashew", "train", "anomaly", "cashew/a.JPG", "cashew/m.png"], "must be normal"),
        (["cashew", "train", "normal", "", ""], "empty image path"),
        (["cashew", "train", "normal", "../outside.JPG", ""], "inside the dataset folder"),
        (["cashew", "train", "normal", "/etc/passwd", ""], "inside the dataset folder"),
        (["cashew", "train", "normal", "C:/Windows/a.JPG", ""], "inside the dataset folder"),
        (["cashew", "test", "anomaly", "cashew/a.JPG", "a/../../m.png"], "inside the dataset"),
    ],
)
def test_parse_rejects_invalid_rows_with_the_line_number(
    tmp_path: Path, row: list[str], message: str
) -> None:
    with pytest.raises(DatasetError, match=rf"line 3: .*{message}"):
        _parse(tmp_path, [GOOD_ROW, row])


def test_dataset_lists_objects_and_filters_samples(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    assert dataset.objects() == ["cashew", "pipe_fryum"]
    assert len(dataset.get_samples("cashew")) == 34
    assert len(dataset.get_samples("cashew", split="train")) == 24
    assert len(dataset.get_samples("cashew", split="test", label="anomaly")) == 5


def test_dataset_rejects_bad_filters_and_absent_objects(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    with pytest.raises(DatasetError, match="Unsupported object"):
        dataset.get_samples("widgets")
    with pytest.raises(DatasetError, match="Invalid split"):
        dataset.get_samples("cashew", split="val")
    with pytest.raises(DatasetError, match="Invalid label"):
        dataset.get_samples("cashew", label="bad")
    with pytest.raises(DatasetError, match="No samples"):
        dataset.get_samples("candle")


def test_load_image_returns_rgb_uint8(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    image = dataset.load_image(dataset.get_samples("cashew")[0])
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8


def test_load_mask_for_an_anomalous_sample_is_binary(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    sample = dataset.get_samples("cashew", split="test", label="anomaly")[0]
    image = dataset.load_image(sample)
    mask = dataset.load_mask(sample, image.shape[:2])
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)) == {0, 1}


def test_load_mask_for_a_normal_sample_is_all_zero(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    sample = dataset.get_samples("cashew", split="train")[0]
    mask = dataset.load_mask(sample, (48, 64))
    assert mask.shape == (48, 64) and not mask.any()


def test_load_mask_rejects_the_wrong_size(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    sample = dataset.get_samples("cashew", split="test", label="anomaly")[0]
    with pytest.raises(ImageError, match="does not match image size"):
        dataset.load_mask(sample, (10, 10))


def test_load_mask_rejects_a_multichannel_mask(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    sample = dataset.get_samples("cashew", split="test", label="anomaly")[0]
    assert sample.mask_path is not None
    write_png(sample.mask_path, np.zeros((48, 64, 3), dtype=np.uint8))
    with pytest.raises(ImageError, match="single-channel"):
        dataset.load_mask(sample, (48, 64))


def test_load_mask_requires_a_mask_for_anomalous_samples(tmp_path: Path) -> None:
    rows = [["cashew", "test", "anomaly", "cashew/Data/Images/Anomaly/000.JPG", ""]]
    dataset = VisADataset(_write_csv(tmp_path / "1cls.csv", rows), tmp_path / "VisA")
    with pytest.raises(DatasetError, match="no mask"):
        dataset.load_mask(dataset.samples[0], (4, 4))


def test_load_image_reports_missing_and_corrupt_files(synthetic_visa: SyntheticVisA) -> None:
    dataset = VisADataset(synthetic_visa.split_csv, synthetic_visa.images_root)
    first, second = dataset.get_samples("cashew", split="train")[:2]
    first.image_path.unlink()
    with pytest.raises(ImageError, match="not found"):
        dataset.load_image(first)
    second.image_path.write_bytes(b"this is not a jpeg")
    with pytest.raises(ImageError, match="readable"):
        dataset.load_image(second)


def test_dataset_loads_from_a_folder_with_non_ascii_characters(tmp_path: Path) -> None:
    root = tmp_path / "données 漢字"
    image = np.full((8, 8, 3), 90, dtype=np.uint8)
    write_jpg(root / "VisA" / "cashew/Data/Images/Normal/0001.JPG", image)
    rows = [["cashew", "train", "normal", "cashew/Data/Images/Normal/0001.JPG", ""]]
    dataset = VisADataset(_write_csv(root / "1cls.csv", rows), root / "VisA")
    assert dataset.load_image(dataset.samples[0]).shape == (8, 8, 3)
