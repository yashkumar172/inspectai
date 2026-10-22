"""Checks against the real VisA dataset. Run with: pytest -m needs_dataset

These tests are skipped by default and when INSPECTAI_DATA_ROOT does not contain the dataset.
"""

import pytest

from inspectai.config import load_paths, resize_mode_for
from inspectai.data.dataset import SUPPORTED_OBJECTS, VisADataset
from inspectai.data.splits import make_splits
from inspectai.data.validation import validate_dataset

pytestmark = pytest.mark.needs_dataset


@pytest.fixture(scope="module")
def dataset() -> VisADataset:
    paths = load_paths()
    if not paths.split_csv.is_file() or not paths.visa_images_root.is_dir():
        pytest.skip("VisA dataset not found under INSPECTAI_DATA_ROOT")
    return VisADataset.from_paths(paths)


def test_split_file_matches_the_official_statistics(dataset: VisADataset) -> None:
    labels = [s.label for s in dataset.samples]
    assert len(labels) == 10_821
    assert labels.count("normal") == 9_621 and labels.count("anomaly") == 1_200
    assert dataset.objects() == sorted(SUPPORTED_OBJECTS)


def test_splits_can_be_built_for_every_object(dataset: VisADataset) -> None:
    for name in dataset.objects():
        result = make_splits(dataset.samples, name)
        assert result.fit and result.calibration_normal and result.test


def test_the_extracted_dataset_validates_without_errors(dataset: VisADataset) -> None:
    report = validate_dataset(dataset.samples, dataset.images_root)
    assert report.ok, report.to_dict()["issues"]


def test_the_chosen_resize_modes_follow_from_the_measured_aspect_ratios(
    dataset: VisADataset,
) -> None:
    report = validate_dataset(dataset.samples, dataset.images_root)
    assert report.ok
    for name, obj in report.objects.items():
        assert resize_mode_for(name) == obj.recommended_resize_mode, name
