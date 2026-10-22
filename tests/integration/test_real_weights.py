"""Checks with real pretrained weights and the real dataset. Run with: pytest -m slow

Skipped by default, and skipped when the weights are not in the torchvision cache. Fetch them
first with: python scripts/fetch_weights.py --backbone resnet18
"""

import numpy as np
import pytest
import torch

from inspectai.models.features import FeatureExtractor, load_backbone, weights_cache_path

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def resnet18():
    if not weights_cache_path("resnet18").is_file():
        pytest.skip("resnet18 weights are not cached; run scripts/fetch_weights.py")
    backbone, info = load_backbone("resnet18", allow_download=False)
    return backbone, info


def test_real_resnet18_weights_load_and_give_the_expected_feature_shape(resnet18) -> None:
    backbone, info = resnet18
    assert info.identifier == "IMAGENET1K_V1" and len(info.sha256) == 64
    assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
    embedding = FeatureExtractor(backbone, ("layer2", "layer3")).embed(torch.zeros(1, 3, 224, 224))
    assert embedding.shape == (1, 128 + 256, 28, 28)


def test_real_weights_are_not_random(resnet18) -> None:
    backbone, _ = resnet18
    import torchvision

    random_net = torchvision.models.resnet18(weights=None)
    assert not torch.equal(backbone.conv1.weight, random_net.conv1.weight)
    assert np.isfinite(backbone.conv1.weight.detach().numpy()).all()


@pytest.mark.needs_dataset
def test_model_b_with_real_weights_ranks_anomalies_above_normals(resnet18) -> None:
    """A small real fit (40 images): anomalous test images must score higher on average."""
    from inspectai.config import SplitConfig, load_paths
    from inspectai.data.dataset import VisADataset
    from inspectai.data.splits import make_splits
    from inspectai.models.gaussian import GaussianModel, gaussian_config_for

    paths = load_paths()
    if not paths.split_csv.is_file() or not paths.visa_images_root.is_dir():
        pytest.skip("VisA dataset not found under INSPECTAI_DATA_ROOT")
    dataset = VisADataset.from_paths(paths)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    by_rel = {s.image_rel: s for s in dataset.get_samples("cashew")}
    backbone, weights = resnet18
    config = gaussian_config_for("cashew", "dev")
    model = GaussianModel(config, FeatureExtractor(backbone, config.layers), weights)
    model.fit(dataset.load_image(by_rel[n]) for n in splits.fit[:40])
    test = [by_rel[n] for n in splits.test]
    normal = [s for s in test if not s.is_anomalous][:10]
    anomalous = [s for s in test if s.is_anomalous][:10]
    scores_normal = [model.predict(dataset.load_image(s)).score for s in normal]
    scores_anomalous = [model.predict(dataset.load_image(s)).score for s in anomalous]
    assert np.mean(scores_anomalous) > np.mean(scores_normal)
    prediction = model.predict(dataset.load_image(anomalous[0]))
    assert prediction.anomaly_map.shape == dataset.load_image(anomalous[0]).shape[:2]


def test_real_wide_resnet50_2_weights_give_the_full_profile_feature_shape() -> None:
    if not weights_cache_path("wide_resnet50_2").is_file():
        pytest.skip("wide_resnet50_2 weights are not cached; run scripts/fetch_weights.py")
    backbone, info = load_backbone("wide_resnet50_2", allow_download=False)
    assert info.identifier == "IMAGENET1K_V1" and info.sha256.startswith("95faca4d")
    embedding = FeatureExtractor(backbone, ("layer2", "layer3")).embed(torch.zeros(1, 3, 224, 224))
    assert embedding.shape == (1, 512 + 1024, 28, 28)


@pytest.mark.needs_dataset
def test_model_c_with_real_weights_ranks_anomalies_above_normals(resnet18) -> None:
    """A small real fit (40 images, 10% coreset): anomalous test images score higher."""
    from inspectai.config import SplitConfig, load_paths
    from inspectai.data.dataset import VisADataset
    from inspectai.data.splits import make_splits
    from inspectai.models.patchcore import PatchCoreModel, patchcore_config_for

    paths = load_paths()
    if not paths.split_csv.is_file() or not paths.visa_images_root.is_dir():
        pytest.skip("VisA dataset not found under INSPECTAI_DATA_ROOT")
    dataset = VisADataset.from_paths(paths)
    splits = make_splits(dataset.samples, "cashew", SplitConfig())
    by_rel = {s.image_rel: s for s in dataset.get_samples("cashew")}
    backbone, weights = resnet18
    config = patchcore_config_for("cashew", "dev", coreset_ratio=0.1)
    model = PatchCoreModel(config, FeatureExtractor(backbone, config.layers), weights)
    model.fit([dataset.load_image(by_rel[n]) for n in splits.fit[:40]])
    assert model.n_candidates == 40 * 784 and model.bank_size == round(0.1 * 40 * 784)
    assert model.feature_dim == 384 and model.feature_map_shape == (28, 28)
    test = [by_rel[n] for n in splits.test]
    normal = [s for s in test if not s.is_anomalous][:10]
    anomalous = [s for s in test if s.is_anomalous][:10]
    scores_normal = [model.predict(dataset.load_image(s)).score for s in normal]
    scores_anomalous = [model.predict(dataset.load_image(s)).score for s in anomalous]
    assert np.mean(scores_anomalous) > np.mean(scores_normal)
    image = dataset.load_image(anomalous[0])
    assert model.predict(image).anomaly_map.shape == image.shape[:2]
