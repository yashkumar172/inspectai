"""Backbone loading and feature extraction. No real weights are used or downloaded."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from fake_backbone import FakeBackbone

from inspectai.errors import ConfigError, ImageError, ModelError, WeightsNotFoundError
from inspectai.models import features
from inspectai.models.features import (
    FeatureExtractor,
    freeze,
    images_to_batch,
    load_backbone,
    weights_cache_path,
)


def test_embedding_concatenates_layers_at_the_first_layers_resolution() -> None:
    extractor = FeatureExtractor(FakeBackbone(), ("layer2", "layer3"))
    embedding = extractor.embed(torch.zeros(2, 3, 64, 64))
    assert embedding.shape == (2, 16 + 24, 16, 16)  # layer2 is 1/4 of the input; layer3 upsampled


def test_a_single_layer_is_returned_unchanged_in_size() -> None:
    extractor = FeatureExtractor(FakeBackbone(), ("layer3",))
    assert extractor.embed(torch.zeros(1, 3, 64, 64)).shape == (1, 24, 8, 8)


def test_features_are_deterministic_and_depend_on_the_input() -> None:
    extractor = FeatureExtractor(FakeBackbone(), ("layer2", "layer3"))
    rng = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, 64, 64, generator=rng)
    assert torch.equal(extractor.embed(x), extractor.embed(x))
    assert not torch.equal(extractor.embed(x), extractor.embed(x + 1.0))
    other = FeatureExtractor(FakeBackbone(seed=1), ("layer2", "layer3"))
    assert not torch.equal(extractor.embed(x), other.embed(x))


def test_the_backbone_is_frozen_and_features_carry_no_gradient() -> None:
    backbone = FakeBackbone()
    extractor = FeatureExtractor(backbone, ("layer2",))
    assert not backbone.training
    assert all(not p.requires_grad for p in backbone.parameters())
    assert not extractor.embed(torch.zeros(1, 3, 32, 32)).requires_grad


def test_features_are_computed_on_the_cpu() -> None:
    embedding = FeatureExtractor(FakeBackbone(), ("layer2",)).embed(torch.zeros(1, 3, 32, 32))
    assert embedding.device.type == "cpu" and embedding.dtype == torch.float32


@pytest.mark.parametrize("layers", [(), ("layer2", "layer2"), ("does_not_exist",)])
def test_invalid_layers_are_rejected(layers: tuple[str, ...]) -> None:
    with pytest.raises(ModelError):
        FeatureExtractor(FakeBackbone(), layers)


def test_embed_rejects_a_wrong_input_shape() -> None:
    extractor = FeatureExtractor(FakeBackbone(), ("layer2",))
    with pytest.raises(ModelError):
        extractor.embed(torch.zeros(3, 32, 32))
    with pytest.raises(ModelError):
        extractor.embed(torch.zeros(1, 1, 32, 32))


def test_images_to_batch_resizes_normalizes_and_stacks() -> None:
    images = [np.full((30, 40, 3), 128, dtype=np.uint8), np.zeros((20, 20, 3), dtype=np.uint8)]
    batch, infos = images_to_batch(images, 32, "stretch")
    assert batch.shape == (2, 3, 32, 32) and batch.dtype == torch.float32
    assert [i.original_hw for i in infos] == [(30, 40), (20, 20)]
    expected = (128 / 255 - 0.485) / 0.229  # ImageNet mean/std on the red channel
    assert batch[0, 0, 0, 0].item() == pytest.approx(expected, rel=1e-4)


def test_images_to_batch_rejects_bad_input() -> None:
    with pytest.raises(ImageError):
        images_to_batch([], 32, "stretch")
    with pytest.raises(ImageError):
        images_to_batch([np.zeros((8, 8, 3), dtype=np.uint8)], 32, "crop")


def test_freeze_disables_gradients() -> None:
    model = freeze(torch.nn.Linear(2, 2))
    assert not model.training and not any(p.requires_grad for p in model.parameters())


# ----- weights ------------------------------------------------------------------------------


@pytest.fixture
def empty_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch home"))
    monkeypatch.delenv("INSPECTAI_ALLOW_WEIGHT_DOWNLOAD", raising=False)
    return tmp_path / "torch home"


def test_the_weights_path_follows_torch_home(empty_cache: Path) -> None:
    path = weights_cache_path("resnet18")
    assert path.parent == empty_cache / "hub" / "checkpoints"
    assert path.name.startswith("resnet18-") and path.suffix == ".pth"


def test_missing_weights_raise_an_actionable_error_and_never_download(
    empty_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(name: str) -> Path:
        raise AssertionError("a download must not be attempted")

    monkeypatch.setattr(features, "fetch_weights", forbidden)
    with pytest.raises(WeightsNotFoundError) as caught:
        load_backbone("resnet18")
    message = str(caught.value)
    assert "fetch_weights.py --backbone resnet18" in message
    assert "INSPECTAI_ALLOW_WEIGHT_DOWNLOAD" in message


def test_download_is_attempted_only_when_allowed(
    empty_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_fetch(name: str) -> Path:
        calls.append(name)
        path = weights_cache_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the real weights")
        return path

    monkeypatch.setattr(features, "fetch_weights", fake_fetch)
    with pytest.raises(WeightsNotFoundError, match="corrupted"):  # hash check catches the fake
        load_backbone("resnet18", allow_download=True)
    assert calls == ["resnet18"]
    monkeypatch.setenv("INSPECTAI_ALLOW_WEIGHT_DOWNLOAD", "1")
    calls.clear()
    weights_cache_path("resnet18").unlink()
    with pytest.raises(WeightsNotFoundError, match="corrupted"):
        load_backbone("resnet18")
    assert calls == ["resnet18"]


def test_a_corrupted_cached_file_is_rejected(empty_cache: Path) -> None:
    path = weights_cache_path("resnet18")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"truncated")
    with pytest.raises(WeightsNotFoundError, match="corrupted"):
        load_backbone("resnet18")


def test_an_unsupported_backbone_is_rejected(empty_cache: Path) -> None:
    with pytest.raises(ConfigError, match="Unsupported backbone"):
        load_backbone("vgg16")


def test_cached_weights_are_loaded_frozen_and_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real-shaped state dict under a file name without a hash prefix, so no network is needed.
    reference = torchvision.models.resnet18(weights=None)
    path = tmp_path / "resnet18-local.pth"
    torch.save(reference.state_dict(), path)
    monkeypatch.setattr(features, "weights_cache_path", lambda name, hub_dir=None: path)
    model, info = load_backbone("resnet18")
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert info.backbone == "resnet18" and info.identifier == "IMAGENET1K_V1"
    assert info.sha256 is not None and len(info.sha256) == 64
    for key, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_a_real_resnet_yields_the_expected_feature_shapes() -> None:
    # Architecture only (random weights, built directly): no pretrained file is involved.
    extractor = FeatureExtractor(torchvision.models.resnet18(weights=None), ("layer2", "layer3"))
    embedding = extractor.embed(torch.zeros(1, 3, 224, 224))
    assert embedding.shape == (1, 128 + 256, 28, 28)


def test_a_hub_dir_overrides_the_torch_home_cache(tmp_path: Path, empty_cache: Path) -> None:
    hub = tmp_path / "bundled hub"
    assert weights_cache_path("resnet18", hub) == hub / "checkpoints" / (
        weights_cache_path("resnet18").name
    )
    assert weights_cache_path("resnet18", hub).parent != weights_cache_path("resnet18").parent
    with pytest.raises(WeightsNotFoundError, match="bundled hub"):
        load_backbone("resnet18", False, hub)
    cached = weights_cache_path("resnet18", hub)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"truncated")
    with pytest.raises(WeightsNotFoundError, match="corrupted"):  # found there, and hash-checked
        load_backbone("resnet18", False, hub)
