"""INT8 quantization: what it converts, that artifacts are untouched, and that results stay sane."""

import numpy as np
import pytest
import torch
import torchvision
from fake_backbone import FakeBackbone

from inspectai.errors import ModelError
from inspectai.models.features import FeatureExtractor, WeightsInfo, images_to_batch
from inspectai.models.patchcore import PatchCoreConfig, PatchCoreModel
from inspectai.models.quantize import StaticInt8Extractor, dynamic_quantize

WEIGHTS = WeightsInfo("fake", "injected", None)
LAYERS = ("layer1", "layer2", "layer3")

needs_engine = pytest.mark.skipif(
    not [e for e in torch.backends.quantized.supported_engines if e != "none"],
    reason="this PyTorch build has no quantized CPU engine",
)


def _images(count: int = 8, size: int = 64) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    ramp = np.linspace(40, 200, size, dtype=np.float32)[None, :, None]
    return [
        np.clip(ramp + rng.normal(0, 6, (size, size, 3)), 0, 255).astype(np.uint8)
        for _ in range(count)
    ]


# ----- dynamic quantization ---------------------------------------------------------------------


@needs_engine
def test_dynamic_quantization_converts_only_linear_layers_and_leaves_the_input_alone() -> None:
    backbone = FakeBackbone()
    quantized, report = dynamic_quantize(backbone, LAYERS)
    assert report.n_conv_layers == 3 and report.n_linear_layers == 1
    assert report.n_quantized_modules == 1  # the unused classification head, nothing else
    assert not report.covers_feature_path
    assert isinstance(backbone.head, torch.nn.Linear)  # the caller's module is not modified
    assert not isinstance(quantized.head, torch.nn.Linear)
    # the convolutions stay float: dynamic INT8 has no convolution kernel
    assert all(isinstance(getattr(quantized, name)[0], torch.nn.Conv2d) for name in LAYERS)


@needs_engine
def test_dynamic_quantization_leaves_the_features_exactly_unchanged() -> None:
    backbone = FakeBackbone()
    quantized, _ = dynamic_quantize(backbone, LAYERS)
    batch = torch.randn(2, 3, 64, 64)
    before = FeatureExtractor(backbone, LAYERS, "layer2").embed(batch)
    after = FeatureExtractor(quantized, LAYERS, "layer2").embed(batch)
    assert torch.equal(before, after)


@needs_engine
def test_the_report_flags_a_converted_layer_on_the_feature_path() -> None:
    backbone = FakeBackbone()
    backbone.layer3 = torch.nn.Sequential(torch.nn.Flatten(0, 0), torch.nn.Linear(24, 24))
    _, report = dynamic_quantize(backbone, LAYERS)
    assert report.covers_feature_path


# ----- artifact compatibility -------------------------------------------------------------------


def _fitted_model() -> PatchCoreModel:
    config = PatchCoreConfig(
        layers=LAYERS,
        grid_layer="layer2",
        image_size=64,
        resize_mode="stretch",
        coreset_ratio=0.2,
        batch_size=4,
        num_threads=1,
    )
    model = PatchCoreModel(config, FeatureExtractor(FakeBackbone(), LAYERS, "layer2"), WEIGHTS)
    model.fit(_images())
    return model


@needs_engine
def test_swapping_in_a_dynamically_quantized_extractor_keeps_tensors_and_scores() -> None:
    model = _fitted_model()
    tensors = {k: v.copy() for k, v in model.tensors().items()}
    probe = _images()[3]
    reference = model.predict(probe)
    quantized, _ = dynamic_quantize(FakeBackbone(), LAYERS)
    model.use_extractor(FeatureExtractor(quantized, LAYERS, "layer2"))
    assert all(np.array_equal(tensors[k], v) for k, v in model.tensors().items())
    swapped = model.predict(probe)
    assert swapped.score == reference.score
    assert np.array_equal(swapped.anomaly_map, reference.anomaly_map)


# ----- static INT8 ------------------------------------------------------------------------------


def _resnet() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None)
    for module in model.modules():  # non-trivial normalization statistics
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0.0, 0.1)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def _calibration(model_size: int = 64) -> list[torch.Tensor]:
    images = _images(16, model_size)
    return [images_to_batch(images[i : i + 8], model_size, "stretch")[0] for i in range(0, 16, 8)]


@needs_engine
def test_static_int8_features_track_the_float_features() -> None:
    backbone = _resnet()
    layers = ("layer1", "layer2", "layer3")
    extractor = StaticInt8Extractor(backbone, layers, "layer2", _calibration())
    reference = FeatureExtractor(backbone, layers, "layer2")
    batch = images_to_batch(_images(4), 64, "stretch")[0]
    quantized, exact = extractor.embed(batch), reference.embed(batch)
    assert quantized.shape == exact.shape == (4, 64 + 128 + 256, 8, 8)
    assert torch.isfinite(quantized).all()
    cosine = torch.nn.functional.cosine_similarity(quantized.flatten(1), exact.flatten(1))
    assert float(cosine.min()) > 0.95  # INT8 changes the values a little, not the content
    assert not torch.equal(quantized, exact)  # and it really is a different (quantized) path


@needs_engine
def test_a_statically_quantized_extractor_scores_a_float_fitted_artifact() -> None:
    backbone = _resnet()
    layers = ("layer2", "layer3")
    config = PatchCoreConfig(
        layers=layers, image_size=64, resize_mode="stretch", coreset_ratio=0.2, num_threads=1
    )
    model = PatchCoreModel(config, FeatureExtractor(backbone, layers), WEIGHTS)
    model.fit(_images())
    tensors = {k: v.copy() for k, v in model.tensors().items()}
    probe = _images()[2]
    float_score = model.predict(probe).score
    model.use_extractor(StaticInt8Extractor(backbone, layers, None, _calibration()))
    prediction = model.predict(probe)
    assert all(np.array_equal(tensors[k], v) for k, v in model.tensors().items())  # bank intact
    assert prediction.anomaly_map.shape == probe.shape[:2]
    assert np.isfinite(prediction.anomaly_map).all()
    assert prediction.score == pytest.approx(float_score, rel=0.35)  # same scale, small drift


@needs_engine
def test_static_quantization_rejects_unsupported_layers_and_missing_calibration() -> None:
    backbone = _resnet()
    with pytest.raises(ModelError, match="layer1 to layer3"):
        StaticInt8Extractor(backbone, ("layer4",), None, _calibration())
    with pytest.raises(ModelError, match="grid_layer"):
        StaticInt8Extractor(backbone, ("layer2", "layer3"), "layer1", _calibration())
    with pytest.raises(ModelError, match="calibration batch"):
        StaticInt8Extractor(backbone, ("layer2", "layer3"), None, [])
