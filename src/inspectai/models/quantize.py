"""Optional INT8 quantization of the frozen backbone (inference only; SPEC.md Section 5.5).

Quantization never touches a fitted artifact: the memory bank or Gaussian parameters stay float32
and are read exactly as stored. Only the feature extractor that produces the query features is
swapped (`use_extractor` on a loaded model), so a quantized run is a different inference setting
of the same artifact, and its effect on image AUROC has to be measured, not assumed.

Two paths:

- `dynamic_quantize` is PyTorch's `quantize_dynamic`. It converts only layers with a dynamic INT8
  implementation (`nn.Linear`, recurrent layers). A ResNet's feature trunk contains convolutions
  only, so on the layers this project uses it converts nothing; `QuantizationReport` says so
  instead of reporting a speedup that does not exist.
- `StaticInt8Extractor` is eager-mode post-training static quantization of torchvision's
  quantizable ResNet-18 (activations and weights INT8, calibrated on normal fit images), which
  does cover the convolutions. Only `resnet18` (the `dev` profile) is supported.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torchvision
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

from inspectai.errors import ConfigError, ModelError
from inspectai.models.features import _to_grid

QUANTIZABLE_STATIC_BACKBONES = ("resnet18",)


@dataclass(frozen=True)
class QuantizationReport:
    """What a dynamic-quantization attempt actually converted."""

    n_conv_layers: int
    n_linear_layers: int
    n_quantized_modules: int
    covers_feature_path: bool  # True only if a layer the feature extractor runs was converted

    def to_dict(self) -> dict[str, object]:
        return {
            "n_conv_layers": self.n_conv_layers,
            "n_linear_layers": self.n_linear_layers,
            "n_quantized_modules": self.n_quantized_modules,
            "covers_feature_path": self.covers_feature_path,
        }


def dynamic_quantize(
    backbone: nn.Module, feature_layers: Sequence[str] = ("layer1", "layer2", "layer3")
) -> tuple[nn.Module, QuantizationReport]:
    """INT8 dynamic quantization of a copy of `backbone` (the input module is not modified).

    `feature_layers` names the sub-modules the feature extractor runs; the report states whether
    any converted module lies inside them.
    """
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(backbone), {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    converted = [
        name for name, module in quantized.named_modules() if isinstance(module, DynamicLinear)
    ]
    in_path = any(name.split(".")[0] in feature_layers for name in converted)
    report = QuantizationReport(
        n_conv_layers=sum(isinstance(m, nn.Conv2d) for m in backbone.modules()),
        n_linear_layers=sum(isinstance(m, nn.Linear) for m in backbone.modules()),
        n_quantized_modules=len(converted),
        covers_feature_path=in_path,
    )
    return quantized, report


class StaticInt8Extractor:
    """ResNet-18 trunk (conv1 .. layer3) in INT8, with the `FeatureExtractor` interface.

    Weights and activations are quantized after calibration on normal images (post-training, no
    training). Each requested layer's output is dequantized to float32 before the layers are
    resized to one grid and concatenated, exactly like `FeatureExtractor.embed`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        layers: Sequence[str],
        grid_layer: str | None,
        calibration_batches: Iterable[torch.Tensor],
    ) -> None:
        self.layers = tuple(layers)
        self.grid_layer = grid_layer
        unknown = [name for name in self.layers if name not in ("layer1", "layer2", "layer3")]
        if unknown or not self.layers:
            raise ModelError(f"Static INT8 supports layer1 to layer3, got {self.layers}")
        if grid_layer is not None and grid_layer not in self.layers:
            raise ModelError(f"grid_layer {grid_layer!r} is not one of the layers {self.layers}")
        engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
        if not engines:
            raise ConfigError("This PyTorch build has no quantized CPU engine")
        self._engine = "x86" if "x86" in engines else engines[0]
        torch.backends.quantized.engine = self._engine
        self._model = self._quantize(backbone, calibration_batches)

    def _quantize(self, backbone: nn.Module, batches: Iterable[torch.Tensor]) -> nn.Module:
        model = torchvision.models.quantization.resnet18(weights=None, quantize=False)
        model.load_state_dict(backbone.state_dict())
        model.eval()
        model.fuse_model(is_qat=False)
        model.layer4 = model.avgpool = model.fc = nn.Identity()  # the trunk never runs them
        model.qconfig = torch.ao.quantization.get_default_qconfig(self._engine)
        torch.ao.quantization.prepare(model, inplace=True)
        seen = 0
        with torch.inference_mode():
            for batch in batches:
                self._trunk(model, batch)
                seen += 1
        if seen == 0:
            raise ModelError("Static quantization needs at least one calibration batch")
        torch.ao.quantization.convert(model, inplace=True)
        return model

    @staticmethod
    def _trunk(model: nn.Module, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        x = model.quant(batch)
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        layer1 = model.layer1(x)
        layer2 = model.layer2(layer1)
        layer3 = model.layer3(layer2)
        return {"layer1": layer1, "layer2": layer2, "layer3": layer3}

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Features of a (batch, 3, H, W) tensor, on one grid, concatenated along channels."""
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ModelError(f"Expected a (batch, 3, H, W) tensor, got {tuple(batch.shape)}")
        outputs = self._trunk(self._model, batch)
        maps = [outputs[name].dequantize() for name in self.layers]
        reference = maps[0] if self.grid_layer is None else maps[self.layers.index(self.grid_layer)]
        size = reference.shape[-2:]
        return torch.cat([m if m.shape[-2:] == size else _to_grid(m, size) for m in maps], dim=1)
