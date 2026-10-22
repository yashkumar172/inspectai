"""Model C: PatchCore-inspired anomaly detection (SPEC.md Section 5.3).

Algorithm:

1. Every `fit` image is resized (category resize mode) and passed through the frozen backbone.
   The layer2 and layer3 feature maps are upsampled to the layer2 resolution and concatenated,
   exactly as for Model B, giving one D-dimensional vector per spatial position (a "patch").
2. Each patch vector is averaged with its `neighborhood` x `neighborhood` spatial neighbors
   (stride 1), so a vector describes a small local region, as in PatchCore.
3. All patch vectors of all fit images form the candidate memory bank.
4. The bank is reduced with greedy k-center coreset selection: the vectors are randomly
   projected to `projection_dim` (default 128) dimensions, a seeded random vector starts the
   selection, and the vector farthest from the already selected ones is added until
   `coreset_ratio` (default 0.10) of the candidates are selected. `coreset_ratio = 1` keeps
   every vector. Selection stops with an error if it exceeds `max_fit_minutes`.
5. A test patch is scored by its Euclidean distance to the nearest bank vector (exact search
   with chunked `torch.cdist`). The patch scores form a map, which is upsampled to the input
   size, smoothed with a Gaussian filter, and mapped back to the original image size.
6. The image score is the maximum of the anomaly map.

Deviations from PatchCore (Roth et al., CVPR 2022), so this is "inspired by", not equivalent to,
the reference method:

- The 3x3 neighborhood averaging is applied after the layers are upsampled and concatenated,
  not to each layer at its own resolution.
- There is no adaptive average pooling of features to a lower dimension: D is the number of
  concatenated channels (384 for `resnet18`, 1536 for `wide_resnet50_2`).
- The image score is the maximum of the smoothed map; PatchCore's re-weighting of the maximum
  by the neighbor distances of the nearest bank vector is not implemented.
- Nearest-neighbor search is exact (k = 1), without an approximate index.
- Only the greedy coreset selection runs on the random projection; the bank itself keeps the
  full-dimensional vectors.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from inspectai.config import RESIZE_MODES, profile_settings, resize_mode_for
from inspectai.errors import ModelError
from inspectai.models.base import Prediction
from inspectai.models.features import FeatureExtractor, WeightsInfo, images_to_batch
from inspectai.preprocessing.images import ImageArray, map_back

logger = logging.getLogger(__name__)

_BYTES_PER_FLOAT32 = 4
_DEADLINE_CHECK_EVERY = 64


@dataclass(frozen=True)
class PatchCoreConfig:
    """Settings of Model C.

    `to_dict` returns the settings that define the fitted model; an artifact fitted with
    different values is stale. The resource settings (`chunk_size`, `batch_size`, `num_threads`,
    `max_bank_gb`, `max_fit_minutes`) only limit cost. `snapshot` combines both, with the
    resource settings under a `runtime` key, for the artifact's configuration snapshot.
    """

    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 224
    resize_mode: str = "letterbox"
    coreset_ratio: float = 0.10
    projection_dim: int = 128
    neighborhood: int = 3
    smoothing_sigma: float = 4.0
    seed: int = 42
    chunk_size: int = 16384
    batch_size: int = 8
    num_threads: int = 6
    max_bank_gb: float = 4.0
    max_fit_minutes: float = 30.0

    def __post_init__(self) -> None:
        if self.image_size < 8:
            raise ModelError(f"image_size must be at least 8, got {self.image_size}")
        if self.resize_mode not in RESIZE_MODES:
            raise ModelError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise ModelError(f"coreset_ratio must be in (0, 1], got {self.coreset_ratio}")
        if self.projection_dim < 1:
            raise ModelError(f"projection_dim must be at least 1, got {self.projection_dim}")
        if self.neighborhood < 1 or self.neighborhood % 2 == 0:
            raise ModelError(f"neighborhood must be a positive odd number, got {self.neighborhood}")
        if self.smoothing_sigma < 0:
            raise ModelError(f"smoothing_sigma must not be negative, got {self.smoothing_sigma}")
        if min(self.chunk_size, self.batch_size, self.num_threads) < 1:
            raise ModelError("chunk_size, batch_size, and num_threads must be at least 1")
        if self.max_bank_gb <= 0 or self.max_fit_minutes <= 0:
            raise ModelError("max_bank_gb and max_fit_minutes must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "image_size": self.image_size,
            "resize_mode": self.resize_mode,
            "coreset_ratio": self.coreset_ratio,
            "projection_dim": self.projection_dim,
            "neighborhood": self.neighborhood,
            "smoothing_sigma": self.smoothing_sigma,
            "seed": self.seed,
        }

    def runtime_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "max_bank_gb": self.max_bank_gb,
            "max_fit_minutes": self.max_fit_minutes,
        }

    def snapshot(self) -> dict[str, object]:
        """The exact configuration used: model settings plus a `runtime` section."""
        return {**self.to_dict(), "runtime": self.runtime_dict()}

    @classmethod
    def from_dict(
        cls, data: dict[str, object], runtime: dict[str, object] | None = None
    ) -> PatchCoreConfig:
        values = {**data, **(runtime or {})}
        values["layers"] = tuple(values.get("layers", ()))  # type: ignore[arg-type]
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ModelError(f"Invalid PatchCore configuration: {exc}") from exc


def patchcore_config_for(
    category: str,
    profile: str = "dev",
    coreset_ratio: float = 0.10,
    seed: int = 42,
    **limits: float,
) -> PatchCoreConfig:
    """The configuration of a category under a profile (`dev` or `full`)."""
    settings = profile_settings(profile)
    return PatchCoreConfig(
        backbone=str(settings["backbone"]),
        layers=tuple(settings["layers"]),  # type: ignore[arg-type]
        image_size=int(settings["image_size"]),  # type: ignore[call-overload]
        resize_mode=resize_mode_for(category),
        coreset_ratio=coreset_ratio,
        seed=seed,
        **limits,  # type: ignore[arg-type]
    )


def estimate_bank_gb(n_images: int, patches_per_image: int, dim: int) -> float:
    """Size of the unreduced memory bank in GiB (SPEC.md Section 5.5)."""
    return n_images * patches_per_image * dim * _BYTES_PER_FLOAT32 / 1024**3


def greedy_coreset(
    features: torch.Tensor,
    n_select: int,
    projection_dim: int = 128,
    seed: int = 42,
    max_seconds: float | None = None,
) -> torch.Tensor:
    """Indices of a greedy k-center coreset of `features` (N, D).

    The features are randomly projected to `projection_dim` dimensions (seeded). A seeded random
    point starts the selection; then the point farthest from all selected points is added
    repeatedly. Raises ModelError if `max_seconds` elapses first, never returning a partial
    selection.
    """
    n_points = features.shape[0]
    if not 1 <= n_select <= n_points:
        raise ModelError(f"n_select must be between 1 and {n_points}, got {n_select}")
    if n_select == n_points:
        return torch.arange(n_points)
    generator = torch.Generator().manual_seed(seed)
    dim = features.shape[1]
    projection = torch.randn(dim, projection_dim, generator=generator) / projection_dim**0.5
    projected = torch.empty(n_points, projection_dim)
    for start in range(0, n_points, 32768):
        projected[start : start + 32768] = features[start : start + 32768] @ projection
    squared = (projected * projected).sum(dim=1)

    start_index = int(torch.randint(n_points, (1,), generator=generator))
    selected = torch.empty(n_select, dtype=torch.int64)
    selected[0] = start_index
    nearest = squared - 2.0 * torch.mv(projected, projected[start_index]) + squared[start_index]
    nearest[start_index] = -1.0  # selected points are never picked again
    begin = time.perf_counter()
    for count in range(1, n_select):
        if (
            max_seconds is not None
            and count % _DEADLINE_CHECK_EVERY == 0
            and time.perf_counter() - begin > max_seconds
        ):
            raise ModelError(
                f"Coreset selection exceeded the time limit after {count} of {n_select} "
                f"points (max_fit_minutes). Use random subsampling instead: lower "
                f"max_fit_images, or raise max_fit_minutes if you can wait."
            )
        chosen = int(torch.argmax(nearest))
        selected[count] = chosen
        distance = squared - 2.0 * torch.mv(projected, projected[chosen]) + squared[chosen]
        torch.minimum(nearest, distance, out=nearest)
        nearest[chosen] = -1.0
    return selected


def nearest_neighbor_distances(
    queries: torch.Tensor, bank: torch.Tensor, chunk_size: int = 16384
) -> torch.Tensor:
    """Euclidean distance from each query (Q, D) to its nearest bank vector (M, D).

    Exact search with chunked `torch.cdist`: only a (Q, chunk_size) distance block exists at a
    time.
    """
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ModelError(f"Incompatible shapes {tuple(queries.shape)} and {tuple(bank.shape)}")
    if chunk_size < 1:
        raise ModelError("chunk_size must be at least 1")
    best = torch.full((queries.shape[0],), float("inf"))
    for start in range(0, bank.shape[0], chunk_size):
        block = torch.cdist(queries, bank[start : start + chunk_size])
        best = torch.minimum(best, block.min(dim=1).values)
    return best


class PatchCoreModel:
    """Patch-level memory bank with exact nearest-neighbor scoring."""

    def __init__(
        self, config: PatchCoreConfig, extractor: FeatureExtractor, weights: WeightsInfo
    ) -> None:
        self.config = config
        self.weights = weights
        self._extractor = extractor
        self.n_fit_images = 0
        self.n_candidates = 0
        self.coreset_seconds = 0.0
        self.estimated_bank_gb = 0.0
        self.reference_aspect_ratio: float | None = None
        self.feature_map_shape: tuple[int, int] | None = None
        self._bank: torch.Tensor | None = None

    @property
    def bank_size(self) -> int:
        """Number of vectors in the (reduced) memory bank."""
        return 0 if self._bank is None else int(self._bank.shape[0])

    @property
    def feature_dim(self) -> int | None:
        return None if self._bank is None else int(self._bank.shape[1])

    # ----- fitting -------------------------------------------------------------------------

    def _patches(self, batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Neighborhood-averaged patch vectors (B, h*w, D) of an image batch."""
        embedding = self._extractor.embed(batch)
        k = self.config.neighborhood
        if k > 1:
            embedding = F.avg_pool2d(
                embedding, k, stride=1, padding=k // 2, count_include_pad=False
            )
        batch_size, channels, height, width = embedding.shape
        vectors = embedding.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return vectors, (height, width)

    @torch.inference_mode()
    def fit(self, images: Iterable[ImageArray]) -> None:
        """Build the memory bank from normal images and reduce it with the coreset.

        If `images` has a length, the bank size is estimated and checked against `max_bank_gb`
        before any image is processed beyond the first batch; otherwise the check runs as
        vectors accumulate. Either way the fit stops with an error and never silently shrinks
        the fit set.
        """
        cfg = self.config
        torch.set_num_threads(cfg.num_threads)
        started = time.perf_counter()
        expected = len(images) if isinstance(images, Sized) else None
        aspects: list[float] = []

        def counted() -> Iterator[ImageArray]:
            for image in images:
                aspects.append(image.shape[1] / image.shape[0])
                yield image

        pieces: list[torch.Tensor] = []
        bank: torch.Tensor | None = None
        filled = 0
        shape: tuple[int, int] | None = None
        batch_images: list[ImageArray] = []

        def flush() -> None:
            nonlocal bank, filled, shape
            batch, _ = images_to_batch(batch_images, cfg.image_size, cfg.resize_mode)
            vectors, map_shape = self._patches(batch)
            batch_images.clear()
            patches, dim = map_shape[0] * map_shape[1], vectors.shape[2]
            if shape is None:
                shape = map_shape
                count = expected if expected is not None else vectors.shape[0]
                self.estimated_bank_gb = estimate_bank_gb(count, patches, dim)
                logger.info(
                    "Memory bank estimate: %d images x %d patches x %d dims x 4 bytes = %.2f GB "
                    "before reduction (limit %.2f GB)",
                    count, patches, dim, self.estimated_bank_gb, cfg.max_bank_gb,
                )  # fmt: skip
                if expected is not None:
                    self._check_bank_size(self.estimated_bank_gb)
                    bank = torch.empty(expected * patches, dim)
            flat = vectors.reshape(-1, dim)
            if bank is not None:
                if filled + flat.shape[0] > bank.shape[0]:
                    raise ModelError("More images were provided than the length they reported")
                bank[filled : filled + flat.shape[0]] = flat
            else:
                pieces.append(flat.clone())
                self._check_bank_size(sum(p.numel() for p in pieces) * _BYTES_PER_FLOAT32 / 1024**3)
            filled += flat.shape[0]

        for image in counted():
            batch_images.append(image)
            if len(batch_images) == cfg.batch_size:
                flush()
        if batch_images:
            flush()
        if shape is None or filled == 0:
            raise ModelError("Fitting needs at least 1 image, got 0")
        candidates = bank[:filled] if bank is not None else torch.cat(pieces)
        pieces.clear()
        n_images = filled // (shape[0] * shape[1])
        if n_images < 2:
            raise ModelError(f"Fitting needs at least 2 images, got {n_images}")

        self.n_candidates = int(candidates.shape[0])
        n_select = max(1, round(cfg.coreset_ratio * self.n_candidates))
        select_start = time.perf_counter()
        remaining = cfg.max_fit_minutes * 60.0 - (select_start - started)
        if remaining <= 0:
            raise ModelError(
                "Feature extraction alone exceeded max_fit_minutes; use random subsampling "
                "(max_fit_images) or raise the limit."
            )
        logger.info("Coreset: selecting %d of %d vectors", n_select, self.n_candidates)
        indices = greedy_coreset(
            candidates, n_select, cfg.projection_dim, cfg.seed, max_seconds=remaining
        )
        self.coreset_seconds = time.perf_counter() - select_start
        self._bank = candidates[indices].contiguous().clone()
        self.feature_map_shape = shape
        self.n_fit_images = n_images
        self.reference_aspect_ratio = float(np.median(aspects))

    def _check_bank_size(self, gigabytes: float) -> None:
        if gigabytes > self.config.max_bank_gb:
            raise ModelError(
                f"The memory bank would need about {gigabytes:.2f} GB before reduction, above "
                f"max_bank_gb={self.config.max_bank_gb}. Lower max_fit_images (a recorded, seeded "
                f"subset of the fit split), use the dev profile, or raise max_bank_gb."
            )

    # ----- inference -----------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, image: ImageArray) -> Prediction:
        """Score one image; the map has the image's own height and width."""
        if self._bank is None or self.feature_map_shape is None:
            raise ModelError("The model has not been fitted")
        cfg = self.config
        torch.set_num_threads(cfg.num_threads)
        batch, infos = images_to_batch([image], cfg.image_size, cfg.resize_mode)
        vectors, _ = self._patches(batch)
        distance = nearest_neighbor_distances(vectors[0], self._bank, cfg.chunk_size)
        height, width = self.feature_map_shape
        upsampled = F.interpolate(
            distance.reshape(1, 1, height, width),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        if cfg.smoothing_sigma > 0:
            upsampled = cv2.GaussianBlur(
                upsampled, (0, 0), cfg.smoothing_sigma, borderType=cv2.BORDER_REFLECT
            )
        anomaly_map = map_back(upsampled.astype(np.float32), infos[0])
        return Prediction(score=float(anomaly_map.max()), anomaly_map=anomaly_map)

    # ----- persistence ---------------------------------------------------------------------

    def tensors(self) -> dict[str, NDArray]:
        """The fitted parameters: the (reduced) memory bank."""
        if self._bank is None:
            raise ModelError("The model has not been fitted")
        return {"bank": self._bank.numpy().copy()}

    @classmethod
    def from_tensors(
        cls,
        config: PatchCoreConfig,
        tensors: dict[str, NDArray],
        extractor: FeatureExtractor,
        weights: WeightsInfo,
        feature_map_shape: tuple[int, int],
        reference_aspect_ratio: float | None,
        n_fit_images: int,
        n_candidates: int,
    ) -> PatchCoreModel:
        """Rebuild a fitted model from a stored bank."""
        bank = tensors.get("bank")
        if bank is None or bank.ndim != 2 or bank.shape[0] < 1:
            found = "missing" if bank is None else tuple(bank.shape)
            raise ModelError(f"Tensor 'bank' must be a non-empty (M, D) array, found {found}")
        model = cls(config, extractor, weights)
        model._bank = torch.from_numpy(bank.astype(np.float32))
        model.feature_map_shape = feature_map_shape
        model.reference_aspect_ratio = reference_aspect_ratio
        model.n_fit_images = n_fit_images
        model.n_candidates = n_candidates
        return model
