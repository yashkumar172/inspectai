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

Index backends (`PatchCoreConfig.index_backend`, a resource setting that does not define the
fitted model): `exact` is the original full scan per selection step and `torch.cdist` search.
`tiled` selects the same coreset with a triangle-inequality prune (see `greedy_coreset`) and
searches with explicit query x bank tiles. Both are exact; there is no approximate index.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from inspectai.config import RESIZE_MODES, profile_settings, resize_mode_for
from inspectai.errors import ModelError
from inspectai.models.base import Prediction
from inspectai.models.features import (
    Embedder,
    FeatureExtractor,
    WeightsInfo,
    images_to_batch,
)
from inspectai.models.refine import (
    DEFAULT_GUIDED_EPS,
    DEFAULT_GUIDED_RADIUS,
    DEFAULT_REFINEMENT,
    luminance_guide,
    refine_map,
    validate_refinement,
)
from inspectai.preprocessing.images import ImageArray, map_back

logger = logging.getLogger(__name__)

_BYTES_PER_FLOAT32 = 4
_DEADLINE_CHECK_EVERY = 64
_PROJECTION_BLOCK = 32768
_PRUNE_SLACK = 1e-3  # relative safety margin of the triangle-inequality test (see _greedy_pruned)
INDEX_BACKENDS = ("exact", "tiled")
DEFAULT_INDEX_BACKEND = "exact"


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
    index_backend: str = DEFAULT_INDEX_BACKEND
    refinement_filter: str = DEFAULT_REFINEMENT
    guided_radius: int = DEFAULT_GUIDED_RADIUS
    guided_eps: float = DEFAULT_GUIDED_EPS
    grid_layer: str | None = None

    def __post_init__(self) -> None:
        validate_refinement(self.refinement_filter, self.guided_radius, self.guided_eps)
        if self.grid_layer is not None and self.grid_layer not in self.layers:
            raise ModelError(f"grid_layer {self.grid_layer!r} is not in layers {self.layers}")
        if self.index_backend not in INDEX_BACKENDS:
            raise ModelError(
                f"index_backend must be one of {INDEX_BACKENDS}, got {self.index_backend!r}"
            )
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
        data: dict[str, object] = {
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
        # Added settings appear only when they differ from the defaults, so the configuration
        # snapshot of a default model is unchanged and earlier artifacts are not stale.
        if self.grid_layer is not None:
            data["grid_layer"] = self.grid_layer
        if self.refinement_filter != DEFAULT_REFINEMENT:
            data["refinement_filter"] = self.refinement_filter
            data["guided_radius"] = self.guided_radius
            data["guided_eps"] = self.guided_eps
        return data

    def runtime_dict(self) -> dict[str, object]:
        runtime: dict[str, object] = {
            "chunk_size": self.chunk_size,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "max_bank_gb": self.max_bank_gb,
            "max_fit_minutes": self.max_fit_minutes,
        }
        if self.index_backend != DEFAULT_INDEX_BACKEND:
            runtime["index_backend"] = self.index_backend  # omitted by default: old snapshots equal
        return runtime

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
    multiscale: bool = False,
    **limits: float | str,
) -> PatchCoreConfig:
    """The configuration of a category under a profile (`dev` or `full`).

    `multiscale` adds `layer1`, average-pooled to the `layer2` grid (see FeatureExtractor).
    """
    settings = profile_settings(profile)
    layers = tuple(settings["layers"])  # type: ignore[arg-type]
    if multiscale:
        layers, limits = ("layer1", *layers), {**limits, "grid_layer": layers[0]}
    return PatchCoreConfig(
        backbone=str(settings["backbone"]),
        layers=layers,
        image_size=int(settings["image_size"]),  # type: ignore[call-overload]
        resize_mode=resize_mode_for(category),
        coreset_ratio=coreset_ratio,
        seed=seed,
        **limits,  # type: ignore[arg-type]
    )


def estimate_bank_gb(n_images: int, patches_per_image: int, dim: int) -> float:
    """Size of the unreduced memory bank in GiB (SPEC.md Section 5.5)."""
    return n_images * patches_per_image * dim * _BYTES_PER_FLOAT32 / 1024**3


def _project(
    features: torch.Tensor, projection_dim: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random projection (Johnson-Lindenstrauss) of (N, D) features, one BLAS call per block.

    Returns the projected vectors (N, P) and their squared norms (N,). The projection matrix is
    drawn first, so the result does not depend on the block size.
    """
    n_points, dim = features.shape
    projection = torch.randn(dim, projection_dim, generator=generator) / projection_dim**0.5
    projected = torch.empty(n_points, projection_dim)
    for start in range(0, n_points, _PROJECTION_BLOCK):
        stop = start + _PROJECTION_BLOCK
        torch.matmul(features[start:stop], projection, out=projected[start:stop])
    return projected, (projected * projected).sum(dim=1)


def _check_deadline(count: int, n_select: int, begin: float, max_seconds: float | None) -> None:
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


def greedy_coreset(
    features: torch.Tensor,
    n_select: int,
    projection_dim: int = 128,
    seed: int = 42,
    max_seconds: float | None = None,
    backend: str = DEFAULT_INDEX_BACKEND,
) -> torch.Tensor:
    """Indices of a greedy k-center coreset of `features` (N, D).

    The features are randomly projected to `projection_dim` dimensions (seeded). A seeded random
    point starts the selection; then the point farthest from all selected points is added
    repeatedly. Both backends keep a running minimum-distance vector `nearest[i]` (squared
    distance of point i to its closest selected point) and fold in each new center with one
    distance computation per point instead of recomputing distances to all selected points.

    `exact` scans every point at every step (memory-bandwidth bound: it reads the whole projected
    matrix each time). `tiled` skips points that provably cannot be closer to the new center
    (triangle inequality), which selects the same points far faster; see `_greedy_pruned`.
    Raises ModelError if `max_seconds` elapses first, never returning a partial selection.
    """
    if backend not in INDEX_BACKENDS:
        raise ModelError(f"backend must be one of {INDEX_BACKENDS}, got {backend!r}")
    n_points = features.shape[0]
    if not 1 <= n_select <= n_points:
        raise ModelError(f"n_select must be between 1 and {n_points}, got {n_select}")
    if n_select == n_points:
        return torch.arange(n_points)
    generator = torch.Generator().manual_seed(seed)
    projected, squared = _project(features, projection_dim, generator)
    start_index = int(torch.randint(n_points, (1,), generator=generator))
    if backend == "tiled":
        return _greedy_pruned(projected, squared, n_select, start_index, max_seconds)
    return _greedy_scan(projected, squared, n_select, start_index, max_seconds)


def _greedy_scan(
    projected: torch.Tensor,
    squared: torch.Tensor,
    n_select: int,
    start_index: int,
    max_seconds: float | None,
) -> torch.Tensor:
    """The original selection loop: one full pass over all points per step."""
    selected = torch.empty(n_select, dtype=torch.int64)
    selected[0] = start_index
    nearest = squared - 2.0 * torch.mv(projected, projected[start_index]) + squared[start_index]
    nearest[start_index] = -1.0  # selected points are never picked again
    begin = time.perf_counter()
    for count in range(1, n_select):
        _check_deadline(count, n_select, begin, max_seconds)
        chosen = int(torch.argmax(nearest))
        selected[count] = chosen
        distance = squared - 2.0 * torch.mv(projected, projected[chosen]) + squared[chosen]
        torch.minimum(nearest, distance, out=nearest)
        nearest[chosen] = -1.0
    return selected


def _greedy_pruned(
    projected: torch.Tensor,
    squared: torch.Tensor,
    n_select: int,
    start_index: int,
    max_seconds: float | None,
) -> torch.Tensor:
    """Greedy k-center with triangle-inequality pruning (Elkan-style), same result as the scan.

    Every point remembers the selected center it is closest to (`owner`). A new center c can only
    be closer to point x than x's owner o if d(c, o) < 2 d(x, o): otherwise
    d(c, x) >= d(c, o) - d(x, o) >= d(x, o). Only points passing that test get a fresh distance
    to c, a few percent of all points once the selection covers the data. The center-to-center
    distances cost k x P per step against N x P for the scan, and pruned points never touch the
    projected matrix. `_PRUNE_SLACK` widens the test so rounding cannot prune a point the scan
    would update.
    """
    n_points, dim = projected.shape
    selected = torch.empty(n_select, dtype=torch.int64)
    centers = torch.empty(n_select, dim)  # projected selected points, in selection order
    center_sq = torch.empty(n_select)
    selected[0] = start_index
    centers[0], center_sq[0] = projected[start_index], squared[start_index]
    nearest = squared - 2.0 * torch.mv(projected, projected[start_index]) + squared[start_index]
    nearest[start_index] = -1.0
    owner = torch.zeros(n_points, dtype=torch.int64)
    begin = time.perf_counter()
    for count in range(1, n_select):
        _check_deadline(count, n_select, begin, max_seconds)
        chosen = int(torch.argmax(nearest))
        point, point_sq = projected[chosen], squared[chosen]
        selected[count] = chosen
        centers[count], center_sq[count] = point, point_sq
        # squared distance from the new center to every earlier center
        to_centers = center_sq[:count] - 2.0 * torch.mv(centers[:count], point) + point_sq
        keep = to_centers[owner] < 4.0 * nearest * (1.0 + _PRUNE_SLACK) + 1e-4 * point_sq
        candidates = keep.nonzero().squeeze(1)
        distance = squared[candidates] - 2.0 * torch.mv(projected[candidates], point) + point_sq
        closer = distance < nearest[candidates]
        updated = candidates[closer]
        nearest[updated] = distance[closer]
        owner[updated] = count
        nearest[chosen] = -1.0
    return selected


def nearest_neighbor_distances(
    queries: torch.Tensor,
    bank: torch.Tensor,
    chunk_size: int = 16384,
    backend: str = DEFAULT_INDEX_BACKEND,
) -> torch.Tensor:
    """Euclidean distance from each query (Q, D) to its nearest bank vector (M, D).

    Exact search. `exact` uses chunked `torch.cdist`: only a (Q, chunk_size) distance block exists
    at a time. `tiled` computes the same distances from squared norms and one `addmm` per bank
    tile; its results agree with `exact` to float32 rounding (about 1e-6 relative), not bit for
    bit.
    """
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ModelError(f"Incompatible shapes {tuple(queries.shape)} and {tuple(bank.shape)}")
    if chunk_size < 1:
        raise ModelError("chunk_size must be at least 1")
    if backend not in INDEX_BACKENDS:
        raise ModelError(f"backend must be one of {INDEX_BACKENDS}, got {backend!r}")
    if backend == "tiled":
        return _tiled_nearest(queries, bank, chunk_size)
    best = torch.full((queries.shape[0],), float("inf"))
    for start in range(0, bank.shape[0], chunk_size):
        block = torch.cdist(queries, bank[start : start + chunk_size])
        best = torch.minimum(best, block.min(dim=1).values)
    return best


def _tiled_nearest(queries: torch.Tensor, bank: torch.Tensor, tile: int) -> torch.Tensor:
    """Blocked nearest-neighbor search: min over bank tiles of |q|^2 + |b|^2 - 2 q.b."""
    query_sq = (queries * queries).sum(dim=1)
    bank_sq = (bank * bank).sum(dim=1)
    best = torch.full((queries.shape[0],), float("inf"))
    for start in range(0, bank.shape[0], tile):
        stop = start + tile
        # -2 q.b + |b|^2 in one addmm; |q|^2 is constant per query and added once at the end
        block = torch.addmm(bank_sq[start:stop], queries, bank[start:stop].T, alpha=-2.0)
        best = torch.minimum(best, block.min(dim=1).values)
    return (best + query_sq).clamp_(min=0.0).sqrt_()


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

    def use_extractor(self, extractor: Embedder) -> None:
        """Replace the feature extractor, e.g. by an INT8 one (the fitted tensors are untouched)."""
        self._extractor = extractor  # type: ignore[assignment]

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
            candidates,
            n_select,
            cfg.projection_dim,
            cfg.seed,
            max_seconds=remaining,
            backend=cfg.index_backend,
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
        distance = nearest_neighbor_distances(
            vectors[0], self._bank, cfg.chunk_size, cfg.index_backend
        )
        height, width = self.feature_map_shape
        upsampled = F.interpolate(
            distance.reshape(1, 1, height, width),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        guide = luminance_guide(batch[0]) if cfg.refinement_filter == "guided" else None
        upsampled = refine_map(
            upsampled,
            guide,
            cfg.refinement_filter,
            cfg.smoothing_sigma,
            cfg.guided_radius,
            cfg.guided_eps,
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
