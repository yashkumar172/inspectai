# Benchmarks: coreset speed, localization variants, latency, and INT8

Everything here was measured on the reference laptop (AMD Ryzen 5 5500U, 6 cores / 12 threads, 16 GB,
no GPU, Windows **Balanced** power plan, plugged in, PyTorch 2.14.1+cpu, 6 threads), one run per
configuration, seed 42. Nothing was tuned on the test split: the variants below were fixed in advance
and every one that was run is reported, including those that did not help. Commands to reproduce are
at the end.

## 1. Greedy coreset selection (Objective 1)

**What was already true.** The greedy k-center loop already kept a running minimum-distance vector
`D[i] = min over selected c of dist(x_i, c)` and folded in each new center with one pass, instead of
recomputing all-pairs distances. It is memory-bandwidth bound: each step streams the whole 128-dimensional
projected matrix (317,520 x 128 x 4 bytes = 162 MB) for about 19 ms of work. Measured per step:
unchunked 19.4 ms; chunked into 4,096 / 8,192 / 16,384 / 32,768 candidates 22.6 / 22.1 / 21.4 / 20.5 ms.
**Chunking does not help** (it only adds per-chunk overhead), so it was not adopted. The random projection
costs 0.19 s in total, whether done in one matrix product or in 32,768-row blocks writing into a
preallocated buffer (`torch.matmul(..., out=)`, bit-identical), so there was nothing to gain there either;
the blocked form is kept because it avoids a temporary of the full projected matrix.

**What was changed.** `index_backend: tiled` prunes the update with the triangle inequality: each vector
remembers its owning center `o`, and a new center `c` can only be closer to `x` than `o` is if
`d(c, o) < 2 d(x, o)`. Only vectors passing that test are re-measured (4.7% to 7.8% of them in the
measurements below), and the test needs `c`'s distances to the selected centers (`k x 128`), not to all
`N` vectors. The selection is exactly the same.

| Measurement | `exact` | `tiled` | Speedup |
|---|---|---|---|
| Selecting 31,752 of 317,520 vectors (cashew `dev`, 384 dims), standalone | 662 s | 189 s | 3.5x |
| Same, 15,680 of 156,800 vectors (half the images) | 154 s | 47 s | 3.3x |
| Same, 3,136 of 31,360 vectors | 6.7 s | 2.9 s | 2.3x |
| Coreset step inside real `fit_model.py` runs, all seven categories (table below) | 596 to 635 s | 117 to 252 s | 2.5x to 5.1x |
| Whole `fit_model.py` run, all seven categories (table below) | 609 to 649 s | 133 to 272 s | 2.4x to 4.6x |
| Peak working set during selection | 1.07 GB | 1.05 GB | unchanged |

The speedup grows with the number of vectors because the scan's cost per step is proportional to N, while
the pruned step's cost is dominated by the few vectors that still need re-measuring. The standalone
cashew benchmark (3.5x) and the real fits (2.5x to 5.1x, table below) were measured separately, and the
standalone run overlapped with light editing work. A per-step profile of the pruned loop (N = 78,400) is spread out
(candidate update 46%, center distances 19%, argmax 18%, gather 11%, nonzero 6%), so no single further
hot spot was found.

**All seven categories (`dev`, pure `tiled` runs, machine otherwise idle).** Each category was fitted and
evaluated twice, once with the default `exact` backend (the numbers in the README) and once with
`--index-backend tiled` (`patchcore-tiled/<category>` in the reports folder). Candle, pcb1 and pcb2 used
the 400-image fit subset. In all seven the stored `tensors.safetensors` is byte-identical between the two
backends and the image AUROC is identical to four decimals.

| Category | Coreset exact | Coreset tiled | Speedup | Fit exact | Fit tiled | Speedup | Latency exact | Latency tiled | Image AUROC (both) |
|---|---|---|---|---|---|---|---|---|---|
| cashew | 635 s | 204 s | 3.1x | 649 s | 225 s | 2.9x | 205 ms | 169 ms | 0.9546 |
| pipe_fryum | 621 s | 160 s | 3.9x | 635 s | 180 s | 3.5x | 196 ms | 156 ms | 0.9824 |
| chewinggum | 625 s | 252 s | 2.5x | 639 s | 272 s | 2.4x | 199 ms | 164 ms | 0.9684 |
| fryum | 630 s | 124 s | 5.1x | 644 s | 144 s | 4.5x | 208 ms | 136 ms | 0.8642 |
| candle | 596 s | 117 s | 5.1x | 609 s | 133 s | 4.6x | 198 ms | 140 ms | 0.9134 |
| pcb1 | 596 s | 120 s | 5.0x | 610 s | 136 s | 4.5x | 198 ms | 138 ms | 0.8833 |
| pcb2 | 596 s | 129 s | 4.6x | 609 s | 145 s | 4.2x | 198 ms | 138 ms | 0.8968 |

Latency is the mean of `model.predict` over the 150 or 200 test images during evaluation (decoding
excluded). So on this machine a Model C `dev` fit takes **2.2 to 4.5 minutes with `tiled` against about
10 to 11 minutes with `exact`** (2.4x to 4.6x), the coreset step is 2.5x to 5.1x faster, and prediction
takes 136 to 169 ms instead of 196 to 208 ms. The speedup depends on the category (chewinggum gains
least, fryum and the three added categories most) because how many vectors the pruning can skip depends
on how the features cluster. The first category measured (cashew) was also run once while other work was
using the CPU (coreset 219 s); that run was discarded and repeated on an idle machine.

**Identical results.** The selected indices are identical at full scale (cashew, 31,752 of them), and the
stored `tensors.safetensors` of the `tiled` fits are byte for byte equal to the `exact` ones for all
seven categories (compared by SHA-256). A unit test compares both backends with a brute-force greedy
implementation that shares no update logic with them.

**Nearest-neighbor search.** The `tiled` kernel computes `-2 q.b + |b|^2` per bank tile with one `addmm`,
adds `|q|^2` once, and agrees with `torch.cdist` to float32 rounding (maximum absolute error 3e-6 on
scores around 2, relative 3e-6). It is **not** bit-identical, which is why `exact` stays the default.
Per-image latency of Model C (cashew `dev`, section 3): 209 ms to 142 ms median (-32%), image AUROC
identical to four decimals.

**HNSW was not added.** An approximate index would add a dependency, make scores depend on index
construction, and give up the exact-search guarantee the project documents. The exact blocked kernel
above is the dependency-free alternative the task allowed.

## 2. Localization refinement (Objective 2)

Goal: raise pixel IoU / Dice, which sit at 0.02 to 0.10 on these three categories. Two changes were tried,
each as a separate, pre-declared variant on `candle`, `pcb1` and `pcb2`, with the same seed-42 splits,
the same 400-image fit subset, the same 256x256 evaluation resolution and the same threshold policy as the
baselines. Thresholds were recalibrated on `calibration_normal` with each variant's own filter. The guided
filter used its defaults (radius 8, eps 0.01) and was not tuned. Every variant was refitted from scratch
(coreset backend `tiled`; the guided-only banks are byte-identical to the baseline banks).

- **Guided filter** (`--refinement-filter guided`): replaces the sigma-4 Gaussian by an edge-aware filter
  guided by the luminance of the preprocessed input.
- **Multiscale** (`--multiscale`): adds `layer1` features (64 channels), average-pooled to the `layer2`
  28x28 grid, so the patch count is unchanged and D grows 384 to 448.

Pixel IoU / Dice at the calibrated pixel threshold (change against the baseline in parentheses):

| Variant | candle | pcb1 | pcb2 | Mean IoU | Mean Dice | Mean image AUROC |
|---|---|---|---|---|---|---|
| C baseline | 0.023 / 0.046 | 0.096 / 0.175 | 0.034 / 0.067 | 0.051 | 0.096 | 0.898 |
| C guided | 0.022 / 0.042 | 0.090 / 0.165 | 0.032 / 0.062 | 0.048 | 0.090 | 0.901 |
| C + layer1 | 0.022 / 0.044 | 0.097 / 0.177 | 0.035 / 0.068 | 0.051 | 0.096 | 0.911 |
| C + layer1 + guided | 0.021 / 0.041 | 0.092 / 0.169 | 0.032 / 0.062 | 0.048 | 0.091 | 0.913 |
| B baseline | 0.024 / 0.047 | 0.089 / 0.163 | 0.053 / 0.100 | 0.055 | 0.104 | 0.806 |
| B guided | 0.024 / 0.046 | 0.083 / 0.153 | 0.052 / 0.099 | 0.053 | 0.099 | 0.829 |

**Result: the goal was not met.** No variant improved pixel IoU or Dice. The guided filter lowered them
or left them unchanged in every category for both models (mean IoU 0.051 to 0.048 for Model C, 0.055 to
0.053 for Model B). Adding `layer1` left them unchanged (differences of at most 0.001 per category). The likely reason is
that the limiting factor is the 28x28 score grid and the conservative pixel threshold (the 99th percentile
of normal pixels), not the smoothing: IoU is pooled over all pixels of all 200 test images, and these
defects are small and low-contrast, so a sharper edge around a weak response does not move the pooled
overlap. This is an explanation consistent with the numbers, not something these runs isolate.

What did change is image-level ranking, which was not the goal: mean image AUROC moved 0.898 to 0.911
(`layer1`) and 0.913 (`layer1` + guided) for Model C, and 0.806 to 0.829 (guided) for Model B, mostly
from `pcb2` (+0.029 and +0.035 for Model C; +0.045 for Model B). The 95% bootstrap intervals of the
baseline and the variants overlap in every category, and no paired interval was computed, so **these
AUROC gains are not demonstrated to be real**; one seed and one 200-image test set cannot separate them
from noise. Both options stay off by default. Pixel AUROC changed by at most +0.005 and pixel AP
between -0.003 and +0.063 (the largest: Model B guided, pcb1); IoU / Dice are the metrics the objective
named.

Costs: `layer1` makes the unreduced bank 17% larger (peak fit memory 1.24 to 1.30 GB); the guided filter
adds one set of box filters per prediction (not separately timed).

## 3. Latency, memory, and INT8 (Objective 3)

`scripts/benchmark_latency.py`: 100 timed single-image `predict` calls after 5 warm-up calls, images
pre-decoded (decoding excluded), one fresh process per row so each peak working set is independent. Image
AUROC is from scoring the whole test split once with that variant (cashew: 50 normal, 100 anomalous).
With 100 calls, p99 is close to the maximum.

| Model | Variant | p50 ms | p95 ms | p99 ms | Images/s | Peak working set | Image AUROC | vs fp32 |
|---|---|---|---|---|---|---|---|---|
| B Gaussian dev | fp32 | 36.8 | 42.8 | 56.2 | 26.80 | 1.14 GB | 0.9164 | - |
| B Gaussian dev | dynamic INT8 | 37.5 | 45.0 | 52.5 | 26.24 | 1.25 GB | 0.9164 | +0.0000 |
| B Gaussian dev | static INT8 | 31.7 | 36.0 | 38.6 | 31.85 | 1.56 GB | 0.9122 | -0.0042 |
| C PatchCore dev | fp32 (`exact`) | 209.0 | 229.0 | 255.0 | 4.75 | 1.27 GB | 0.9546 | - |
| C PatchCore dev | fp32, `tiled` search | 142.3 | 157.8 | 217.4 | 6.86 | 1.27 GB | 0.9546 | +0.0000 |
| C PatchCore dev | dynamic INT8 | 216.1 | 240.0 | 259.9 | 4.54 | 1.38 GB | 0.9546 | +0.0000 |
| C PatchCore dev | static INT8 | 220.8 | 238.2 | 274.1 | 4.57 | 1.65 GB | 0.9062 | -0.0484 |

- **Dynamic INT8 is a no-op here, by construction.** `quantize_dynamic` only has kernels for `nn.Linear`
  and recurrent layers. The report counts what it converted: 1 module of 20 convolutions and 1 linear
  layer, and the one converted module is the classification head, which the feature extractor never runs
  (`covers_feature_path: false`). Latency and AUROC are unchanged within noise, as expected; the small
  differences in the table are run-to-run variation.
- **Static INT8** (post-training, calibrated on 64 normal fit images, ResNet-18 only, via the deprecated
  `torch.ao.quantization` eager API) does cover the convolutions. For Model B it is 14% faster at the
  median (31.7 vs 36.8 ms) for -0.004 AUROC. For Model C it is **not faster** (220.8 vs 209.0 ms) because
  its time is spent in nearest-neighbor search, not in the backbone, and it **costs 0.048 AUROC** (0.9546
  to 0.9062): the bank holds float32 features while the queries are now INT8-approximated. It is not
  recommended for Model C. Static INT8 also raised the peak working set (a second backbone copy).
- The fitted artifacts are never modified; INT8 only swaps the feature extractor of the loaded model, and
  a test checks the stored tensors are unchanged.
- PyTorch's own deprecation notice says `torch.ao.quantization` is moving to `torchao`; the static path
  will need porting when it is removed. Only the `onednn` engine exists in this CPU build.
- One category (cashew), one machine, one run per row.

## 4. Reproducing

```powershell
# coreset speed: fit the same model with each backend and compare tensors.safetensors
python scripts/fit_model.py --model patchcore --category candle --profile dev --max-fit-images 400
python scripts/fit_model.py --model patchcore --category candle --profile dev --max-fit-images 400 `
    --index-backend tiled --output-dir $env:INSPECTAI_ARTIFACT_DIR\patchcore-tiled\candle

# localization variants (use a separate --output-dir / --artifact-dir for each)
python scripts/fit_model.py --model patchcore --category pcb1 --profile dev --max-fit-images 400 `
    --index-backend tiled --multiscale --refinement-filter guided `
    --output-dir $env:INSPECTAI_ARTIFACT_DIR\patchcore-msguided\pcb1
python scripts/evaluate_model.py --model patchcore --category pcb1 --profile dev `
    --index-backend tiled --multiscale --refinement-filter guided `
    --artifact-dir $env:INSPECTAI_ARTIFACT_DIR\patchcore-msguided\pcb1 `
    --output-dir $env:INSPECTAI_REPORT_DIR\patchcore-msguided\pcb1

# latency and INT8 (writes <INSPECTAI_REPORT_DIR>\benchmarks\latency_cashew.json and .md)
python scripts/benchmark_latency.py --category cashew --models gaussian patchcore `
    --variants fp32 tiled dynamic static
```
