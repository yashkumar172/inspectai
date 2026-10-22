# Architecture

InspectAI is a modular Python package (`src/inspectai/`) plus command-line scripts (`scripts/`) and a
Streamlit app (`app.py`). This document explains how data flows through it and how each part works.
Numbers quoted here are defaults taken from `configs/default.yaml`; measured results are in the
[README](../README.md) and [evaluation.md](evaluation.md).

## 1. Data flow

```text
 VisA split file (1cls.csv) + images + masks            (outside the repository)
        |
        v
 data/dataset.py ----- parse and validate rows, load images and masks safely
        |
        v
 data/splits.py ------ fit | calibration_normal | test   (seeded, disjoint, saved as split.json)
        |
        v
 preprocessing/images.py   resize (stretch or letterbox) -> ImageNet normalization
        |
        +--> models/classical.py ----------------------+   per-pixel mean / std z-score
        |                                              |
        +--> models/features.py (frozen CNN) --+       |
                                               |       |
                 models/gaussian.py (Model B) -+-------+--> anomaly map + image score
                 models/patchcore.py (Model C) -+
                                                        |
        +-----------------------------------------------+
        |
        v
 evaluation/runner.py   fit -> score calibration_normal -> thresholds -> score test -> metrics
        |                                   |
        |                                   v
        |                         artifacts/store.py   manifest + safetensors + split + config
        v
 evaluation/report.py   predictions.csv + summary.json (+ example figures)
        |
        v
 evaluation/comparison.py   verify, compare, tables  -->  comparison.json / .csv / .md / figures
        |
        v
 inference/predictor.py + ui/*   load artifacts -> inspect one image -> Streamlit pages
```

Everything on the left of `artifacts/store.py` happens when a model is **fitted**; everything that
reads artifacts (evaluation of saved models, the comparison, the app) never refits and never sees
training images again. The one deliberate exception is the classical baseline, which
`evaluate_model.py` refits in about 8 seconds; its artifact exists for the app.

## 2. Module map

| Package / module | Responsibility |
|---|---|
| `config.py` | Environment-driven paths, `SplitConfig`, `PreprocessConfig`, the `dev` / `full` profiles, per-category resize modes |
| `settings.py` | Loads and validates `configs/default.yaml` (environment references, CPU-only device) |
| `data/dataset.py` | Split-file parsing, `VisADataset`, the `DatasetAdapter` interface, mask rules |
| `data/validation.py` | Row-by-row dataset validation, per-object size and aspect-ratio statistics |
| `data/splits.py` | `fit` / `calibration_normal` / `test` splits, `split.json`, the recorded `max_fit_images` cap |
| `preprocessing/images.py` | Safe image and mask reading, size limits, resizing, mapping maps back |
| `models/base.py` | `Prediction` (score + map) and the `AnomalyModel` interface |
| `models/classical.py` | Model A |
| `models/features.py` | Frozen torchvision backbone, weight loading, feature extraction |
| `models/gaussian.py` | Model B |
| `models/patchcore.py` | Model C, greedy coreset, exact nearest-neighbor search |
| `artifacts/store.py` | Artifact format, integrity checks, stale-artifact detection |
| `artifacts/*_io.py` | Model-specific save / load on top of the store |
| `evaluation/metrics.py` | Image and pixel metrics, bootstrap intervals, paired bootstrap |
| `evaluation/thresholds.py` | Threshold policies |
| `evaluation/runner.py` | The harness: fit, calibrate, evaluate, measure time and memory |
| `evaluation/report.py` | Per-image CSV and JSON summary writers |
| `evaluation/comparison.py` | Cross-model verification, tables, paired differences, macro averages |
| `evaluation/dashboard.py` | Validated result data for the app's dashboard |
| `inference/uploads.py` | Upload validation |
| `inference/predictor.py` | Model registry, loading, single-image inspection, downloads |
| `visualization/` | Heatmaps, overlays, panels, comparison figures |
| `ui/` | The four Streamlit pages and their shared helpers |

Each module has one responsibility and its own tests (`tests/unit`, `tests/integration`).

## 3. Preprocessing and map geometry

Images are RGB `uint8` arrays. Each category has a resize mode chosen from its measured median
aspect ratio: `stretch` (direct resize to a square, no cropping) or `letterbox` (keep the aspect
ratio, pad with black). All 12 VisA objects are non-square (median ratios 1.08 to 1.63), so all use
`letterbox`. The resize returns a `ResizeInfo`; `map_back` uses it to remove the padding of a
letterboxed map and resize it (bilinear) to the original image size, so every model's anomaly map
has exactly the input image's height and width. Inputs are normalized with ImageNet mean and
standard deviation.

## 4. Feature extraction (Models B and C)

`FeatureExtractor` wraps any `torch.nn.Module` (tests inject a small deterministic fake backbone)
with torchvision's `create_feature_extractor`. The backbone is frozen (`eval()`, no gradients) and
used only for inference under `torch.inference_mode()`. It returns the `layer2` and `layer3`
feature maps; `layer3` is upsampled bilinearly to the `layer2` resolution and the two are
concatenated along channels. At 224x224 input this gives a 28x28 grid of 784 positions:

| Profile | Backbone | Channels (layer2 + layer3) |
|---|---|---|
| `dev` | ResNet-18 (ImageNet-1k `IMAGENET1K_V1`) | 128 + 256 = 384 |
| `full` | Wide ResNet-50-2 (`IMAGENET1K_V1`) | 512 + 1024 = 1536 |

Weights come from the torchvision cache (`TORCH_HOME`), are verified against the hash in their file
name, and are loaded with `weights_only=True`. If they are not cached, `WeightsNotFoundError` explains
how to fetch them; nothing is downloaded implicitly and there is no fallback to random weights.

## 5. Data splits and threshold policy

The official one-class split has `train` (normal images only) and `test` (normal and anomalous).
For each category the splits are built once with a fixed seed (42):

| Split | Source | Used for |
|---|---|---|
| `fit` | official `train` minus the hold-out | fitting the model |
| `calibration_normal` | random 10% of `train` (at least 10 images) | selecting thresholds |
| `test` | all official `test` rows | final evaluation only |

The three parts are disjoint (enforced in code and tested), the split is stored as `split.json`
(relative file names plus a SHA-256 of each sorted list) inside every artifact, and the selection is
seeded per object, so adding another category never changes an existing split. An optional
`--max-fit-images N` replaces `fit` by a seeded random subset that is recorded in `split.json`.

The default threshold policy, `normal_percentile`, is the 0.99 quantile of the image scores of
`calibration_normal`. The pixel threshold for IoU and Dice is the 0.99 quantile of the pooled
anomaly-map values of the same images. Neither can see test labels, test scores, or test masks: the
policy function receives only calibration scores. The decision rule is `score > threshold`. A second
policy, `labeled_calibration`, selects a maximum-F1 threshold on half of the test rows and reports
every threshold-dependent metric on the other half. Thresholds are stored per model and category;
there is no global threshold because score scales differ.

## 6. Model A: classical baseline

For every pixel position the model stores the mean and standard deviation (floored at 5 colour
levels) of the resized `fit` images. A new image's map is the root-mean-square z-score over the three
colour channels, smoothed with a Gaussian filter (sigma 4) and mapped back to the original size. It
works only if the object sits at nearly the same place in every image; it is a measurable reference,
not a competitive model.

## 7. Model B: Gaussian feature model (PaDiM-inspired)

At each of the 784 grid positions a multivariate Gaussian is fitted over the `fit` images on a fixed
random subset of d = 100 channels. Fitting is streaming: per-position sums and outer-product sums are
accumulated in float64, so memory does not grow with the number of images. The covariance is
`(S2 - n m mT) / (n - 1) + 0.01 I`, inverted per position, and stored as float32.

The anomaly score at a position is the Mahalanobis distance `sqrt((x - m)T S^-1 (x - m))` of the
test feature to that position's Gaussian. The 28x28 score grid is upsampled bilinearly to the model
input size, smoothed (Gaussian filter, sigma 4), and mapped back to the original image size. Only the
means, inverse covariances, and the selected channel indices are stored (about 32 MB).

## 8. Model C: reference feature memory and nearest-neighbor scoring (PatchCore-inspired)

1. Each position's feature vector is averaged with its 3x3 spatial neighbors (stride 1).
2. All vectors of all `fit` images form the candidate memory bank (for example 405 images x 784 =
   317,520 vectors). The size estimate `images x 784 x D x 4 bytes` is logged first and the fit stops
   with an explanation if it exceeds `max_bank_gb` (default 4).
3. The bank is reduced to 10% by **greedy k-center coreset selection**: vectors are randomly
   projected to 128 dimensions (seeded), a seeded random vector starts the selection, and the vector
   farthest from the already selected ones is added repeatedly. If selection exceeds
   `max_fit_minutes` (default 30) the fit stops with a message suggesting random subsampling; it
   never returns a partial coreset.
4. A test vector's score is its Euclidean distance to the nearest bank vector, computed exactly with
   chunked `torch.cdist` (no approximate index). The 28x28 distance grid becomes the anomaly map
   exactly as in Model B.

The selection dominates the fit time (about 11 minutes per category on the reference CPU); the stored
bank is 49 MB (`dev`) or 195 MB (`full`).

## 9. Anomaly map and image score

All three models return a `Prediction`: a float32 map with the input image's height and width and an
image score equal to the **maximum of the map**. A score is a distance-like number, not a
probability; its scale differs between models and categories, so it is only compared with the
threshold and the normal-score range of the same model and category. In the app and in the comparison
figures, heatmaps use a fixed colour scale anchored at the stored pixel threshold (blue is 0, the
middle is the threshold, red is twice the threshold), so a normal image is not stretched to look
alarming. (The per-model example panels written by `evaluate_model.py` use one scale from 0 to the
largest score of that run instead.)

## 10. Model persistence

A fitted model is the directory `<artifact_root>/<model>/<category>/`:

| File | Content |
|---|---|
| `manifest.json` | schema version, model type, category, backbone and weights identifier with SHA-256, layers, image size, feature dimension, tensor shapes and dtypes, resize mode, reference aspect ratio, seed, thresholds and calibration scores, fit time and peak memory, library versions, creation time, and the SHA-256 of each data file |
| `tensors.safetensors` | the parameters (Model A: mean and std; B: means, inverse covariances, channel indices; C: the reduced bank) |
| `split.json` | the split used |
| `config.snapshot.yaml` | the exact configuration, including a `runtime` section of resource limits |

Nothing is stored with `pickle`. On load the store verifies the schema version, the checksums, tensor
names, shapes, dtypes and the absence of NaN or infinity, that `split.json` has no overlap between
`fit` and `test`, and (optionally) that the configuration snapshot equals the requested one; a
snapshot that differs in anything except `runtime` makes the artifact stale. Any problem raises
`ArtifactError`. The manifest is written last, so an interrupted save cannot leave a loadable
artifact. Saved models reload bit-identically (tested for all three).

## 11. Reports and the comparison

`scripts/evaluate_model.py` loads (or, for Model A, refits) a model, scores the test split once, and
writes `predictions.csv` (one row per image) and `summary.json` (settings, split hashes, thresholds,
metrics with sample counts and bootstrap intervals, timings, memory, environment).
`scripts/compare_models.py` then refuses to compare runs that did not use identical splits and
settings, re-derives every image-level metric from the saved `predictions.csv`, and writes the
comparison tables, paired differences, macro averages, and example figures.

## 12. UI integration

The Streamlit pages in `ui/` contain no model logic. `inference/predictor.py` finds the fitted models
from the artifact folders, loads one through the artifact store (all checks above), and runs it on an
image; `inference/uploads.py` validates uploads (size, type, real format, dimensions, aspect ratio)
before any model code runs. Loaded models and validated result data are cached with
`st.cache_resource`, so a rerun never refits or reloads. The decision always uses the threshold stored
in the artifact. Failures are mapped to fixed, path-free messages (`Model unavailable` with the
command that fixes it); technical details go to the server log only. The dashboard reads the same
report files as the comparison and shows a category only after its results pass the same
verification.

## 13. Error handling and exit codes

Every error the project raises on purpose derives from `InspectAIError`. The scripts map them to
exit codes:

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | configuration, data, image, split, model, or evaluation error (including a failed comparability check) |
| 3 | artifact error (missing, corrupted, or stale artifact) |
| 4 | pretrained weights unavailable |
| 1 | any other failure (logged with its traceback) |

## 14. Resource handling and determinism

Inference runs on the CPU with `torch.set_num_threads` set explicitly (default 6, the physical core
count). Fitting is streaming or bounded: Model B uses constant memory, Model C estimates and caps its
memory bank, and peak process memory is measured and recorded. Seeds are fixed (42) for the splits,
the channel subset, the coreset projection and start, and the bootstrap. Refitting on the same
machine with the same thread count and batch size reproduces the stored tensors bit for bit
(verified for Models B and C); different thread counts or batch sizes can change floating-point
summation order and are recorded in the configuration.
