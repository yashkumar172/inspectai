# Evaluation

How InspectAI measures its models, how thresholds are chosen without touching the test set, how
models are compared, and how to reproduce every number. The measured results themselves are in the
[README](../README.md) (per-model sections and "Comparison report") and in the generated
`comparison.md`.

## 1. Protocol in one paragraph

Each category has its own models. A model is fitted on the `fit` split only; thresholds come from the
held-out normal `calibration_normal` split only; the official `test` split is scored once and is used
for nothing else. Models are compared on identical splits at one evaluation resolution (256x256) with
bootstrap confidence intervals. Everything reported was produced by running the code; nothing is
assumed or copied from a paper.

## 2. Data splits

Dataset: VisA, official one-class split (`split_csv/1cls.csv`), 12 objects. Per object (seed 42):

| Split | Source | Typical size (cashew) |
|---|---|---|
| `fit` | official `train` rows minus the hold-out | 405 normal images |
| `calibration_normal` | random 10% of `train` (at least 10) | 45 normal images |
| `test` | official `test` rows | 50 normal + 100 anomalous images |

Rules that are enforced in code and tested: the three splits are disjoint; they are reproducible for a
fixed seed and differ for another seed; a manifest that lists a test file in `fit` is rejected; the
model is never refitted on `calibration_normal` after the threshold is chosen (that would change the
score scale). Each artifact stores `split.json` (relative file names and a SHA-256 of each sorted
list), and every report records the split hashes, so a reader can check that two models saw the same
data.

Hyperparameters (backbone, layers, image size, coreset ratio, d) are fixed in advance from the
defaults and were **not** tuned on the test split. No sweep was run.

## 3. Threshold selection

| Policy | How the threshold is chosen | Reported on |
|---|---|---|
| `normal_percentile` (default, used for all reported results) | the 0.99 quantile of image scores on `calibration_normal`; the pixel threshold is the 0.99 quantile of pooled anomaly-map values on the same images | the full `test` split |
| `labeled_calibration` (implemented, not used for the reported results) | maximum F1 on half of the test rows (stratified by label, seeded) | the other half, `test_eval`, only |

With about 45 calibration images the 0.99 quantile is close to their maximum, so thresholds are
conservative: very few false alarms, but many subtle defects score below the threshold and are missed.
Thresholds are stored per model and per category. The decision rule is `score > threshold`. An anomaly
score is not a calibrated probability and is never presented as one.

## 4. Metrics

**Image level** (from the model's image score, labels from the split file, never from predictions):

- AUROC and average precision (AP): rank all test images; threshold-free.
- Precision, recall, F1, the confusion matrix, false-positive rate and false-negative rate at the
  stored threshold.
- A 95% percentile bootstrap confidence interval for AUROC (1000 resamples of the test images, fixed
  seed). Every metric is reported with the number of normal and anomalous images.
- A metric that cannot be defined (for example AUROC with only one class) is returned as `None` and
  printed as `undefined`; it is never replaced by a number and is skipped, with a count, in averages.

**Pixel level** (where ground-truth masks exist):

- Maps are resized bilinearly and masks with nearest neighbor to a common 256x256 resolution.
- Pixel AUROC and pixel AP are computed over all pixels of all test images. Normal images have no
  stored mask and count as all-normal, so they contribute valid negative pixels (the same policy as
  the official evaluation code).
- IoU and Dice are computed at the pixel threshold, **pooled** over all test images (intersections
  and unions are summed before dividing), so false-positive pixels on normal images count against
  the score. They are undefined only if neither prediction nor ground truth has a positive pixel.
- Pixel AUROC is high for every deep model mostly because most pixels are background; read it with
  IoU, Dice, and AP.

## 5. Comparison methodology

`scripts/compare_models.py` implements the rules below; it stops with an error rather than comparing
runs that break them.

1. **Identical data and settings.** Runs are compared only if they share the same split file hashes,
   split seed, evaluated split, evaluation resolution, threshold policy and quantile, bootstrap
   settings, sample counts, and the same test images with the same labels.
2. **Metrics from predictions.** Every image-level metric in a report is re-derived from the run's
   saved `predictions.csv` and must equal the harness summary. Pixel metrics need the full maps and are
   taken from the summaries (and labeled as such).
3. **Paired differences.** All models score the same test images, so AUROC differences are given with
   a paired bootstrap interval (the same resampled images for both models). An interval that
   includes 0 means no difference is demonstrated on that test set. This is more informative than
   comparing two separate intervals, which overlap even when the paired difference is clear.
4. **Measured versus not measured.** A configuration with no results for a category is listed as not
   measured and left out; categories with fewer than two measured models are not compared.
5. **Macro averages** are unweighted means over the categories where a model was measured, with the
   number of categories shown. Per-category results always come first.
6. **No external comparison.** Published PatchCore and PaDiM numbers use other protocols (and other
   benchmarks); they are not compared with these results.
7. **Figures.** Example images are chosen without looking at any model output: two normal images at
   random (seed 42) and the small, median, and large defect by ground-truth mask area.

## 6. Performance measurements

Fit time, peak process memory (peak working set on Windows) during fitting and during scoring,
per-image prediction latency (mean, median, maximum; image decoding excluded), images per second,
memory-bank size, thread count, power plan, and library versions are recorded in every `summary.json`
and in the artifact manifest. All benchmarks ran on one machine (AMD Ryzen 5 5500U, 6 cores, 16 GB,
no GPU) plugged in, on the Windows **Balanced** power plan; SPEC.md recommends High performance, so
the times are probably conservative. Provisional targets (`dev` at most 1 s per image, `full` at most
3 s) were confirmed with large margins.

## 7. Deviations from the intended algorithms

The models are inspired by, not equivalent to, the published methods.

**Model B versus PaDiM (Defard et al., 2020):** layers are upsampled with bilinear interpolation (the
paper uses nearest-neighbor-style embedding concatenation); images are resized with the category's
mode (`letterbox`) and never center-cropped, so border defects are not cut off; d = 100 channels for
both backbones (the paper uses a per-backbone value); only means, inverse covariances, and channel
indices are stored, with float32 inverses of float64 covariances.

**Model C versus PatchCore (Roth et al., CVPR 2022):** the 3x3 neighborhood averaging is applied after
the layers are upsampled and concatenated (PatchCore pools each layer at its own resolution); there is
no adaptive pooling of features to a lower dimension, so D is the number of concatenated channels (384
or 1536); the image score is the plain maximum of the smoothed map (PatchCore re-weights it by
neighbor distances); search is exact with k = 1; the random projection is used only to choose the
coreset, and the bank keeps full-dimensional vectors.

**Classical baseline:** a per-pixel-position statistic; it only works when the object is positioned
consistently, and it is a reference, not a competitor.

## 8. Reproducing the results

Prerequisites: the environment in the README, the VisA data and `1cls.csv` under
`INSPECTAI_DATA_ROOT`, and the cached weights (`fetch_weights.py`).

```powershell
python scripts/validate_dataset.py                       # 0 errors expected; checks every split row
python scripts/validate_dataset.py --write-splits         # writes split.json per object (seed 42)

# one category, all five configurations (replace <category>)
python scripts/evaluate_model.py --model classical --category <category>
python scripts/fit_model.py --model gaussian --category <category> --profile dev
python scripts/evaluate_model.py --model gaussian --category <category> --profile dev
python scripts/fit_model.py --model patchcore --category <category> --profile dev
python scripts/evaluate_model.py --model patchcore --category <category> --profile dev
# the `full` profile: add --profile full and a separate --output-dir / --artifact-dir
# (see the README or the "How to reproduce" section of comparison.md)

python scripts/compare_models.py                          # verification, tables, figures
```

What to expect: refitting Models B and C reproduces the stored tensors bit for bit on the same machine
(checked), so metrics are identical and only timings vary by a few percent. A full Model C fit takes
about 11 minutes per category; Model B takes under a minute; the baseline takes seconds.

Without the dataset or weights, `python scripts/run_demo.py` runs the same pipeline on synthetic
images with a fake backbone (about 15 seconds) and checks that every saved prediction equals the
evaluation report. Its metrics are meaningless; it verifies the machinery, not the models.

## 9. What the tests verify

The automated tests (`pytest`) cover the splits (disjointness, reproducibility, tampering), the
threshold policy (it cannot receive test data), metrics on toy inputs with known answers (including
undefined cases), artifact integrity (wrong checksum, shape, NaN, schema version, pickle, stale
configuration), leakage (replacing every test image changes neither the fitted parameters nor the
thresholds), the comparison rules (every refusal path), and the app. Tests marked `slow` or
`needs_dataset` use the real dataset and pretrained weights and are run separately
(`pytest -m "needs_dataset or slow"`).
