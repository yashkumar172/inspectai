# Model comparison

Generated 2026-10-23T01:07:25+00:00 UTC. Dataset: VisA (official one-class split, 1cls.csv).
Every number below was measured by running the model; nothing is planned or assumed. Split: `test`, evaluation resolution 256x256, threshold policy `normal_percentile` (q = 0.99), split seed 42, 1000 bootstrap resamples (seed 42).

## cashew

50 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.701 (0.603-0.794) | 0.818 | 0.809 | 0.21 / 0.35 | 1.000 | 0.180 | 0.000 | 0.305 | 7.8 s | 14 ms |
| B Gaussian dev | 0.916 (0.871-0.955) | 0.962 | 0.987 | 0.19 / 0.32 | 1.000 | 0.540 | 0.000 | 0.701 | 17.4 s | 39 ms |
| B Gaussian full | 0.876 (0.818-0.928) | 0.937 | 0.990 | 0.26 / 0.41 | 1.000 | 0.140 | 0.000 | 0.246 | 50.9 s | 122 ms |
| C PatchCore-inspired dev | 0.955 (0.922-0.981) | 0.978 | 0.990 | 0.20 / 0.34 | 1.000 | 0.580 | 0.000 | 0.734 | 649.4 s | 205 ms |
| C PatchCore-inspired full | 0.968 (0.944-0.987) | 0.984 | 0.992 | 0.19 / 0.33 | 1.000 | 0.640 | 0.000 | 0.780 | 681.8 s | 592 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | +0.215 | +0.131 to +0.308 | yes |
| C PatchCore-inspired dev | A classical | +0.254 | +0.158 to +0.352 | yes |
| C PatchCore-inspired dev | B Gaussian dev | +0.038 | +0.001 to +0.079 | yes |
| B Gaussian full | A classical | +0.175 | +0.083 to +0.267 | yes |
| C PatchCore-inspired full | A classical | +0.267 | +0.170 to +0.365 | yes |
| C PatchCore-inspired full | B Gaussian full | +0.092 | +0.046 to +0.147 | yes |

## pipe_fryum

50 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.674 (0.587-0.759) | 0.818 | 0.963 | 0.24 / 0.39 | 1.000 | 0.080 | 0.000 | 0.148 | 7.2 s | 14 ms |
| B Gaussian dev | 0.933 (0.885-0.967) | 0.963 | 0.992 | 0.20 / 0.34 | 0.965 | 0.550 | 0.040 | 0.701 | 16.2 s | 37 ms |
| B Gaussian full | 0.932 (0.882-0.973) | 0.960 | 0.994 | 0.24 / 0.39 | 0.934 | 0.570 | 0.080 | 0.708 | 52.4 s | 130 ms |
| C PatchCore-inspired dev | 0.982 (0.964-0.995) | 0.992 | 0.992 | 0.18 / 0.31 | 1.000 | 0.600 | 0.000 | 0.750 | 635.0 s | 196 ms |
| C PatchCore-inspired full | 0.990 (0.977-0.998) | 0.995 | 0.994 | 0.19 / 0.33 | 1.000 | 0.520 | 0.000 | 0.684 | 686.9 s | 650 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | +0.259 | +0.172 to +0.345 | yes |
| C PatchCore-inspired dev | A classical | +0.308 | +0.226 to +0.391 | yes |
| C PatchCore-inspired dev | B Gaussian dev | +0.050 | +0.015 to +0.096 | yes |
| B Gaussian full | A classical | +0.258 | +0.169 to +0.350 | yes |
| C PatchCore-inspired full | A classical | +0.316 | +0.232 to +0.402 | yes |
| C PatchCore-inspired full | B Gaussian full | +0.058 | +0.017 to +0.105 | yes |

## chewinggum

50 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.790 (0.706-0.865) | 0.873 | 0.810 | 0.07 / 0.13 | 1.000 | 0.060 | 0.000 | 0.113 | 7.6 s | 14 ms |
| B Gaussian dev | 0.974 (0.947-0.994) | 0.989 | 0.985 | 0.10 / 0.18 | 1.000 | 0.780 | 0.000 | 0.876 | 17.1 s | 34 ms |
| B Gaussian full | 0.973 (0.948-0.992) | 0.988 | 0.983 | 0.11 / 0.20 | 1.000 | 0.690 | 0.000 | 0.817 | 44.9 s | 106 ms |
| C PatchCore-inspired dev | 0.968 (0.940-0.991) | 0.987 | 0.984 | 0.08 / 0.14 | 0.989 | 0.890 | 0.020 | 0.937 | 638.7 s | 199 ms |
| C PatchCore-inspired full | 0.967 (0.938-0.991) | 0.987 | 0.982 | 0.08 / 0.14 | 0.989 | 0.900 | 0.020 | 0.942 | 716.0 s | 578 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | +0.184 | +0.117 to +0.260 | yes |
| C PatchCore-inspired dev | A classical | +0.178 | +0.106 to +0.260 | yes |
| C PatchCore-inspired dev | B Gaussian dev | -0.006 | -0.027 to +0.012 | no |
| B Gaussian full | A classical | +0.183 | +0.116 to +0.261 | yes |
| C PatchCore-inspired full | A classical | +0.177 | +0.102 to +0.261 | yes |
| C PatchCore-inspired full | B Gaussian full | -0.006 | -0.033 to +0.020 | no |

## fryum

50 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.785 (0.700-0.856) | 0.868 | 0.621 | 0.05 / 0.10 | 0.875 | 0.210 | 0.060 | 0.339 | 7.4 s | 13 ms |
| B Gaussian dev | 0.848 (0.778-0.908) | 0.911 | 0.967 | 0.27 / 0.43 | 1.000 | 0.170 | 0.000 | 0.291 | 15.2 s | 35 ms |
| B Gaussian full | 0.866 (0.803-0.922) | 0.921 | 0.965 | 0.26 / 0.42 | 0.950 | 0.380 | 0.040 | 0.543 | 46.1 s | 110 ms |
| C PatchCore-inspired dev | 0.864 (0.799-0.915) | 0.939 | 0.970 | 0.30 / 0.46 | 1.000 | 0.540 | 0.000 | 0.701 | 644.1 s | 208 ms |
| C PatchCore-inspired full | 0.914 (0.864-0.952) | 0.959 | 0.967 | 0.25 / 0.40 | 1.000 | 0.480 | 0.000 | 0.649 | 680.2 s | 577 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | +0.062 | -0.010 to +0.134 | no |
| C PatchCore-inspired dev | A classical | +0.079 | +0.004 to +0.158 | yes |
| C PatchCore-inspired dev | B Gaussian dev | +0.017 | -0.051 to +0.082 | no |
| B Gaussian full | A classical | +0.081 | +0.027 to +0.134 | yes |
| C PatchCore-inspired full | A classical | +0.129 | +0.061 to +0.206 | yes |
| C PatchCore-inspired full | B Gaussian full | +0.048 | +0.001 to +0.100 | yes |

## candle

100 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified). Every model was fitted on the same seeded random subset of 400 of the 810 fit images (`--max-fit-images`, SPEC.md Section 5.5).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.838 (0.784-0.889) | 0.856 | 0.832 | 0.02 / 0.04 | 0.930 | 0.400 | 0.030 | 0.559 | 8.2 s | 13 ms |
| B Gaussian dev | 0.816 (0.754-0.871) | 0.755 | 0.977 | 0.02 / 0.05 | 0.700 | 0.070 | 0.030 | 0.127 | 18.7 s | 46 ms |
| B Gaussian full | 0.869 (0.819-0.914) | 0.857 | 0.987 | 0.03 / 0.06 | 0.920 | 0.230 | 0.020 | 0.368 | 65.9 s | 110 ms |
| C PatchCore-inspired dev | 0.913 (0.873-0.947) | 0.900 | 0.984 | 0.02 / 0.05 | 0.935 | 0.290 | 0.020 | 0.443 | 608.8 s | 198 ms |
| C PatchCore-inspired full | 0.935 (0.897-0.967) | 0.910 | 0.989 | 0.03 / 0.06 | 0.957 | 0.450 | 0.020 | 0.612 | 662.0 s | 580 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | -0.021 | -0.098 to +0.053 | no |
| C PatchCore-inspired dev | A classical | +0.076 | +0.017 to +0.137 | yes |
| C PatchCore-inspired dev | B Gaussian dev | +0.097 | +0.048 to +0.145 | yes |
| B Gaussian full | A classical | +0.032 | -0.032 to +0.098 | no |
| C PatchCore-inspired full | A classical | +0.098 | +0.039 to +0.158 | yes |
| C PatchCore-inspired full | B Gaussian full | +0.066 | +0.032 to +0.102 | yes |

## pcb1

100 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified). Every model was fitted on the same seeded random subset of 400 of the 814 fit images (`--max-fit-images`, SPEC.md Section 5.5).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.834 (0.778-0.885) | 0.813 | 0.833 | 0.03 / 0.06 | 0.917 | 0.110 | 0.010 | 0.196 | 7.7 s | 13 ms |
| B Gaussian dev | 0.856 (0.797-0.907) | 0.794 | 0.990 | 0.09 / 0.16 | 0.833 | 0.450 | 0.090 | 0.584 | 15.5 s | 35 ms |
| B Gaussian full | 0.856 (0.805-0.907) | 0.807 | 0.990 | 0.09 / 0.16 | 0.870 | 0.400 | 0.060 | 0.548 | 45.5 s | 114 ms |
| C PatchCore-inspired dev | 0.883 (0.828-0.927) | 0.874 | 0.992 | 0.10 / 0.17 | 0.918 | 0.450 | 0.040 | 0.604 | 609.7 s | 198 ms |
| C PatchCore-inspired full | 0.902 (0.857-0.945) | 0.884 | 0.995 | 0.09 / 0.16 | 0.909 | 0.400 | 0.040 | 0.556 | 660.9 s | 582 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | +0.022 | -0.033 to +0.081 | no |
| C PatchCore-inspired dev | A classical | +0.049 | -0.013 to +0.112 | no |
| C PatchCore-inspired dev | B Gaussian dev | +0.027 | -0.021 to +0.075 | no |
| B Gaussian full | A classical | +0.022 | -0.031 to +0.072 | no |
| C PatchCore-inspired full | A classical | +0.068 | +0.011 to +0.128 | yes |
| C PatchCore-inspired full | B Gaussian full | +0.046 | +0.016 to +0.079 | yes |

## pcb2

100 normal and 100 anomalous test images; every model used identical splits and evaluation settings (verified). Every model was fitted on the same seeded random subset of 400 of the 811 fit images (`--max-fit-images`, SPEC.md Section 5.5).

| Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | Precision | Recall | FPR | F1 | Fit time | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| A classical | 0.919 (0.875-0.955) | 0.905 | 0.696 | 0.00 / 0.01 | 0.926 | 0.250 | 0.020 | 0.394 | 7.7 s | 13 ms |
| B Gaussian dev | 0.745 (0.675-0.809) | 0.728 | 0.969 | 0.05 / 0.10 | 0.824 | 0.140 | 0.030 | 0.239 | 15.0 s | 35 ms |
| B Gaussian full | 0.819 (0.760-0.872) | 0.825 | 0.968 | 0.05 / 0.10 | 0.950 | 0.380 | 0.020 | 0.543 | 45.0 s | 116 ms |
| C PatchCore-inspired dev | 0.897 (0.849-0.940) | 0.882 | 0.977 | 0.03 / 0.07 | 0.932 | 0.410 | 0.030 | 0.569 | 609.4 s | 198 ms |
| C PatchCore-inspired full | 0.914 (0.870-0.950) | 0.900 | 0.980 | 0.04 / 0.07 | 0.922 | 0.590 | 0.050 | 0.720 | 663.3 s | 580 ms |

Paired AUROC differences (same test images, paired bootstrap 95% interval):

| A | B | AUROC(A) - AUROC(B) | 95% interval | Excludes 0 |
|---|---|---|---|---|
| B Gaussian dev | A classical | -0.174 | -0.246 to -0.110 | yes |
| C PatchCore-inspired dev | A classical | -0.023 | -0.080 to +0.028 | no |
| C PatchCore-inspired dev | B Gaussian dev | +0.152 | +0.100 to +0.208 | yes |
| B Gaussian full | A classical | -0.100 | -0.163 to -0.045 | yes |
| C PatchCore-inspired full | A classical | -0.005 | -0.058 to +0.046 | no |
| C PatchCore-inspired full | B Gaussian full | +0.095 | +0.054 to +0.140 | yes |

## Macro average over the measured categories

Unweighted mean over the categories in which a model was measured (the number of categories is shown); undefined values are skipped.

| Model | Categories | Image AUROC | Image AP | Pixel AUROC | Pixel IoU | Pixel Dice | F1 |
|---|---|---|---|---|---|---|---|
| A classical | 7 (cashew, pipe_fryum, chewinggum, fryum, candle, pcb1, pcb2) | 0.792 | 0.850 | 0.795 | 0.09 | 0.15 | 0.294 |
| B Gaussian dev | 7 (cashew, pipe_fryum, chewinggum, fryum, candle, pcb1, pcb2) | 0.870 | 0.872 | 0.981 | 0.13 | 0.23 | 0.503 |
| B Gaussian full | 7 (cashew, pipe_fryum, chewinggum, fryum, candle, pcb1, pcb2) | 0.884 | 0.899 | 0.982 | 0.15 | 0.25 | 0.539 |
| C PatchCore-inspired dev | 7 (cashew, pipe_fryum, chewinggum, fryum, candle, pcb1, pcb2) | 0.923 | 0.936 | 0.984 | 0.13 | 0.22 | 0.677 |
| C PatchCore-inspired full | 7 (cashew, pipe_fryum, chewinggum, fryum, candle, pcb1, pcb2) | 0.942 | 0.946 | 0.986 | 0.12 | 0.21 | 0.706 |

## Protocol and caveats

- Image-level metrics in this report were re-derived from each run's saved `predictions.csv` (scores and split-file labels) and agree with the harness summaries.
- Pixel metrics are copied from the harness summaries; pooled over all test images at the evaluation resolution; the pixel threshold comes from calibration_normal anomaly-map values, never from test masks.
- Thresholds are selected from the held-out normal training images only; test images and test labels never influence fitting or thresholds.
- Published results: not compared: published VisA numbers use other protocols.
- Test sets are small (50 to 100 images per class); read the confidence intervals, and prefer the paired differences when comparing two models on the same images.

## How to reproduce

Replace `<category>` by `cashew`, `pipe_fryum`, `chewinggum`, `fryum`, `candle`, `pcb1`, or
`pcb2` (PowerShell; the data root and the report and artifact folders come from the
`INSPECTAI_*` environment variables). `candle`, `pcb1`, and `pcb2` have about 900 normal
training images; their runs add `--max-fit-images 400` to every `fit_model.py` command and to
the classical `evaluate_model.py` command (the other models take the subset from their
artifact), so all five configurations share one seeded subset. Fitting and thresholds use only the `fit` and `calibration_normal` splits (seed 42); the test
split is only scored. Every run's exact configuration, split file hashes, weights SHA-256,
and library versions are in `comparison.json`.

```powershell
python scripts/fetch_weights.py --backbone resnet18
python scripts/fetch_weights.py --backbone wide_resnet50_2
python scripts/evaluate_model.py --model classical --category <category>
python scripts/fit_model.py --model gaussian --category <category> --profile dev
python scripts/evaluate_model.py --model gaussian --category <category> --profile dev
python scripts/fit_model.py --model gaussian --category <category> --profile full --output-dir <artifacts>\gaussian-full\<category>
python scripts/evaluate_model.py --model gaussian --category <category> --profile full --artifact-dir <artifacts>\gaussian-full\<category> --output-dir <reports>\gaussian-full\<category>
python scripts/fit_model.py --model patchcore --category <category> --profile dev
python scripts/evaluate_model.py --model patchcore --category <category> --profile dev
python scripts/fit_model.py --model patchcore --category <category> --profile full --output-dir <artifacts>\patchcore-full\<category>
python scripts/evaluate_model.py --model patchcore --category <category> --profile full --artifact-dir <artifacts>\patchcore-full\<category> --output-dir <reports>\patchcore-full\<category>
python scripts/compare_models.py
```

## Example figures

`cashew_examples.png`, `pipe_fryum_examples.png`, `chewinggum_examples.png`, `fryum_examples.png`, `candle_examples.png`, `pcb1_examples.png`, `pcb2_examples.png`: one row per test image, with the original image, the ground-truth mask, and one heatmap overlay per model. Images were chosen without looking at any model output: two normal images at random (seed 42) and the small, median, and large defect (15th, 50th, and 85th percentile of ground-truth mask area). Each model's colors are anchored at its own pixel threshold (the 99th percentile of normal calibration pixels): blue is 0, the middle of the scale is the threshold, and red is twice the threshold. Text under each tile shows the image score, the image threshold, and the decision.
