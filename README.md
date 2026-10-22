# InspectAI — Industrial Visual Anomaly Detection

CPU-friendly visual anomaly detection on the VisA dataset: a classical baseline, a Gaussian feature model, and a PatchCore-inspired model, with a Streamlit app. The full specification is in [SPEC.md](SPEC.md).

**Status: Milestone 3 (feature extractor, Model B, artifact store) implemented.** The package contains the dataset loader, splits, preprocessing, the metrics and threshold harness, the classical OpenCV baseline, a pretrained-CNN feature extractor, the Gaussian feature model (Model B), and a checked artifact store. The PatchCore-inspired model and the Streamlit app do not exist yet; nothing under "Planned" below exists.

## Requirements

- Windows with PowerShell, Python 3.12
- No GPU needed (CPU-only PyTorch)
- Reference machine: Ryzen 5 5500U (6 cores / 12 threads), 16 GB RAM (see [docs/environment.md](docs/environment.md))

## Installation (Windows PowerShell)

Keep the repository, virtual environment, dataset, and artifacts **outside OneDrive** (SPEC.md Section 9.2).

```powershell
# 1. Virtual environment (outside OneDrive)
New-Item -ItemType Directory -Force C:\dev\venvs | Out-Null
py -3.12 -m venv C:\dev\venvs\inspectai
C:\dev\venvs\inspectai\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. CPU-only PyTorch first, then the project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
cd C:\dev\inspectai
pip install -e ".[dev]"
```

`requirements-lock.txt` lists the exact package versions the project was last verified with.

### Environment variables

Machine-specific locations come from environment variables and are never committed. Set them once for your user account, then open a new terminal:

```powershell
[Environment]::SetEnvironmentVariable('INSPECTAI_DATA_ROOT',    'C:\dev\inspectai-data',           'User')
[Environment]::SetEnvironmentVariable('INSPECTAI_ARTIFACT_DIR', 'C:\dev\inspectai-data\artifacts', 'User')
[Environment]::SetEnvironmentVariable('INSPECTAI_REPORT_DIR',   'C:\dev\inspectai-data\reports',   'User')
[Environment]::SetEnvironmentVariable('TORCH_HOME',             'C:\dev\inspectai-data\torch_cache', 'User')
```

## Dataset

The dataset (VisA) is not part of this repository. It lives under `INSPECTAI_DATA_ROOT`:

```text
C:\dev\inspectai-data\visa\
├── split_csv\1cls.csv
└── VisA\<object>\Data\...
```

Download and verification steps, the licence (CC BY 4.0), and the citation are in SPEC.md Section 3.1. Downloads happen only when you ask for them; no script in this project downloads the dataset.

## Verify the setup

```powershell
python -c "import inspectai; print(inspectai.__version__)"
pytest
ruff check .
```

## Validate the dataset

Requires the VisA data under `INSPECTAI_DATA_ROOT` (see above). The command checks every row of `1cls.csv`, reports counts and per-object image-size and aspect-ratio statistics (with a recommended resize mode), and writes a JSON report to `INSPECTAI_REPORT_DIR`.

```powershell
python scripts/validate_dataset.py
python scripts/validate_dataset.py --objects cashew pipe_fryum --write-splits
```

`--write-splits` writes `split.json` (fit / calibration_normal / test, seed 42 by default) per object to `<INSPECTAI_ARTIFACT_DIR>\splits\<object>\`, and refuses to overwrite a different existing split. Exit codes: 0 success, 2 configuration or data error, 1 other failure.

## Evaluate the classical baseline (Model A)

```powershell
python scripts/evaluate_model.py --model classical --category cashew
python scripts/evaluate_model.py --model classical --category cashew --threshold-policy labeled_calibration
```

The baseline stores a per-pixel mean and standard deviation of the `fit` images. It has no artifact and is fitted inside the command; test images are only scored. The default threshold is the 0.99 quantile of scores on `calibration_normal` (SPEC.md Section 3.5); `labeled_calibration` picks max-F1 on half of the test rows and reports on the other half (`test_eval`) only, stratified by label (defect-type stratification from `image_anno.csv` is not implemented). Pixel IoU/Dice are pooled over all test images at 256x256 and use a threshold from `calibration_normal` map values.

Output goes to `<INSPECTAI_REPORT_DIR>\classical\<category>\`: `predictions.csv` (one row per image), `summary.json` (settings, split hashes, thresholds, metrics with sample counts and a 95% bootstrap CI for image AUROC, timings, memory, environment), and `examples\*.png`. Undefined metrics are `null` and listed in `undefined_metrics`.

### Measured results (classical baseline, seed 42, split `test`, 256x256)

These come from the runs above on the reference laptop (power mode not recorded). They are the baseline only; no other model is evaluated yet.

| Category | Normal / anomalous | Image AUROC (95% CI) | Image AP | Pixel AUROC | F1 at threshold | Fit time | Predict latency |
|---|---|---|---|---|---|---|---|
| cashew | 50 / 100 | 0.701 (0.603-0.794) | 0.818 | 0.809 | 0.305 (recall 0.18, FPR 0.00) | 7.8 s | 13.6 ms |
| pipe_fryum | 50 / 100 | 0.674 (0.587-0.759) | 0.818 | 0.963 | 0.148 (recall 0.08, FPR 0.00) | 7.2 s | 13.4 ms |

The baseline is weak, as expected: it compares each pixel position with the average of the training images, so it responds to object shifts and shape differences more than to the small defects (see `examples\anomaly_highest_score.png`). With only 45 calibration images, the 0.99 quantile sits near their maximum, so the threshold is conservative (no false positives, but low recall). Pixel AUROC is high partly because most pixels are background that every map ranks low; read it with IoU and Dice in `summary.json`. A random ranking has image AP equal to the anomaly share of the test set (0.667).

## Model B: Gaussian feature model (PaDiM-inspired)

Model B fits a Gaussian (mean and regularized covariance) at every position of the layer2+layer3 feature map of a frozen ImageNet-pretrained ResNet, on a fixed random subset of d = 100 channels. The anomaly score at a position is the Mahalanobis distance. The map is upsampled, smoothed (sigma 4), and mapped back to the original image size; the image score is the maximum of the map. Deviations from the PaDiM paper are listed in the docstring of `src/inspectai/models/gaussian.py`.

Profiles (SPEC.md Section 5.5): `dev` = `resnet18`, `full` = `wide_resnet50_2`, both at 224x224 with layers `layer2` and `layer3`.

```powershell
# 1. Fetch the pretrained weights once (an explicit download; nothing else downloads them)
python scripts/fetch_weights.py --backbone resnet18

# 2. Fit on the fit split, select thresholds on calibration_normal, save the artifact
python scripts/fit_model.py --model gaussian --category cashew --profile dev

# 3. Evaluate the saved artifact on the test split (loads it; never refits)
python scripts/evaluate_model.py --model gaussian --category cashew --profile dev
```

Without cached weights, `fit_model.py` and `evaluate_model.py` stop with `WeightsNotFoundError` (exit code 4) and print the fetch command; they never fall back to random weights. Set `INSPECTAI_ALLOW_WEIGHT_DOWNLOAD=1` only for a hosted deployment that must download at load time.

### Artifacts

`fit_model.py` writes `<INSPECTAI_ARTIFACT_DIR>\gaussian\<category>\` with `manifest.json`, `tensors.safetensors` (means, inverse covariances, channel indices), `split.json`, and `config.snapshot.yaml`. The manifest records the backbone and weights identifier with the weights file SHA-256, layers, image size, d, resize mode, reference aspect ratio, seed, the image and pixel thresholds, the calibration score range, fit time and peak memory, library versions, and the SHA-256 of every data file. Nothing is read with pickle. On load, the schema version, checksums, tensor shapes and dtypes, absence of NaN/Inf, and the split (no overlap between `fit` and `test`) are verified; any problem raises `ArtifactError` (exit code 3). An artifact whose configuration snapshot differs from the requested settings (for example a different `--feature-dim`) is rejected as stale. Artifacts are not committed to Git.

### Measured results (Model B, `dev` profile, seed 42, split `test`, 256x256)

Measured on the reference laptop (Ryzen 5 5500U, 6 threads), plugged in, with the Windows **Balanced** power plan (recorded in each `summary.json` and artifact manifest; SPEC.md Section 5.5 recommends High performance, so these times are probably conservative), after `fetch_weights.py --backbone resnet18`. Same splits and evaluation resolution as the classical baseline above.

| Category | Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | F1 at threshold (recall, FPR) |
|---|---|---|---|---|---|
| cashew | classical | 0.701 (0.603-0.794) | 0.818 | 0.809 | 0.305 (0.18, 0.00) |
| cashew | gaussian | 0.916 (0.871-0.955) | 0.962 | 0.987 | 0.701 (0.54, 0.00) |
| pipe_fryum | classical | 0.674 (0.587-0.759) | 0.818 | 0.963 | 0.148 (0.08, 0.00) |
| pipe_fryum | gaussian | 0.933 (0.885-0.967) | 0.963 | 0.992 | 0.701 (0.55, 0.04) |

Each category has 50 normal and 100 anomalous test images. Model B IoU / Dice at the pixel threshold: cashew 0.19 / 0.32, pipe_fryum 0.20 / 0.34 (pooled over all test images; the threshold comes from `calibration_normal`, so it is conservative).

#### `full` profile (`wide_resnet50_2`, 1536 feature channels, d = 100)

Weights fetched with `fetch_weights.py --backbone wide_resnet50_2` (SHA-256 `95faca4d...`, hash verified). Same splits, seed, thresholds policy, and evaluation resolution; artifacts were written to a separate folder (`--output-dir`) so the `dev` artifacts stay reproducible. Both profiles use the same 405 fit images, 45 calibration images and 150 test images per category.

| Category | Profile | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | F1 (recall, FPR) |
|---|---|---|---|---|---|---|
| cashew | dev | 0.916 (0.871-0.955) | 0.962 | 0.987 | 0.19 / 0.32 | 0.701 (0.54, 0.00) |
| cashew | full | 0.876 (0.818-0.928) | 0.937 | 0.990 | 0.26 / 0.41 | 0.246 (0.14, 0.00) |
| pipe_fryum | dev | 0.933 (0.885-0.967) | 0.963 | 0.992 | 0.20 / 0.34 | 0.701 (0.55, 0.04) |
| pipe_fryum | full | 0.932 (0.882-0.973) | 0.960 | 0.994 | 0.24 / 0.39 | 0.708 (0.57, 0.08) |

`full` did not improve image-level ranking on these two categories: the cashew difference (0.876 vs 0.916) is within overlapping confidence intervals, and pipe_fryum is equal. It improved pixel-level overlap (IoU/Dice) and pixel AUROC slightly. The `full` cashew F1 is low only because its calibrated threshold (10.43) is near the top of the calibration scores while test anomalies score lower; this is a threshold-policy effect, not a ranking one (AUROC is 0.88). Nothing was tuned on the test split, and the profiles are fixed in advance by SPEC.md Section 5.5, so these are reported as measured, not selected.

| Quantity (`full`) | cashew | pipe_fryum |
|---|---|---|
| Fit images | 405 | 405 |
| Fit time | 50.9 s | 52.4 s |
| Peak memory after fitting | 0.98 GB | 0.98 GB |
| Peak memory while scoring (separate process) | 0.88 GB | 0.88 GB |
| `predict` latency (mean / median / max) | 122 / 121 / 181 ms | 130 / 124 / 468 ms |
| Model parameters | 31.7 MB | 31.7 MB |

With about 16 ms of image decoding (measured earlier on cashew), end-to-end latency is roughly 0.14 s per image. The provisional targets of SPEC.md Section 5.5 are therefore **confirmed with large margins**: `dev` at most 1 s per image (measured about 0.05 s) and `full` at most 3 s per image (measured about 0.14 s), on the Balanced power plan, plugged in, 6 threads. Peak RAM while fitting stayed under 1 GB against the 8 GB budget.

Resource measurements (Model B, `dev`):

| Quantity | cashew | pipe_fryum |
|---|---|---|
| Fit images | 405 | 405 |
| Fit time | 17.4 s | 16.2 s |
| Peak memory after fitting (process peak working set) | 0.74 GB | 0.75 GB |
| Peak memory while scoring the test split (separate process) | 0.59 GB | 0.59 GB |
| `predict` latency (mean / median / max) | 39 / 38 / 99 ms | 37 / 37 / 55 ms |
| Model parameters (means, inverse covariances, channel indices) | 31.7 MB | 31.7 MB |

Image decoding took 16 ms per image on average (40 cashew test images), so end-to-end decode + predict was about 54 ms per image. The provisional `dev` target of at most 1 s per image (SPEC.md Section 5.5) is met by a wide margin. The `full` profile is measured in the next subsection. Refitting and re-evaluating reproduced the tensors bit for bit and every metric exactly; only timings varied (about 3%). The 0.99-quantile thresholds sit near the maximum of only 45 calibration scores, which gives few false positives but recall of about 0.55.

## Planned (not implemented yet)

These commands are described in SPEC.md Section 11 and do not exist yet (the PatchCore-inspired model is Milestone 4; the app is Milestone 6):

```powershell
python scripts/fit_model.py --model patchcore --category cashew --profile dev
streamlit run app.py
```

## Milestone checklist

- [x] Milestone 0 — Repository inspection and setup
- [x] Milestone 1 — Dataset and preprocessing (125 unit tests plus 3 real-dataset tests pass; real VisA validates with 0 errors)
- [x] Milestone 2 — Evaluation harness and classical baseline
- [x] Milestone 3 — Feature extractor, Model B, and artifact store
- [ ] Milestone 4 — PatchCore-inspired model
- [ ] Milestone 5 — Comparison reports and additional categories
- [ ] Milestone 6 — Streamlit application
- [ ] Milestone 7 — Final verification and delivery

## Troubleshooting

- **`import inspectai` fails:** activate the virtual environment and run `pip install -e ".[dev]"` from `C:\dev\inspectai`.
- **"libiomp5md.dll already initialized":** fix duplicate OpenMP runtimes first. Use `KMP_DUPLICATE_LIB_OK=TRUE` only as a last resort.
- **Environment variables not visible:** open a new terminal after setting them.
- **`WeightsNotFoundError` (exit code 4):** run `python scripts/fetch_weights.py --backbone resnet18` once; the weights are cached under `TORCH_HOME`.
- **`ArtifactError` (exit code 3):** the artifact is missing, corrupted, or stale. Run `scripts/fit_model.py` again with the same settings you evaluate with.

## Dataset attribution and licensing

VisA is released under CC BY 4.0. Cite: Zou, Jeong, Pemula, Zhang, and Dabeer, "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022 (arXiv:2207.14315). See SPEC.md Section 3.1 for the attribution rules that apply to figures derived from the dataset.
