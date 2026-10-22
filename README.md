# InspectAI — Industrial Visual Anomaly Detection

CPU-friendly visual anomaly detection on the VisA dataset: a classical baseline, a Gaussian feature model, and a PatchCore-inspired model, with a Streamlit app. The full specification is in [SPEC.md](SPEC.md).

**Status: Milestone 4 (PatchCore-inspired model) implemented.** The package contains the dataset loader, splits, preprocessing, the metrics and threshold harness, the classical OpenCV baseline (Model A), a pretrained-CNN feature extractor, the Gaussian feature model (Model B), the PatchCore-inspired model (Model C), and a checked artifact store. The Streamlit app does not exist yet; nothing under "Planned" below exists.

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

These come from the runs above on the reference laptop (plugged in, Windows Balanced power plan; the numbers were reproduced exactly after the Milestone 3 and 4 code changes). They are the classical baseline only; the deep models are measured in the sections below.

| Category | Normal / anomalous | Image AUROC (95% CI) | Image AP | Pixel AUROC | F1 at threshold | Fit time | Predict latency |
|---|---|---|---|---|---|---|---|
| cashew | 50 / 100 | 0.701 (0.603-0.794) | 0.818 | 0.809 | 0.305 (recall 0.18, FPR 0.00) | 7.8 s | 13.5 ms |
| pipe_fryum | 50 / 100 | 0.674 (0.587-0.759) | 0.818 | 0.963 | 0.148 (recall 0.08, FPR 0.00) | 7.2 s | 13.5 ms |

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

## Model C: PatchCore-inspired anomaly detection

Model C keeps the patch features of the normal images in a memory bank and scores a test patch by its distance to the nearest stored patch. This is **inspired by** PatchCore (Roth et al., CVPR 2022), not equivalent to it; the deviations are listed below.

Algorithm (`src/inspectai/models/patchcore.py`):

1. Each `fit` image is resized (category resize mode) and passed through the frozen backbone. The layer2 and layer3 maps are upsampled to the layer2 resolution and concatenated (as in Model B): one D-dimensional vector per spatial position (384 dims for `resnet18`, 1536 for `wide_resnet50_2`; 784 positions per image at 224x224).
2. Each vector is averaged with its 3x3 spatial neighbors (stride 1).
3. All vectors of all fit images form the candidate memory bank.
4. The bank is reduced with **greedy k-center coreset selection**: vectors are randomly projected to 128 dimensions (seeded), a seeded random vector starts the selection, and the vector farthest from those already selected is added until 10% (`--coreset-ratio`, default 0.10) are kept. A ratio of 1 keeps every vector.
5. A test patch score is the Euclidean distance to its nearest bank vector (exact search with chunked `torch.cdist`; no approximate index, no faiss). The patch scores form a map that is upsampled to the input size, smoothed (Gaussian, sigma 4), and mapped back to the original image size.
6. The image score is the maximum of the anomaly map.

Deviations from PatchCore: the 3x3 averaging is applied after the layers are upsampled and concatenated (PatchCore pools each layer at its own resolution); there is no adaptive pooling of features to a lower dimension (D is the number of concatenated channels); the image score is the plain maximum of the smoothed map (PatchCore re-weights it by neighbor distances); nearest-neighbor search is exact with k = 1; the projection is used only to choose the coreset, and the bank keeps full-dimensional vectors.

```powershell
python scripts/fit_model.py --model patchcore --category cashew --profile dev
python scripts/evaluate_model.py --model patchcore --category cashew --profile dev
```

Options: `--coreset-ratio`, `--max-fit-images N` (also available for `--model gaussian`), `--max-bank-gb` (default 4), `--max-fit-minutes` (default 30), `--chunk-size` (nearest-neighbor block size, default 16384). Evaluation loads the saved artifact and never refits.

Resource limits (SPEC.md Section 5.5), all enforced with an error and never by silently shrinking the data:

- Before the bank is built, the estimate `images x 784 patches x D x 4 bytes` is logged (for example `405 images x 784 patches x 384 dims x 4 bytes = 0.45 GB before reduction (limit 4.00 GB)`). If it exceeds `--max-bank-gb`, the fit stops and tells you to lower `--max-fit-images`, use the `dev` profile, or raise the limit.
- `--max-fit-images N` fits on a **seeded random subset** of N images of the fit split. The subset is printed and recorded in `split.json` (`max_fit_images`, `fit_used`, and its SHA-256), so an evaluation always knows what the model saw.
- If feature extraction plus coreset selection exceed `--max-fit-minutes`, the fit stops with a message suggesting random subsampling (`--max-fit-images`); it never returns a partial coreset.

The artifact (`<INSPECTAI_ARTIFACT_DIR>\patchcore\<category>\`) has the same four files as Model B. `tensors.safetensors` holds the reduced bank. `config.snapshot.yaml` holds the exact configuration used: the model settings plus a `runtime` section (chunk size, batch size, threads, memory and time limits). The `runtime` section is stored but ignored when deciding whether an artifact is stale, so changing `--chunk-size` or `--max-bank-gb` is allowed, while a different `--coreset-ratio` or profile is rejected as stale (exit code 3).

### Measured results (Model C, 10% coreset, seed 42, split `test`, 256x256)

Measured on the reference laptop (Ryzen 5 5500U, 6 threads), plugged in, Windows **Balanced** power plan (recorded in every summary and manifest). Fitted with the spec defaults (coreset 0.10, d = all concatenated channels, `max_bank_gb` 4, `max_fit_minutes` 30), fixed in advance, with no tuning on the test split. Each category has 50 normal and 100 anomalous test images. The `full` results were produced with `--profile full` and a separate folder so they do not replace the `dev` artifact, for example:

```powershell
python scripts/fit_model.py --model patchcore --category cashew --profile full --output-dir C:\dev\inspectai-data\artifacts\patchcore-full\cashew
python scripts/evaluate_model.py --model patchcore --category cashew --profile full --artifact-dir C:\dev\inspectai-data\artifacts\patchcore-full\cashew --output-dir C:\dev\inspectai-data\reports\patchcore-full\cashew
```

| Category | Profile | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | F1 (recall, FPR) |
|---|---|---|---|---|---|---|
| cashew | dev | 0.955 (0.922-0.981) | 0.978 | 0.990 | 0.20 / 0.34 | 0.734 (0.58, 0.00) |
| cashew | full | 0.968 (0.944-0.987) | 0.984 | 0.992 | 0.19 / 0.33 | 0.780 (0.64, 0.00) |
| pipe_fryum | dev | 0.982 (0.964-0.995) | 0.992 | 0.992 | 0.18 / 0.31 | 0.750 (0.60, 0.00) |
| pipe_fryum | full | 0.990 (0.977-0.998) | 0.995 | 0.994 | 0.20 / 0.33 | 0.684 (0.52, 0.00) |

| Quantity | cashew dev | pipe_fryum dev | cashew full | pipe_fryum full |
|---|---|---|---|---|
| Fit images | 405 | 405 | 405 | 405 |
| Candidate patches (405 x 784) -> bank vectors (10%) | 317,520 -> 31,752 | 317,520 -> 31,752 | 317,520 -> 31,752 | 317,520 -> 31,752 |
| Unreduced bank estimate | 0.45 GB | 0.45 GB | 1.82 GB | 1.82 GB |
| Reduced bank (stored) | 48.8 MB | 48.8 MB | 195.1 MB | 195.1 MB |
| Coreset selection time | 635 s | 621 s | 630 s | 640 s |
| Total fit time | 649 s (10.8 min) | 635 s (10.6 min) | 682 s (11.4 min) | 687 s (11.4 min) |
| Peak memory after fitting | 1.23 GB | 1.23 GB | 2.87 GB | 2.87 GB |
| Peak memory while scoring (separate process) | 0.75 GB | 0.75 GB | 1.35 GB | 1.34 GB |
| `predict` latency (mean / median / max) | 205 / 205 / 230 ms | 197 / 197 / 233 ms | 593 / 577 / 774 ms | 650 / 621 / 1466 ms |

With about 16 ms of image decoding, end-to-end latency is roughly 0.22 s per image for `dev` and 0.61-0.67 s for `full`. The provisional targets of SPEC.md Section 5.5 (`dev` at most 1 s, `full` at most 3 s per image) are met. Peak fitting memory stayed under 3 GB against the 8 GB budget.

**Bottleneck.** 92-98% of the fit time is the greedy coreset selection: each of the 31,752 selection steps scans all 317,520 projected vectors (about 20 ms per step, memory-bound), so a fit takes roughly 11 minutes per category on this CPU, within the 30-minute limit. Feature extraction takes about 15 s (`dev`) or 50 s (`full`). Selection time is proportional to the number of selected vectors, so a smaller `--coreset-ratio` or `--max-fit-images` reduces it linearly (a 1% coreset took about 65 s to select in a smoke run). No optimization was attempted, in line with SPEC.md Section 5.3.

**Reproducibility.** Fitting `cashew` (`dev`) a second time from scratch produced a bit-identical memory bank: the SHA-256 of `tensors.safetensors`, `split.json`, and `config.snapshot.yaml` and the stored thresholds and calibration scores were all identical (Model B refits were bit-identical too). This was checked on one machine with the same thread count. A different thread count or batch size can change floating-point summation order, and in principle the last bits of the features and which near-tied vector the coreset picks, which is why both are recorded in the configuration snapshot (`runtime`). Test-suite checks use a fixed seed and a deterministic fake backbone.

### Model comparison (classical, Model B, Model C)

All rows use the same seed-42 splits (fit 405 / calibration 45 / test 150 images per category), the same 256x256 evaluation resolution, the same threshold policy (`normal_percentile`, q = 0.99 on `calibration_normal`), and the same bootstrap settings. The split file hashes recorded in the `summary.json` files are identical across all models.

| Category | Model | Image AUROC (95% CI) | Image AP | Pixel AUROC | Pixel IoU / Dice | F1 (recall, FPR) |
|---|---|---|---|---|---|---|
| cashew | A classical | 0.701 (0.603-0.794) | 0.818 | 0.809 | 0.21 / 0.35 | 0.305 (0.18, 0.00) |
| cashew | B Gaussian dev | 0.916 (0.871-0.955) | 0.962 | 0.987 | 0.19 / 0.32 | 0.701 (0.54, 0.00) |
| cashew | B Gaussian full | 0.876 (0.818-0.928) | 0.937 | 0.990 | 0.26 / 0.41 | 0.246 (0.14, 0.00) |
| cashew | C PatchCore-insp. dev | 0.955 (0.922-0.981) | 0.978 | 0.990 | 0.20 / 0.34 | 0.734 (0.58, 0.00) |
| cashew | C PatchCore-insp. full | 0.968 (0.944-0.987) | 0.984 | 0.992 | 0.19 / 0.33 | 0.780 (0.64, 0.00) |
| pipe_fryum | A classical | 0.674 (0.587-0.759) | 0.818 | 0.963 | 0.24 / 0.39 | 0.148 (0.08, 0.00) |
| pipe_fryum | B Gaussian dev | 0.933 (0.885-0.967) | 0.963 | 0.992 | 0.20 / 0.34 | 0.701 (0.55, 0.04) |
| pipe_fryum | B Gaussian full | 0.932 (0.882-0.973) | 0.960 | 0.994 | 0.24 / 0.39 | 0.708 (0.57, 0.08) |
| pipe_fryum | C PatchCore-insp. dev | 0.982 (0.964-0.995) | 0.992 | 0.992 | 0.18 / 0.31 | 0.750 (0.60, 0.00) |
| pipe_fryum | C PatchCore-insp. full | 0.990 (0.977-0.998) | 0.995 | 0.994 | 0.20 / 0.33 | 0.684 (0.52, 0.00) |

Reading the comparison with care:

- Model C has the highest image AUROC and AP in both categories and both profiles. Its intervals overlap Model B's in some cases (for example cashew `dev`: 0.922-0.981 against 0.871-0.955), and only two categories are measured, so this is a consistent ranking, not a proven significant difference.
- Both deep models are far better than the classical baseline at image level (AUROC 0.88-0.99 against 0.67-0.70).
- Model C does **not** improve pixel-level overlap: IoU/Dice at the calibrated pixel threshold are 0.18-0.20 / 0.31-0.34 for Model C, at or slightly below Model B (0.19-0.26 / 0.32-0.41) and slightly below the classical baseline (0.21-0.24 / 0.35-0.39) in both categories. Pixel AUROC is about 0.99 for both deep models and is inflated by the many background pixels.
- F1 at the calibrated threshold depends on a threshold taken from only 45 calibration images, so recall is limited (0.52-0.64 for Model C) while false positives are rare. The F1 difference between `dev` and `full` for the same model is mostly this threshold effect, not ranking quality.
- Model C costs more: about 11 minutes to fit and 0.2-0.65 s per image against 17-52 s and 0.04-0.13 s for Model B. No model has been selected as the default; that follows the evaluation protocol in a later milestone.

## Planned (not implemented yet)

These commands are described in SPEC.md Section 11 and do not exist yet (the app is Milestone 6):

```powershell
streamlit run app.py
```

## Milestone checklist

- [x] Milestone 0 — Repository inspection and setup
- [x] Milestone 1 — Dataset and preprocessing (125 unit tests plus 3 real-dataset tests pass; real VisA validates with 0 errors)
- [x] Milestone 2 — Evaluation harness and classical baseline
- [x] Milestone 3 — Feature extractor, Model B, and artifact store
- [x] Milestone 4 — PatchCore-inspired model
- [ ] Milestone 5 — Comparison reports and additional categories
- [ ] Milestone 6 — Streamlit application
- [ ] Milestone 7 — Final verification and delivery

## Troubleshooting

- **`import inspectai` fails:** activate the virtual environment and run `pip install -e ".[dev]"` from `C:\dev\inspectai`.
- **"libiomp5md.dll already initialized":** fix duplicate OpenMP runtimes first. Use `KMP_DUPLICATE_LIB_OK=TRUE` only as a last resort.
- **Environment variables not visible:** open a new terminal after setting them.
- **`WeightsNotFoundError` (exit code 4):** run `python scripts/fetch_weights.py --backbone resnet18` once; the weights are cached under `TORCH_HOME`.
- **"The memory bank would need about N GB ... above max_bank_gb" (exit code 2):** lower `--max-fit-images` (a recorded, seeded subset), use the `dev` profile, or raise `--max-bank-gb` if you have the RAM.
- **"Coreset selection exceeded the time limit" / "exceeded max_fit_minutes" (exit code 2):** use `--max-fit-images` or a smaller `--coreset-ratio`, or raise `--max-fit-minutes` and wait (a default fit takes about 11 minutes on the reference laptop).
- **`ArtifactError` (exit code 3):** the artifact is missing, corrupted, or stale. Run `scripts/fit_model.py` again with the same settings you evaluate with.

## Dataset attribution and licensing

VisA is released under CC BY 4.0. Cite: Zou, Jeong, Pemula, Zhang, and Dabeer, "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022 (arXiv:2207.14315). See SPEC.md Section 3.1 for the attribution rules that apply to figures derived from the dataset.
