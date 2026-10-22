# InspectAI — Industrial Visual Anomaly Detection

CPU-friendly visual anomaly detection on the VisA dataset: a classical baseline, a Gaussian feature model, and a PatchCore-inspired model, with a Streamlit app. The full specification is in [SPEC.md](SPEC.md).

**Status: Milestone 2 (evaluation harness and classical baseline) implemented.** The package contains the dataset loader, splits, preprocessing, the metrics and threshold harness, the classical OpenCV baseline, and heatmap visualization. There is no deep-learning model yet; nothing under "Planned" below exists.

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

The baseline stores a per-pixel mean and standard deviation of the `fit` images. Until the artifact store exists (Milestone 3) it is fitted inside the command; test images are only scored. The default threshold is the 0.99 quantile of scores on `calibration_normal` (SPEC.md Section 3.5); `labeled_calibration` picks max-F1 on half of the test rows and reports on the other half (`test_eval`) only, stratified by label (defect-type stratification from `image_anno.csv` is not implemented). Pixel IoU/Dice are pooled over all test images at 256x256 and use a threshold from `calibration_normal` map values.

Output goes to `<INSPECTAI_REPORT_DIR>\classical\<category>\`: `predictions.csv` (one row per image), `summary.json` (settings, split hashes, thresholds, metrics with sample counts and a 95% bootstrap CI for image AUROC, timings, memory, environment), and `examples\*.png`. Undefined metrics are `null` and listed in `undefined_metrics`.

### Measured results (classical baseline, seed 42, split `test`, 256x256)

These come from the runs above on the reference laptop (power mode not recorded). They are the baseline only; no other model is evaluated yet.

| Category | Normal / anomalous | Image AUROC (95% CI) | Image AP | Pixel AUROC | F1 at threshold | Fit time | Predict latency |
|---|---|---|---|---|---|---|---|
| cashew | 50 / 100 | 0.701 (0.603-0.794) | 0.818 | 0.809 | 0.305 (recall 0.18, FPR 0.00) | 7.8 s | 13.6 ms |
| pipe_fryum | 50 / 100 | 0.674 (0.587-0.759) | 0.818 | 0.963 | 0.148 (recall 0.08, FPR 0.00) | 7.2 s | 13.4 ms |

The baseline is weak, as expected: it compares each pixel position with the average of the training images, so it responds to object shifts and shape differences more than to the small defects (see `examples\anomaly_highest_score.png`). With only 45 calibration images, the 0.99 quantile sits near their maximum, so the threshold is conservative (no false positives, but low recall). Pixel AUROC is high partly because most pixels are background that every map ranks low; read it with IoU and Dice in `summary.json`. A random ranking has image AP equal to the anomaly share of the test set (0.667).

## Planned (not implemented yet)

These commands are described in SPEC.md Section 11 and do not exist yet:

```powershell
python scripts/fetch_weights.py --backbone resnet18
python scripts/fit_model.py --model patchcore --category cashew --profile dev
streamlit run app.py
```

## Milestone checklist

- [x] Milestone 0 — Repository inspection and setup
- [x] Milestone 1 — Dataset and preprocessing (125 unit tests plus 3 real-dataset tests pass; real VisA validates with 0 errors)
- [x] Milestone 2 — Evaluation harness and classical baseline
- [ ] Milestone 3 — Feature extractor, Model B, and artifact store
- [ ] Milestone 4 — PatchCore-inspired model
- [ ] Milestone 5 — Comparison reports and additional categories
- [ ] Milestone 6 — Streamlit application
- [ ] Milestone 7 — Final verification and delivery

## Troubleshooting

- **`import inspectai` fails:** activate the virtual environment and run `pip install -e ".[dev]"` from `C:\dev\inspectai`.
- **"libiomp5md.dll already initialized":** fix duplicate OpenMP runtimes first. Use `KMP_DUPLICATE_LIB_OK=TRUE` only as a last resort.
- **Environment variables not visible:** open a new terminal after setting them.

## Dataset attribution and licensing

VisA is released under CC BY 4.0. Cite: Zou, Jeong, Pemula, Zhang, and Dabeer, "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022 (arXiv:2207.14315). See SPEC.md Section 3.1 for the attribution rules that apply to figures derived from the dataset.
