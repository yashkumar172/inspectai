# Final project report

**InspectAI: industrial visual anomaly detection on VisA.** This report summarizes the finished
project as required by SPEC.md Section 19. Every number comes from an executed run and is traceable to
a report file; nothing is estimated. Details: [README](../README.md), [architecture](architecture.md),
[evaluation](evaluation.md), [limitations](limitations.md).

## 1. Problem and proposed solution

On a production line, defective products are rare and unpredictable, so there are often no defective
examples to train a classifier on. InspectAI learns what **normal** looks like for each product and
flags deviations: for an image it returns an anomaly score, a threshold-based decision, and a pixel
heatmap showing where the image looks unusual. Three methods are implemented and compared under one
protocol (same splits, same metrics, thresholds chosen without test data), and a Streamlit app serves
the saved models. The system runs on a CPU-only Windows laptop.

## 2. Implemented models and architecture

| Model | Idea | Stored parameters |
|---|---|---|
| **A: classical baseline** | per-pixel-position mean and standard deviation of normal images; z-score map | two 256x256x3 arrays |
| **B: Gaussian model (PaDiM-inspired)** | frozen ImageNet CNN (layer2 + layer3); per-position Gaussian on 100 channels; Mahalanobis distance | means, inverse covariances, channel indices (about 32 MB) |
| **C: PatchCore-inspired** | the same features, 3x3 averaged, as a coreset memory bank (10%); nearest-neighbor distance | the reduced bank (49 MB `dev`, 195 MB `full`) |

Backbones: ResNet-18 (`dev` profile) and Wide ResNet-50-2 (`full` profile), 224x224 input, frozen,
CPU only. All models return an anomaly map with the input image's size and an image score equal to the
map's maximum. The modules, data flow, and the deviations from PaDiM and PatchCore are in
[architecture.md](architecture.md) and [evaluation.md](evaluation.md).

## 3. Dataset categories and data split

VisA (official one-class split, CC BY 4.0). All 12 objects are supported by the loader; the four initial
categories were evaluated: `cashew`, `pipe_fryum`, `chewinggum`, `fryum`. Per category (seed 42): about
405 `fit` images, 45 `calibration_normal` images (thresholds only), and the official test split of 50
normal and 100 anomalous images (final evaluation only). The splits are disjoint, reproducible, stored in
every artifact, and enforced by tests. The other eight objects were not run.

## 4. Actual evaluation results

Image AUROC with 95% bootstrap confidence intervals (1000 resamples), 50 normal and 100 anomalous test
images per category, thresholds from `calibration_normal`, 256x256 evaluation resolution:

| Model | cashew | pipe_fryum | chewinggum | fryum | Macro mean |
|---|---|---|---|---|---|
| A classical | 0.701 (0.603-0.794) | 0.674 (0.587-0.759) | 0.790 (0.706-0.865) | 0.785 (0.700-0.856) | 0.738 |
| B Gaussian dev | 0.916 (0.871-0.955) | 0.933 (0.885-0.967) | 0.974 (0.947-0.994) | 0.848 (0.778-0.908) | 0.918 |
| B Gaussian full | 0.876 (0.818-0.928) | 0.932 (0.882-0.973) | 0.973 (0.948-0.992) | 0.866 (0.803-0.922) | 0.912 |
| C PatchCore-inspired dev | 0.955 (0.922-0.981) | 0.982 (0.964-0.995) | 0.968 (0.940-0.991) | 0.864 (0.799-0.915) | 0.942 |
| C PatchCore-inspired full | 0.968 (0.944-0.987) | 0.990 (0.977-0.998) | 0.967 (0.938-0.991) | 0.914 (0.864-0.952) | 0.960 |

Full tables (AP, pixel AUROC, IoU, Dice, precision, recall, FPR, F1, confusion matrices, performance)
are in the README and in the generated `comparison.md` / `comparison.json`. Every image-level metric
was re-derived from the saved per-image predictions and agrees with the harness summaries.

## 5. Baseline comparison

Both deep models rank images better than the classical baseline in every category: all
16 paired AUROC differences are positive (+0.062 to
+0.316) and 15 of 16 paired intervals exclude 0 (the exception
is Model B `dev` on `fryum`). Model C is **not consistently better than Model B**: it has the higher
AUROC in 6 of 8 same-profile comparisons and the paired interval excludes 0 in
5; on `chewinggum` (both profiles) and on `fryum` `dev` no difference is demonstrated. F1 at the
calibrated threshold does not follow ranking quality; compare models by AUROC and paired intervals.

## 6. CPU performance, fitting time, and peak RAM

Measured on one machine (AMD Ryzen 5 5500U, 6 cores, 16 GB, no GPU), plugged in, Windows **Balanced**
power plan, 6 threads:

| Model | Fit time (405 images) | Peak RAM while fitting | Prediction latency (mean) | Stored size |
|---|---|---|---|---|
| A classical | 7-8 s | 0.4 GB | 13-14 ms | 1.6 MB |
| B Gaussian `dev` / `full` | 15-17 s / 45-52 s | 0.74-0.78 GB / 0.98-1.04 GB | 34-39 ms / 106-130 ms | 31.7 MB |
| C PatchCore-inspired `dev` / `full` | 10.6-10.8 min / 11.3-11.9 min | 1.2 GB / 2.9 GB | 0.20 s / 0.58-0.65 s | 49 MB / 195 MB |

The provisional targets of SPEC.md Section 5.5 (`dev` at most 1 s and `full` at most 3 s per image) were
confirmed with large margins (about 0.05 s and 0.14 s end to end for Model B; 0.22 s and 0.67 s for
Model C). About 92-98% of a Model C fit is the greedy coreset selection. Peak RAM stayed under 3 GB
against the 8 GB budget. Model C refits reproduce the stored tensors bit for bit.

## 7. Representative examples

Normal images, small / median / large defects, and every model's heatmap, chosen without looking at
model output and including false alarms and misses:
[cashew](images/example_cashew.jpg), [pipe_fryum](images/example_pipe_fryum.jpg) (also shown in the
README). *Dataset images: VisA (Zou et al., ECCV 2022), CC BY 4.0; modified by resizing and adding
heatmap overlays and text.* Full-size figures for all four categories are generated by
`scripts/compare_models.py`.

## 8. Test results

| Check | Result |
|---|---|
| Default test suite (no dataset, no weights, no `INSPECTAI_*` variables) | **600 passed**, 9 deselected, about 2 minutes |
| Real dataset and real pretrained weights (`pytest -m "needs_dataset or slow"`) | **9 passed** |
| `ruff check` and `ruff format --check` | all checks passed; 91 files formatted |
| `pip check` | no broken requirements; `requirements-lock.txt` matches the installed packages |
| Clean-copy check: the exact set of files that would be committed, copied to a new folder | 600 passed, ruff clean, `run_demo.py` succeeded, every script answers `--help` |
| End-to-end smoke test (`scripts/run_demo.py`, synthetic data, about 13 s) | passed: 30 predictions equal the saved reports |
| Real-data regression (re-run of a saved evaluation and of the comparison) | metrics, thresholds, and split hashes identical to the stored reports |
| App | 30 AppTest UI tests; 20 real-model inspections through the app matched the saved evaluation scores and decisions; `streamlit run app.py` starts and passes its health check (listens on localhost only) |

Six pandas deprecation warnings in the test output come from inside Streamlit's chart code and are not
suppressed. The tests cover the leakage rules, metrics with known answers, artifact corruption, every
comparison refusal, and the app's failure paths ([evaluation.md](evaluation.md), section 9).

## 9. Deployment status

**Prepared, not deployed.** Deployment (SPEC.md Section 15) is optional. A dataset-free hosted bundle
(`hosted/`, about 408 MB) holds the dashboard results, the Model C `dev` artifacts for all seven
categories with the ResNet-18 weights they need, and the classical baseline; `requirements.txt` installs
the CPU-only dependencies. From the bundle alone, Model C reproduced the stored evaluation scores exactly
on 28 test images (about 0.2 s per image, about 1.1 GB peak memory). Nothing was hosted or tested online,
so no working hosted deployment is claimed. Model B, the Model C `full` profile, and a float16 profile
are not bundled.

## 10. Limitations and potential future improvements

Full list: [limitations.md](limitations.md). In short: four of twelve objects evaluated with small test
sets (50 + 100 images), one seed and one machine; model rankings depend on the category; thresholds from
about 45 images are conservative (recall 0.06 to 0.90); localisation is coarse (pixel IoU 0.05 to 0.30);
scores are not probabilities; the app's layout was not reviewed in a browser; benchmarks used the
Balanced power plan; no static type checker or CI. Future work: more and cross-validated calibration
data, the remaining eight objects and several seeds, a faster coreset or approximate search and a
float16 bank for a smaller hosted bundle, finer feature resolution, and a browser-based UI review before any
deployment.

## Definition of Done (SPEC.md Section 18)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | A working, documented dataset loader exists | Done | `data/`, `scripts/validate_dataset.py` (10,821 rows, 0 errors on the real data), README dataset setup |
| 2 | The classical baseline runs | Done | `models/classical.py`; evaluated on 4 categories |
| 3 | Model B runs | Done | `models/gaussian.py`; evaluated on 4 categories, `dev` and `full` |
| 4 | An anomaly heatmap and overlay are generated | Done | `visualization/heatmaps.py`, app, example figures |
| 5 | A normal-reference fitting procedure exists | Done | `scripts/fit_model.py` (fit split only) |
| 6 | At least two methods are evaluated and compared | Done | three methods, four categories, one protocol |
| 7 | Image-level metrics are computed correctly | Done | toy-input tests with known answers; re-derived from predictions in every comparison |
| 8 | Pixel-level evaluation works where annotations exist | Done | pixel AUROC, AP, IoU, Dice at 256x256 |
| 9 | Thresholds are stored per category and selected without test-set leakage | Done | `normal_percentile` policy cannot receive test data; leakage tests for all models |
| 10 | Artifacts reload only after integrity checks (no pickle) | Done | safetensors, checksums, shapes, NaN, schema, stale detection; corruption tests |
| 11 | The Streamlit application performs real inference | Done | 20 real-model inspections matched the evaluation reports |
| 12 | CPU-only execution is supported | Done | CPU PyTorch; `device: cpu` is the only accepted value |
| 13 | Automated unit and integration tests pass | Done | 600 default + 9 real-data/weights tests |
| 14 | README and architecture documentation are complete | Done | README covers the 14 required items; `docs/architecture.md`, `evaluation.md`, `limitations.md` |
| 15 | Experimental results are based on actual executions | Done | every figure from `summary.json` / `comparison.json`; none estimated |
| 16 | Known limitations are clearly documented | Done | README, `docs/limitations.md`, this report |
| 17 | Git contains no datasets, secrets, or unnecessary large artifacts | Done | scans for data/weights/caches/secrets; largest tracked non-text files are two example JPEGs (under 300 KB) |
| 18 | Splits are disjoint and reproducible; a test enforces this | Done | `tests/unit/test_splits.py` |
| 19 | Fit time, peak RAM, bank size, and per-image latency are measured and reported | Done | every `summary.json` and manifest; README tables |
| 20 | Results include bootstrap confidence intervals and sample counts | Done | every metric table |
| 21 | Repository, virtual environment, dataset, and artifacts are outside OneDrive | Done | `C:\dev\inspectai`, `C:\dev\venvs\inspectai`, `C:\dev\inspectai-data` |

PatchCore-inspired functionality was implemented and evaluated, not deferred. Deployment (optional) was
prepared locally but not hosted; the unfinished items are listed in [limitations.md](limitations.md).

## Milestones

| Milestone | Delivered |
|---|---|
| 0 | repository, environment, package, documentation scaffold |
| 1 | dataset loader and validator, preprocessing, reproducible splits |
| 2 | evaluation harness, threshold policies, classical baseline |
| 3 | feature extractor, Model B, artifact store, resource measurements |
| 4 | PatchCore-inspired model, memory and time limits, `max_fit_images` |
| 5 | verified comparison reports, paired differences, four categories |
| 6 | Streamlit app, upload validation, validated dashboard, baseline artifact |
| 7 | smoke test, default-settings file, documentation, clean-copy verification, this report |

## Addendum: `candle`, `pcb1`, `pcb2`

After Milestone 7 the benchmark was extended to three more VisA objects with the same models and
protocol (no tuning). The tables above describe the original four categories and are unchanged; the
seven-category results, the macro averages, and the paired differences are in the README ("Results by
category") and in the regenerated `comparison.md`. Differences from the original four: these objects
have about 810 fit images, so all five configurations were fitted on the same seeded subset of 400
(`--max-fit-images 400`); the test sets have 100 normal and 100 anomalous images; and the classical
baseline is not clearly worse than the deep models on `pcb2`, where it has the highest image AUROC
(0.919). Five objects remain unevaluated.

Test status after this extension (counted, not estimated): 603 default tests passed (9 deselected)
and 9 real-dataset/real-weights tests passed; the 600 quoted in Section 6 belong to the Milestone 7
snapshot. Ruff check and format are clean.
