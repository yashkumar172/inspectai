# SPEC.md — Industrial Defect Detection System

**Project name:** InspectAI — Industrial Visual Anomaly Detection  
**Type:** Computer Vision / Deep Learning / Anomaly Detection  
**Primary language:** Python 3.12  
**Primary framework:** PyTorch  
**Interface:** Streamlit  
**Execution target:** CPU-first Windows laptop (reference hardware: Ryzen 5 5500U, 6 cores / 12 threads, 16 GB RAM, no GPU)  
**Dataset:** VisA (Visual Anomaly dataset, Amazon)  
**Repository location:** `C:\dev\inspectai` (outside OneDrive; see Section 9.2)  
**Project style:** Modular, tested, reproducible, portfolio-ready

---

## 1. Project Overview

Build a complete industrial visual anomaly detection system that identifies defective products from images, estimates where defects occur, and displays interpretable visual results.

The system must be capable of learning normal visual patterns from training images and detecting anomalous patterns in previously unseen images.

Examples of defects include scratches, cracks, dents, contamination, broken components, and unusual textures.

The project must go beyond a basic binary CNN classifier. It should implement a reproducible computer vision pipeline, compare multiple approaches, measure performance using appropriate metrics, and expose the final pipeline through a usable web application.

### Core capabilities

1. Load and validate industrial images.
2. Train or fit an anomaly detection model using normal training images.
3. Predict whether an image contains an anomaly.
4. Generate a pixel-level anomaly heatmap.
5. Overlay the heatmap on the original image.
6. Evaluate image-level and pixel-level performance.
7. Compare the proposed method against a baseline.
8. Provide an interactive Streamlit interface.
9. Run automated tests without downloading the full dataset.
10. Document experimental results, limitations, and reproduction steps.

### Non-goals

Do not initially implement:

- Training a large CNN from scratch.
- Cloud GPU dependencies.
- Paid APIs or hosted inference services.
- Industrial camera hardware integration.
- Production PLC or manufacturing-line integration.
- Medical diagnosis or safety-critical decisions.
- Multiple complicated models before establishing a reliable baseline.
- A single unified model across all categories (models are fitted per category).
- GPU-specific optimizations.

The project must remain useful on a CPU-only laptop.

---

## 2. Problem Definition

Industrial anomaly detection differs from ordinary image classification.

In a conventional supervised classifier, training data typically contains labeled examples of both normal and defective products. In industrial anomaly detection, the model may receive only normal examples during training.

The system should learn a representation of normal appearance and identify deviations at inference time.

For an input image x, the system should produce:

- An image-level anomaly score s(x).
- A binary decision based on a configurable threshold.
- An anomaly map A(x) indicating suspicious image locations.
- An optional binary defect mask derived from the anomaly map.

The anomaly map should have the same spatial dimensions as the input image after resizing or postprocessing.

The binary decision must be configurable rather than permanently hard-coded.

Important: An anomaly score is not automatically a calibrated probability that an image is defective. The application must not display an anomaly score as a probability unless a separate calibration procedure has been implemented and evaluated.

---

## 3. Dataset

### 3.1 Primary dataset: VisA

Use the official VisA (Visual Anomaly) dataset released by Amazon with the paper cited below.

Official source (README, download link, and split files):

<https://github.com/amazon-science/spot-diff>

The dataset contains 12 subsets (objects) with 10,821 images in total (9,621 normal and 1,200 anomalous), and pixel-level annotation masks for the anomalous images. The objects fall into three groups: complex-structure PCBs (`pcb1`, `pcb2`, `pcb3`, `pcb4`), multiple-instance objects (`capsules`, `candle`, `macaroni1`, `macaroni2`), and single-instance objects (`cashew`, `chewinggum`, `fryum`, `pipe_fryum`). The names are the folder names used by the official `prepare_data.py`. VisA is a benchmark for industrial-style inspection; it is not data from a production line, and it includes food items as well as PCBs.

The loader is written behind a small dataset-adapter interface so that another dataset can be added later without changing model code.

Do not commit the dataset or its image files to Git.

**License.** VisA is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0), as stated in the official repository README and its `LICENSE-DATASET` file. When the data or material derived from it is shared, including modified material such as heatmap overlays, the license requires: credit to the creators, a link to the source and to the license, and a statement that changes were made. Example figures derived from VisA images may appear in the README, reports, and the deployed application only with this attribution (Section 14). Tests and fixtures use synthetic images only. The official repository's code is covered by a separate license (shown on GitHub as Apache-2.0); this project does not copy it. Verify the license text on the official repository before publishing anything.

**Download process.**

- Archive: `VisA_20220922.tar`, linked from the official README at `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`. A HEAD request on 2026-10-22 returned HTTP 200 and a size of 1,929,840,640 bytes (about 1.93 GB). The README offers no per-object downloads, so the whole archive is downloaded.
- Split file: `split_csv/1cls.csv` from the official repository (the one-class setting). Fetch it from a specific commit, and record the commit hash and the file's SHA-256 in the project documentation.
- Do not run the official `utils/prepare_data.py`. It copies every image into a second folder, which would roughly double disk use. The loader reads the split CSV directly (Section 3.3).
- Disk budget: the extracted images are expected to take about as much space as the archive (JPEG files do not compress further). Before extracting, require free space of at least the archive size plus a 1 GB margin, delete the archive after verification, and extract only the needed objects when space is tight. Check member names with `tar -tf` before choosing paths.
- Download into a staging folder outside the repository and OneDrive (Section 9.2), for example `C:\dev\inspectai-data\downloads`, then extract so that the object folders end up under `<INSPECTAI_DATA_ROOT>\visa\VisA\` and place the split file at `<INSPECTAI_DATA_ROOT>\visa\split_csv\1cls.csv`.
- The README documents no checksum. Before extracting, list the archive and reject it if any entry has an absolute path or contains `..`. After extracting, run `validate_dataset.py` (Section 3.3), and record the SHA-256 of the archive in the project documentation.
- Use only the links in the official README. Do not use third-party mirrors, whose contents and license labels can differ.
- Downloads happen only when the user explicitly asks for them. The application, scripts, and tests never download the dataset; `validate_dataset.py` explains where to place the data if it is missing (Section 11). A command-line download with a tool such as `curl.exe` from the official links is allowed only after the user approves it.

**Citations.** Cite the VisA paper: Zou, Jeong, Pemula, Zhang, and Dabeer, "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", European Conference on Computer Vision (ECCV) 2022, arXiv:2207.14315 (DOI 10.48550/arXiv.2207.14315). The title, authors, and venue were checked against the official repository README and the arXiv page on 2026-10-22; the ECCV proceedings page numbers were not verified.

Also cite PatchCore (Roth et al., CVPR 2022) and PaDiM (Defard et al., 2020) for the corresponding methods. Verify the details of these two before publishing.

### 3.2 Initial categories

All 12 VisA objects (Section 3.1) are supported by the loader. Start with a small, configurable subset:

- `cashew`
- `pipe_fryum`
- `chewinggum`
- `fryum`

These are single-instance objects with 500–503 normal images each in total, according to the official statistics (the training share is set by the split file). Their memory banks and fit times stay modest on a CPU. They are initial candidates, not a requirement to process every category immediately.

Start with `cashew` and `pipe_fryum` if their data are available and the pipeline supports them correctly.

The PCB, capsule, candle, and macaroni objects have about 600–1,000 normal images each. They may need the `dev` profile or `max_fit_images` (Section 5.5) to fit within the memory and time limits.

The dataset loader must support additional categories without requiring modifications to the model implementation.

### 3.3 Expected dataset structure

Support the VisA layout as distributed in the official archive, plus the official split file. The dataset lives outside the repository, under `INSPECTAI_DATA_ROOT` (see Section 9.2):

```text
<INSPECTAI_DATA_ROOT>/
└── visa/
    ├── split_csv/
    │   └── 1cls.csv
    └── VisA/
        ├── cashew/
        │   ├── Data/
        │   │   ├── Images/
        │   │   │   ├── Normal/
        │   │   │   └── Anomaly/
        │   │   └── Masks/
        │   │       └── Anomaly/
        │   └── image_anno.csv
        ├── pipe_fryum/
        └── ...
```

**Split file.** `1cls.csv` has the header `object,split,label,image,mask`. The `split` values are `train` and `test`, and the `label` values are `normal` and `anomaly`. The `image` and `mask` paths are relative to `visa/VisA/` (for example `candle/Data/Images/Normal/0836.JPG`). The `mask` column is empty for the normal rows seen at the top of the file; `validate_dataset.py` checks every row. In the official one-class setting the training split contains only normal images.

**Masks.** The official README states that masks for normal images are not stored; treat a normal image as having an all-zero mask. Read each stored mask as a single-channel image and treat every nonzero pixel as anomalous (the official script does the same). Multi-class mask IDs are not used. A mask whose height and width differ from its image is an error, as is a multi-channel mask.

**Annotation file.** The per-object `image_anno.csv` gives an image-level label and mask path per image. It is optional and is used only for defect-type information when present.

**Validation.** `validate_dataset.py` checks every row of the split file instead of assuming the layout. It confirms that:

- the files exist and the object name is one of the 12 supported objects;
- the `split` and `label` values are valid;
- training rows are all normal;
- anomalous test rows have a mask and normal rows have none.

It reports counts per object, split, and label, plus per-object image-size and aspect-ratio statistics. It does not assume any counts.

### 3.4 Data integrity requirements

- Fit the model only on the *fit split* defined in Section 3.5.
- Never use test images or test annotations to fit the model, select the threshold, or select hyperparameters.
- Do not silently replace missing or corrupted images.
- Report how many files were discovered, loaded, skipped, or rejected.

### 3.5 Data splits and threshold policy

The official one-class split provides a `train` split with only normal images and a `test` split with normal and anomalous images; it has no validation split. For each category, build the splits once with a fixed seed (default 42), save them to `split.json` inside the model artifact (relative file names plus a SHA-256 of the sorted file list), and reuse them for every later step.

| Split | Source | Used for |
|---|---|---|
| `fit` | Official `train` rows minus the calibration hold-out | Fitting the model / memory bank |
| `calibration_normal` | Random 10% of the official `train` rows (minimum 10 images) | Threshold selection |
| `test` | All official `test` rows | Final evaluation only |

Rules:

1. The splits are disjoint. This is enforced in code and covered by a test.
2. The final model is fit on `fit` only. Do **not** refit on `calibration_normal` after selecting the threshold, because that would change the score scale and invalidate the threshold.
3. Default threshold policy `normal_percentile`: threshold = the q-quantile (default q = 0.99) of image-level scores on `calibration_normal`. With about 20 hold-out images this is close to their maximum. State this in the report and report the resulting false-positive rate on the test normals. The pixel-level threshold used for IoU and Dice is derived the same way, from anomaly-map values on `calibration_normal`, never from test masks.
4. Optional policy `labeled_calibration`: split the official `test` rows once, stratified by label (normal or anomaly) and, when `image_anno.csv` provides a defect type, by defect type, into `test_calibration` and `test_eval` (50/50, fixed seed). Select the threshold (for example, maximum F1) on `test_calibration`. Report every threshold-dependent metric (precision, recall, F1, confusion matrix, IoU, Dice) on `test_eval` only, labeled as such.
5. Hyperparameters (backbone, layers, image size, coreset ratio) are fixed in advance from the defaults in Sections 5.5 and 9. Any sweep must use only the `labeled_calibration` policy and be reported on `test_eval`.
6. Threshold-free metrics (AUROC, AP, pixel AUROC) may be reported on the full `test` split only if no hyperparameter was chosen using test labels.
7. Thresholds are stored per model and per category. A single global threshold is not allowed, because score scales differ.

---

## 4. Technical Architecture

Implement a modular Python package with the following pipeline:

```text
Dataset
   |
   v
Image Validation and Preprocessing
   |
   v
Pretrained CNN Feature Extractor
   |
   v
Normal Feature Memory / Reference Model
   |
   v
Anomaly Scoring
   |
   +--------------------+
   |                    |
   v                    v
Image-Level Score    Pixel-Level Map
   |                    |
   v                    v
Threshold Decision   Heatmap and Overlay
   |                    |
   +----------+---------+
              |
              v
       Streamlit Interface
              |
              v
       Evaluation and Reports
```

Keep dataset loading, split construction, preprocessing, feature extraction, anomaly scoring, threshold selection, artifact storage, evaluation, and UI code in separate modules.

Use typed interfaces where practical, clear error messages, and explicit configuration values.

---

## 5. Models and Experimental Strategy

Implement the models in the following order.

### 5.1 Model A: Classical image-processing baseline

Implement a simple baseline using OpenCV and normal-image statistics.

Possible components:

- Consistent resizing and normalization.
- Grayscale or color statistics.
- Local texture or intensity differences.
- A configurable thresholding approach.

This is a reference baseline, not the primary model. Compute its normal-image statistics on the `fit` split only, and select its threshold per Section 3.5.

Document its limitations. Do not imply that a simple thresholding method should perform well across different industrial categories.

### 5.2 Model B: Gaussian feature model (PaDiM-inspired)

Use a frozen pretrained CNN available through PyTorch and torchvision (see Section 5.5). For each spatial position of the intermediate feature map, fit a multivariate Gaussian (mean and regularized covariance) over the `fit` split images, using a fixed random subset of d channels (default d = 100). The anomaly score at each position is the Mahalanobis distance. The anomaly map is the per-position score map, smoothed and resized to the input image size. The image-level score is the maximum of the map.

Requirements:

- Support CPU inference.
- Freeze the backbone.
- Extract intermediate spatial features rather than relying on final classification logits.
- Resize or align feature maps consistently.
- Store only the means, inverse covariances, and selected channel indices.
- Record the selected architecture, pretrained weights identifier, image dimensions, feature layers, d, and the covariance regularization epsilon in the configuration and the artifact manifest (Section 9.1).
- Document deviations from the PaDiM paper.

### 5.3 Model C: PatchCore-inspired anomaly detection

Implement a PatchCore-style method only after Model B is correct and tested.

Model C reuses the feature extractor from Model B. It differs by using a patch-level memory bank and nearest-neighbor scoring instead of a parametric Gaussian model.

The intended approach is:

1. Extract spatial features from the `fit` split images.
2. Construct a collection of normal reference feature vectors.
3. Optionally reduce the memory bank using a reproducible coreset selection method.
4. Compute nearest-neighbor distances between test features and reference features.
5. Convert these distances into a spatial anomaly map.
6. Aggregate spatial scores into an image-level score.
7. Resize the anomaly map to the evaluation image dimensions.

Use a CPU-compatible nearest-neighbor implementation. A simple, exact implementation is acceptable for a small dataset; optimize only when measurements justify it.

Do not claim full PatchCore equivalence if the implementation uses a simplified architecture, different feature aggregation, or an approximate scoring procedure. Document deviations from the reference method.

If a full implementation is too expensive on CPU, implement and evaluate a smaller PatchCore-inspired version first.

### 5.4 Model selection

The default final model should be the best method supported by reproducible evaluation results obtained under the protocol in Section 3.5 (never by tuning on the test split).

Do not automatically select the most complicated model.

The baseline must remain available even if the more advanced model performs worse.

### 5.5 Backbone, CPU, and memory budget

Reference hardware: Ryzen 5 5500U (6 cores / 12 threads), 16 GB RAM, no GPU.

Defaults (all configurable, see Section 9):

| Profile | Backbone | Layers | Input | Use |
|---|---|---|---|---|
| `dev` | `resnet18` | layer2, layer3 | 224×224 | Development, tests, demo |
| `full` | `wide_resnet50_2` | layer2, layer3 | 224×224 | Final reported results |

Pretrained weights come from the torchvision cache (`TORCH_HOME`). If they are not cached, raise `WeightsNotFoundError` explaining how to fetch them. Downloading happens only through an explicit action: `python scripts/fetch_weights.py`, or `INSPECTAI_ALLOW_WEIGHT_DOWNLOAD=1` for hosted deployment. Never fall back to random weights.

Rules:

- Memory-bank size estimate (Model C): N_fit_images × 784 patches × D dims × 4 bytes (for example, about 2.2 GB for 450 fit images with `full` before reduction). Model B instead accumulates running means and covariances and does not store all patches. Log the estimate before fitting and abort with a clear message if it exceeds `max_bank_gb` (default 4). To fit within the limit, lower `max_fit_images` (a seeded random subset of the `fit` split, recorded in `split.json`), use the `dev` profile, or increase the coreset reduction. Never reduce the fit set silently.
- Peak RAM during fitting should stay below 8 GB, leaving room for the OS and other applications. Measure it (process RSS) and report it.
- Run all inference under `torch.inference_mode()`. Set `torch.set_num_threads(cfg.num_threads)` explicitly (default 6, the physical core count). DataLoader `num_workers` defaults to 0 on Windows.
- Exact k-nearest-neighbor search uses chunked `torch.cdist` (configurable chunk size). Do not add faiss.
- Coreset reduction: greedy k-center on features randomly projected to 128 dimensions, seeded, default ratio 0.10. If fitting time exceeds `max_fit_minutes` (default 30), stop with a clear message suggesting random subsampling. Never fall back silently.
- Provisional latency targets, to be confirmed or revised at Milestone 3 from measurements: `dev` ≤ 1 s/image, `full` ≤ 3 s/image. If missed, document the measured values and do not claim the targets.
- Plug the laptop in and use a high-performance power plan when benchmarking, and record that in the report.

---

## 6. Image Processing with OpenCV

Use OpenCV for image operations, not as a replacement for the deep learning model.

Required functionality:

- Load supported image formats safely.
- Validate image dimensions and channels.
- Handle grayscale and RGB images consistently.
- Convert between OpenCV's BGR convention and the RGB convention used by Streamlit.
- Resize images using a consistent strategy.
- Generate heatmap visualizations.
- Overlay heatmaps on original images.
- Save prediction visualizations when requested.

Requirements:

- Avoid unintended distortion of image geometry.
- Preserve original images.
- Make image-size assumptions configurable.
- Reject unsupported or malformed inputs with understandable errors.
- Do not silently interpret a failed image read as an empty image.
- Resize mode is configurable per category: `stretch` (direct resize to a square of side `image_size`, with no center crop, so border defects are not cut off) or `letterbox` (pad to a square while keeping the aspect ratio, and remove the padding when mapping the anomaly map back to the original image). The official VisA README does not document image sizes, so `validate_dataset.py` reports per-object size and aspect-ratio statistics (Section 3.3). Choose the mode for each category from those measurements and record it in the artifact manifest. Normalize with ImageNet mean and standard deviation. Convert grayscale to three channels by replication.
- Uploaded images whose aspect ratio differs by more than 25% from the category's reference aspect ratio (the median of the `fit` images, stored in the artifact manifest) are rejected with a clear message.
- Smooth anomaly maps with a Gaussian filter (default sigma = 4, applied at the model input resolution and configurable), then resize them to the original image size.
- Cap image decoding: set `Image.MAX_IMAGE_PIXELS` and reject images over 4096×4096 before decoding them fully.

Create unit tests for color conversion, resizing, output dimensions, and overlay generation.

---

## 7. Evaluation Framework

Evaluation is a core deliverable, not an optional extra.

### 7.1 Image-level metrics

Calculate:

- AUROC for distinguishing normal and anomalous images.
- Average precision (AP) for image-level anomaly ranking.
- Precision, recall, and F1-score at a documented decision threshold.
- Confusion matrix.
- False-positive rate and false-negative rate at the selected threshold.

Use correct sample-level labels. Do not derive ground-truth image labels from model predictions.

If a metric is undefined because the evaluation subset contains only one class, report it as undefined rather than inventing a value.

Report 95% bootstrap confidence intervals (1000 resamples, fixed seed) for image AUROC, because each object's test set is small (the official statistics list 100 anomalous images per object). Report the number of normal and anomalous test images with every metric. Report per-category results first, and macro-average only across the categories actually evaluated.

### 7.2 Pixel-level metrics

Where pixel-level ground-truth masks exist, calculate:

- Pixel AUROC.
- Pixel-level average precision where appropriate.
- Intersection over Union (IoU) at a documented threshold.
- Dice coefficient at a documented threshold.

Exclude normal images without anomaly masks from metrics that require positive defect pixels only when methodologically appropriate. For pixel AUROC, normal images can contribute valid all-normal pixel labels.

Handle missing masks explicitly and document the policy. In VisA, masks are stored only for anomalous images (Section 3.3). Normal images are treated as all-normal, so they contribute to pixel AUROC; the official evaluation code likewise includes normal samples in localization metrics. The IoU and Dice thresholds follow Section 3.5, rule 3.

Pixel metrics are computed after resizing both anomaly maps (bilinear) and ground-truth masks (nearest-neighbor) to a common evaluation resolution (default 256×256, configurable). Do not compute ROC on full-resolution masks for all images at once. Record the resolution in the report.

### 7.3 Threshold selection

Threshold selection follows Section 3.5. Do not tune the decision threshold using the final test set.

Save the selected threshold, the policy name, its parameters, and `split.json` in the model artifact.

If no calibration data are available, use a configurable threshold and clearly label the decision as an *unvalidated default* rather than a validated optimum.

### 7.4 Performance measurements

Measure and report:

- Model fitting time.
- Inference latency per image.
- Images processed per second.
- Peak RAM (process RSS) during fitting and inference.
- Thread count and power mode used.
- Feature memory-bank size.
- Image resolution and hardware used.
- Number of images used in each experiment.

Do not fabricate performance numbers.

### 7.5 Comparison report

Compare at least two approaches on the same evaluation subset:

1. Classical OpenCV baseline.
2. Gaussian feature model (Model B).

Include the PatchCore-inspired model (Model C) when implemented successfully.

Report all comparison tables per category, with the same splits and the same evaluation resolution for every model.

PatchCore and PaDiM publish results on other benchmarks. Do not compare VisA numbers with those tables. Compare with published VisA results only when the protocol matches (official one-class split, localization metrics that include normal images), and state the protocol used.

Create:

- A machine-readable CSV of per-image predictions.
- A machine-readable JSON summary of experiment settings and aggregate metrics.
- A comparison table.
- A visualization of representative normal and anomalous cases.
- Heatmaps with ground-truth masks where available.

The report must distinguish measured results from planned experiments.

---

## 8. Streamlit Application

Build a polished interface suitable for a project demonstration or technical interview.

### Page 1: Home

Display:

- Project name and purpose.
- Brief explanation of anomaly detection.
- Selected model and supported categories.
- Instructions for uploading an image.
- Limitations and intended use.

### Page 2: Inspect an Image

Allow the user to:

1. Upload a JPG, JPEG, or PNG image.
2. Select the model and supported category where applicable.
3. Run inference.
4. View the original image.
5. View the anomaly heatmap.
6. View the heatmap overlay.
7. View the anomaly score and threshold-based decision.
8. Download the generated visualization and prediction details.

Use clear status labels, such as `Normal`, `Anomaly detected`, or `Model unavailable`.

The application must not show fabricated predictions when a model is missing or inference fails.

Additional requirements for this page:

- Show the score next to the category's threshold and the range (min/median/max) of the `calibration_normal` scores, so users can see how far the score is from normal.
- Show a warning: "Scores are only meaningful if the uploaded object matches the selected category."
- Maximum upload size is 10 MB (`server.maxUploadSize`), validated before inference.
- If the local dataset is available, allow choosing a test image from it instead of uploading one.

### Page 3: Evaluation Dashboard

Display evaluation metrics only when valid evaluation results exist.

Include:

- Image-level AUROC and average precision.
- Pixel-level metrics where ground-truth masks are available.
- Model comparison.
- Inference performance.
- Representative predictions.
- Confusion matrix where applicable.

Clearly label the dataset split used to calculate the displayed metrics.

### Page 4: About and Limitations

Explain:

- The difference between anomaly detection and supervised classification.
- How the model creates an anomaly map.
- Which pretrained weights and algorithms are used.
- Known failure cases.
- Dataset limitations.
- That the application is a research and educational prototype, not a certified industrial quality-control system.

### UI requirements

- Responsive layout.
- Clear progress indicators.
- Graceful error handling.
- No unnecessary external API calls.
- Cache model initialization and other expensive reusable operations appropriately.
- Avoid retraining the model on every Streamlit rerun.
- Validate file size and image dimensions before expensive processing.
- Do not expose local filesystem paths or internal stack traces to users.

---

## 9. Configuration and Reproducibility

Use a centralized configuration file or equivalent validated settings.

Support configuration of:

- Dataset root.
- Selected categories.
- Image resolution.
- Backbone architecture.
- Feature layers.
- Random seed.
- Batch size.
- Nearest-neighbor settings.
- Memory-bank reduction parameters.
- Inference threshold policy and its parameters (`normal_percentile` with q, or `labeled_calibration`).
- Calibration hold-out fraction and split seed.
- Profile (`dev` or `full`, see Section 5.5).
- Number of CPU threads, maximum memory-bank size (`max_bank_gb`), and maximum fitting time (`max_fit_minutes`).
- Evaluation resolution for pixel metrics.
- Output directory.
- Device selection.

Default device: CPU.

If automatic device selection is implemented, it must work correctly when CUDA is unavailable. Never require a CUDA installation.

Use deterministic seeds where practical and document any remaining sources of nondeterminism.

Record configuration, library versions, random seed, data split, and model settings with each experiment.

Do not include absolute paths specific to the developer's machine in committed files. Machine-specific locations are supplied through environment variables (Section 9.2).

### 9.1 Model artifact format

Each fitted model is a directory `<artifact_root>/<model>/<category>/` containing:

- `manifest.json`: `schema_version`, `model_type`, `category`, backbone name, weights identifier and SHA-256, layers, image size, feature dimension, tensor shapes and dtypes, resize mode and reference aspect ratio, seed, threshold policy and value, library versions, creation time, and the SHA-256 of each data file.
- `tensors.safetensors`: memory bank or Gaussian parameters (float32, or float16 if configured).
- `split.json`: the split from Section 3.5.
- `config.snapshot.yaml`: the exact configuration used.

Rules:

- Never use `pickle`, or `torch.load` without `weights_only=True`, for artifacts.
- On load, verify `schema_version`, checksums, shapes, dtype, absence of NaN/Inf, and that `split.json` has no overlap between `fit` and `test`. Raise a specific `ArtifactError` on any mismatch.
- Do not reuse a stale artifact if its config snapshot differs from the requested configuration.
- Artifacts are not committed to Git.

### 9.2 Paths and Windows/OneDrive

- The repository, virtual environment, dataset, `TORCH_HOME` cache, artifacts, and reports must **not** live in a OneDrive-synced folder. Layout:
  - Repository: `C:\dev\inspectai`
  - Virtual environment: `C:\dev\venvs\inspectai`
  - Data root (`INSPECTAI_DATA_ROOT`): `C:\dev\inspectai-data`, with the dataset at `C:\dev\inspectai-data\visa\` (the `VisA\` object folders and `split_csv\`)
  - Artifacts: `C:\dev\inspectai-data\artifacts\`; reports: `C:\dev\inspectai-data\reports\`; weights cache (`TORCH_HOME`): `C:\dev\inspectai-data\torch_cache\`
- Machine-specific locations are supplied through environment variables, never committed: `INSPECTAI_DATA_ROOT`, `INSPECTAI_ARTIFACT_DIR`, `INSPECTAI_REPORT_DIR`, `TORCH_HOME`. `configs/default.yaml` refers to them (for example `data_root: ${INSPECTAI_DATA_ROOT}`).
- If a user places these in a synced folder anyway, the README must document pausing sync or excluding those directories.
- Use `pathlib` everywhere. Tests must include paths containing spaces and non-ASCII characters.
- Scripts that use multiprocessing must use an `if __name__ == "__main__":` guard.
- Troubleshooting entry: if a "libiomp5md.dll already initialized" error appears, fix duplicate OpenMP runtimes first, and use `KMP_DUPLICATE_LIB_OK=TRUE` only as a documented last resort.

---

## 10. Project Structure

Use a structure similar to the following. Modify it only when there is a concrete architectural reason.

```text
inspectai/
├── app.py
├── README.md
├── SPEC.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml
├── src/
│   └── inspectai/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── splits.py
│       │   └── validation.py
│       ├── artifacts/
│       │   ├── __init__.py
│       │   └── store.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── images.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── classical.py
│       │   ├── features.py
│       │   ├── gaussian.py
│       │   └── patchcore.py
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predictor.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── thresholds.py
│       │   └── report.py
│       └── visualization/
│           ├── __init__.py
│           └── heatmaps.py
├── scripts/
│   ├── fetch_weights.py
│   ├── validate_dataset.py
│   ├── fit_model.py
│   ├── evaluate_model.py
│   └── run_demo.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│       └── make_synthetic.py
├── data/
│   └── README.md
├── artifacts/
│   └── README.md
├── reports/
│   └── README.md
└── docs/
    ├── architecture.md
    ├── evaluation.md
    └── limitations.md
```

The tree root is the repository root, `C:\dev\inspectai` (Section 9.2). The `data/`, `artifacts/`, and `reports/` folders in the repository contain only a `README.md` that explains where the real locations are (`INSPECTAI_DATA_ROOT`, `INSPECTAI_ARTIFACT_DIR`, `INSPECTAI_REPORT_DIR`). Datasets, artifacts, and generated reports are not stored in the repository.

`pyproject.toml` is the single source of truth for dependencies. A `requirements.txt` exists only if generated for hosting (Section 15). There is no `.env.example`, because the project uses no secrets.

The `src/inspectai/` layout must be installable as a Python package. Configure `pyproject.toml` accordingly.

Do not create empty modules merely to match this tree. Every implemented module must have a defined responsibility.

Do not commit datasets, large generated artifacts, caches, virtual environments, or secrets.

---

## 11. CLI Commands

Provide clear commands for the main workflow.

The exact command syntax can be finalized during implementation, but the following capabilities must exist:

```powershell
python scripts/fetch_weights.py --backbone resnet18
python scripts/validate_dataset.py
python scripts/fit_model.py --model patchcore --category cashew --profile dev
python scripts/evaluate_model.py --model patchcore --category cashew
streamlit run app.py
pytest
```

Adapt arguments to the implemented CLI and document the final commands in the README.

Exit codes: 0 success; 2 configuration or data error; 3 artifact error; 4 pretrained weights unavailable; 1 any other failure.

Requirements:

- Missing dataset errors must explain where to place the data.
- Model fitting must save reusable artifacts.
- Evaluation must load the saved model rather than refit on test data.
- Evaluation output must include machine-readable results.
- Inference must not require the original training process to run again.
- Commands return the exit codes listed above.
- Use logging rather than scattered debug prints.

Provide a small synthetic-data or fixture-based smoke test so that basic functionality can be checked without downloading VisA. `scripts/run_demo.py` runs this end-to-end smoke path (generate synthetic data, fit with the fake backbone, predict, write a report) and is the only purpose of that script.

---

## 12. Testing Requirements

Use `pytest`.

### Unit tests

Test:

- Configuration validation.
- Dataset discovery.
- Missing files and malformed images.
- Image color conversion.
- Image resizing.
- Ground-truth mask loading.
- Model input and output shapes.
- Anomaly-map resizing.
- Image-level score aggregation.
- Threshold-based classification.
- Visualization generation.
- CPU-only inference.

Metric, split, threshold, and artifact tests are listed under "Additional required tests" below.

### Integration tests

Test:

- Loading a small fixture dataset.
- Fitting the classical baseline and Model B (with the fake backbone).
- Saving and reloading model artifacts.
- Producing a prediction.
- Producing a heatmap.
- Generating an evaluation report.
- Running the inference pipeline without the full dataset.

### Test constraints

- Ordinary tests must not download model weights or datasets.
- Do not run expensive model training during the normal unit-test suite.
- Mock external downloads and isolate filesystem operations.
- Test CPU behavior explicitly.
- Ensure tests fail when predictions, masks, or metrics have invalid shapes or values.
- Do not weaken or remove tests simply to make the suite pass.
- Do not report tests as passing unless they were actually executed.
- The feature extractor must accept an injected backbone. Tests use a small deterministic `FakeBackbone` (random-seeded conv layers) and never touch torchvision weights.
- Provide `tests/fixtures/make_synthetic.py`, which generates a small VisA-style tree in `tmp_path`: `Data/Images/Normal` and `Data/Images/Anomaly` folders, masks for the anomalous images only, and a matching `1cls.csv` (columns `object,split,label,image,mask`, with an empty `mask` for normal rows). Normal images are textured; anomalous images have drawn defects. Do not commit VisA images.
- Markers: `slow` (real weights or real dataset) and `needs_dataset`. Default `addopts = -m "not slow and not needs_dataset"`.

Additional required tests:

- Splits: `fit`, `calibration_normal`, and `test` are disjoint; splits are reproducible for the same seed and differ for another seed.
- The `normal_percentile` policy never receives test labels or test scores.
- A manifest listing any test file in `fit` is rejected.
- Artifact integrity: wrong checksum, wrong shape, NaN values, unknown `schema_version`, and a pickle file are each rejected with `ArtifactError`.
- Metrics: single-class subsets return "undefined"; known toy inputs give known AUROC, AP, IoU, and Dice values.
- Paths containing spaces and non-ASCII characters.
- Split-file parsing: the header and column order, valid and invalid `split` and `label` values, a training row labeled anomalous (rejected), a row whose file is missing (reported), an unsupported object name (rejected), and an anomalous test row without a mask (reported).
- Masks: nonzero pixels become anomalous, normal images get an all-zero mask, and a multi-channel mask or an image/mask size mismatch raises an error.
- Resize modes: `stretch` and `letterbox` give maps that map back to the original image size, and the reference aspect-ratio check accepts and rejects the expected cases.

Run the complete test suite after meaningful changes and report the results.

---

## 13. Code Quality and Engineering Standards

The codebase must follow these standards:

- Python type hints for public functions.
- Docstrings for important classes and public APIs.
- Clear separation of concerns.
- Small, testable functions.
- Consistent naming.
- Explicit exception handling.
- No hard-coded machine-specific paths.
- No secrets in source control.
- No silent fallback to random model weights.
- No fake model outputs or hard-coded metrics.
- No unexplained suppression of warnings.
- No broad exception handling that hides failures.
- No unnecessary dependencies.

Use a formatter and linter such as Ruff if appropriate.

Pin or constrain dependencies in a way that supports reproducible installation on Python 3.12. Verify that selected package versions are compatible before adding them.

Avoid installing multiple overlapping deep learning frameworks.

Core dependencies (verify versions for Python 3.12 at Milestone 0 and record them in a lock file): `torch` and `torchvision` (CPU wheels from `https://download.pytorch.org/whl/cpu`), `numpy`, `opencv-python-headless`, `pillow`, `scikit-learn` (metrics only), `safetensors`, `psutil`, `pyyaml`, `streamlit`. Development: `pytest`, `ruff`. Do not install both `opencv-python` and `opencv-python-headless`.

---

## 14. Documentation Requirements

### README.md

Include:

1. Project overview.
2. Key features.
3. Architecture diagram.
4. Technologies used.
5. Installation instructions for Windows PowerShell, including the environment variables and the non-OneDrive locations from Section 9.2.
6. Dataset download and setup instructions.
7. Training and evaluation commands.
8. Instructions for running Streamlit.
9. Example outputs. Any figure derived from VisA images carries the attribution required by Section 3.1.
10. Actual evaluation results, when available.
11. Hardware and runtime information.
12. Limitations.
13. Troubleshooting.
14. Dataset attribution and licensing notes (CC BY 4.0 attribution and the citation in Section 3.1).

### Architecture documentation

Explain:

- Data flow.
- Feature extraction.
- Data splits and threshold policy.
- Gaussian (Mahalanobis) scoring for Model B.
- Reference feature memory.
- Nearest-neighbor scoring.
- Anomaly-map generation.
- Image-level score aggregation.
- Model persistence.
- UI integration.

### Evaluation documentation

Explain the evaluation metrics, data splits, threshold selection, comparison methodology, and reproducibility procedure.

Document all deviations from the intended algorithms.

---

## 15. Deployment

Deployment is optional and is attempted only after the local application (Milestone 6) works.

Prepare the application for deployment on Streamlit Community Cloud if practical. The hosted profile uses `dev`, a 5% coreset, and float16 tensors. Pretrained weights download at runtime only with `INSPECTAI_ALLOW_WEIGHT_DOWNLOAD=1`. Free hosting has limited memory, so a documented local-only mode is an acceptable final state.

Requirements:

- No mandatory paid APIs.
- No dependency on local absolute paths.
- No requirement to upload the full dataset to the deployment environment.
- A clear strategy for model artifacts.
- A small, manageable default model configuration.
- Graceful behavior if the model artifact is unavailable.
- Clear installation instructions and dependency management.

Do not assume that model weights, memory banks, or the full dataset will fit within a hosting provider's storage or memory limits.

If the complete model is too large for free deployment, provide a lightweight demo configuration or document a local-only mode.

Deployment must not be claimed successful until the deployed application has actually been tested.

---

## 16. Milestone-Based Implementation Plan

Work incrementally. Complete and verify each milestone before moving to the next.

### Milestone 0 — Repository inspection and setup

Tasks:

- Inspect the current state before changing anything: whether `C:\dev\inspectai` already exists and, if so, its code, dependencies, tests, and Git state. Do not overwrite unrelated work.
- If it does not exist, create the repository at `C:\dev\inspectai` (outside OneDrive) and run `git init` there. Copy this `SPEC.md` into it. Commits remain subject to the Git policy in Section 17.
- Create the virtual environment at `C:\dev\venvs\inspectai`, install the CPU-only PyTorch wheels, and set the environment variables from Section 9.2.
- Create or update project configuration.
- Add `.gitignore`, README scaffolding, and the package structure.
- Verify that the package imports correctly.
- Record the Python version, installed package versions, and hardware (CPU, RAM) in the project documentation.

Acceptance criteria:

- Installation instructions are documented.
- The repository and virtual environment are outside OneDrive.
- `python -c "import inspectai"` works in the selected environment.
- The Git working tree is understood.
- No dataset or model artifacts are accidentally tracked.

### Milestone 1 — Dataset and preprocessing

Tasks:

- Implement dataset discovery.
- Parse and validate the VisA split file and directory layout (Section 3.3).
- Measure per-object image sizes and aspect ratios, and choose each category's resize mode from them (Section 6).
- Load images and masks.
- Implement image preprocessing.
- Add unit tests using temporary fixtures.
- Add a dataset validation command.
- Implement reproducible splits per Section 3.5 (`fit`, `calibration_normal`, `test`) and save `split.json`.

Acceptance criteria:

- Dataset paths are configurable through environment variables.
- Splits are disjoint and reproducible for a fixed seed, and a test enforces this.
- Missing and malformed files are handled clearly.
- Image and mask dimensions are validated.
- Tests pass without the real dataset.

### Milestone 2 — Evaluation harness and classical baseline

Tasks:

- Implement image-level and pixel-level metrics (Section 7), including undefined-metric handling and bootstrap confidence intervals.
- Implement threshold selection per Section 3.5 (`normal_percentile`, and optionally `labeled_calibration`).
- Implement the CSV/JSON report writer.
- Implement a simple OpenCV-based baseline.
- Produce an anomaly score and spatial map where feasible.
- Add prediction visualization.
- Implement a baseline evaluation path using the harness.

Acceptance criteria:

- Metrics are computed from actual predictions and labels, and tested on toy inputs with known answers.
- The threshold is selected without test labels or test scores.
- The baseline runs on CPU.
- Predictions are reproducible.
- The output shapes are correct.
- The baseline is measurable rather than merely visual.

### Milestone 3 — Feature extractor, Model B, and artifact store

Tasks:

- Load a pretrained CNN backbone (with an injectable backbone for tests).
- Extract spatial feature representations.
- Implement Model B (Gaussian feature model, Section 5.2).
- Generate image-level scores and pixel-level anomaly maps.
- Implement the artifact store (Section 9.1): save, load, and integrity checks.
- Measure fitting time, peak RAM, and per-image latency, and confirm or revise the provisional targets in Section 5.5.

Acceptance criteria:

- No training of the full CNN from scratch.
- CPU inference works.
- Feature extraction is tested with the fake backbone.
- Inference results are not hard-coded.
- Missing pretrained weights produce a clear, actionable error (`WeightsNotFoundError`).
- Corrupted or mismatched artifacts are rejected with `ArtifactError`.
- Measured resource numbers are recorded, not assumed.

### Milestone 4 — PatchCore-inspired model

Tasks:

- Implement normal feature memory.
- Implement nearest-neighbor scoring.
- Add configurable memory-bank reduction.
- Generate anomaly maps.
- Compare against the classical baseline and Model B using the Milestone 2 harness.

Acceptance criteria:

- The implementation matches its documented algorithm.
- CPU resource usage is measured.
- Model configuration is reproducible.
- The comparison uses the same splits and evaluation resolution as the other models.

If the advanced method proves too expensive, document the bottleneck and deliver a smaller working version before optimizing.

### Milestone 5 — Comparison reports and additional categories

Tasks:

- Run all implemented models on the same splits and generate CSV predictions and JSON summaries with the Milestone 2 harness.
- Generate per-category comparison tables, confidence intervals, and visualization examples.
- Extend to additional categories from Section 3.2 as time and resources allow.

Acceptance criteria:

- Comparison tables for at least two categories use identical splits and the same evaluation resolution for every model.
- Metrics are computed from actual predictions and labels.
- Test data do not influence fitting or threshold selection.
- Undefined metrics are handled correctly.
- Results are reproducible and include configuration metadata.

### Milestone 6 — Streamlit application

Tasks:

- Implement image upload and validation (size, dimensions, aspect ratio; Sections 6 and 8).
- Load fitted model artifacts through the artifact store, with integrity checks.
- Display predictions, scores, per-category thresholds, calibration ranges, heatmaps, and overlays.
- Build an evaluation dashboard that reads actual report files.
- Add user-friendly errors and progress indicators.

Acceptance criteria:

- The application works locally.
- The decision uses the category's stored threshold, and the wrong-category warning is shown.
- A valid image produces an actual model prediction.
- Missing artifacts do not generate fake results.
- Large or malformed inputs are handled safely.
- UI tests or testable application logic cover critical behavior.

### Milestone 7 — Final verification and delivery

Tasks:

- Run all tests.
- Run formatting and lint checks.
- Run a small end-to-end smoke test.
- Review the Git diff for unrelated changes or secrets.
- Update README and architecture documentation.
- Record actual experimental results.
- Verify the application startup command.
- Provide a concise final project report.
- Optional: attempt deployment per Section 15, only after the local application works.

Acceptance criteria:

- All required checks pass, or remaining failures are explicitly documented.
- The end-to-end path is verified.
- No fabricated metrics are included.
- Setup and reproduction instructions are complete.
- Remaining limitations and unfinished tasks are disclosed.

---

## 17. Claude CLI Working Protocol

Claude must follow these rules throughout implementation.

### Before making changes

1. Inspect the repository (if it exists yet) and its current Git status.
2. Read the existing README, SPEC.md, configuration, and relevant source files.
3. Identify the current milestone and any unfinished work.
4. Propose a short implementation plan.
5. Avoid rewriting existing working code without a reason.

### During implementation

1. Work on one milestone at a time.
2. Make small, coherent changes.
3. Run relevant tests after each meaningful change.
4. Investigate failures before attempting workarounds.
5. Do not weaken tests to obtain a passing result.
6. Do not fabricate model outputs, evaluation metrics, or successful command results.
7. Do not silently download large datasets or model weights.
8. Ask before introducing a paid service, changing project scope, deleting user data, or performing consequential Git operations.
9. Keep documentation synchronized with actual behavior.
10. Maintain a short milestone checklist in the project documentation.
11. Report the expected duration before starting any run likely to exceed 5 minutes.
12. Use PowerShell syntax in documentation examples.

### Before declaring a milestone complete

Report:

- What was implemented.
- Files created or modified.
- Commands executed.
- Actual test results.
- Known limitations.
- Acceptance criteria that passed.
- Acceptance criteria that remain incomplete.
- The next recommended milestone.

Do not proceed to the next milestone if a critical acceptance criterion fails without documenting and resolving or explicitly deferring it.

### Git policy

Do not create commits or push to a remote unless explicitly authorized. Preserve unrelated changes in the working tree. Do not discard user changes.

---

## 18. Definition of Done

The project is complete when all of the following are satisfied:

- [ ] A working, documented dataset loader exists.
- [ ] The classical baseline runs.
- [ ] Model B (pretrained-CNN Gaussian feature model) runs.
- [ ] An anomaly heatmap and overlay are generated.
- [ ] A normal-reference training or fitting procedure exists.
- [ ] At least two methods are evaluated and compared.
- [ ] Image-level metrics are computed correctly.
- [ ] Pixel-level evaluation works where annotations exist.
- [ ] Thresholds are stored per category and selected without test-set leakage (Section 3.5).
- [ ] Model artifacts can be saved and reloaded, and are loaded only after integrity checks (no pickle).
- [ ] The Streamlit application performs real inference.
- [ ] CPU-only execution is supported.
- [ ] Automated unit and integration tests pass.
- [ ] README and architecture documentation are complete.
- [ ] Experimental results are based on actual executions.
- [ ] Known limitations are clearly documented.
- [ ] Git contains no datasets, secrets, or unnecessary large artifacts.
- [ ] Splits are disjoint and reproducible; a test enforces this.
- [ ] Fit time, peak RAM, bank size, and per-image latency are measured and reported.
- [ ] Results include bootstrap confidence intervals and sample counts.
- [ ] Repository, virtual environment, dataset, and artifacts are outside OneDrive.

PatchCore-inspired functionality may be marked as a documented, deferred enhancement only if a concrete resource or implementation blocker prevents it. The core application, Model B (Gaussian), and the classical baseline must still work and be evaluated. Deployment (Section 15) is optional and is not required for completion.

---

## 19. Final Expected Deliverable

Deliver a repository that another developer can clone, configure, test, and run by following the README.

The final report must summarize:

1. The problem and proposed solution.
2. Implemented models and architecture.
3. Dataset categories and data split.
4. Actual evaluation results.
5. Baseline comparison.
6. CPU inference performance, fitting time, and peak RAM.
7. Representative normal, defective, and heatmap examples.
8. Test results.
9. Deployment status.
10. Limitations and potential future improvements.

The goal is not to claim perfect defect detection. The goal is to build a scientifically evaluated, reproducible computer vision system that demonstrates practical deep learning and sound software engineering.

**Start with Milestone 0. Inspect the current state (including whether `C:\dev\inspectai` already exists) and report it before making changes. Do not implement the entire project in one unverified pass.**
