# Limitations, unfinished work, and future improvements

InspectAI is a research and educational prototype. It is **not** a certified industrial
quality-control system and must not be used for safety-critical decisions. This document lists what
the system does not do, what was measured only partially, and what was left unfinished, so that no
claim in the README is read more broadly than the evidence allows.

## 1. What the results do and do not show

- **Seven of twelve VisA objects were evaluated** (`cashew`, `pipe_fryum`, `chewinggum`, `fryum`,
  and later `candle`, `pcb1`, `pcb2`). Nothing is claimed for `capsules`, `macaroni1`, `macaroni2`,
  `pcb3`, and `pcb4`. `candle`, `pcb1`, and `pcb2` have about 900 normal training images; all five
  configurations were fitted on the same seeded subset of 400 of their ~810 fit images
  (`--max-fit-images 400`, SPEC.md Section 5.5), because the `full` Model C memory bank and coreset
  time would otherwise exceed the default limits. Their numbers describe that setting and may differ
  for a fit on all images.
- **Small test sets.** Each category has 50-100 normal and 100 anomalous test images, so every metric
  carries sampling uncertainty (the reports give 95% bootstrap intervals). Differences of a few AUROC
  points should not be over-read. One seed (42) and one machine were used; there is no multi-seed
  study.
- **The model ranking depends on the category.** The deep models rank images much better than the
  classical baseline in the first four categories, but not in the added ones: on `pcb2` the baseline
  has the highest image AUROC (0.919) and beats Model B (paired intervals exclude 0), and on `candle`
  and `pcb1` only Model C's advantage is demonstrated in some profiles. Model C is not consistently
  better than Model B: its paired AUROC advantage excludes zero in 10 of 14 same-profile comparisons
  and is absent on `chewinggum`, on `fryum` `dev`, and on `pcb1` `dev`.
  The `full` profile is not always better than `dev`. No default model has been declared the winner.
- **Hyperparameters were fixed in advance** and not tuned; better settings may exist, and any tuning
  would have to follow the leakage rules in [evaluation.md](evaluation.md).
- **Published VisA numbers are not compared** because their protocols differ.

## 2. Technical limitations

- **Thresholds are conservative and noisy.** Each comes from the 0.99 quantile of only about 45
  held-out normal scores, so false alarms are rare but recall at the threshold is moderate (0.06 to
  0.90 depending on model and category). F1 at the threshold does not follow ranking quality
  (on `fryum` the baseline's F1 exceeds Model B `dev`'s although Model B ranks images better). Compare
  models by AUROC and paired intervals, not by F1.
- **Defect localisation is coarse.** Maps come from a 28x28 feature grid upsampled to the image
  size; pixel IoU is 0.05 to 0.30 for every model. Small, faint, or low-contrast defects are often
  missed, and heatmaps indicate regions, not exact outlines.
- **Scores are not probabilities and not comparable across models or categories.** The app states
  this next to every result. A score for an object of the wrong category is meaningless.
- **Per-category models.** There is no single model for all objects, and a model fitted on one
  object's images cannot be reused for another.
- **The classical baseline** works only when the object is positioned consistently; it fails on
  objects that move.
- **Input handling.** Uploads must be JPEG or PNG, between 32 and 4096 pixels per side, up to 10 MB,
  with an aspect ratio within 25% of the category's. EXIF orientation is ignored (to match the
  stored masks), so a photo that relies on an EXIF rotation tag appears unrotated.
- **Cost.** A Model C fit takes about 11 minutes per category on the reference CPU (the greedy
  coreset dominates) and the `full` profile peaks at about 2.9 GB of RAM; inference takes 14 ms to
  0.65 s per image depending on the model. These were measured on one machine on the Balanced power
  plan, so absolute times will differ elsewhere.
- **Determinism.** Refits on the same machine with the same thread count and batch size reproduced
  stored tensors bit for bit; a different thread count or batch size can change floating-point
  summation order (and, in principle, near-tied coreset choices).
- **Dataset.** VisA is an industrial-style benchmark with controlled imaging, not data from a
  production line, and includes food items as well as PCBs. Results do not transfer to other cameras,
  lighting, or products without re-evaluation.

## 3. Unfinished or not attempted

| Item | Status |
|---|---|
| Deployment (SPEC.md Section 15) | **Not attempted.** It is optional; the project has no remote repository, and hosting was not tried. The documented local-only mode is the final state. The hosted profile (`dev` profile, 5% coreset, float16 tensors) is not implemented (the store supports float16, the models write float32). |
| Visual check of the app in a browser | **Not done.** The browser extension was unavailable in the development environment. The app is tested in-process with Streamlit's `AppTest` against synthetic and real artifacts, and `streamlit run app.py` starts and passes its health check, but the on-screen layout and responsiveness have not been reviewed. |
| Other five VisA objects | Not run (see above). |
| Added categories fitted on all images | `candle`, `pcb1`, and `pcb2` were fitted on 400 of ~810 images; fits on all images were not run. |
| Device selection | CPU only by design; `configs/default.yaml` accepts only `device: cpu` and the code never moves tensors to a GPU. |
| Defect-type stratification | The `labeled_calibration` policy can stratify by defect type, but `image_anno.csv` is not parsed, so only the label is used. No reported result uses this policy. |
| Power plan | Benchmarks ran on Windows **Balanced**, not High performance, as SPEC.md recommends. |
| Evaluation of the baseline from its artifact | `evaluate_model.py` refits the classical baseline (8 s) instead of loading its artifact; the artifact is used by the app and reproduces the evaluation scores exactly. |
| Static type checking and CI | Public functions are annotated, but no type checker is configured (the code carries `# type: ignore` comments for untyped JSON access), and there is no continuous-integration configuration. |

## 4. Possible future work

- **Better thresholds:** more calibration images per category, cross-validated quantiles, or a
  separately evaluated calibration step (isotonic regression) so scores could responsibly be shown as
  probabilities; defect-type-stratified labeled calibration using `image_anno.csv`.
- **Broader evaluation:** the remaining five objects, fits of the added categories on all images, several seeds, and an unseen-category check.
- **Faster Model C:** a cheaper coreset selection or an approximate nearest-neighbor index (the exact
  search was kept for clarity and correctness, as SPEC.md asks), and the float16 bank for hosting.
- **Finer localisation:** higher-resolution feature maps or multi-scale scoring.
- **Deployment:** a hosted demo with the lightweight profile, after a browser-based UI review.
- **Engineering:** a type checker and CI, and loading the
  classical artifact in `evaluate_model.py`.
