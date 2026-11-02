# Troubleshooting

- **`import inspectai` fails:** activate the virtual environment and run `pip install -e ".[dev]"` from `C:\dev\inspectai`.
- **"libiomp5md.dll already initialized":** fix duplicate OpenMP runtimes first. Use `KMP_DUPLICATE_LIB_OK=TRUE` only as a last resort.
- **Environment variables not visible:** open a new terminal after setting them.
- **`WeightsNotFoundError` (exit code 4):** run `python scripts/fetch_weights.py --backbone resnet18` once; the weights are cached under `TORCH_HOME`.
- **"The memory bank would need about N GB ... above max_bank_gb" (exit code 2):** lower `--max-fit-images` (a recorded, seeded subset), use the `dev` profile, or raise `--max-bank-gb` if you have the RAM.
- **"Coreset selection exceeded the time limit" / "exceeded max_fit_minutes" (exit code 2):** use `--max-fit-images` or a smaller `--coreset-ratio`, or raise `--max-fit-minutes` and wait (a default fit takes about 11 minutes on the reference laptop; `--index-backend tiled` is faster).
- **The app says `Model unavailable`:** the selected model has no saved artifact for that category, the artifact failed its integrity checks, or the pretrained weights are not cached. The message names the fix (`fit_model.py` or `fetch_weights.py`); use "Try loading again" afterwards. Details are in the app's console log, never in the browser.
- **The app cannot find any model or result:** check that `INSPECTAI_DATA_ROOT`, `INSPECTAI_ARTIFACT_DIR`, and `INSPECTAI_REPORT_DIR` are set in the terminal that runs `streamlit run app.py`.
- **`ArtifactError` (exit code 3):** the artifact is missing, corrupted, or stale. Run `scripts/fit_model.py` again with the same settings you evaluate with.
