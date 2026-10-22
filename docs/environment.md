# Environment record

Recorded for Milestone 0 on 2026-10-22 and re-checked for Milestone 7 on 2026-10-23 (`pip check` reports no broken requirements, and `requirements-lock.txt` matches the installed packages exactly). Update this file when the environment changes.

## Hardware and operating system

| Item | Value |
|---|---|
| CPU | AMD Ryzen 5 5500U, 6 cores / 12 threads |
| RAM | 15.3 GB reported by Windows (16 GB installed) |
| GPU | None used (integrated AMD Radeon graphics; no CUDA) |
| OS | Microsoft Windows 11 Home Single Language 10.0.26200 |

## Python and key packages

| Item | Version |
|---|---|
| Python | 3.12.2 (`C:\dev\venvs\inspectai`) |
| pip | 26.2.2 |
| setuptools | 78.1.0 |
| torch | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| numpy | 2.5.2 |
| opencv-python-headless | 5.0.0.93 |
| pillow | 12.3.0 |
| scikit-learn | 1.9.1 |
| safetensors | 0.8.0 |
| psutil | 7.2.2 |
| PyYAML | 6.0.3 |
| streamlit | 1.66.1 |
| pytest | 9.1.1 |
| ruff | 0.17.1 |

The full pinned list is in `requirements-lock.txt`. PyTorch was installed from the CPU wheel index (`https://download.pytorch.org/whl/cpu`); `torch.cuda.is_available()` returns `False`.

## Conditions of the reported measurements

| Item | Value |
|---|---|
| Power | plugged in, Windows **Balanced** power plan (SPEC.md Section 5.5 recommends High performance, so the times are probably conservative); recorded in every `summary.json` and artifact manifest |
| Threads | 6 (the physical core count) for PyTorch inference; `batch_size` 8 |
| Cached pretrained weights | `resnet18-f37072fd.pth` and `wide_resnet50_2-95faca4d.pth` under `TORCH_HOME`, hash-verified |
| Python launcher | `py -3.12`; virtual environment outside OneDrive |

The file `requirements-lock.txt` is the output of `pip freeze` for this environment (the editable install of the project itself aside).

## Locations (all outside OneDrive)

| Purpose | Path |
|---|---|
| Repository | `C:\dev\inspectai` |
| Virtual environment | `C:\dev\venvs\inspectai` |
| Data root (`INSPECTAI_DATA_ROOT`) | `C:\dev\inspectai-data` |
| Artifacts (`INSPECTAI_ARTIFACT_DIR`) | `C:\dev\inspectai-data\artifacts` |
| Reports (`INSPECTAI_REPORT_DIR`) | `C:\dev\inspectai-data\reports` |
| PyTorch weights cache (`TORCH_HOME`) | `C:\dev\inspectai-data\torch_cache` |

## Dataset provenance

VisA was downloaded and verified outside the repository. Archive and split-file hashes and the pinned repository commit are recorded in `C:\dev\inspectai-data\visa\PROVENANCE.txt`.
