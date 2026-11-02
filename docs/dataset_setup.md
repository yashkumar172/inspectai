# Dataset download and setup

The VisA dataset is **not** part of this repository and no script downloads it; downloading is something you do yourself, from the official links only. It is released under CC BY 4.0 (see the attribution in the [README](../README.md#dataset-attribution-and-licensing)). The official source is <https://github.com/amazon-science/spot-diff>.

1. **Disk space.** You need the archive size plus a margin (about 1.93 GB + 1 GB) and about 1.92 GB for the extracted images.
2. **Download the archive** from the link in the official README (do not use mirrors) into a staging folder outside the repository and OneDrive:

   ```powershell
   New-Item -ItemType Directory -Force C:\dev\inspectai-data\downloads | Out-Null
   curl.exe -L -o C:\dev\inspectai-data\downloads\VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
   ```

3. **Verify it.** The archive was 1,929,840,640 bytes with SHA-256 `2EB8690C803AB37DE0324772964100169EC8BA1FA3F7E94291C9CA673F40F362` when this project was built (the official README documents no checksum):

   ```powershell
   (Get-Item C:\dev\inspectai-data\downloads\VisA_20220922.tar).Length
   Get-FileHash -Algorithm SHA256 C:\dev\inspectai-data\downloads\VisA_20220922.tar
   tar -tf C:\dev\inspectai-data\downloads\VisA_20220922.tar | Select-String -Pattern '^/|\.\.' | Select-Object -First 3   # must print nothing
   ```

4. **Extract** so that the object folders end up under `visa\VisA\` (the archive has no top-level `VisA` folder; it also contains `split_csv\` and `LICENSE-DATASET`):

   ```powershell
   New-Item -ItemType Directory -Force C:\dev\inspectai-data\visa\VisA | Out-Null
   tar -xf C:\dev\inspectai-data\downloads\VisA_20220922.tar -C C:\dev\inspectai-data\visa\VisA
   ```

5. **Place the split file** where the loader expects it. The archive's own copy is identical to the official one (SHA-256 `A48557E6033318CB90556F706196BC9D247A776A23EA51AECEE5A80DD0332995`, repository commit `2a692ab575001cbde74d402d897a7286086c6199`):

   ```powershell
   New-Item -ItemType Directory -Force C:\dev\inspectai-data\visa\split_csv | Out-Null
   Copy-Item C:\dev\inspectai-data\visa\VisA\split_csv\1cls.csv C:\dev\inspectai-data\visa\split_csv\1cls.csv
   ```

6. **Validate** (expect 10,821 rows and 0 errors), then delete the archive:

   ```powershell
   python scripts/validate_dataset.py
   Remove-Item C:\dev\inspectai-data\downloads\VisA_20220922.tar
   ```

The loader reads `1cls.csv` directly and never copies images, so the official `prepare_data.py` is not needed. The resulting layout is:

```text
C:\dev\inspectai-data\visa\
├── split_csv\1cls.csv
└── VisA\<object>\Data\Images\{Normal,Anomaly}\... and Data\Masks\Anomaly\...
```

## Pretrained weights

The deep models use torchvision's ImageNet weights, which are fetched only on request and cached under `TORCH_HOME`:

```powershell
python scripts/fetch_weights.py --backbone resnet18          # 45 MB, the `dev` profile
python scripts/fetch_weights.py --backbone wide_resnet50_2   # 132 MB, the `full` profile
```

Without cached weights, the scripts and the app stop with an explanation (exit code 4) and never use random weights.
