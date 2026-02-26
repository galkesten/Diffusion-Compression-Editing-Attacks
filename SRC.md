# Source Code Architecture and Usage

This document describes the `src/` module: its architecture, file roles, and how to run experiments.

## Overview

The codebase evaluates **image compression algorithms** (JPEG, DDCM, Turbo-DDCM, Robust Turbo-DDCM) under a **noisy channel model**: bit flips are applied to compressed bitstreams, then decompression is attempted. Metrics (PSNR, NIQE, LPIPS, patch FID) measure reconstruction quality.

**Typical workflow:**
1. **Find compression params** — Determine algorithm parameters (e.g. JPEG quality, DDCM/Turbo M) to hit a target BPP.
2. **Noisy channel experiment** — Compress images, flip bits at various BERs, decompress, and record metrics.
3. **Analyze results** — Plot PSNR/NIQE vs BER and error counts.

---

## Directory Layout

```
src/
├── experiment_utils.py      # Shared utilities (BPP, flip_bits, CSV, etc.)
├── fid_utils.py             # Patch FID metric
├── find_compression_params.py   # Step 1: find params for target BPP
├── noisy_channel_runner.py  # Step 2: noisy-channel experiment
├── analyze_noisy_channel_results.py  # Step 3: plotting / analysis
└── runners/
    ├── base.py              # Abstract BaseModelRunner
    ├── jpeg_runner.py       # JPEG compression/decompression
    ├── ddcm_runner.py       # DDCM (diffusion-based)
    └── turbo_runner.py      # Turbo-DDCM and Robust Turbo-DDCM
```

External dependencies (expected in project root):
- `ddcm-compressed-image-generation-main/` — DDCM API
- `Turbo-DDCM-master/` — Turbo-DDCM API

---

## Architecture

### Runner Abstraction

All compression algorithms are exposed through `BaseModelRunner` in `runners/base.py`. Each runner implements:

| Method | Purpose |
|--------|---------|
| `get_model_params()` | Return current compression params |
| `run_compression(input_dir, output_dir, img_height, img_width)` | Compress images, return `{image_file: error_msg}` |
| `run_decompression(compressed_dir, img_height, img_width)` | Decompress in place, return errors dict |
| `path_for_compressed(compressed_dir, base)` | Path to compressed file for image base |
| `prepare_temp_for_noisy_channel(compressed_path, temp_dir, base)` | Copy into temp layout, return path to flip |
| `find_decompressed_png(temp_dir, base)` | Path to decompressed PNG after run |

**Runners:**

- **JpegModelRunner** — Uses `quality_by_image: Dict[str, int]` for per-image JPEG quality. Output: `{base}_compressed.jpg` → `{base}_decompressed.png`.
- **DdcmModelRunner** — Wraps DDCM; outputs `T=..._K=.../` subdirs with `*_noise_indices.bin` and `*_decomp.png`.
- **TurboModelRunner** — Wraps Turbo-DDCM. `robust=True` → Robust Turbo-DDCM (old protocol); `robust=False` → standard Turbo-DDCM.

### Noisy Channel Flow

1. **Baseline** (optional): Compress + decompress all images if no precomputed `compressed_dir` exists.
2. For each (BER, trial, image):
   - Copy compressed file(s) into `temp_work/ber{ber}_trial{trial}_{base}/`
   - Apply `flip_bits(ber)` to the bitstream
   - Run decompression on temp dir
   - If decompression fails (e.g. corrupted JPEG), record error in CSV
   - Otherwise compute PSNR, NIQE, LPIPS and save sample images
3. Compute patch FID per (BER, trial) across all decompressed images.
4. Write CSV: `image_file, ber, trial, num_bit_flips, bpp, psnr, niqe, lpips, error`

### MetricsComputer

- **PSNR** — skimage `peak_signal_noise_ratio`
- **NIQE** — pyiqa (GPU if available)
- **LPIPS** — pyiqa (GPU if available)
- **Patch FID** — `fid_utils.compute_patch_fid` via `neuralcompression` + torchmetrics

---

## File Descriptions

### `experiment_utils.py`

- `calculate_actual_bpp(binary_file_path, img_height, img_width)` — BPP from file size
- `flip_bits(binary_path, ber, output_path)` — Apply BER bit flips, return number flipped
- `get_images(dataset_path, num_samples=None)` — Sorted list of `.png` files
- `load_quality_csv(csv_path, target_bpp=None)` — Load `{image_file: quality}` for JPEG/BPG
- `parse_subset(s)`, `resolve_subset(image_files, subset)` — Parse `"0-4"` or `"0,2,5"` indices
- `prepare_input_dir(dataset_path, image_files, input_dir, subset)` — Copy images to staging
- `binary_search_best_int`, `collect_actual_bpp_from_output`, `collect_actual_bpp_by_glob` — Used by `find_compression_params`

### `fid_utils.py`

- `compute_patch_fid(ref_dir, gen_dir, image_files, img_height, img_width, ...)` — Patch FID
- `get_fid_patch_size(dataset_name)` — Patch size by dataset (e.g. Kodak24 → 64)

### `find_compression_params.py`

Finds algorithm parameters to achieve target BPP. Config-driven via `ALGO_CONFIG`:

- **jpeg** — Binary search quality per image
- **ddcm** — Search M given T, K, C
- **turbo_ddcm** / **robust_turbo_ddcm** — Search M given T, K, C

Writes CSVs under `results/compression_ratio_estimate/{algorithm}/` with dataset in the filename and a `dataset` column (e.g. `dataset_Kodak24_jpeg_compression_params.csv`, `dataset_Kodak24_ddcm_bpp.csv`).

### `noisy_channel_runner.py`

Main noisy-channel experiment. `NoisyChannelExperiment` orchestrates:

1. Baseline compression/decompression (or use existing `compressed_dir`)
2. For each BER × trial × image: flip bits, decompress, compute metrics
3. FID per trial
4. CSV output: `noisy_channel_{algorithm}.csv`
5. Samples: `samples/baseline/`, `samples/ber{ber}/`
6. FID summary: `fid_{algorithm}.csv`

### `analyze_noisy_channel_results.py`

Loads noisy-channel CSVs and produces PSNR/NIQE plots and error-count plots. Expects column names like `PSNR`, `BER`, `original_image_name`, `actual_bit_flips`, `error_reason`; you may need to map from the noisy_channel runner CSV format (`psnr`, `ber`, `image_file`, `num_bit_flips`, `error`).

---

## How to Use

### 1. Find Compression Parameters

Run for one or more algorithms and target BPPs:

```bash
# JPEG: find quality per image for BPP 1.0
python src/find_compression_params.py --algorithms jpeg --bpp 1.0 --dataset dataset_Kodak24

# DDCM / Turbo: find M for BPP 0.1 (subset 0-4 for quick test)
python src/find_compression_params.py --algorithms robust_turbo_ddcm ddcm --bpp 0.1 --dataset dataset_Kodak24 --subset 0-4

# All algorithms, all default BPPs
python src/find_compression_params.py --algorithms all --dataset dataset_Kodak24
```

Outputs go to `results/compression_ratio_estimate/{algorithm}/`.

### 2. Run Noisy Channel Experiment

**Structure:** The experiment loops over **BER** → **trial** → **image**. For each (BER, trial, image):

1. Copy the compressed file into `temp_work/ber{ber}_trial{trial}_{base}/`
2. Apply bit flips at the given BER
3. Run decompression
4. If successful: compute PSNR, NIQE, LPIPS; optionally save a sample
5. If decompression fails (e.g. corrupted bitstream): record the error in the CSV

After each (BER, trial), patch FID is computed over all decompressed images. Temp dirs are cleaned up; only samples and CSVs are kept.

**Output layout** (under `output_dir`, e.g. `results/noisy_channel/jpeg/dataset_Kodak24/` — one folder per dataset):

```
results/noisy_channel/{algorithm}/{dataset_name}/
├── noisy_channel_{algorithm}.csv   # Per-image metrics: image_file, ber, trial, num_bit_flips, bpp, psnr, niqe, lpips, error
├── fid_{algorithm}.csv            # Per-trial FID: ber, trial, fid
├── compressed/                    # Baseline compressed outputs (or precomputed via --compressed_dir)
│   └── {dataset_name}_{algo}_bpp{bpp}/
└── samples/
    ├── baseline/                  # Decompressed samples at BER=0 (no bit flips)
    │   └── {base}.png
    └── ber{ber}/                  # Samples at each BER (e.g. ber0.001)
        └── {base}_trial{trial}.png
```

`dataset_name` is the basename of `--dataset` (e.g. `dataset_Kodak24`, `DIV2K_valid_HR_512`).

The CSV has one row per (image, BER, trial). Baseline rows have `ber=0`, `trial=0`, `num_bit_flips=0`. Rows where decompression failed have `error` set and empty `psnr`/`niqe`/`lpips`.

```bash
# JPEG (requires jpeg_quality_csv from step 1)
python src/noisy_channel_runner.py \
  --algorithm jpeg \
  --dataset dataset_Kodak24 \
  --bpp 1.0 \
  --num_trials 1 \
  --jpeg_quality_csv results/compression_ratio_estimate/jpeg/dataset_Kodak24_jpeg_compression_params.csv \
  --seed 42 \
  --sample_images 0-5

# Turbo-DDCM / Robust Turbo-DDCM
python src/noisy_channel_runner.py \
  --algorithm robust_turbo_ddcm \
  --dataset dataset_Kodak24 \
  --bpp 0.1 \
  --num_trials 50 \
  --ber 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --seed 42

# DDCM
python src/noisy_channel_runner.py --algorithm ddcm --dataset dataset_Kodak24 --bpp 0.1 --num_trials 50
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | (required) | `jpeg`, `ddcm`, `turbo_ddcm`, `robust_turbo_ddcm` |
| `--dataset` | `dataset` | Path to image folder (PNG) |
| `--bpp` | 0.1 | Target BPP (used for dir names / JPEG quality lookup) |
| `--resolution` | 512 | Image size |
| `--ber` | 1e-6 .. 1e-1 | BER values to test |
| `--num_trials` | 50 | Trials per BER |
| `--jpeg_quality_csv` | — | Required for JPEG: CSV from find_compression_params |
| `--sample_images` | all | Indices to save samples, e.g. `0-5` or `0,2,5` |
| `--compressed_dir` | — | Skip baseline; use this precomputed dir |

Results: `results/noisy_channel/{algorithm}/{dataset_name}/` (CSV, samples, FID).

### 3. Analyze Results

```bash
python src/analyze_noisy_channel_results.py \
  --jpeg_csv results/noisy_channel/jpeg/dataset_Kodak24/noisy_channel_jpeg.csv \
  --turbo_csv results/noisy_channel/turbo_ddcm/dataset_Kodak24/noisy_channel_turbo_ddcm.csv \
  --turbo_improved_csv results/noisy_channel/robust_turbo_ddcm/dataset_Kodak24/noisy_channel_robust_turbo_ddcm.csv \
  --ddcm_csv results/noisy_channel/ddcm/dataset_Kodak24/noisy_channel_ddcm.csv \
  --output_dir results/plots
```

Note: `analyze_noisy_channel_results` expects specific column names; if your CSV format differs, a small adapter or column mapping may be needed.

---

## Adding a New Runner

1. Create `runners/new_runner.py` extending `BaseModelRunner`.
2. Implement all abstract methods.
3. Register in `runners/__init__.py` and in `_runner_factory` in `noisy_channel_runner.py`.
4. Add a `_prepare_params_*` function and config entry in `find_compression_params.py` if you want automatic BPP tuning.
