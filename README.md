# On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors

This project evaluates the robustness of various image compression methods (including diffusion-based approaches) to bit-flip errors in noisy channels. It supports JPEG, BPG, DDCM, Turbo-DDCM, Robust Turbo-DDCM, DiffC, ILLM (MS-ILLM), and StableCodec.

---

## Repo Structure

```
├── src/
│   ├── noisy_channel_runner.py      # Main experiment: bit-flip + decompress
│   ├── find_compression_params.py   # Precompute quality params for JPEG/BPG
│   ├── plots.ipynb                  # Rate-distortion curves, metrics vs BER
│   ├── qualitative_results.ipynb    # Visual comparison of samples
│   ├── runners/                     # Per-method compression/decompression
│   │   ├── jpeg_runner.py
│   │   ├── bpg_runner.py
│   │   ├── ddcm_runner.py
│   │   ├── turbo_runner.py
│   │   ├── diffc_runner.py
│   │   ├── illm_runner.py
│   │   └── stable_codec_runner.py
│   └── ...
├── DiffC/                           # DiffC (requires zipf_encoding build)
├── ILLM/                            # MS-ILLM
├── StableCodec/                     # StableCodec (requires ckpts)
├── ddcm-compressed-image-generation-main/
├── Turbo-DDCM-master/
├── results/
│   ├── noisy_channel/               # Experiment outputs (CSV, FID, samples)
│   ├── compression_ratio_estimate/  # JPEG/BPG quality CSVs
│   └── plots/                       # Generated tables/figures
├── run_main_experiment.sh           # Run all noisy-channel experiments
└── environment.yml
```

---

## Set Up the Environment

### 1. Base Environment

```bash
cd /path/to/Diffusion-Compression-Editing-Attacks
conda env create -f environment.yml
conda activate bit_flips_analysis
```

### 2. PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

We use CUDA 12.1 (`cu121`). If your system has a different CUDA version, replace `cu121` with the matching PyTorch suffix (e.g. `cu118` for CUDA 11.8, `cu124` for CUDA 12.4). See [pytorch.org](https://pytorch.org/get-started/locally/) for options.

---

## Additional Setup by Method

### BPG (Better Portable Graphics)

BPG requires system binaries `bpgenc` and `bpgdec` in your PATH.

**Build from source:**
- Download from https://bellard.org/bpg/
- Extract, run `make`, then `make install`

---

### DiffC

**1. Install Rust** (for the `zipf_encoding` extension):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Restart the terminal or run `source ~/.cargo/env`.

**2. Build zipf_encoding:**
```bash
conda activate bit_flips_analysis
cd DiffC/lib/diffc/rcc/arithmetic-coding/python-bindings
pip install .
cd -
```

**3. CuPy** is installed via `environment.yml` (`cupy-cuda12x<14`). DiffC uses custom CUDA kernels that CuPy compiles at runtime.

**4. CUDA toolkit** must be installed on the system (for `curand` headers). If CUDA is not in `/usr/local/cuda/include`, edit `DiffC/lib/diffc/rcc/pfr.py` and change the include path.


---

### StableCodec

**1. Create the checkpoints directory:**
```bash
mkdir -p StableCodec/ckpts
```

**2. Download pretrained checkpoints** from [Google Drive](https://drive.google.com/drive/folders/1itiVVAPSTATGPcHLp_bLI9r9Qi3YcM12):
- `elic_official.pth`
- `stablecodec_ft2.pkl`

**3. Place them in** `StableCodec/ckpts/`:
```
StableCodec/ckpts/
├── elic_official.pth
└── stablecodec_ft2.pkl
```

---

## Run the Main Experiment

**JPEG and BPG** need per-image quality settings for a target bpp. We already ran this step; the resulting CSVs are in `results/compression_ratio_estimate/`. These are the exact commands we used:

```bash
python src/find_compression_params.py --algorithms jpeg --bpp 1.0 --dataset dataset_Kodak24
python src/find_compression_params.py --algorithms jpeg --bpp 1.0 --dataset DIV2K_valid_HR_512

python src/find_compression_params.py --algorithms bpg --bpp 0.5 --dataset dataset_Kodak24
python src/find_compression_params.py --algorithms bpg --bpp 0.5 --dataset DIV2K_valid_HR_512
```

This writes CSV files to `results/compression_ratio_estimate/` that the noisy-channel runner uses.

We also ran `find_compression_params` for **DDCM, Turbo-DDCM, and Robust Turbo-DDCM**. For these three methods, the params are configured manually in the runners (see `src/runners/`), not via quality CSV.

**Run the main experiment:**
```bash
./run_main_experiment.sh
```

This runs the noisy-channel experiment for all methods (jpeg, bpg, robust_turbo_ddcm, turbo_ddcm, ddcm, diffc, illm, stable_codec) on both datasets.

**Results structure** (`results/noisy_channel/{algorithm}/{dataset}/`):

| File / Folder | Description |
|---------------|-------------|
| `noisy_channel_{algorithm}.csv` | Per-image, per-BER, per-trial metrics: `image_file`, `ber`, `trial`, `num_bit_flips`, `bpp`, `psnr`, `niqe`, `lpips`, `error` |
| `fid_{algorithm}.csv` | Per-BER, per-trial patch FID: `ber`, `trial`, `fid` |
| `samples/` | Visual samples of decompressed images (only when `--sample_images` is set). Subfolders: `baseline/` (no bit flips), `ber1e-06/`, `ber1e-05/`, etc. |
| `compressed/` | Baseline compressed bitstreams (before bit-flipping), e.g. `.jpg`, `.bpg`, `.turbo_ddcm`, `.diffc`, `.bin` |

---

## Plots and Qualitative Results

To generate graphs and qualitative results, run the notebooks:

- **`src/plots.ipynb`** — Rate-distortion curves, FID/PSNR/LPIPS/NIQE vs. BER, and summary tables
- **`src/qualitative_results.ipynb`** — Visual comparison of decompressed samples across methods
