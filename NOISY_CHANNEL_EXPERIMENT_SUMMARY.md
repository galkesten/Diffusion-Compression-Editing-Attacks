# Noisy Channel Experiment Summary

## Experiment Methodology

We conducted a comprehensive noisy channel experiment to evaluate the robustness of three image compression methods—JPEG, DDCM (Diffusion-based Deep Compression Model), and Turbo-DDCM—under various bit error rates (BER). The experiment simulates transmission errors by randomly flipping bits in the compressed bitstreams according to specified BER values, then decompressing and evaluating the reconstructed images.

### Experimental Setup

**Dataset**: Kodak dataset (24 images, 512×512 pixels)

**Resolution**: 512×512 pixels

**Target Compression Rates**: 0.1, 0.5, and 1.0 bits per pixel (BPP)

**Bit Error Rates (BER)**: 0 (baseline), 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹

**Number of Trials**: 50 independent trials per BER value per image

**Evaluation Metrics**: 
- Peak Signal-to-Noise Ratio (PSNR)
- Natural Image Quality Evaluator (NIQE)

### Experimental Procedure

1. **Compression Phase**: Each image is compressed using the specified method and target BPP, with compression parameters optimized per image to achieve the target rate.

2. **Baseline Measurement**: For BER=0, images are decompressed without bit errors to establish baseline quality metrics.

3. **Noisy Channel Simulation**: For each BER > 0:
   - The compressed file is read as binary data
   - Each bit is independently flipped with probability equal to the BER
   - The corrupted binary file is saved
   - The corrupted file is decompressed
   - Quality metrics (PSNR and NIQE) are computed

4. **Random State Management**: To ensure reproducibility and prevent interference between trials, the global random state is saved before decompression and restored afterward, isolating the bit-flipping process from any internal randomness in the decompression algorithms.

## Compression Parameters

### JPEG

JPEG compression uses quality-based encoding with per-image quality factors optimized to achieve the target BPP.

| Target BPP | Quality Factor Range | Average Actual BPP |
|------------|---------------------|-------------------|
| 0.1        | 1                   | 0.1823            |
| 0.5        | 9-39                | 0.5020             |
| 1.0        | 25-79               | 0.9967             |

### DDCM (Diffusion-based Deep Compression Model)

DDCM uses a diffusion model-based compression with the following parameters:

| Target BPP | M (Pursuit Noises) | T (Timesteps) | K (Candidate Noises) | C (Coefficient Bits) | Theoretical BPP |
|------------|-------------------|---------------|---------------------|---------------------|----------------|
| 0.1        | 2                 | 1000          | 8192                | 3                   | 0.1105         |
| 0.5        | 8                 | 1000          | 8192                | 3                   | 0.4764         |
| 1.0        | 17                | 1000          | 8192                | 3                   | 1.0251         |

**Model**: Stable Diffusion 2.1 Base (Manojb/stable-diffusion-2-1-base)

**Timestep Range**: [999, 0] (reverse diffusion from T-1 to 0)

### Turbo-DDCM

Turbo-DDCM uses an optimized diffusion-based compression with the following parameters:

| Target BPP | M (Pursuit Noises) | T (Timesteps) | K (Candidate Noises) | C (Coefficient Bits) | Theoretical BPP |
|------------|-------------------|---------------|---------------------|---------------------|----------------|
| 0.1        | 150               | 20            | 16384               | 1                   | 0.0998         |
| 0.5        | 1097              | 20            | 16384               | 1                   | 0.5000         |
| 1.0        | 2860              | 20            | 16384               | 1                   | 1.0001         |

## Expected Bit Flips

For 512×512 pixel images, the expected number of bit flips is calculated as:

**Expected Bit Flips = BER × (512 × 512) × Actual BPP**

The following tables provide the expected number of bit flips for each method and BPP at different BER values:

### BPP 0.1

| BER     | JPEG (BPP=0.1823) | Turbo-DDCM (BPP=0.0998) | DDCM (BPP=0.1105) |
|---------|-------------------|-------------------------|-------------------|
| 10⁻⁶    | 0.05              | 0.03                    | 0.03              |
| 10⁻⁵    | 0.48              | 0.26                    | 0.29              |
| 10⁻⁴    | 4.78              | 2.62                    | 2.90              |
| 10⁻³    | 47.80             | 26.17                   | 28.98             |
| 10⁻²    | 477.97            | 261.68                  | 289.76            |
| 10⁻¹    | 4,779.73          | 2,616.80                | 2,897.60          |

### BPP 0.5

| BER     | JPEG (BPP=0.5020) | Turbo-DDCM (BPP=0.5001) | DDCM (BPP=0.4764) |
|---------|-------------------|-------------------------|-------------------|
| 10⁻⁶    | 0.13              | 0.13                    | 0.12              |
| 10⁻⁵    | 1.32              | 1.31                    | 1.25              |
| 10⁻⁴    | 13.16             | 13.11                   | 12.50             |
| 10⁻³    | 131.60            | 131.09                  | 125.00            |
| 10⁻²    | 1,316.05          | 1,310.88                | 1,250.00          |
| 10⁻¹    | 13,160.47         | 13,108.80               | 12,500.00         |

### BPP 1.0

| BER     | JPEG (BPP=0.9967) | Turbo-DDCM (BPP=1.0002) | DDCM (BPP=1.0251) |
|---------|-------------------|-------------------------|-------------------|
| 10⁻⁶    | 0.26              | 0.26                    | 0.27              |
| 10⁻⁵    | 2.61              | 2.62                    | 2.68              |
| 10⁻⁴    | 26.13             | 26.22                   | 26.84             |
| 10⁻³    | 261.27            | 262.18                  | 268.44             |
| 10⁻²    | 2,612.74          | 2,621.84                | 2,684.35          |
| 10⁻¹    | 26,127.37         | 26,218.40               | 26,843.52          |

## Notes

- The actual BPP values may vary slightly from the target BPP due to image-specific compression characteristics.
- For JPEG, quality factors are optimized per image to achieve the target BPP, resulting in image-dependent actual BPP values.
- For DDCM and Turbo-DDCM, parameters are fixed per target BPP, with slight variations in actual BPP due to the compression algorithm's characteristics.
- The expected bit flips are calculated using the average actual BPP across all images for each method and target BPP.

