#!/bin/bash
# Run noisy-channel experiments for all methods on dataset_Kodak24 and DIV2K_valid_HR_512.
# Results: results/noisy_channel/{algorithm}/{dataset}/
# Prerequisite: run find_compression_params.py for jpeg/bpg first (see README).

set -e
cd "$(dirname "$0")"

# --- dataset_Kodak24 ---

python src/noisy_channel_runner.py \
  --algorithm jpeg \
  --dataset dataset_Kodak24 \
  --bpp 1.0 \
  --num_trials 10 \
  --quality_csv results/compression_ratio_estimate/jpeg/dataset_Kodak24_jpeg_compression_params.csv \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm bpg \
  --dataset dataset_Kodak24 \
  --bpp 0.5 \
  --num_trials 10 \
  --quality_csv results/compression_ratio_estimate/bpg/dataset_Kodak24_bpg_compression_params.csv \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm robust_turbo_ddcm \
  --dataset dataset_Kodak24 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm turbo_ddcm \
  --dataset dataset_Kodak24 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm ddcm \
  --dataset dataset_Kodak24 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm diffc \
  --dataset dataset_Kodak24 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm illm \
  --dataset dataset_Kodak24 \
  --bpp 0.07 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm stable_codec \
  --dataset dataset_Kodak24 \
  --bpp 0.035 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

# --- DIV2K_valid_HR_512 ---

python src/noisy_channel_runner.py \
  --algorithm jpeg \
  --dataset DIV2K_valid_HR_512 \
  --bpp 1.0 \
  --num_trials 10 \
  --quality_csv results/compression_ratio_estimate/jpeg/dataset_DIV2K_valid_HR_512_jpeg_compression_params.csv \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm bpg \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.5 \
  --num_trials 10 \
  --quality_csv results/compression_ratio_estimate/bpg/dataset_DIV2K_valid_HR_512_bpg_compression_params.csv \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm robust_turbo_ddcm \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm turbo_ddcm \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm ddcm \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm diffc \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.1 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm illm \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.07 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5

python src/noisy_channel_runner.py \
  --algorithm stable_codec \
  --dataset DIV2K_valid_HR_512 \
  --bpp 0.035 \
  --num_trials 10 \
  --seed 42 \
  --sample_images 0-5
