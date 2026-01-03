#!/usr/bin/env python3
"""
Analyze noisy channel experiment results and create visualization plots.

Creates:
1. Per-image plots: 6×4 grid showing PSNR/NIQE vs BER for each image (separate figures for PSNR and NIQE)
2. Averaged plots: Mean PSNR/NIQE vs BER across all images and trials (separate figures for PSNR and NIQE)
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load_csv_data_psnr(csv_path):
    """Load PSNR data from CSV file, including rows where PSNR is valid (even if NIQE is N/A)."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr = row['PSNR']
            
            if psnr != 'N/A':
                try:
                    data.append({
                        'image': row['original_image_name'],
                        'ber': float(row['BER']),
                        'psnr': float(psnr),
                        'actual_bit_flips': int(row['actual_bit_flips']),
                        'error_reason': row['error_reason']
                    })
                except (ValueError, KeyError) as e:
                    continue
    return data


def load_csv_data_niqe(csv_path):
    """Load NIQE data from CSV file, including rows where NIQE is valid (even if PSNR is N/A)."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            niqe = row['NIQE']
            
            if niqe != 'N/A':
                try:
                    data.append({
                        'image': row['original_image_name'],
                        'ber': float(row['BER']),
                        'niqe': float(niqe),
                        'actual_bit_flips': int(row['actual_bit_flips']),
                        'error_reason': row['error_reason']
                    })
                except (ValueError, KeyError) as e:
                    continue
    return data


def load_all_csv_data(csv_path):
    """Load all data from CSV file, including failures (N/A values)."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    'image': row['original_image_name'],
                    'ber': float(row['BER']),
                    'psnr': row['PSNR'],
                    'niqe': row['NIQE'],
                    'error_reason': row['error_reason']
                })
            except (ValueError, KeyError) as e:
                continue
    return data


def get_image_number(image_name):
    """Extract image number from filename like '1.png' -> 1."""
    try:
        return int(image_name.replace('.png', ''))
    except:
        return 0


def plot_per_image_results(jpeg_data, turbo_data, output_dir, metric='PSNR'):
    """Create 6×4 grid of subplots, one per image, showing metric vs BER.
    
    Grouping: For each image, group by BER value, then average across all trials.
    Baseline (BER=0) is shown as a horizontal dashed line.
    """
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
    fig.suptitle(f'{metric} vs BER (Log Scale) - Per Image', fontsize=16, y=0.995)
    
    all_images = sorted(set([d['image'] for d in jpeg_data] + [d['image'] for d in turbo_data]),
                       key=get_image_number)
    
    metric_key = metric.lower()
    
    for idx, image_name in enumerate(all_images):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        jpeg_image_data = [d for d in jpeg_data if d['image'] == image_name]
        turbo_image_data = [d for d in turbo_data if d['image'] == image_name]
        
        jpeg_bers_plot = []
        jpeg_means = []
        jpeg_stds = []
        jpeg_baseline_value = None
        
        if jpeg_image_data:
            jpeg_bers = sorted(set([d['ber'] for d in jpeg_image_data if d['ber'] > 0]))
            jpeg_baseline_data = [d[metric_key] for d in jpeg_image_data if d['ber'] == 0]
            
            if jpeg_baseline_data:
                jpeg_baseline_value = np.mean(jpeg_baseline_data)
            
            for ber in jpeg_bers:
                values = [d[metric_key] for d in jpeg_image_data if d['ber'] == ber]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    jpeg_bers_plot.append(ber)
                    jpeg_means.append(mean_val)
                    jpeg_stds.append(std_val)
            
        ax.set_xscale('log')
        
        if jpeg_bers_plot:
            ax.fill_between(jpeg_bers_plot,
                           [m - s for m, s in zip(jpeg_means, jpeg_stds)],
                           [m + s for m, s in zip(jpeg_means, jpeg_stds)],
                           color='blue', alpha=0.2, zorder=1)
            ax.plot(jpeg_bers_plot, jpeg_means, 'o-', label='JPEG', color='blue', 
                   markersize=4, linewidth=1.5, zorder=2)
        if jpeg_baseline_value is not None:
            ax.axhline(y=jpeg_baseline_value, color='blue', linestyle='--', linewidth=1.5, 
                      alpha=0.7, label='JPEG (baseline)', zorder=5)
        
        turbo_bers_plot = []
        turbo_means = []
        turbo_stds = []
        turbo_baseline_value = None
        
        if turbo_image_data:
            turbo_bers = sorted(set([d['ber'] for d in turbo_image_data if d['ber'] > 0]))
            turbo_baseline_data = [d[metric_key] for d in turbo_image_data if d['ber'] == 0]
            
            if turbo_baseline_data:
                turbo_baseline_value = np.mean(turbo_baseline_data)
            
            for ber in turbo_bers:
                values = [d[metric_key] for d in turbo_image_data if d['ber'] == ber]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    turbo_bers_plot.append(ber)
                    turbo_means.append(mean_val)
                    turbo_stds.append(std_val)
            
            if turbo_bers_plot:
                ax.fill_between(turbo_bers_plot,
                               [m - s for m, s in zip(turbo_means, turbo_stds)],
                               [m + s for m, s in zip(turbo_means, turbo_stds)],
                               color='red', alpha=0.2, zorder=1)
                ax.plot(turbo_bers_plot, turbo_means, 's-', label='Turbo-DDCM', color='red',
                       markersize=4, linewidth=1.5, zorder=2)
        if turbo_baseline_value is not None:
            ax.axhline(y=turbo_baseline_value, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.7, label='Turbo-DDCM (baseline)', zorder=5)
        ax.set_xlabel('BER', fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f'{image_name}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{metric.lower()}_per_image.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_averaged_results(jpeg_data, turbo_data, output_dir, metric='PSNR'):
    """Create averaged plot showing mean metric vs BER across all images and trials.
    
    Grouping: Group all measurements by BER value (across all images and trials), 
    then compute mean and std for each BER.
    Baseline (BER=0) is excluded from noisy data plots and shown separately as a horizontal dashed line.
    Variance bands (mean ± std) are shown as shaded regions around the noisy data curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_bers = sorted(set([d['ber'] for d in jpeg_data if d['ber'] > 0] + 
                          [d['ber'] for d in turbo_data if d['ber'] > 0]))
    
    jpeg_means = []
    jpeg_stds = []
    jpeg_bers_plot = []
    jpeg_counts = []
    jpeg_baseline_mean = None
    jpeg_baseline_std = None
    
    turbo_means = []
    turbo_stds = []
    turbo_bers_plot = []
    turbo_counts = []
    turbo_baseline_mean = None
    turbo_baseline_std = None
    
    metric_key = metric.lower()
    
    for ber in all_bers:
        jpeg_values = [d[metric_key] for d in jpeg_data if d['ber'] == ber]
        turbo_values = [d[metric_key] for d in turbo_data if d['ber'] == ber]
        
        if jpeg_values:
            mean_val = np.mean(jpeg_values)
            std_val = np.std(jpeg_values)
            jpeg_bers_plot.append(ber)
            jpeg_means.append(mean_val)
            jpeg_stds.append(std_val)
            jpeg_counts.append(len(jpeg_values))
        
        if turbo_values:
            mean_val = np.mean(turbo_values)
            std_val = np.std(turbo_values)
            turbo_bers_plot.append(ber)
            turbo_means.append(mean_val)
            turbo_stds.append(std_val)
            turbo_counts.append(len(turbo_values))
    
    jpeg_baseline_values = [d[metric_key] for d in jpeg_data if d['ber'] == 0]
    turbo_baseline_values = [d[metric_key] for d in turbo_data if d['ber'] == 0]
    
    if jpeg_baseline_values:
        jpeg_baseline_mean = np.mean(jpeg_baseline_values)
        jpeg_baseline_std = np.std(jpeg_baseline_values)
    
    if turbo_baseline_values:
        turbo_baseline_mean = np.mean(turbo_baseline_values)
        turbo_baseline_std = np.std(turbo_baseline_values)
    
    ax.set_xscale('log')
    
    if jpeg_bers_plot:
        ax.fill_between(jpeg_bers_plot, 
                        [m - s for m, s in zip(jpeg_means, jpeg_stds)],
                        [m + s for m, s in zip(jpeg_means, jpeg_stds)],
                        color='blue', alpha=0.2, zorder=1)
        ax.plot(jpeg_bers_plot, jpeg_means, 'o-', label='JPEG', color='blue', 
               markersize=6, linewidth=2, alpha=0.8, zorder=2)
        print(f"  JPEG: {len(jpeg_bers_plot)} BER points, sample counts: {jpeg_counts}")
    
    if turbo_bers_plot:
        ax.fill_between(turbo_bers_plot,
                       [m - s for m, s in zip(turbo_means, turbo_stds)],
                       [m + s for m, s in zip(turbo_means, turbo_stds)],
                       color='red', alpha=0.2, zorder=1)
        ax.plot(turbo_bers_plot, turbo_means, 's-', label='Turbo-DDCM', color='red',
               markersize=6, linewidth=2, alpha=0.8, zorder=2)
        print(f"  Turbo-DDCM: {len(turbo_bers_plot)} BER points, sample counts: {turbo_counts}")
    
    if jpeg_baseline_mean is not None:
        ax.axhline(y=float(jpeg_baseline_mean), color='blue', linestyle='--', linewidth=2,
                  alpha=0.7, label='JPEG (baseline)', zorder=5)
        print(f"    Baseline (BER=0): {jpeg_baseline_mean:.2f} ± {jpeg_baseline_std:.2f}")
    
    if turbo_baseline_mean is not None:
        ax.axhline(y=float(turbo_baseline_mean), color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Turbo-DDCM (baseline)', zorder=5)
        print(f"    Baseline (BER=0): {turbo_baseline_mean:.2f} ± {turbo_baseline_std:.2f}")
    ax.set_xlabel('BER (Log Scale)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Average {metric} vs BER (Log Scale) - All Images and Trials', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{metric.lower()}_averaged.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_counts(jpeg_csv, turbo_csv, output_dir):
    """Plot decoding failure counts per BER for both methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    jpeg_all_data = load_all_csv_data(jpeg_csv)
    turbo_all_data = load_all_csv_data(turbo_csv)
    
    jpeg_bers = sorted(set([d['ber'] for d in jpeg_all_data if d['ber'] > 0]))
    turbo_bers = sorted(set([d['ber'] for d in turbo_all_data if d['ber'] > 0]))
    all_bers = sorted(set(jpeg_bers + turbo_bers))
    
    jpeg_error_counts = []
    jpeg_total_counts = []
    jpeg_bers_plot = []
    
    turbo_error_counts = []
    turbo_total_counts = []
    turbo_bers_plot = []
    
    for ber in all_bers:
        jpeg_rows = [d for d in jpeg_all_data if d['ber'] == ber]
        turbo_rows = [d for d in turbo_all_data if d['ber'] == ber]
        
        if jpeg_rows:
            jpeg_failures = sum(1 for d in jpeg_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
            jpeg_total = len(jpeg_rows)
            jpeg_bers_plot.append(ber)
            jpeg_error_counts.append(jpeg_failures)
            jpeg_total_counts.append(jpeg_total)
        
        if turbo_rows:
            turbo_failures = sum(1 for d in turbo_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
            turbo_total = len(turbo_rows)
            turbo_bers_plot.append(ber)
            turbo_error_counts.append(turbo_failures)
            turbo_total_counts.append(turbo_total)
    
    if jpeg_bers_plot:
        ax.plot(jpeg_bers_plot, jpeg_error_counts, 'o-', label='JPEG failures', 
               color='blue', markersize=6, linewidth=2, alpha=0.8)
        print(f"  JPEG error counts: {dict(zip(jpeg_bers_plot, jpeg_error_counts))}")
        print(f"  JPEG total counts: {dict(zip(jpeg_bers_plot, jpeg_total_counts))}")
    
    if turbo_bers_plot:
        ax.plot(turbo_bers_plot, turbo_error_counts, 's-', label='Turbo-DDCM failures',
               color='red', markersize=6, linewidth=2, alpha=0.8)
        print(f"  Turbo-DDCM error counts: {dict(zip(turbo_bers_plot, turbo_error_counts))}")
        print(f"  Turbo-DDCM total counts: {dict(zip(turbo_bers_plot, turbo_total_counts))}")
    
    ax.set_xscale('log')
    ax.set_xlabel('BER (Log Scale)', fontsize=12)
    ax.set_ylabel('Number of Decoding Failures', fontsize=12)
    ax.set_title('Decoding Failures vs BER - All Images and Trials', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_counts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze noisy channel experiment results')
    parser.add_argument('--jpeg_csv', type=str, required=True,
                       help='Path to JPEG results CSV file')
    parser.add_argument('--turbo_csv', type=str, required=True,
                       help='Path to Turbo-DDCM results CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--bpp', type=float, required=True,
                       help='BPP value (for output naming)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading JPEG PSNR data from: {args.jpeg_csv}")
    jpeg_psnr_data = load_csv_data_psnr(args.jpeg_csv)
    print(f"  Loaded {len(jpeg_psnr_data)} rows with valid PSNR")
    
    print(f"Loading JPEG NIQE data from: {args.jpeg_csv}")
    jpeg_niqe_data = load_csv_data_niqe(args.jpeg_csv)
    print(f"  Loaded {len(jpeg_niqe_data)} rows with valid NIQE")
    
    print(f"Loading Turbo-DDCM PSNR data from: {args.turbo_csv}")
    turbo_psnr_data = load_csv_data_psnr(args.turbo_csv)
    print(f"  Loaded {len(turbo_psnr_data)} rows with valid PSNR")
    
    print(f"Loading Turbo-DDCM NIQE data from: {args.turbo_csv}")
    turbo_niqe_data = load_csv_data_niqe(args.turbo_csv)
    print(f"  Loaded {len(turbo_niqe_data)} rows with valid NIQE")
    
    print("\nCreating PSNR plots...")
    print("  Per-image: Grouping by (image, BER), averaging across trials")
    plot_per_image_results(jpeg_psnr_data, turbo_psnr_data, args.output_dir, metric='PSNR')
    print("  Averaged: Grouping by BER only, averaging across all images and trials")
    plot_averaged_results(jpeg_psnr_data, turbo_psnr_data, args.output_dir, metric='PSNR')
    
    print("\nCreating NIQE plots...")
    print("  Per-image: Grouping by (image, BER), averaging across trials")
    plot_per_image_results(jpeg_niqe_data, turbo_niqe_data, args.output_dir, metric='NIQE')
    print("  Averaged: Grouping by BER only, averaging across all images and trials")
    plot_averaged_results(jpeg_niqe_data, turbo_niqe_data, args.output_dir, metric='NIQE')
    
    print("\nCreating error count plot...")
    plot_error_counts(args.jpeg_csv, args.turbo_csv, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

