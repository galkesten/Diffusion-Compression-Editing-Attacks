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


def load_csv_data_psnr(csv_path, ber_min=None, ber_max=None):
    """Load PSNR data from CSV file, including rows where PSNR is valid (even if NIQE is N/A)."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr = row['PSNR']
            
            if psnr != 'N/A':
                try:
                    ber = float(row['BER'])
                    if ber == 0:
                        data.append({
                            'image': row['original_image_name'],
                            'ber': ber,
                            'psnr': float(psnr),
                            'actual_bit_flips': int(row['actual_bit_flips']),
                            'error_reason': row['error_reason']
                        })
                    else:
                        if ber_min is not None and ber < ber_min:
                            continue
                        if ber_max is not None and ber > ber_max:
                            continue
                        data.append({
                            'image': row['original_image_name'],
                            'ber': ber,
                            'psnr': float(psnr),
                            'actual_bit_flips': int(row['actual_bit_flips']),
                            'error_reason': row['error_reason']
                        })
                except (ValueError, KeyError) as e:
                    continue
    return data


def load_csv_data_niqe(csv_path, ber_min=None, ber_max=None):
    """Load NIQE data from CSV file, including rows where NIQE is valid (even if PSNR is N/A)."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            niqe = row['NIQE']
            
            if niqe != 'N/A':
                try:
                    ber = float(row['BER'])
                    if ber == 0:
                        data.append({
                            'image': row['original_image_name'],
                            'ber': ber,
                            'niqe': float(niqe),
                            'actual_bit_flips': int(row['actual_bit_flips']),
                            'error_reason': row['error_reason']
                        })
                    else:
                        if ber_min is not None and ber < ber_min:
                            continue
                        if ber_max is not None and ber > ber_max:
                            continue
                        data.append({
                            'image': row['original_image_name'],
                            'ber': ber,
                            'niqe': float(niqe),
                            'actual_bit_flips': int(row['actual_bit_flips']),
                            'error_reason': row['error_reason']
                        })
                except (ValueError, KeyError) as e:
                    continue
    return data


def load_all_csv_data(csv_path, ber_min=None, ber_max=None):
    """Load all data from CSV file, including failures (N/A values)."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ber = float(row['BER'])
                if ber_min is not None and ber < ber_min:
                    continue
                if ber_max is not None and ber > ber_max:
                    continue
                data.append({
                    'image': row['original_image_name'],
                    'ber': ber,
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


def plot_per_image_results(jpeg_data, turbo_data, ddcm_data, models, output_dir, metric='PSNR'):
    """Create 6×4 grid of subplots, one per image, showing metric vs BER.
    
    Grouping: For each image, group by BER value, then average across all trials.
    Baseline (BER=0) is shown as a horizontal dashed line.
    """
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
    fig.suptitle(f'{metric} vs BER (Log Scale) - Per Image', fontsize=16, y=0.995)
    
    all_images = sorted(set([d['image'] for d in jpeg_data] + 
                           [d['image'] for d in turbo_data] + 
                           [d['image'] for d in ddcm_data]),
                       key=get_image_number)
    
    metric_key = metric.lower()
    colors = {'jpeg': 'blue', 'turbo': 'red', 'ddcm': 'green'}
    markers = {'jpeg': 'o', 'turbo': 's', 'ddcm': '^'}
    labels = {'jpeg': 'JPEG', 'turbo': 'Turbo-DDCM', 'ddcm': 'DDCM'}
    
    for idx, image_name in enumerate(all_images):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        jpeg_image_data = [d for d in jpeg_data if d['image'] == image_name] if 'jpeg' in models else []
        turbo_image_data = [d for d in turbo_data if d['image'] == image_name] if 'turbo' in models else []
        ddcm_image_data = [d for d in ddcm_data if d['image'] == image_name] if 'ddcm' in models else []
        
        ax.set_xscale('log')
        
        if 'jpeg' in models and jpeg_image_data:
            jpeg_bers = sorted(set([d['ber'] for d in jpeg_image_data if d['ber'] > 0]))
            jpeg_baseline_data = [d[metric_key] for d in jpeg_image_data if d['ber'] == 0]
            jpeg_baseline_value = np.mean(jpeg_baseline_data) if jpeg_baseline_data else None
            
            jpeg_bers_plot = []
            jpeg_means = []
            jpeg_stds = []
            for ber in jpeg_bers:
                values = [d[metric_key] for d in jpeg_image_data if d['ber'] == ber]
                if values:
                    jpeg_bers_plot.append(ber)
                    jpeg_means.append(np.mean(values))
                    jpeg_stds.append(np.std(values))
            
            if jpeg_bers_plot:
                ax.plot(jpeg_bers_plot, jpeg_means, f'{markers["jpeg"]}-', 
                       label=labels['jpeg'], color=colors['jpeg'], 
                       markersize=4, linewidth=1.5, zorder=2)
            if jpeg_baseline_value is not None:
                ax.axhline(y=jpeg_baseline_value, color=colors['jpeg'], linestyle='--', 
                          linewidth=1.5, alpha=0.7, label=f'{labels["jpeg"]} (baseline)', zorder=5)
        
        if 'turbo' in models and turbo_image_data:
            turbo_bers = sorted(set([d['ber'] for d in turbo_image_data if d['ber'] > 0]))
            turbo_baseline_data = [d[metric_key] for d in turbo_image_data if d['ber'] == 0]
            turbo_baseline_value = np.mean(turbo_baseline_data) if turbo_baseline_data else None
            
            turbo_bers_plot = []
            turbo_means = []
            turbo_stds = []
            for ber in turbo_bers:
                values = [d[metric_key] for d in turbo_image_data if d['ber'] == ber]
                if values:
                    turbo_bers_plot.append(ber)
                    turbo_means.append(np.mean(values))
                    turbo_stds.append(np.std(values))
            
            if turbo_bers_plot:
                ax.plot(turbo_bers_plot, turbo_means, f'{markers["turbo"]}-', 
                       label=labels['turbo'], color=colors['turbo'],
                       markersize=4, linewidth=1.5, zorder=2)
            if turbo_baseline_value is not None:
                ax.axhline(y=turbo_baseline_value, color=colors['turbo'], linestyle='--', 
                          linewidth=1.5, alpha=0.7, label=f'{labels["turbo"]} (baseline)', zorder=5)
        
        if 'ddcm' in models and ddcm_image_data:
            ddcm_bers = sorted(set([d['ber'] for d in ddcm_image_data if d['ber'] > 0]))
            ddcm_baseline_data = [d[metric_key] for d in ddcm_image_data if d['ber'] == 0]
            ddcm_baseline_value = np.mean(ddcm_baseline_data) if ddcm_baseline_data else None
            
            ddcm_bers_plot = []
            ddcm_means = []
            ddcm_stds = []
            for ber in ddcm_bers:
                values = [d[metric_key] for d in ddcm_image_data if d['ber'] == ber]
                if values:
                    ddcm_bers_plot.append(ber)
                    ddcm_means.append(np.mean(values))
                    ddcm_stds.append(np.std(values))
            
            if ddcm_bers_plot:
                ax.plot(ddcm_bers_plot, ddcm_means, f'{markers["ddcm"]}-', 
                       label=labels['ddcm'], color=colors['ddcm'],
                       markersize=4, linewidth=1.5, zorder=2)
            if ddcm_baseline_value is not None:
                ax.axhline(y=ddcm_baseline_value, color=colors['ddcm'], linestyle='--', 
                          linewidth=1.5, alpha=0.7, label=f'{labels["ddcm"]} (baseline)', zorder=5)
        
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


def plot_averaged_results(jpeg_data, turbo_data, ddcm_data, models, output_dir, metric='PSNR'):
    """Create averaged plot showing mean metric vs BER across all images and trials.
    
    Grouping: Group all measurements by BER value (across all images and trials), 
    then compute mean and std for each BER.
    Baseline (BER=0) is excluded from noisy data plots and shown separately as a horizontal dashed line.
    Variance bands (mean ± std) are shown as shaded regions around the noisy data curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_bers = sorted(set([d['ber'] for d in jpeg_data if d['ber'] > 0] + 
                          [d['ber'] for d in turbo_data if d['ber'] > 0] +
                          [d['ber'] for d in ddcm_data if d['ber'] > 0]))
    
    metric_key = metric.lower()
    colors = {'jpeg': 'blue', 'turbo': 'red', 'ddcm': 'green'}
    markers = {'jpeg': 'o', 'turbo': 's', 'ddcm': '^'}
    labels = {'jpeg': 'JPEG', 'turbo': 'Turbo-DDCM', 'ddcm': 'DDCM'}
    
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
    
    ddcm_means = []
    ddcm_stds = []
    ddcm_bers_plot = []
    ddcm_counts = []
    ddcm_baseline_mean = None
    ddcm_baseline_std = None
    
    for ber in all_bers:
        if 'jpeg' in models:
            jpeg_values = [d[metric_key] for d in jpeg_data if d['ber'] == ber]
            if jpeg_values:
                jpeg_bers_plot.append(ber)
                jpeg_means.append(np.mean(jpeg_values))
                jpeg_stds.append(np.std(jpeg_values))
                jpeg_counts.append(len(jpeg_values))
        
        if 'turbo' in models:
            turbo_values = [d[metric_key] for d in turbo_data if d['ber'] == ber]
            if turbo_values:
                turbo_bers_plot.append(ber)
                turbo_means.append(np.mean(turbo_values))
                turbo_stds.append(np.std(turbo_values))
                turbo_counts.append(len(turbo_values))
        
        if 'ddcm' in models:
            ddcm_values = [d[metric_key] for d in ddcm_data if d['ber'] == ber]
            if ddcm_values:
                ddcm_bers_plot.append(ber)
                ddcm_means.append(np.mean(ddcm_values))
                ddcm_stds.append(np.std(ddcm_values))
                ddcm_counts.append(len(ddcm_values))
    
    if 'jpeg' in models:
        jpeg_baseline_values = [d[metric_key] for d in jpeg_data if d['ber'] == 0]
        if jpeg_baseline_values:
            jpeg_baseline_mean = np.mean(jpeg_baseline_values)
            jpeg_baseline_std = np.std(jpeg_baseline_values)
    
    if 'turbo' in models:
        turbo_baseline_values = [d[metric_key] for d in turbo_data if d['ber'] == 0]
        if turbo_baseline_values:
            turbo_baseline_mean = np.mean(turbo_baseline_values)
            turbo_baseline_std = np.std(turbo_baseline_values)
    
    if 'ddcm' in models:
        ddcm_baseline_values = [d[metric_key] for d in ddcm_data if d['ber'] == 0]
        if ddcm_baseline_values:
            ddcm_baseline_mean = np.mean(ddcm_baseline_values)
            ddcm_baseline_std = np.std(ddcm_baseline_values)
    
    ax.set_xscale('log')
    
    if 'jpeg' in models and jpeg_bers_plot:
        ax.plot(jpeg_bers_plot, jpeg_means, f'{markers["jpeg"]}-', 
               label=labels['jpeg'], color=colors['jpeg'], 
               markersize=6, linewidth=2, zorder=2)
        print(f"  JPEG: {len(jpeg_bers_plot)} BER points, sample counts: {jpeg_counts}")
        if jpeg_baseline_mean is not None:
            ax.axhline(y=float(jpeg_baseline_mean), color=colors['jpeg'], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'{labels["jpeg"]} (baseline)', zorder=5)
            print(f"    Baseline (BER=0): {jpeg_baseline_mean:.2f} ± {jpeg_baseline_std:.2f}")
    
    if 'turbo' in models and turbo_bers_plot:
        ax.plot(turbo_bers_plot, turbo_means, f'{markers["turbo"]}-', 
               label=labels['turbo'], color=colors['turbo'],
               markersize=6, linewidth=2, zorder=2)
        print(f"  Turbo-DDCM: {len(turbo_bers_plot)} BER points, sample counts: {turbo_counts}")
        if turbo_baseline_mean is not None:
            ax.axhline(y=float(turbo_baseline_mean), color=colors['turbo'], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'{labels["turbo"]} (baseline)', zorder=5)
            print(f"    Baseline (BER=0): {turbo_baseline_mean:.2f} ± {turbo_baseline_std:.2f}")
    
    if 'ddcm' in models and ddcm_bers_plot:
        ax.plot(ddcm_bers_plot, ddcm_means, f'{markers["ddcm"]}-', 
               label=labels['ddcm'], color=colors['ddcm'],
               markersize=6, linewidth=2, zorder=2)
        print(f"  DDCM: {len(ddcm_bers_plot)} BER points, sample counts: {ddcm_counts}")
        if ddcm_baseline_mean is not None:
            ax.axhline(y=float(ddcm_baseline_mean), color=colors['ddcm'], linestyle='--', 
                      linewidth=2, alpha=0.8, label=f'{labels["ddcm"]} (baseline)', zorder=5)
            print(f"    Baseline (BER=0): {ddcm_baseline_mean:.2f} ± {ddcm_baseline_std:.2f}")
    
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


def plot_error_counts(jpeg_csv, turbo_csv, ddcm_csv, models, output_dir, ber_min=None, ber_max=None):
    """Plot decoding failure counts per BER for selected methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'jpeg': 'blue', 'turbo': 'red', 'ddcm': 'green'}
    markers = {'jpeg': 'o', 'turbo': 's', 'ddcm': '^'}
    labels = {'jpeg': 'JPEG', 'turbo': 'Turbo-DDCM', 'ddcm': 'DDCM'}
    
    jpeg_all_data = load_all_csv_data(jpeg_csv, ber_min, ber_max) if 'jpeg' in models and jpeg_csv else []
    turbo_all_data = load_all_csv_data(turbo_csv, ber_min, ber_max) if 'turbo' in models and turbo_csv else []
    ddcm_all_data = load_all_csv_data(ddcm_csv, ber_min, ber_max) if 'ddcm' in models and ddcm_csv else []
    
    all_bers = sorted(set([d['ber'] for d in jpeg_all_data if d['ber'] > 0] + 
                          [d['ber'] for d in turbo_all_data if d['ber'] > 0] +
                          [d['ber'] for d in ddcm_all_data if d['ber'] > 0]))
    
    if 'jpeg' in models and jpeg_all_data:
        jpeg_error_counts = []
        jpeg_total_counts = []
        jpeg_bers_plot = []
        
        for ber in all_bers:
            jpeg_rows = [d for d in jpeg_all_data if d['ber'] == ber]
            if jpeg_rows:
                jpeg_failures = sum(1 for d in jpeg_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
                jpeg_total = len(jpeg_rows)
                jpeg_bers_plot.append(ber)
                jpeg_error_counts.append(jpeg_failures)
                jpeg_total_counts.append(jpeg_total)
        
        if jpeg_bers_plot:
            ax.plot(jpeg_bers_plot, jpeg_error_counts, f'{markers["jpeg"]}-', 
                   label=f'{labels["jpeg"]} failures', color=colors['jpeg'], 
                   markersize=6, linewidth=2, alpha=0.8)
            print(f"  JPEG error counts: {dict(zip(jpeg_bers_plot, jpeg_error_counts))}")
            print(f"  JPEG total counts: {dict(zip(jpeg_bers_plot, jpeg_total_counts))}")
    
    if 'turbo' in models and turbo_all_data:
        turbo_error_counts = []
        turbo_total_counts = []
        turbo_bers_plot = []
        
        for ber in all_bers:
            turbo_rows = [d for d in turbo_all_data if d['ber'] == ber]
            if turbo_rows:
                turbo_failures = sum(1 for d in turbo_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
                turbo_total = len(turbo_rows)
                turbo_bers_plot.append(ber)
                turbo_error_counts.append(turbo_failures)
                turbo_total_counts.append(turbo_total)
        
        if turbo_bers_plot:
            ax.plot(turbo_bers_plot, turbo_error_counts, f'{markers["turbo"]}-', 
                   label=f'{labels["turbo"]} failures', color=colors['turbo'],
                   markersize=6, linewidth=2, alpha=0.8)
            print(f"  Turbo-DDCM error counts: {dict(zip(turbo_bers_plot, turbo_error_counts))}")
            print(f"  Turbo-DDCM total counts: {dict(zip(turbo_bers_plot, turbo_total_counts))}")
    
    if 'ddcm' in models and ddcm_all_data:
        ddcm_error_counts = []
        ddcm_total_counts = []
        ddcm_bers_plot = []
        
        for ber in all_bers:
            ddcm_rows = [d for d in ddcm_all_data if d['ber'] == ber]
            if ddcm_rows:
                ddcm_failures = sum(1 for d in ddcm_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
                ddcm_total = len(ddcm_rows)
                ddcm_bers_plot.append(ber)
                ddcm_error_counts.append(ddcm_failures)
                ddcm_total_counts.append(ddcm_total)
        
        if ddcm_bers_plot:
            ax.plot(ddcm_bers_plot, ddcm_error_counts, f'{markers["ddcm"]}-', 
                   label=f'{labels["ddcm"]} failures', color=colors['ddcm'],
                   markersize=6, linewidth=2, alpha=0.8)
            print(f"  DDCM error counts: {dict(zip(ddcm_bers_plot, ddcm_error_counts))}")
            print(f"  DDCM total counts: {dict(zip(ddcm_bers_plot, ddcm_total_counts))}")
    
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


def concatenate_csv_files(csv_paths):
    """Concatenate multiple CSV files into a single list of data rows."""
    all_data = []
    for csv_path in csv_paths:
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                all_data.extend(list(reader))
    return all_data


def write_concatenated_csv(all_data, output_path):
    """Write concatenated data to a temporary CSV file."""
    if not all_data:
        return None
    
    fieldnames = all_data[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Analyze noisy channel experiment results')
    parser.add_argument('--jpeg_csv', type=str, default=None,
                       help='Path to JPEG results CSV file')
    parser.add_argument('--turbo_csv', type=str, default=None,
                       help='Path to Turbo-DDCM results CSV file')
    parser.add_argument('--ddcm_csv', type=str, nargs='+', default=None,
                       help='Path(s) to DDCM results CSV file(s). Can provide multiple files to concatenate (e.g., first_half and second_half)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--models', type=str, nargs='+', default=['jpeg', 'turbo', 'ddcm'],
                       choices=['jpeg', 'turbo', 'ddcm'],
                       help='Which models to include in plots (default: all)')
    parser.add_argument('--ber_min', type=float, default=None,
                       help='Minimum BER to include in plots (e.g., 1e-6). If None, include all BERs')
    parser.add_argument('--ber_max', type=float, default=None,
                       help='Maximum BER to include in plots (e.g., 1e-5). If None, include all BERs')
    
    args = parser.parse_args()
    
    if not args.jpeg_csv and not args.turbo_csv and not args.ddcm_csv:
        parser.error("At least one of --jpeg_csv, --turbo_csv, or --ddcm_csv must be provided")
    
    models = [m.lower() for m in args.models]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ddcm_csv_combined = None
    if 'ddcm' in models and args.ddcm_csv:
        if len(args.ddcm_csv) > 1:
            print(f"Concatenating DDCM CSV files: {args.ddcm_csv}")
            all_ddcm_data = concatenate_csv_files(args.ddcm_csv)
            ddcm_csv_combined = os.path.join(args.output_dir, 'ddcm_combined.csv')
            write_concatenated_csv(all_ddcm_data, ddcm_csv_combined)
            print(f"  Combined {len(all_ddcm_data)} rows into: {ddcm_csv_combined}")
        else:
            ddcm_csv_combined = args.ddcm_csv[0]
    
    jpeg_psnr_data = []
    jpeg_niqe_data = []
    if 'jpeg' in models and args.jpeg_csv:
        print(f"Loading JPEG PSNR data from: {args.jpeg_csv}")
        jpeg_psnr_data = load_csv_data_psnr(args.jpeg_csv, args.ber_min, args.ber_max)
        print(f"  Loaded {len(jpeg_psnr_data)} rows with valid PSNR")
        
        print(f"Loading JPEG NIQE data from: {args.jpeg_csv}")
        jpeg_niqe_data = load_csv_data_niqe(args.jpeg_csv, args.ber_min, args.ber_max)
        print(f"  Loaded {len(jpeg_niqe_data)} rows with valid NIQE")
    
    turbo_psnr_data = []
    turbo_niqe_data = []
    if 'turbo' in models and args.turbo_csv:
        print(f"Loading Turbo-DDCM PSNR data from: {args.turbo_csv}")
        turbo_psnr_data = load_csv_data_psnr(args.turbo_csv, args.ber_min, args.ber_max)
        print(f"  Loaded {len(turbo_psnr_data)} rows with valid PSNR")
        
        print(f"Loading Turbo-DDCM NIQE data from: {args.turbo_csv}")
        turbo_niqe_data = load_csv_data_niqe(args.turbo_csv, args.ber_min, args.ber_max)
        print(f"  Loaded {len(turbo_niqe_data)} rows with valid NIQE")
    
    ddcm_psnr_data = []
    ddcm_niqe_data = []
    if 'ddcm' in models and ddcm_csv_combined:
        print(f"Loading DDCM PSNR data from: {ddcm_csv_combined}")
        ddcm_psnr_data = load_csv_data_psnr(ddcm_csv_combined, args.ber_min, args.ber_max)
        print(f"  Loaded {len(ddcm_psnr_data)} rows with valid PSNR")
        
        print(f"Loading DDCM NIQE data from: {ddcm_csv_combined}")
        ddcm_niqe_data = load_csv_data_niqe(ddcm_csv_combined, args.ber_min, args.ber_max)
        print(f"  Loaded {len(ddcm_niqe_data)} rows with valid NIQE")
    
    print("\nCreating PSNR plots...")
    print("  Per-image: Grouping by (image, BER), averaging across trials")
    plot_per_image_results(jpeg_psnr_data, turbo_psnr_data, ddcm_psnr_data, models, args.output_dir, metric='PSNR')
    print("  Averaged: Grouping by BER only, averaging across all images and trials")
    plot_averaged_results(jpeg_psnr_data, turbo_psnr_data, ddcm_psnr_data, models, args.output_dir, metric='PSNR')
    
    print("\nCreating NIQE plots...")
    print("  Per-image: Grouping by (image, BER), averaging across trials")
    plot_per_image_results(jpeg_niqe_data, turbo_niqe_data, ddcm_niqe_data, models, args.output_dir, metric='NIQE')
    print("  Averaged: Grouping by BER only, averaging across all images and trials")
    plot_averaged_results(jpeg_niqe_data, turbo_niqe_data, ddcm_niqe_data, models, args.output_dir, metric='NIQE')
    
    print("\nCreating error count plot...")
    plot_error_counts(args.jpeg_csv, args.turbo_csv, ddcm_csv_combined, models, args.output_dir, args.ber_min, args.ber_max)
    
    if ddcm_csv_combined and ddcm_csv_combined.startswith(args.output_dir):
        os.remove(ddcm_csv_combined)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

