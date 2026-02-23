#!/usr/bin/env python3
"""
Analyze noisy channel experiment results and create visualization plots.

Creates:
1. Per-image plots: 6×4 grid showing PSNR/NIQE vs BER for each image (separate figures for PSNR and NIQE)
2. Averaged plots: Mean PSNR/NIQE vs BER across all images and trials (separate figures for PSNR and NIQE)
"""

# import argparse
# import csv
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from pathlib import Path
#
# COLORS = {'jpeg': 'blue', 'turbo': 'red', 'turbo_improved': 'orange', 'ddcm': 'green'}
# MARKERS = {'jpeg': 'o', 'turbo': 's', 'turbo_improved': 'D', 'ddcm': '^'}
# LABELS = {'jpeg': 'JPEG', 'turbo': 'Turbo-DDCM', 'turbo_improved': 'Robust Turbo-DDCM', 'ddcm': 'DDCM'}
#
#
# def get_image_number(image_name):
#     """Extract image number from filename like '1.png' -> 1."""
#     try:
#         return int(image_name.replace('.png', ''))
#     except:
#         return 0
#
#
# def plot_per_image_results(jpeg_data, turbo_data, turbo_improved_data, ddcm_data, models, output_dir, metric='PSNR'):
#     """Create 6×4 grid of subplots, one per image, showing metric vs BER.
#
#     Grouping: For each image, group by BER value, then average across all trials.
#     Baseline (BER=0) is shown as a horizontal dashed line.
#     """
#     fig, axes = plt.subplots(6, 4, figsize=(16, 24))
#     fig.suptitle(f'{metric} vs BER (Log Scale) - Per Image', fontsize=16, y=0.995)
#
#     all_images = sorted(set([d['image'] for d in jpeg_data] +
#                            [d['image'] for d in turbo_data] +
#                            [d['image'] for d in turbo_improved_data] +
#                            [d['image'] for d in ddcm_data]),
#                        key=get_image_number)
#
#     metric_key = metric.lower()
#
#     for idx, image_name in enumerate(all_images):
#         row = idx // 4
#         col = idx % 4
#         ax = axes[row, col]
#
#         jpeg_image_data = [d for d in jpeg_data if d['image'] == image_name] if 'jpeg' in models else []
#         turbo_image_data = [d for d in turbo_data if d['image'] == image_name] if 'turbo' in models else []
#         turbo_improved_image_data = [d for d in turbo_improved_data if d['image'] == image_name] if 'turbo_improved' in models else []
#         ddcm_image_data = [d for d in ddcm_data if d['image'] == image_name] if 'ddcm' in models else []
#
#         ax.set_xscale('log')
#
#         if 'jpeg' in models and jpeg_image_data:
#             jpeg_bers = sorted(set([d['ber'] for d in jpeg_image_data if d['ber'] > 0]))
#             jpeg_baseline_data = [d[metric_key] for d in jpeg_image_data if d['ber'] == 0]
#             jpeg_baseline_value = np.mean(jpeg_baseline_data) if jpeg_baseline_data else None
#
#             jpeg_bers_plot = []
#             jpeg_means = []
#             jpeg_stds = []
#             for ber in jpeg_bers:
#                 values = [d[metric_key] for d in jpeg_image_data if d['ber'] == ber]
#                 if values:
#                     jpeg_bers_plot.append(ber)
#                     jpeg_means.append(np.mean(values))
#                     jpeg_stds.append(np.std(values))
#
#             if jpeg_bers_plot:
#                 ax.plot(jpeg_bers_plot, jpeg_means, f'{markers["jpeg"]}-',
#                        label=labels['jpeg'], color=colors['jpeg'],
#                        markersize=4, linewidth=1.5, zorder=2)
#             if jpeg_baseline_value is not None:
#                 ax.axhline(y=jpeg_baseline_value, color=colors['jpeg'], linestyle='--',
#                           linewidth=1.5, alpha=0.7, label=f'{labels["jpeg"]} (baseline)', zorder=5)
#
#         if 'turbo' in models and turbo_image_data:
#             turbo_bers = sorted(set([d['ber'] for d in turbo_image_data if d['ber'] > 0]))
#             turbo_baseline_data = [d[metric_key] for d in turbo_image_data if d['ber'] == 0]
#             turbo_baseline_value = np.mean(turbo_baseline_data) if turbo_baseline_data else None
#
#             turbo_bers_plot = []
#             turbo_means = []
#             turbo_stds = []
#             for ber in turbo_bers:
#                 values = [d[metric_key] for d in turbo_image_data if d['ber'] == ber]
#                 if values:
#                     turbo_bers_plot.append(ber)
#                     turbo_means.append(np.mean(values))
#                     turbo_stds.append(np.std(values))
#
#             if turbo_bers_plot:
#                 ax.plot(turbo_bers_plot, turbo_means, f'{markers["turbo"]}-',
#                        label=labels['turbo'], color=colors['turbo'],
#                        markersize=4, linewidth=1.5, zorder=2)
#             if turbo_baseline_value is not None:
#                 ax.axhline(y=turbo_baseline_value, color=colors['turbo'], linestyle='--',
#                           linewidth=1.5, alpha=0.7, label=f'{labels["turbo"]} (baseline)', zorder=5)
#
#         if 'turbo_improved' in models and turbo_improved_image_data:
#             turbo_improved_bers = sorted(set([d['ber'] for d in turbo_improved_image_data if d['ber'] > 0]))
#             turbo_improved_baseline_data = [d[metric_key] for d in turbo_improved_image_data if d['ber'] == 0]
#             turbo_improved_baseline_value = np.mean(turbo_improved_baseline_data) if turbo_improved_baseline_data else None
#
#             turbo_improved_bers_plot = []
#             turbo_improved_means = []
#             turbo_improved_stds = []
#             for ber in turbo_improved_bers:
#                 values = [d[metric_key] for d in turbo_improved_image_data if d['ber'] == ber]
#                 if values:
#                     turbo_improved_bers_plot.append(ber)
#                     turbo_improved_means.append(np.mean(values))
#                     turbo_improved_stds.append(np.std(values))
#
#             if turbo_improved_bers_plot:
#                 ax.plot(turbo_improved_bers_plot, turbo_improved_means, f'{markers["turbo_improved"]}-',
#                        label=labels['turbo_improved'], color=colors['turbo_improved'],
#                        markersize=4, linewidth=1.5, zorder=2)
#             if turbo_improved_baseline_value is not None:
#                 ax.axhline(y=turbo_improved_baseline_value, color=colors['turbo_improved'], linestyle='--',
#                           linewidth=1.5, alpha=0.7, label=f'{labels["turbo_improved"]} (baseline)', zorder=5)
#
#         if 'ddcm' in models and ddcm_image_data:
#             ddcm_bers = sorted(set([d['ber'] for d in ddcm_image_data if d['ber'] > 0]))
#             ddcm_baseline_data = [d[metric_key] for d in ddcm_image_data if d['ber'] == 0]
#             ddcm_baseline_value = np.mean(ddcm_baseline_data) if ddcm_baseline_data else None
#
#             ddcm_bers_plot = []
#             ddcm_means = []
#             ddcm_stds = []
#             for ber in ddcm_bers:
#                 values = [d[metric_key] for d in ddcm_image_data if d['ber'] == ber]
#                 if values:
#                     ddcm_bers_plot.append(ber)
#                     ddcm_means.append(np.mean(values))
#                     ddcm_stds.append(np.std(values))
#
#             if ddcm_bers_plot:
#                 ax.plot(ddcm_bers_plot, ddcm_means, f'{markers["ddcm"]}-',
#                        label=labels['ddcm'], color=colors['ddcm'],
#                        markersize=4, linewidth=1.5, zorder=2)
#             if ddcm_baseline_value is not None:
#                 ax.axhline(y=ddcm_baseline_value, color=colors['ddcm'], linestyle='--',
#                           linewidth=1.5, alpha=0.7, label=f'{labels["ddcm"]} (baseline)', zorder=5)
#
#         ax.set_xlabel('BER', fontsize=9)
#         ax.set_ylabel(metric, fontsize=9)
#         ax.set_title(f'{image_name}', fontsize=10)
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     output_path = os.path.join(output_dir, f'{metric.lower()}_per_image.png')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {output_path}")
#
#
# def plot_averaged_results(data, output_dir, metric='PSNR'):
#     """Create averaged plot showing mean metric vs BER across all images and trials.
#
#     Grouping: Group all measurements by BER value (across all images and trials),
#     then compute mean and std for each BER.
#     Baseline (BER=0) is excluded from noisy data plots and shown separately as a horizontal dashed line.
#     Variance bands (mean ± std) are shown as shaded regions around the noisy data curves.
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     ax.set_xscale('log')
#     ax.set_xlabel('BER (Log Scale)', fontsize=12)
#     ax.set_ylabel(metric, fontsize=12)
#     # ax.set_title(f'Average {metric} vs BER (Log Scale) - All Images and Trials', fontsize=14)
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=11)
#
#     plt.tight_layout()
#     output_path = os.path.join(output_dir, f'{metric.lower()}_averaged.pdf')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"Saved: {output_path}")
#
#
# def plot_error_counts(jpeg_csv, turbo_csv, turbo_improved_csv, ddcm_csv, models, output_dir, ber_min=None, ber_max=None):
#     """Plot decoding failure counts per BER for selected methods."""
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     colors = {'jpeg': 'blue', 'turbo': 'red', 'turbo_improved': 'orange', 'ddcm': 'green'}
#     markers = {'jpeg': 'o', 'turbo': 's', 'turbo_improved': 'D', 'ddcm': '^'}
#     labels = {'jpeg': 'JPEG', 'turbo': 'Turbo-DDCM', 'turbo_improved': 'Robust Turbo-DDCM', 'ddcm': 'DDCM'}
#
#     jpeg_all_data = load_all_csv_data(jpeg_csv, ber_min, ber_max) if 'jpeg' in models and jpeg_csv else []
#     turbo_all_data = load_all_csv_data(turbo_csv, ber_min, ber_max) if 'turbo' in models and turbo_csv else []
#     turbo_improved_all_data = load_all_csv_data(turbo_improved_csv, ber_min, ber_max) if 'turbo_improved' in models and turbo_improved_csv else []
#     ddcm_all_data = load_all_csv_data(ddcm_csv, ber_min, ber_max) if 'ddcm' in models and ddcm_csv else []
#
#     all_bers = sorted(set([d['ber'] for d in jpeg_all_data if d['ber'] > 0] +
#                           [d['ber'] for d in turbo_all_data if d['ber'] > 0] +
#                           [d['ber'] for d in turbo_improved_all_data if d['ber'] > 0] +
#                           [d['ber'] for d in ddcm_all_data if d['ber'] > 0]))
#
#     if 'jpeg' in models and jpeg_all_data:
#         jpeg_error_counts = []
#         jpeg_total_counts = []
#         jpeg_bers_plot = []
#
#         for ber in all_bers:
#             jpeg_rows = [d for d in jpeg_all_data if d['ber'] == ber]
#             if jpeg_rows:
#                 jpeg_failures = sum(1 for d in jpeg_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
#                 jpeg_total = len(jpeg_rows)
#                 jpeg_bers_plot.append(ber)
#                 jpeg_error_counts.append(jpeg_failures)
#                 jpeg_total_counts.append(jpeg_total)
#
#         if jpeg_bers_plot:
#             ax.plot(jpeg_bers_plot, jpeg_error_counts, f'{markers["jpeg"]}-',
#                    label=f'{labels["jpeg"]} failures', color=colors['jpeg'],
#                    markersize=6, linewidth=2, alpha=0.8)
#             print(f"  JPEG error counts: {dict(zip(jpeg_bers_plot, jpeg_error_counts))}")
#             print(f"  JPEG total counts: {dict(zip(jpeg_bers_plot, jpeg_total_counts))}")
#
#     if 'turbo' in models and turbo_all_data:
#         turbo_error_counts = []
#         turbo_total_counts = []
#         turbo_bers_plot = []
#
#         for ber in all_bers:
#             turbo_rows = [d for d in turbo_all_data if d['ber'] == ber]
#             if turbo_rows:
#                 turbo_failures = sum(1 for d in turbo_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
#                 turbo_total = len(turbo_rows)
#                 turbo_bers_plot.append(ber)
#                 turbo_error_counts.append(turbo_failures)
#                 turbo_total_counts.append(turbo_total)
#
#         if turbo_bers_plot:
#             ax.plot(turbo_bers_plot, turbo_error_counts, f'{markers["turbo"]}-',
#                    label=f'{labels["turbo"]} failures', color=colors['turbo'],
#                    markersize=6, linewidth=2, alpha=0.8)
#             print(f"  Turbo-DDCM error counts: {dict(zip(turbo_bers_plot, turbo_error_counts))}")
#             print(f"  Turbo-DDCM total counts: {dict(zip(turbo_bers_plot, turbo_total_counts))}")
#
#     if 'turbo_improved' in models and turbo_improved_all_data:
#         turbo_improved_error_counts = []
#         turbo_improved_total_counts = []
#         turbo_improved_bers_plot = []
#
#         for ber in all_bers:
#             turbo_improved_rows = [d for d in turbo_improved_all_data if d['ber'] == ber]
#             if turbo_improved_rows:
#                 turbo_improved_failures = sum(1 for d in turbo_improved_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
#                 turbo_improved_total = len(turbo_improved_rows)
#                 turbo_improved_bers_plot.append(ber)
#                 turbo_improved_error_counts.append(turbo_improved_failures)
#                 turbo_improved_total_counts.append(turbo_improved_total)
#
#         if turbo_improved_bers_plot:
#             ax.plot(turbo_improved_bers_plot, turbo_improved_error_counts, f'{markers["turbo_improved"]}-',
#                    label=f'{labels["turbo_improved"]} failures', color=colors['turbo_improved'],
#                    markersize=6, linewidth=2, alpha=0.8)
#             print(f"  Robust Turbo-DDCM error counts: {dict(zip(turbo_improved_bers_plot, turbo_improved_error_counts))}")
#             print(f"  Robust Turbo-DDCM total counts: {dict(zip(turbo_improved_bers_plot, turbo_improved_total_counts))}")
#
#     if 'ddcm' in models and ddcm_all_data:
#         ddcm_error_counts = []
#         ddcm_total_counts = []
#         ddcm_bers_plot = []
#
#         for ber in all_bers:
#             ddcm_rows = [d for d in ddcm_all_data if d['ber'] == ber]
#             if ddcm_rows:
#                 ddcm_failures = sum(1 for d in ddcm_rows if d['psnr'] == 'N/A' or d['niqe'] == 'N/A')
#                 ddcm_total = len(ddcm_rows)
#                 ddcm_bers_plot.append(ber)
#                 ddcm_error_counts.append(ddcm_failures)
#                 ddcm_total_counts.append(ddcm_total)
#
#         if ddcm_bers_plot:
#             ax.plot(ddcm_bers_plot, ddcm_error_counts, f'{markers["ddcm"]}-',
#                    label=f'{labels["ddcm"]} failures', color=colors['ddcm'],
#                    markersize=6, linewidth=2, alpha=0.8)
#             print(f"  DDCM error counts: {dict(zip(ddcm_bers_plot, ddcm_error_counts))}")
#             print(f"  DDCM total counts: {dict(zip(ddcm_bers_plot, ddcm_total_counts))}")
#
#     ax.set_xscale('log')
#     ax.set_xlabel('BER (Log Scale)', fontsize=12)
#     ax.set_ylabel('Number of Decoding Failures', fontsize=12)
#     ax.set_title('Decoding Failures vs BER - All Images and Trials', fontsize=14)
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=11)
#
#     plt.tight_layout()
#     output_path = os.path.join(output_dir, 'error_counts.png')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {output_path}")
#
#
# def concatenate_csv_files(csv_paths):
#     """Concatenate multiple CSV files into a single list of data rows."""
#     all_data = []
#     for csv_path in csv_paths:
#         if csv_path and os.path.exists(csv_path):
#             with open(csv_path, 'r') as f:
#                 reader = csv.DictReader(f)
#                 all_data.extend(list(reader))
#     return all_data
#
#
# def write_concatenated_csv(all_data, output_path):
#     """Write concatenated data to a temporary CSV file."""
#     if not all_data:
#         return None
#
#     fieldnames = all_data[0].keys()
#     with open(output_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(all_data)
#     return output_path
#
#
# def main():
#     parser = argparse.ArgumentParser(description='Analyze noisy channel experiment results')
#     parser.add_argument('--jpeg_csv', type=str, default=None,
#                        help='Path to JPEG results CSV file')
#     parser.add_argument('--turbo_csv', type=str, default=None,
#                        help='Path to Turbo-DDCM (improved protocol) results CSV file')
#     parser.add_argument('--turbo_improved_csv', type=str, default=None,
#                        help='Path to Robust Turbo-DDCM (old protocol) results CSV file')
#     parser.add_argument('--ddcm_csv', type=str, nargs='+', default=None,
#                        help='Path(s) to DDCM results CSV file(s). Can provide multiple files to concatenate (e.g., first_half and second_half)')
#     parser.add_argument('--output_dir', type=str, required=True,
#                        help='Output directory for plots')
#     parser.add_argument('--methods', type=str, nargs='+', default=['jpeg', 'turbo', 'ddcm'],
#                        choices=['jpeg', 'turbo', 'turbo_improved', 'ddcm'],
#                        help='Which models to include in plots (default: all)')
#     parser.add_argument('--ber_min', type=float, default=None,
#                        help='Minimum BER to include in plots (e.g., 1e-6). If None, include all BERs')
#     parser.add_argument('--ber_max', type=float, default=None,
#                        help='Maximum BER to include in plots (e.g., 1e-5). If None, include all BERs')
#
#     args = parser.parse_args()
#
#     if not args.jpeg_csv and not args.turbo_csv and not args.turbo_improved_csv and not args.ddcm_csv:
#         parser.error("At least one of --jpeg_csv, --turbo_csv, --turbo_improved_csv, or --ddcm_csv must be provided")
#
#     models = [m.lower() for m in args.models]
#
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     ddcm_csv_combined = None
#     if 'ddcm' in models and args.ddcm_csv:
#         if len(args.ddcm_csv) > 1:
#             print(f"Concatenating DDCM CSV files: {args.ddcm_csv}")
#             all_ddcm_data = concatenate_csv_files(args.ddcm_csv)
#             ddcm_csv_combined = os.path.join(args.output_dir, 'ddcm_combined.csv')
#             write_concatenated_csv(all_ddcm_data, ddcm_csv_combined)
#             print(f"  Combined {len(all_ddcm_data)} rows into: {ddcm_csv_combined}")
#         else:
#             ddcm_csv_combined = args.ddcm_csv[0]
#
#     results_df = pd.DataFrame()
#     for method in args.methods:
#         if args[f'{method}_csv'] is None:
#             continue
#
#         current_df = pd.read_csv(args[f'{method}_csv'])
#         current_df['method'] = method
#         results_df = results_df.concat([results_df, current_df])
#
#     if args.ber_min is not None:
#         results_df = results_df[results_df['ber'] >= args.ber_min]
#     if args.ber_max is not None:
#         results_df = results_df[results_df['ber'] <= args.ber_max]
#
#
#     print("\nCreating PSNR plots...")
#     print("  Per-image: Grouping by (image, BER), averaging across trials")
#     # plot_per_image_results(jpeg_psnr_data, turbo_psnr_data, turbo_improved_psnr_data, ddcm_psnr_data, models, args.output_dir, metric='PSNR')
#     print("  Averaged: Grouping by BER only, averaging across all images and trials")
#     plot_averaged_results(jpeg_psnr_data, turbo_psnr_data, turbo_improved_psnr_data, ddcm_psnr_data, models, args.output_dir, metric='PSNR')
#
#     print("\nCreating NIQE plots...")
#     print("  Per-image: Grouping by (image, BER), averaging across trials")
#     plot_per_image_results(jpeg_niqe_data, turbo_niqe_data, turbo_improved_niqe_data, ddcm_niqe_data, models, args.output_dir, metric='NIQE')
#     print("  Averaged: Grouping by BER only, averaging across all images and trials")
#     plot_averaged_results(jpeg_niqe_data, turbo_niqe_data, turbo_improved_niqe_data, ddcm_niqe_data, models, args.output_dir, metric='NIQE')
#
#     print("\nCreating error count plot...")
#     plot_error_counts(args.jpeg_csv, args.turbo_csv, args.turbo_improved_csv, ddcm_csv_combined, models, args.output_dir, args.ber_min, args.ber_max)
#
#     if ddcm_csv_combined and ddcm_csv_combined.startswith(args.output_dir):
#         os.remove(ddcm_csv_combined)
#
#     print(f"\nAll plots saved to: {args.output_dir}")
#
#
# if __name__ == '__main__':
#     main()
#
