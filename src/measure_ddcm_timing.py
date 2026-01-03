#!/usr/bin/env python3
"""
Script to measure average compression and decompression times for DDCM
at different BPP values (0.1, 0.5, 1.0).
"""

import os
import sys
import time
import pandas as pd
import shutil

# Set up paths for API files
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_ddcm_path = os.path.join(_project_root, 'ddcm-compressed-image-generation-main')

if _ddcm_path not in sys.path:
    sys.path.insert(0, _ddcm_path)

from ddcm_api import main_programmatic as ddcm_main_programmatic


def load_ddcm_params(csv_path, target_bpp):
    """Load DDCM parameters for a given target BPP."""
    df = pd.read_csv(csv_path)
    df = df[df['target_bpp'] == target_bpp]
    if len(df) == 0:
        return None
    row = df.iloc[0]
    return {
        'M': int(row['M']),
        'T': int(row['T']),
        'K': int(row['K']),
        'C': int(row['C'])
    }


def measure_compress_time(img_path, params, output_dir, resolution=512):
    """Measure compression time for a single image."""
    temp_input = os.path.join(output_dir, 'temp_input_timing')
    os.makedirs(temp_input, exist_ok=True)
    shutil.copy(img_path, os.path.join(temp_input, os.path.basename(img_path)))
    
    M = params['M']
    T = params['T']
    K = params['K']
    C = params['C']
    
    try:
        start_time = time.time()
        ddcm_main_programmatic(
            mode='compress',
            input_dir=temp_input,
            output_dir=output_dir,
            gpu=0,
            float16=True,
            model_id='Manojb/stable-diffusion-2-1-base',
            K=K,
            T=T,
            M=M,
            C=C,
            t_range=[T-1, 0]
        )
        elapsed_time = time.time() - start_time
        return elapsed_time
    finally:
        if os.path.exists(temp_input):
            if os.path.islink(temp_input):
                os.unlink(temp_input)
            else:
                shutil.rmtree(temp_input)


def measure_decompress_time(bin_file, output_dir, resolution=512):
    """Measure decompression time for a single binary file."""
    temp_input = os.path.join(output_dir, 'temp_input_timing')
    temp_output = os.path.join(output_dir, 'temp_output_timing')
    
    # Clean directories
    if os.path.exists(temp_input):
        if os.path.islink(temp_input):
            os.unlink(temp_input)
        else:
            shutil.rmtree(temp_input)
    if os.path.exists(temp_output):
        if os.path.islink(temp_output):
            os.unlink(temp_output)
        else:
            shutil.rmtree(temp_output)
    
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)
    
    # Create directory structure matching the bin_file location
    bin_dir = os.path.dirname(bin_file)
    bin_subdir = os.path.basename(bin_dir)
    temp_bin_dir = os.path.join(temp_input, bin_subdir)
    os.makedirs(temp_bin_dir, exist_ok=True)
    
    bin_basename = os.path.basename(bin_file)
    temp_bin_file = os.path.join(temp_bin_dir, bin_basename)
    shutil.copy(bin_file, temp_bin_file)
    
    try:
        start_time = time.time()
        ddcm_main_programmatic(
            mode='decompress',
            input_dir=temp_input,
            output_dir=temp_output,
            gpu=0,
            float16=True
        )
        elapsed_time = time.time() - start_time
        return elapsed_time
    finally:
        if os.path.exists(temp_input):
            if os.path.islink(temp_input):
                os.unlink(temp_input)
            else:
                shutil.rmtree(temp_input)
        if os.path.exists(temp_output):
            if os.path.islink(temp_output):
                os.unlink(temp_output)
            else:
                shutil.rmtree(temp_output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure DDCM compression/decompression timing')
    parser.add_argument('--test_image', required=True, help='Path to test image file')
    parser.add_argument('--params_csv', required=True, help='Path to DDCM parameters CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for compressed files')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of trials to average (default: 3)')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution (default: 512)')
    args = parser.parse_args()
    
    if not os.path.exists(args.test_image):
        print(f"Error: Test image not found: {args.test_image}")
        return
    
    if not os.path.exists(args.params_csv):
        print(f"Error: Parameters CSV not found: {args.params_csv}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    bpp_values = [0.1, 0.5, 1.0]
    
    print("=" * 70)
    print("DDCM Timing Measurement")
    print("=" * 70)
    print(f"Test image: {args.test_image}")
    print(f"Number of trials per BPP: {args.num_trials}")
    print(f"Resolution: {args.resolution}")
    print("=" * 70)
    print()
    
    results = {}
    
    for bpp in bpp_values:
        print(f"\n{'='*70}")
        print(f"BPP: {bpp}")
        print(f"{'='*70}")
        
        # Load parameters
        params = load_ddcm_params(args.params_csv, bpp)
        if params is None:
            print(f"  ⚠ No parameters found for BPP {bpp}, skipping...")
            continue
        
        print(f"  Parameters: T={params['T']}, K={params['K']}, M={params['M']}, C={params['C']}")
        print()
        
        # Measure compression times
        compress_times = []
        print("  Measuring compression times...")
        for trial in range(args.num_trials):
            print(f"    Trial {trial+1}/{args.num_trials}...", end=' ', flush=True)
            compress_time = measure_compress_time(args.test_image, params, args.output_dir, args.resolution)
            compress_times.append(compress_time)
            print(f"{compress_time:.2f}s")
        
        avg_compress = sum(compress_times) / len(compress_times)
        print(f"  Average compression time: {avg_compress:.2f}s")
        print()
        
        # Find the compressed file
        base_name = os.path.splitext(os.path.basename(args.test_image))[0]
        T = params['T']
        K = params['K']
        M = params['M']
        C = params['C']
        out_prefix = f'T={T}_in{T-1}-0_K={K}_M={M}_C={C}_model=stable-diffusion-2-1-base'
        bin_file = os.path.join(args.output_dir, out_prefix, f'{base_name}_noise_indices.bin')
        
        if not os.path.exists(bin_file):
            print(f"  ⚠ Compressed file not found: {bin_file}, skipping decompression timing")
            results[bpp] = {
                'compress_avg': avg_compress,
                'compress_times': compress_times,
                'decompress_avg': None,
                'decompress_times': []
            }
            continue
        
        # Measure decompression times
        decompress_times = []
        print("  Measuring decompression times...")
        for trial in range(args.num_trials):
            print(f"    Trial {trial+1}/{args.num_trials}...", end=' ', flush=True)
            decompress_time = measure_decompress_time(bin_file, args.output_dir, args.resolution)
            decompress_times.append(decompress_time)
            print(f"{decompress_time:.2f}s")
        
        avg_decompress = sum(decompress_times) / len(decompress_times)
        print(f"  Average decompression time: {avg_decompress:.2f}s")
        
        results[bpp] = {
            'compress_avg': avg_compress,
            'compress_times': compress_times,
            'decompress_avg': avg_decompress,
            'decompress_times': decompress_times
        }
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'BPP':<10} {'Compress (avg)':<20} {'Decompress (avg)':<20}")
    print("-" * 70)
    for bpp in bpp_values:
        if bpp in results:
            compress_avg = results[bpp]['compress_avg']
            decompress_avg = results[bpp]['decompress_avg']
            decompress_str = f"{decompress_avg:.2f}s" if decompress_avg is not None else "N/A"
            print(f"{bpp:<10} {compress_avg:<20.2f} {decompress_str:<20}")
    print("=" * 70)


if __name__ == '__main__':
    main()

