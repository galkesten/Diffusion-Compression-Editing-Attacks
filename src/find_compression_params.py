import os
import sys
import csv
import subprocess
import shutil
import argparse
from PIL import Image
import io
import torch

import turbo_ddcm.utils as utils


def find_jpeg_quality_for_bpp(img, target_bpp, resize_to=(512, 512)):
    low, high = 1, 100
    best_quality, best_bpp, best_diff = 1, 0, float('inf')
    for _ in range(20):
        mid = (low + high) // 2
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=mid)
        file_size_bits = len(buffer.getvalue()) * 8
        pixels = resize_to[0] * resize_to[1]
        bpp = file_size_bits / pixels
        diff = abs(bpp - target_bpp)
        if diff < best_diff:
            best_diff = diff
            best_quality = mid
            best_bpp = bpp
        if bpp < target_bpp:
            low = mid + 1
        else:
            high = mid - 1
    return best_quality, best_bpp

def find_turbo_ddcm_M_for_bpp(target_bpp, T, K, C):
    img_height, img_width = 512, 512
    
    low, high = 1, min(K, 10000)
    best_M, best_diff = 1, float('inf')
    for _ in range(30):
        mid = (low + high) // 2
        try:
            bpp = utils.turbo_ddcm_bpp(T, K, mid, C, 0, img_height, img_width)
            diff = abs(bpp - target_bpp)
            if diff < best_diff:
                best_diff = diff
                best_M = mid
            if bpp < target_bpp:
                low = mid + 1
            else:
                high = mid - 1
        except:
            high = mid - 1
    return best_M

def turbo_ddcm_bpp_old_protocol(T, K, M, C, NBS, img_height, img_width):
    """Calculate BPP for Turbo-DDCM using old protocol encoding."""
    import math
    # Old protocol: M * (ceil(log2(K)) + C) bits per iteration
    
    
    bits_per_iteration = M * (math.ceil(math.log2(K)) + C)
    bits = (T - NBS - 1) * bits_per_iteration
    bpp = bits / (img_height * img_width)
    return round(bpp, 8)

def turbo_ddcm_bpp_new_protocol_improved_psnr(T, K, M, C, img_height, img_width):

    bpp = turbo_ddcm_bpp_old_protocol(T, K, M, C, 0, img_height, img_width)
    bins = torch.logspace(
    start=torch.log10(torch.tensor(0.01)), end=torch.log10(torch.tensor(0.15)), steps=70)


    nbs = torch.max(torch.tensor(0),
                torch.min(torch.tensor(T - 2),
                            # NBS <= T - 2 (we have to do one step with bits and we have one DDIM step
                            # either way at the end of the DDPM process).
                            70 - torch.bucketize(bpp, bins) - 1)).item()
    
    return turbo_ddcm_bpp_old_protocol(T, K, M, C, nbs, img_height, img_width)

def find_turbo_ddcm_M_for_bpp_old_protocol(target_bpp, T, K, C):
    img_height, img_width = 512, 512
    
    low, high = 1, min(K, 10000)
    best_M, best_diff = 1, float('inf')
    for _ in range(30):
        mid = (low + high) // 2
        try:
            bpp = turbo_ddcm_bpp_old_protocol(T, K, mid, C, 0, img_height, img_width)
            diff = abs(bpp - target_bpp)
            if diff < best_diff:
                best_diff = diff
                best_M = mid
            if bpp < target_bpp:
                low = mid + 1
            else:
                high = mid - 1
        except:
            high = mid - 1
    return best_M

def find_turbo_ddcm_M_for_bpp_old_protocol_improved_psnr(target_bpp, T, K, C):
    print("Finding Turbo-DDCM M for BPP using old protocol and improved PSNR")
    print(f"Target BPP: {target_bpp}")
    print(f"T: {T}")
    print(f"K: {K}")
    print(f"C: {C}")
    print(f"--------------------------------")
    img_height, img_width = 512, 512
    
    low, high = 1, min(K, 10000)
    best_M, best_diff = 1, float('inf')
    for _ in range(30):
        mid = (low + high) // 2
        try:
            bpp = turbo_ddcm_bpp_new_protocol_improved_psnr(T, K, mid, C, img_height, img_width)
            diff = abs(bpp - target_bpp)
            if diff < best_diff:
                best_diff = diff
                best_M = mid
            if bpp < target_bpp:
                low = mid + 1
            else:
                high = mid - 1
        except:
            high = mid - 1
    return best_M

def ddcm_bpp(T, K, M, C, imsize, optimized_Ts):
    import numpy as np
    return (optimized_Ts - 1) * (M * np.log2(K) + (M - 1) * C) / (imsize ** 2)

def find_ddcm_M_for_bpp(target_bpp, T, K, C, imsize=512, t_range=(999, 0)):
    import numpy as np
    optimized_Ts = T  # Approximation: t_range=(999,0) covers all timesteps
    
    low, high = 1, 100
    best_M, best_diff = 1, float('inf')
    for _ in range(30):
        mid = (low + high) // 2
        try:
            bpp = ddcm_bpp(T, K, mid, C, imsize, optimized_Ts)
            diff = abs(bpp - target_bpp)
            if diff < best_diff:
                best_diff = diff
                best_M = mid
            if bpp < target_bpp:
                low = mid + 1
            else:
                high = mid - 1
        except:
            high = mid - 1
    return best_M

def get_sample_images(dataset_path, num_samples=2):
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    return image_files[:num_samples]

def get_all_images(dataset_path):
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    return image_files

def run_jpeg_experiment(dataset_path, all_images, target_bpps, project_root, csv_path):
    resize_to = (512, 512)
    img_height, img_width = resize_to
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'jpeg')
    os.makedirs(results_dir, exist_ok=True)
    
    for target_bpp in target_bpps:
        output_dir = os.path.join(results_dir, f'dataset_Kodack24_jpeg_bpp{target_bpp}')
        os.makedirs(output_dir, exist_ok=True)
        
        for img_file in all_images:
            img_path = os.path.join(dataset_path, img_file)
            
            img = Image.open(img_path).convert('RGB')
            if resize_to and img.size != resize_to:
                img = img.resize(resize_to)
            
            quality, _ = find_jpeg_quality_for_bpp(img, target_bpp, resize_to)
            
            base_name = os.path.splitext(img_file)[0]
            jpeg_path = os.path.join(output_dir, f'{base_name}_quality{quality}.jpg')
            img.save(jpeg_path, format='JPEG', quality=quality)
            
            actual_bpp = calculate_actual_bpp(jpeg_path, img_height, img_width)
            append_to_jpeg_csv(csv_path, img_file, target_bpp, quality, actual_bpp)
            
            reconstructed_img = Image.open(jpeg_path).convert('RGB')
            png_path = os.path.join(output_dir, f'{base_name}_quality{quality}.png')
            reconstructed_img.save(png_path, format='PNG')
            os.remove(jpeg_path)
    
    return {}, {}

def calculate_actual_bpp(binary_file_path, img_height=512, img_width=512):
    file_size_bytes = os.path.getsize(binary_file_path)
    file_size_bits = file_size_bytes * 8
    pixels = img_height * img_width
    return file_size_bits / pixels

def run_turbo_ddcm_experiment(dataset_path, sample_images, target_bpps, project_root, theoretical_csv_path, actual_csv_path):
    T = 20
    K = 16384
    C = 1
    seed = 88888888
    turbo_ddcm_path = os.path.join(project_root, 'Turbo-DDCM-master')
    img_height, img_width = 512, 512
    
    turbo_ddcm_Ms = {}
    turbo_ddcm_actual_bpps = {}
    
    for target_bpp in target_bpps:
        M = find_turbo_ddcm_M_for_bpp(target_bpp, T, K, C)
        turbo_ddcm_Ms[target_bpp] = M
        theoretical_bpp = utils.turbo_ddcm_bpp(T, K, M, C, 0, img_height, img_width)
        append_to_turbo_ddcm_theoretical_csv(theoretical_csv_path, target_bpp, M, theoretical_bpp, T, K, C)
    
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'turbo_ddcm')
    os.makedirs(results_dir, exist_ok=True)
    
    for target_bpp in target_bpps:
        compression_dir = os.path.join(results_dir, f'dataset_Kodack24_turbo_ddcm_bpp{target_bpp}_compressed')
        decompression_dir = os.path.join(results_dir, f'dataset_Kodack24_turbo_ddcm_bpp{target_bpp}_decompressed')
        os.makedirs(compression_dir, exist_ok=True)
        os.makedirs(decompression_dir, exist_ok=True)
        M = turbo_ddcm_Ms[target_bpp]
        
        temp_input_dir = os.path.join(compression_dir, 'temp_input')
        os.makedirs(temp_input_dir, exist_ok=True)
        for img_file in sample_images:
            src_path = os.path.join(dataset_path, img_file)
            dst_path = os.path.join(temp_input_dir, img_file)
            shutil.copy(src_path, dst_path)
        
        roundtrip_script = os.path.join(turbo_ddcm_path, 'roundtrip.py')
        cmd = [
            sys.executable, roundtrip_script,
            '--input_dir', os.path.abspath(temp_input_dir),
            '--output_compression_dir', os.path.abspath(compression_dir),
            '--output_decompression_dir', os.path.abspath(decompression_dir),
            '--M', str(M),
            '--T', str(T),
            '--K', str(K),
            '--seed', str(seed),
            '--save_reconstructions'
        ]
        subprocess.run(cmd, check=True, cwd=turbo_ddcm_path)
        
        actual_bpps = []
        for img_file in sample_images:
            base_name = os.path.splitext(img_file)[0]
            binary_path = os.path.join(compression_dir, f'{base_name}{utils.BIN_SUFFIX}')
            if os.path.exists(binary_path):
                actual_bpp = calculate_actual_bpp(binary_path, img_height, img_width)
                actual_bpps.append(actual_bpp)
                new_binary = os.path.join(compression_dir, f'{base_name}_M{M}{utils.BIN_SUFFIX}')
                os.rename(binary_path, new_binary)
        
        avg_actual_bpp = sum(actual_bpps) / len(actual_bpps) if actual_bpps else 0
        turbo_ddcm_actual_bpps[target_bpp] = avg_actual_bpp
        append_to_turbo_ddcm_actual_csv(actual_csv_path, target_bpp, M, avg_actual_bpp, T, K, C)
        
        for img_file in sample_images:
            base_name = os.path.splitext(img_file)[0]
            recon_path = os.path.join(decompression_dir, img_file)
            new_recon = os.path.join(decompression_dir, f'{base_name}_M{M}.png')
            if os.path.exists(recon_path):
                os.rename(recon_path, new_recon)
        
        shutil.rmtree(temp_input_dir)
    
    return turbo_ddcm_Ms, T, K, turbo_ddcm_actual_bpps

def run_turbo_ddcm_old_protocol_experiment(dataset_path, sample_images, target_bpps, project_root, theoretical_csv_path, actual_csv_path, manual_list_ind=False):
    T = 30 if manual_list_ind else 20
    K = 16384
    C = 1
    seed = 88888888
    turbo_ddcm_path = os.path.join(project_root, 'Turbo-DDCM-master')
    img_height, img_width = 512, 512
    
    turbo_ddcm_Ms = {}
    turbo_ddcm_actual_bpps = {}
    
    for target_bpp in target_bpps:
        M = find_turbo_ddcm_M_for_bpp_old_protocol_improved_psnr(target_bpp, T, K, C) if manual_list_ind else find_turbo_ddcm_M_for_bpp_old_protocol(target_bpp, T, K, C)
        turbo_ddcm_Ms[target_bpp] = M
        theoretical_bpp = turbo_ddcm_bpp_new_protocol_improved_psnr(T, K, M, C, img_height, img_width) if manual_list_ind else turbo_ddcm_bpp_old_protocol(T, K, M, C, 0, img_height, img_width)
        append_to_turbo_ddcm_old_protocol_theoretical_csv(theoretical_csv_path, target_bpp, M, theoretical_bpp, T, K, C)
    
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'turbo_ddcm_old_protocol')
    os.makedirs(results_dir, exist_ok=True)
    
    for target_bpp in target_bpps:
        compression_dir = os.path.join(results_dir, f'dataset_Kodack24_turbo_ddcm_old_protocol_bpp{target_bpp}_compressed')
        decompression_dir = os.path.join(results_dir, f'dataset_Kodack24_turbo_ddcm_old_protocol_bpp{target_bpp}_decompressed')
        os.makedirs(compression_dir, exist_ok=True)
        os.makedirs(decompression_dir, exist_ok=True)
        M = turbo_ddcm_Ms[target_bpp]
        
        temp_input_dir = os.path.join(compression_dir, 'temp_input')
        os.makedirs(temp_input_dir, exist_ok=True)
        for img_file in sample_images:
            src_path = os.path.join(dataset_path, img_file)
            dst_path = os.path.join(temp_input_dir, img_file)
            shutil.copy(src_path, dst_path)
        
        roundtrip_script = os.path.join(turbo_ddcm_path, 'roundtrip.py')
        cmd = [
            sys.executable, roundtrip_script,
            '--input_dir', os.path.abspath(temp_input_dir),
            '--output_compression_dir', os.path.abspath(compression_dir),
            '--output_decompression_dir', os.path.abspath(decompression_dir),
            '--M', str(M),
            '--T', str(T),
            '--K', str(K),
            '--seed', str(seed),
            '--old_protocol',
            '--save_reconstructions'
        ]
        if manual_list_ind:
            cmd.append('--manual_list_ind')
        subprocess.run(cmd, check=True, cwd=turbo_ddcm_path)
        
        actual_bpps = []
        for img_file in sample_images:
            base_name = os.path.splitext(img_file)[0]
            binary_path = os.path.join(compression_dir, f'{base_name}{utils.BIN_SUFFIX}')
            if os.path.exists(binary_path):
                actual_bpp = calculate_actual_bpp(binary_path, img_height, img_width)
                actual_bpps.append(actual_bpp)
                new_binary = os.path.join(compression_dir, f'{base_name}_M{M}{utils.BIN_SUFFIX}')
                os.rename(binary_path, new_binary)
        
        avg_actual_bpp = sum(actual_bpps) / len(actual_bpps) if actual_bpps else 0
        turbo_ddcm_actual_bpps[target_bpp] = avg_actual_bpp
        append_to_turbo_ddcm_old_protocol_actual_csv(actual_csv_path, target_bpp, M, avg_actual_bpp, T, K, C)
        
        for img_file in sample_images:
            base_name = os.path.splitext(img_file)[0]
            recon_path = os.path.join(decompression_dir, img_file)
            new_recon = os.path.join(decompression_dir, f'{base_name}_M{M}.png')
            if os.path.exists(recon_path):
                os.rename(recon_path, new_recon)
        
        shutil.rmtree(temp_input_dir)
    
    return turbo_ddcm_Ms, T, K, turbo_ddcm_actual_bpps

def run_ddcm_experiment(dataset_path, sample_images, target_bpps, project_root, theoretical_csv_path, actual_csv_path):
    T = 1000
    K = 8192
    C = 3
    t_range = (999, 0)
    model_id = "Manojb/stable-diffusion-2-1-base"
    imsize = 512
    optimized_Ts = T  # Approximation for t_range=(999,0)
    ddcm_path = os.path.join(project_root, 'ddcm-compressed-image-generation-main')
    
    ddcm_Ms = {}
    
    for target_bpp in target_bpps:
        M = find_ddcm_M_for_bpp(target_bpp, T, K, C, imsize, t_range)
        ddcm_Ms[target_bpp] = M
        theoretical_bpp = ddcm_bpp(T, K, M, C, imsize, optimized_Ts)
        append_to_ddcm_theoretical_csv(theoretical_csv_path, target_bpp, M, theoretical_bpp, T, K, C)
    
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'ddcm')
    os.makedirs(results_dir, exist_ok=True)
    
    for target_bpp in target_bpps:
        output_dir = os.path.join(results_dir, f'dataset_Kodack24_ddcm_bpp{target_bpp}')
        os.makedirs(output_dir, exist_ok=True)
        M = ddcm_Ms[target_bpp]
        
        temp_input_dir = os.path.join(output_dir, 'temp_input')
        images_subdir = os.path.join(temp_input_dir, 'images')
        os.makedirs(images_subdir, exist_ok=True)
        for img_file in sample_images:
            src_path = os.path.join(dataset_path, img_file)
            dst_path = os.path.join(images_subdir, img_file)
            shutil.copy(src_path, dst_path)
        
        roundtrip_script = os.path.join(ddcm_path, 'latent_compression.py')
        # Use relative paths to avoid DDCM's path stripping bug
        rel_input_dir = os.path.relpath(images_subdir, ddcm_path)
        rel_output_dir = os.path.relpath(output_dir, ddcm_path)
        cmd = [
            sys.executable, roundtrip_script,
            'roundtrip',
            '--input_dir', rel_input_dir,
            '--output_dir', rel_output_dir,
            '--model_id', model_id,
            '--num_noises', str(K),
            '--timesteps', str(T),
            '--num_pursuit_noises', str(M),
            '--num_pursuit_coef_bits', str(C),
            '--gpu', '0',
            '--float16'
        ]
        subprocess.run(cmd, check=True, cwd=ddcm_path)
        
        actual_bpps = []
        out_prefix = f'T={T}_in{t_range[0]}-{t_range[1]}_K={K}_M={M}_C={C}_model={model_id.split("/")[1]}'
        for img_file in sample_images:
            base_name = os.path.splitext(img_file)[0]
            # DDCM saves files directly under out_prefix/ (not in images/ subdirectory)
            bin_file = os.path.join(output_dir, out_prefix, f'{base_name}_noise_indices.bin')
            if os.path.exists(bin_file):
                actual_bpp = calculate_actual_bpp(bin_file, imsize, imsize)
                actual_bpps.append(actual_bpp)
        
        avg_actual_bpp = sum(actual_bpps) / len(actual_bpps) if actual_bpps else 0
        append_to_ddcm_actual_csv(actual_csv_path, target_bpp, M, avg_actual_bpp, T, K, C)
        
        shutil.rmtree(temp_input_dir)
    
    return ddcm_Ms

def init_jpeg_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'jpeg')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'jpeg_compression_params.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_file', 'target_bpp', 'quality', 'actual_bpp'])
    return csv_path

def append_to_jpeg_csv(csv_path, image_file, target_bpp, quality, actual_bpp):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_file, target_bpp, quality, actual_bpp])

def init_turbo_ddcm_theoretical_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'turbo_ddcm')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'turbo_ddcm_theoretical_bpp.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_bpp', 'M', 'theoretical_bpp', 'T', 'K', 'C'])
    return csv_path

def append_to_turbo_ddcm_theoretical_csv(csv_path, target_bpp, M, theoretical_bpp, T, K, C):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([target_bpp, M, theoretical_bpp, T, K, C])

def init_turbo_ddcm_actual_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'turbo_ddcm')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'turbo_ddcm_actual_bpp.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_bpp', 'M', 'actual_bpp', 'T', 'K', 'C'])
    return csv_path

def append_to_turbo_ddcm_actual_csv(csv_path, target_bpp, M, actual_bpp, T, K, C):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([target_bpp, M, actual_bpp, T, K, C])

def init_turbo_ddcm_old_protocol_theoretical_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'turbo_ddcm_old_protocol')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'turbo_ddcm_old_protocol_theoretical_bpp.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_bpp', 'M', 'theoretical_bpp', 'T', 'K', 'C'])
    return csv_path

def append_to_turbo_ddcm_old_protocol_theoretical_csv(csv_path, target_bpp, M, theoretical_bpp, T, K, C):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([target_bpp, M, theoretical_bpp, T, K, C])

def init_turbo_ddcm_old_protocol_actual_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'turbo_ddcm_old_protocol')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'turbo_ddcm_old_protocol_actual_bpp.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_bpp', 'M', 'actual_bpp', 'T', 'K', 'C'])
    return csv_path

def append_to_turbo_ddcm_old_protocol_actual_csv(csv_path, target_bpp, M, actual_bpp, T, K, C):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([target_bpp, M, actual_bpp, T, K, C])

def init_ddcm_theoretical_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'ddcm')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'ddcm_theoretical_bpp.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_bpp', 'M', 'theoretical_bpp', 'T', 'K', 'C'])
    return csv_path

def append_to_ddcm_theoretical_csv(csv_path, target_bpp, M, theoretical_bpp, T, K, C):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([target_bpp, M, theoretical_bpp, T, K, C])

def init_ddcm_actual_csv(project_root):
    results_dir = os.path.join(project_root, 'results', 'compression_ratio_estimate', 'ddcm')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'ddcm_actual_bpp.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_bpp', 'M', 'actual_bpp', 'T', 'K', 'C'])
    return csv_path

def append_to_ddcm_actual_csv(csv_path, target_bpp, M, actual_bpp, T, K, C):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([target_bpp, M, actual_bpp, T, K, C])

def main():
    parser = argparse.ArgumentParser(description='Find compression parameters for different algorithms')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['jpeg', 'ddcm', 'turbo_ddcm', 'turbo_ddcm_old_protocol', 'all'],
                       default=['all'],
                       help='Algorithms to run (default: all)')
    parser.add_argument('--bpp', nargs='+', type=float, 
                       choices=[0.1, 0.5, 1.0],
                       default=[0.1, 0.5, 1.0],
                       help='Target BPP values (default: 0.1 0.5 1.0)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset directory (default: dataset_Kodack24 in project root)')
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of sample images to use for DDCM and Turbo-DDCM experiments (default: 2)')
    parser.add_argument('--manual_list_ind', action='store_true', default=False,
                       help='Use manual list index for Turbo-DDCM (default: False)')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Determine which algorithms to run
    if 'all' in args.algorithms:
        algorithms_to_run = ['jpeg', 'ddcm', 'turbo_ddcm', 'turbo_ddcm_old_protocol']
    else:
        algorithms_to_run = args.algorithms
    
    # Set dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = os.path.join(project_root, 'dataset_Kodack24')
    
    target_bpps = args.bpp
    num_samples = args.num_samples
    
    print(f"\n{'='*60}")
    print(f"Finding compression parameters")
    print(f"Algorithms: {', '.join(algorithms_to_run)}")
    print(f"Target BPPs: {target_bpps}")
    print(f"Dataset: {dataset_path}")
    print(f"Sample images for DDCM/Turbo-DDCM: {num_samples}")
    print(f"{'='*60}\n")
    
    sample_images = get_sample_images(dataset_path, num_samples=num_samples)
    all_images = get_all_images(dataset_path)
    
    # Initialize CSV files only for algorithms we'll run
    if 'jpeg' in algorithms_to_run:
        jpeg_csv_path = init_jpeg_csv(project_root)
    if 'turbo_ddcm' in algorithms_to_run:
        turbo_ddcm_theoretical_csv = init_turbo_ddcm_theoretical_csv(project_root)
        turbo_ddcm_actual_csv = init_turbo_ddcm_actual_csv(project_root)
    if 'turbo_ddcm_old_protocol' in algorithms_to_run:
        turbo_ddcm_old_protocol_theoretical_csv = init_turbo_ddcm_old_protocol_theoretical_csv(project_root)
        turbo_ddcm_old_protocol_actual_csv = init_turbo_ddcm_old_protocol_actual_csv(project_root)
    if 'ddcm' in algorithms_to_run:
        ddcm_theoretical_csv = init_ddcm_theoretical_csv(project_root)
        ddcm_actual_csv = init_ddcm_actual_csv(project_root)
    
    # Run experiments
    if 'jpeg' in algorithms_to_run:
        print(f"\n{'='*60}")
        print(f"Running JPEG experiments")
        print(f"{'='*60}")
        jpeg_qualities, jpeg_actual_bpps = run_jpeg_experiment(dataset_path, all_images, target_bpps, project_root, jpeg_csv_path)
        print(f"✓ JPEG experiments complete")
    
    if 'ddcm' in algorithms_to_run:
        print(f"\n{'='*60}")
        print(f"Running DDCM experiments")
        print(f"{'='*60}")
        ddcm_Ms = run_ddcm_experiment(dataset_path, sample_images, target_bpps, project_root, ddcm_theoretical_csv, ddcm_actual_csv)
        print(f"✓ DDCM experiments complete")
    
    if 'turbo_ddcm' in algorithms_to_run:
        print(f"\n{'='*60}")
        print(f"Running Turbo-DDCM experiments")
        print(f"{'='*60}")
        turbo_ddcm_Ms, T, K, turbo_ddcm_actual_bpps = run_turbo_ddcm_experiment(dataset_path, sample_images, target_bpps, project_root, turbo_ddcm_theoretical_csv, turbo_ddcm_actual_csv)
        print(f"✓ Turbo-DDCM experiments complete")
    
    if 'turbo_ddcm_old_protocol' in algorithms_to_run:
        print(f"\n{'='*60}")
        print(f"Running Turbo-DDCM Old Protocol experiments")
        print(f"{'='*60}")
        turbo_ddcm_old_protocol_Ms, T_old, K_old, turbo_ddcm_old_protocol_actual_bpps = run_turbo_ddcm_old_protocol_experiment(dataset_path, sample_images, target_bpps, project_root, turbo_ddcm_old_protocol_theoretical_csv, turbo_ddcm_old_protocol_actual_csv, args.manual_list_ind)
        print(f"✓ Turbo-DDCM Old Protocol experiments complete")
    
    print(f"\n{'='*60}")
    print(f"All selected experiments complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
