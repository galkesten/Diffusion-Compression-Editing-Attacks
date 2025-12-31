import os
import sys
import csv
import shutil
import random
import numpy as np
from PIL import Image
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from pyiqa import create_metric
import torch

# Set up paths for API files (ddcm_api.py and turbo_ddcm_api.py are at root level)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_ddcm_path = os.path.join(_project_root, 'ddcm-compressed-image-generation-main')
_turbo_ddcm_path = os.path.join(_project_root, 'Turbo-DDCM-master')

if _ddcm_path not in sys.path:
    sys.path.insert(0, _ddcm_path)
if _turbo_ddcm_path not in sys.path:
    sys.path.insert(0, _turbo_ddcm_path)

# Import API functions (packages ddcm and turbo_ddcm are installed, but API files need sys.path)
from ddcm_api import main_programmatic as ddcm_main_programmatic
from turbo_ddcm_api import compress_main_programmatic as turbo_compress_main_programmatic
from turbo_ddcm_api import decompress_main_programmatic as turbo_decompress_main_programmatic
import turbo_ddcm.utils as utils


def get_images(dataset_path):
    return sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')], 
                  key=lambda x: int(x.split('.')[0]))


def load_jpeg_params(csv_path, target_bpp):
    df = pd.read_csv(csv_path)
    df = df[df['target_bpp'] == target_bpp]
    return dict(zip(df['image_file'], df['quality']))


def load_ddcm_params(csv_path, target_bpp):
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


def load_turbo_ddcm_params(csv_path, target_bpp):
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


def compress_jpeg(img_path, quality, output_path, resolution=512):
    img = Image.open(img_path).convert('RGB')
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution))
    img.save(output_path, format='JPEG', quality=quality)


def decompress_jpeg(jpeg_path, resolution=512):
    try:
        img = Image.open(jpeg_path).convert('RGB')
        if img.size != (resolution, resolution):
            raise ValueError(f"Decompressed JPEG image {img.size} does not match resolution {resolution}")
        return img, None
    except Exception as e:
        error_msg = str(e)
        print(f"Error decompressing JPEG {jpeg_path}: {error_msg}")
        return None, error_msg


def compress_ddcm(img_path, params, output_dir, project_root, resolution=512):
    temp_input = os.path.join(output_dir, 'temp_input')
    print(f"    [DDCM compress] Creating temp_input dir: {temp_input}")
    os.makedirs(temp_input, exist_ok=True)
    shutil.copy(img_path, os.path.join(temp_input, os.path.basename(img_path)))
    print(f"    [DDCM compress] Copied {os.path.basename(img_path)} to temp_input")
    
    M = params['M']
    T = params['T']
    K = params['K']
    C = params['C']
    
    print(f" original image path: {img_path}, temp input path: {temp_input}, output directory: {output_dir}")
    try:
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
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_prefix = f'T={T}_in{T-1}-0_K={K}_M={M}_C={C}_model=stable-diffusion-2-1-base'
        bin_file = os.path.join(output_dir, out_prefix, f'{base_name}_noise_indices.bin')
        return bin_file if os.path.exists(bin_file) else None
    finally:
        if os.path.exists(temp_input):
            shutil.rmtree(temp_input)


def decompress_ddcm(bin_file, output_dir, project_root, resolution=512):
    temp_input = os.path.join(output_dir, 'temp_ddcm_input')
    temp_output = os.path.join(output_dir, 'temp_decomp')
    # Clean directories first to avoid leftover files from previous decompressions
    if os.path.exists(temp_input):
        shutil.rmtree(temp_input)
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    print(f"    [DDCM decompress] Creating temp_input dir: {temp_input}")
    print(f"    [DDCM decompress] Creating temp_output dir: {temp_output}")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)
    
    # Create directory structure matching the bin_file location
    bin_dir = os.path.dirname(bin_file)
    bin_subdir = os.path.basename(bin_dir)  # e.g., "T=1000_in999-0_K=8192_M=2_C=3_model=..."
    temp_bin_dir = os.path.join(temp_input, bin_subdir)
    print(f"    [DDCM decompress] Creating temp_bin_dir: {temp_bin_dir}")
    os.makedirs(temp_bin_dir, exist_ok=True)
    
    bin_basename = os.path.basename(bin_file)
    temp_bin_file = os.path.join(temp_bin_dir, bin_basename)
    print(f"    [DDCM decompress] Copying {bin_basename} to {temp_bin_file}")
    shutil.copy(bin_file, temp_bin_file)
    
    print(f"bin file path: {bin_file}, input directory: {temp_input}, output directory: {temp_output}")
    try:
        ddcm_main_programmatic(
            mode='decompress',
            input_dir=temp_input,
            output_dir=temp_output,
            gpu=0,
            float16=True
        )
        # DDCM saves files in subdirectories matching the input structure, search recursively
        # Since we use a clean temp directory for each decompression, taking the first PNG is safe
        png_files = []
        for root, dirs, files in os.walk(temp_output):
            for f in files:
                if f.endswith('.png'):
                    png_files.append(os.path.join(root, f))
        if png_files:
            img = Image.open(png_files[0]).convert('RGB')
            if img.size != (resolution, resolution):
                raise ValueError(f"Decompressed DDCM image {img.size} does not match resolution {resolution}")
            return img, None
        else:
            return None, "No PNG files found in decompression output"
    except Exception as e:
        error_msg = str(e)
        print(f"Error decompressing DDCM {bin_file}: {error_msg}")
        return None, error_msg
    finally:
        if os.path.exists(temp_input):
            shutil.rmtree(temp_input)
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)


def compress_turbo_ddcm(img_path, params, output_dir, project_root, resolution=512):
    temp_input = os.path.join(output_dir, 'temp_input')
    print(f"    [Turbo-DDCM compress] Creating temp_input dir: {temp_input}")
    os.makedirs(temp_input, exist_ok=True)
    shutil.copy(img_path, os.path.join(temp_input, os.path.basename(img_path)))
    print(f"    [Turbo-DDCM compress] Copied {os.path.basename(img_path)} to temp_input")
    
    M = params['M']
    T = params['T']
    K = params['K']
    C = params['C']
    
    print(f" original image path: {img_path}, temp input path: {temp_input}, output directory: {output_dir}")
    try:
        turbo_compress_main_programmatic(
            input_dir=temp_input,
            output_dir=output_dir,
            M=M,
            gpu=0,
            float32=False,
            seed=88888888,
            T=T,
            K=K,
            weights_dir=None,
            save_reconstructions=False,
            save_runtimes=False
        )
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        bin_file = os.path.join(output_dir, f'{base_name}{utils.BIN_SUFFIX}')
        return bin_file if os.path.exists(bin_file) else None
    finally:
        if os.path.exists(temp_input):
            shutil.rmtree(temp_input)


def decompress_turbo_ddcm(bin_file, output_dir, project_root, resolution=512):
    temp_input = os.path.join(output_dir, 'temp_turbo_ddcm_input')
    temp_output = os.path.join(output_dir, 'temp_decomp')
    # Clean directories first to avoid leftover files from previous decompressions
    if os.path.exists(temp_input):
        shutil.rmtree(temp_input)
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    print(f"    [Turbo-DDCM decompress] Creating temp_input dir: {temp_input}")
    print(f"    [Turbo-DDCM decompress] Creating temp_output dir: {temp_output}")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)
    
    bin_dir = os.path.dirname(bin_file)
    config_path = os.path.join(bin_dir, 'compression_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"compression_config.json not found in {bin_dir}. Cannot decompress without compression parameters.")
    
    bin_basename = os.path.basename(bin_file)
    temp_bin_file = os.path.join(temp_input, bin_basename)
    temp_config_file = os.path.join(temp_input, 'compression_config.json')
    print(f"    [Turbo-DDCM decompress] Copying {bin_basename} to {temp_bin_file}")
    print(f"    [Turbo-DDCM decompress] Copying compression_config.json to {temp_config_file}")
    shutil.copy(bin_file, temp_bin_file)
    shutil.copy(config_path, temp_config_file)
    
    print(f"bin file path: {bin_file}, input directory: {temp_input}, output directory: {temp_output}")
    try:
        turbo_decompress_main_programmatic(
            input_dir=temp_input,
            output_dir=temp_output,
            gpu=0,
            save_runtimes=False
        )
        png_files = [f for f in os.listdir(temp_output) if f.endswith('.png')]
        if png_files:
            img = Image.open(os.path.join(temp_output, png_files[0])).convert('RGB')
            if img.size != (resolution, resolution):
                raise ValueError(f"Decompressed Turbo-DDCM image {img.size} does not match resolution {resolution}")
            return img, None
        else:
            return None, "No PNG files found in decompression output"
    except Exception as e:
        error_msg = str(e)
        print(f"Error decompressing Turbo-DDCM {bin_file}: {error_msg}")
        return None, error_msg
    finally:
        if os.path.exists(temp_input):
            shutil.rmtree(temp_input)
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)


def calculate_bpp(binary_file, resolution=512):
    if not os.path.exists(binary_file):
        return 0
    size_bits = os.path.getsize(binary_file) * 8
    return size_bits / (resolution * resolution)


def calculate_psnr(img1, img2):
    try:
        arr1 = np.array(img1).astype(np.float64)
        arr2 = np.array(img2).astype(np.float64)
        
        # Ensure images have the same dimensions
        if arr1.shape != arr2.shape:
            # Resize arr2 to match arr1
            img2_pil = Image.fromarray(arr2.astype(np.uint8))
            img2_pil = img2_pil.resize((arr1.shape[1], arr1.shape[0]))
            arr2 = np.array(img2_pil).astype(np.float64)
        
        # Images are RGB with values 0-255
        assert arr1.min() >= 0 and arr1.max() <= 255, f"Image 1 values out of range [0, 255]: min={arr1.min()}, max={arr1.max()}"
        assert arr2.min() >= 0 and arr2.max() <= 255, f"Image 2 values out of range [0, 255]: min={arr2.min()}, max={arr2.max()}"
        return psnr(arr1, arr2, data_range=255)
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return np.nan


# Initialize NIQE metric once
_niqe_metric = None

def get_niqe_metric():
    global _niqe_metric
    if _niqe_metric is None:
        _niqe_metric = create_metric('niqe', device='cpu')
    return _niqe_metric

def calculate_niqe(img):
    try:
        img_array = np.array(img).astype(np.float32) / 255.0
        # pyiqa expects (1, C, H, W) format and values in range [0, 1]
        if len(img_array.shape) == 2:
            img_array = img_array[np.newaxis, np.newaxis, :, :]
        elif len(img_array.shape) == 3:
            img_array = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :]
        
        img_tensor = torch.from_numpy(img_array)
        niqe_metric = get_niqe_metric()
        score = niqe_metric(img_tensor)
        return float(score.item())
    except Exception as e:
        print(f"Error calculating NIQE: {e}")
        return np.nan


def flip_bits(binary_file, ber, output_file):
    """
    Simulate a symmetric binary noisy channel by flipping each bit independently 
    with probability ber (Bit Error Rate).
    Returns the actual number of bits flipped.
    """
    with open(binary_file, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    num_bits = len(data) * 8
    random_values = np.random.random(num_bits)
    should_flip = random_values < ber
    should_flip = should_flip.reshape(len(data), 8)
    
    powers_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    mask = (should_flip.astype(np.uint8) @ powers_of_2).astype(np.uint8)
    
    flipped_data = data ^ mask
    
    num_flips = int(np.sum(np.unpackbits(mask)))
    
    with open(output_file, 'wb') as f:
        f.write(flipped_data.tobytes())
    
    return num_flips


def compress_all_images(dataset_path, method, target_bpp, params_csv, resolution, output_dir, project_root):
    print(f"\n=== Step 1: Compressing all images ===")
    print(f"Method: {method}, Target BPP: {target_bpp}, Resolution: {resolution}")
    os.makedirs(output_dir, exist_ok=True)
    images = get_images(dataset_path)
    print(f"Found {len(images)} images to compress")
    
    params = None
    if method == 'jpeg':
        params = load_jpeg_params(params_csv, target_bpp)
    elif method == 'ddcm':
        params = load_ddcm_params(params_csv, target_bpp)
    elif method == 'turbo_ddcm':
        params = load_turbo_ddcm_params(params_csv, target_bpp)
    
    compressed_files = {}
    for idx, img_file in enumerate(images, 1):
        print(f"  Compressing image {idx}/{len(images)}: {img_file}")
        img_path = os.path.join(dataset_path, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        if method == 'jpeg':
            if params and img_file in params:
                quality = params[img_file]
                jpeg_path = os.path.join(output_dir, f'{base_name}_compressed.jpg')
                compress_jpeg(img_path, quality, jpeg_path, resolution)
                compressed_files[img_file] = jpeg_path
        elif method == 'ddcm' and params is not None:
            bin_file = compress_ddcm(img_path, params, output_dir, project_root, resolution)
            compressed_files[img_file] = bin_file
        elif method == 'turbo_ddcm' and params is not None:
            bin_file = compress_turbo_ddcm(img_path, params, output_dir, project_root, resolution)
            compressed_files[img_file] = bin_file
    
    print(f"Compression complete: {len(compressed_files)}/{len(images)} images compressed successfully")
    return compressed_files


def measure_baseline_metrics(dataset_path, compressed_files, method, output_dir, project_root, resolution, target_bpp, csv_writer, csv_file):
    print(f"\n=== Step 2: Measuring baseline metrics (BER=0) ===")
    images = get_images(dataset_path)
    
    for idx, img_file in enumerate(images, 1):
        print(f"  Processing image {idx}/{len(images)}: {img_file}")
        original_path = os.path.join(dataset_path, img_file)
        original = Image.open(original_path).convert('RGB')
        original = original.resize((resolution, resolution))
        
        compressed_file = compressed_files[img_file]
        actual_bpp = calculate_bpp(compressed_file, resolution)
        
        decompressed = None
        decompression_success = False
        error_reason = ''
        if method == 'jpeg':
            decompressed, error_reason = decompress_jpeg(compressed_file, resolution)
        elif method == 'ddcm':
            decompressed, error_reason = decompress_ddcm(compressed_file, output_dir, project_root, resolution)
        elif method == 'turbo_ddcm':
            decompressed, error_reason = decompress_turbo_ddcm(compressed_file, output_dir, project_root, resolution)
        if decompressed is not None:
            decompression_success = True
            print(f"    ✓ Decompression successful: {img_file}")
        elif error_reason:
            print(f"    ✗ Decompression failed for {img_file} ({method}): {error_reason}")
        
        if decompressed is None:
            psnr_val = 'N/A'
            niqe_val = 'N/A'
        else:
            try:
                psnr_val = calculate_psnr(original, decompressed)
                if np.isnan(psnr_val):
                    psnr_val = 'N/A'
                    print(f"    ✗ PSNR calculation failed for {img_file}")
            except Exception as e:
                psnr_val = 'N/A'
                print(f"    ✗ PSNR calculation error for {img_file}: {e}")
            
            try:
                niqe_val = calculate_niqe(decompressed)
                if np.isnan(niqe_val):
                    niqe_val = 'N/A'
            except Exception as e:
                niqe_val = 'N/A'
                print(f"    ✗ NIQE calculation error for {img_file}: {e}")
        
        file_size_bits = os.path.getsize(compressed_file) * 8
        expected_bit_flips = 0 * file_size_bits  # BER = 0
        
        row = {
            'original_image_name': img_file,
            'target_bpp': target_bpp,
            'actual_bpp': actual_bpp,
            'BER': 0,
            'expected_bit_flips': expected_bit_flips,
            'actual_bit_flips': 0,
            'PSNR': psnr_val,
            'NIQE': niqe_val,
            'error_reason': error_reason
        }
        csv_writer.writerow(row)
        csv_file.flush()  # Ensure immediate write to disk
    print(f"Baseline metrics complete for {len(images)} images")


def simulate_noisy_channel(compressed_files, method, ber_values, num_trials, output_dir, project_root, 
                          dataset_path, resolution, num_samples_to_save, target_bpp, csv_writer, csv_file):
    print(f"\n=== Step 3: Simulating noisy channel ===")
    print(f"BER values: {ber_values}")
    print(f"Number of trials per BER: {num_trials}")
    images = get_images(dataset_path)
    total_iterations = len(ber_values) * num_trials * len(images)
    print(f"Total iterations: {total_iterations}")
    
    # Create a temporary directory for noisy files (separate from compressed files)
    temp_noisy_dir = os.path.join(output_dir, 'temp_noisy_files')
    os.makedirs(temp_noisy_dir, exist_ok=True)
    
    iteration = 0
    try:
        for ber_idx, ber in enumerate(ber_values, 1):
            print(f"\n  Processing BER {ber_idx}/{len(ber_values)}: {ber}")
            # Track how many samples we've saved for this BER
            samples_saved = {img_file: 0 for img_file in images}
            
            for trial in range(num_trials):
                print(f"    Trial {trial+1}/{num_trials}")
                for img_idx, img_file in enumerate(images, 1):
                    iteration += 1
                    if iteration % 10 == 0 or img_idx == 1:
                        print(f"      Image {img_idx}/{len(images)}: {img_file} (iteration {iteration}/{total_iterations})")
                    compressed_file = compressed_files.get(img_file)
                    if compressed_file is None or not os.path.exists(compressed_file):
                        raise ValueError(f"Compressed file {compressed_file} not found for image {img_file}")
                        continue
                    
                    compressed_dir = os.path.dirname(compressed_file)
                    compressed_basename = os.path.basename(compressed_file)
                    compressed_name, orig_ext = os.path.splitext(compressed_basename)
                    base_name = os.path.splitext(img_file)[0]
                    
                    # Create noisy file in temp directory
                    noisy_file = os.path.join(temp_noisy_dir, f'{base_name}_ber{ber}_trial{trial}{orig_ext}')
                    
                    # For DDCM and Turbo-DDCM, we need to set up the directory structure
                    if method == 'ddcm':
                        # DDCM needs the directory name with parameters, so create a subdirectory
                        # Extract the directory name from compressed_file path
                        ddcm_subdir = os.path.basename(compressed_dir)  # e.g., "T=1000_in999-0_K=8192_M=2_C=3_model=..."
                        noisy_ddcm_dir = os.path.join(temp_noisy_dir, ddcm_subdir)
                        os.makedirs(noisy_ddcm_dir, exist_ok=True)
                        # Create noisy file in this subdirectory with the expected name
                        noisy_file = os.path.join(noisy_ddcm_dir, f'{base_name}_noise_indices.bin')
                    elif method == 'turbo_ddcm':
                        # Turbo-DDCM needs compression_config.json in the same directory
                        config_source = os.path.join(compressed_dir, 'compression_config.json')
                        if os.path.exists(config_source):
                            shutil.copy(config_source, os.path.join(temp_noisy_dir, 'compression_config.json'))
                    
                    num_bit_flips = 0
                    try:
                        num_bit_flips = flip_bits(compressed_file, ber, noisy_file)
                    except Exception as e:
                        print(f"      ✗ Error flipping bits for {img_file} trial {trial} BER {ber}: {e}")
                        continue
                    
                    decompressed = None
                    decompression_success = False
                    error_reason = ''
                    if method == 'jpeg':
                        decompressed, error_reason = decompress_jpeg(noisy_file, resolution)
                    elif method == 'ddcm':
                        decompressed, error_reason = decompress_ddcm(noisy_file, output_dir, project_root, resolution)
                    elif method == 'turbo_ddcm':
                        decompressed, error_reason = decompress_turbo_ddcm(noisy_file, output_dir, project_root, resolution)
                    if decompressed is not None:
                        decompression_success = True
                    elif error_reason:
                        print(f"      ✗ Decompression failed for {img_file} trial {trial} BER {ber} ({method}): {error_reason}")
                    
                    if decompressed is None:
                        print(f"      ✗ Decompression failed for {img_file} trial {trial} BER {ber}")
                        psnr_val = 'N/A'
                        niqe_val = 'N/A'
                    else:
                        original_path = os.path.join(dataset_path, img_file)
                        original = Image.open(original_path).convert('RGB')
                        original = original.resize((resolution, resolution))
                        try:
                            psnr_val = calculate_psnr(original, decompressed)
                            if np.isnan(psnr_val):
                                psnr_val = 'N/A'
                        except Exception as e:
                            psnr_val = 'N/A'
                            print(f"      ✗ PSNR calculation error for {img_file} trial {trial}: {e}")
                        
                        try:
                            niqe_val = calculate_niqe(decompressed)
                            if np.isnan(niqe_val):
                                niqe_val = 'N/A'
                        except Exception as e:
                            niqe_val = 'N/A'
                            print(f"      ✗ NIQE calculation error for {img_file} trial {trial}: {e}")
                        
                        # Save sample only if there were actual bit flips and we haven't reached the limit
                        if num_bit_flips > 0 and samples_saved[img_file] < num_samples_to_save:
                            sample_dir = os.path.join(output_dir, f'samples_ber{ber}')
                            os.makedirs(sample_dir, exist_ok=True)
                            decompressed.save(os.path.join(sample_dir, f'{base_name}_trial{trial}.png'))
                            samples_saved[img_file] += 1
                    
                    file_size_bits = os.path.getsize(compressed_file) * 8
                    expected_bit_flips = ber * file_size_bits
                    
                    row = {
                        'original_image_name': img_file,
                        'target_bpp': target_bpp,
                        'actual_bpp': calculate_bpp(compressed_file, resolution),
                        'BER': ber,
                        'expected_bit_flips': expected_bit_flips,
                        'actual_bit_flips': num_bit_flips,
                        'PSNR': psnr_val,
                        'NIQE': niqe_val,
                        'error_reason': error_reason
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()  # Ensure immediate write to disk
                    
                    # Clean up noisy file immediately after use
                    try:
                        if os.path.exists(noisy_file):
                            os.remove(noisy_file)
                        # For DDCM, also clean up the subdirectory if empty
                        if method == 'ddcm':
                            ddcm_subdir = os.path.dirname(noisy_file)
                            if os.path.exists(ddcm_subdir) and not os.listdir(ddcm_subdir):
                                os.rmdir(ddcm_subdir)
                    except Exception as e:
                        print(f"      ✗ Error cleaning up noisy file {noisy_file}: {e}")
                        pass
            print(f"  Completed BER {ber} ({ber_idx}/{len(ber_values)})")
    finally:
        # Clean up temporary directory at the end
        try:
            if os.path.exists(temp_noisy_dir):
                shutil.rmtree(temp_noisy_dir)
        except Exception as e:
            print(f"  ✗ Error cleaning up temporary directory {temp_noisy_dir}: {e}")
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--resolution', type=int, default=512, choices=[512])
    parser.add_argument('--method', required=True, choices=['jpeg', 'ddcm', 'turbo_ddcm'])
    parser.add_argument('--bpp', type=float, required=True, choices=[0.1, 0.5, 1.0])
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--params_csv', required=True)
    parser.add_argument('--num_samples_to_save_per_BER', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Note: sys.path is already set up at module level for DDCM and Turbo-DDCM imports
    
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'noisy_channel_results_{args.method}_bpp{args.bpp}.csv')
    
    # Open CSV file in append mode, but write header if file doesn't exist
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['original_image_name', 'target_bpp', 'actual_bpp', 'BER', 'expected_bit_flips', 'actual_bit_flips', 'PSNR', 'NIQE', 'error_reason'])
    
    if not file_exists:
        writer.writeheader()
        csv_file.flush()
    
    print(f"\n{'='*60}")
    print(f"Starting noisy channel experiment")
    print(f"Method: {args.method}, BPP: {args.bpp}, Resolution: {args.resolution}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    try:
        compressed_files = compress_all_images(
            args.dataset, args.method, args.bpp, args.params_csv,
            args.resolution, args.output_dir, project_root
        )
        
        measure_baseline_metrics(
            args.dataset, compressed_files, args.method,
            args.output_dir, project_root, args.resolution, args.bpp, writer, csv_file
        )
        
        ber_values = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
        simulate_noisy_channel(
            compressed_files, args.method, ber_values, args.num_trials,
            args.output_dir, project_root, args.dataset, args.resolution,
            args.num_samples_to_save_per_BER, args.bpp, writer, csv_file
        )
    finally:
        csv_file.close()
    
    print(f"\n{'='*60}")
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

