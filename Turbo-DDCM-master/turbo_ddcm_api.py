import os
import torch
import pandas as pd
import time
from tqdm import tqdm
from argparse import Namespace
import json

from turbo_ddcm.turbo_ddcm import TurboDDCM
from turbo_ddcm import utils


# Model cache: keyed by (model_id, T, K, M, old_protocol, seed, device_str, float32)
_model_cache = {}


def _get_cached_model(model_id, T, K, M, old_protocol, seed, float32, device_str):
    """Get or load cached model."""
    global _model_cache
    
    cache_key = (model_id, T, K, M, old_protocol, seed, device_str, float32)
    
    if cache_key not in _model_cache:
        turbo_ddcm = TurboDDCM(model_id, T, K, M, old_protocol, seed, float32, device_str)
        _model_cache[cache_key] = turbo_ddcm
        print(f"Loaded Turbo-DDCM model: {cache_key}")
    
    return _model_cache[cache_key]


def compress_main_programmatic(input_dir, output_dir, M, gpu=0, float32=False, seed=88888888,
                               T=20, K=16384, weights_dir=None, save_reconstructions=False, save_runtimes=False, old_protocol=False):
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    files = sorted(os.listdir(input_dir))
    target_files = [f for f in files if f.endswith('png')]
    if not target_files:
        raise FileNotFoundError(f"No files with suffix png found in {input_dir}.")
    
    # open one img to check for the size to determine the needed diffusion model
    test_img = utils.load_image(os.path.join(input_dir, target_files[0]), None, device_str)
    if max(test_img.shape[2], test_img.shape[3]) < 512:
        # too small images
        raise ValueError(f"Too small images. Minimum size is 512x512.")
    elif min(test_img.shape[2], test_img.shape[3]) >= 768:
        resize_to = (768, 768)
        #model_id = "stabilityai/stable-diffusion-2-1"
        model_id = "CompVis/stable-diffusion-v1-4"
    else:
        resize_to = (512, 512)
        model_id = "Manojb/stable-diffusion-2-1-base"

    if test_img.shape[2:3] != torch.Size(resize_to):
        print(f"images will be resized to {resize_to}")

    turbo_ddcm = _get_cached_model(model_id, T, K, M, old_protocol, seed, float32, device_str)
    runtimes = []
    for file_name in tqdm(target_files):
        img = utils.load_image(os.path.join(input_dir, file_name), resize_to, device_str)
        weight_pixel_vector = None
        if weights_dir is not None:
            weight_pixel_vector = torch.load(os.path.join(weights_dir, os.path.splitext(file_name)[0] + '.pt'), device_str)

        compr_start_time = time.process_time()
        (reconstruction, encoding), compr_end_time = turbo_ddcm.compress(img, weight_pixel_vector)
        
        binary_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + utils.BIN_SUFFIX)
        utils.save_as_binary(encoding, binary_path)
        runtimes.append({'file_name' : file_name, 'compression_time_seconds' : compr_end_time-compr_start_time})
        if save_reconstructions:
            utils.save_decoded_img(os.path.join(output_dir, file_name), reconstruction)

    if save_runtimes:
        pd.DataFrame(runtimes).to_csv(os.path.join(output_dir, 'compression_runtimes.csv'))

    # Save config for decompression
    config = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'M': M,
        'gpu': gpu,
        'float32': float32,
        'seed': seed,
        'T': T,
        'K': K,
        'weights_dir': weights_dir,
        'save_reconstructions': save_reconstructions,
        'save_runtimes': save_runtimes,
        'model_id': model_id,
        'old_protocol': old_protocol
    }
    with open(os.path.join(output_dir, 'compression_config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def decompress_main_programmatic(input_dir, output_dir, gpu=0, save_runtimes=False):
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    files = sorted(os.listdir(input_dir))
    target_files = [f for f in files if f.endswith(utils.BIN_SUFFIX)]
    if not target_files:
        raise FileNotFoundError(f"No files with suffix {utils.BIN_SUFFIX} found in {input_dir}.")

    # open config
    with open(os.path.join(input_dir, 'compression_config.json'), 'r') as f:
        compression_config = json.load(f)
    compression_config = Namespace(**compression_config)
    
    # Handle old configs that may not have old_protocol flag
    old_protocol = getattr(compression_config, 'old_protocol', False)
    
    turbo_ddcm = _get_cached_model(compression_config.model_id, compression_config.T, compression_config.K, 
                                   compression_config.M, old_protocol, compression_config.seed, compression_config.float32, device_str)
    runtimes = []
    for file_name in tqdm(target_files):
        encoding = utils.load_binary(os.path.join(input_dir, file_name))
        
        decompr_start_time = time.process_time()
        reconstruction = turbo_ddcm.decompress(encoding)
        decompr_end_time = time.process_time()
        
        utils.save_decoded_img(os.path.join(output_dir, os.path.splitext(file_name)[0] + '.png'), reconstruction)
        runtimes.append({'file_name' : os.path.splitext(file_name)[0] + '.png', 'decompression_time_seconds' : decompr_end_time-decompr_start_time})

    if save_runtimes:
        decompression_runtime_df = pd.DataFrame(runtimes)
        
        # join to compression runtime file (if exists) and compute the roundtrip time
        if os.path.exists(os.path.join(input_dir, 'compression_runtimes.csv')):
            compression_runtime_df = pd.read_csv(os.path.join(input_dir, 'compression_runtimes.csv'))
            decompression_runtime_df = decompression_runtime_df.merge(compression_runtime_df, on='file_name')
            decompression_runtime_df['roundtrip_time_seconds'] = decompression_runtime_df['compression_time_seconds'] + decompression_runtime_df['decompression_time_seconds']
            decompression_runtime_df = decompression_runtime_df[['file_name', 'compression_time_seconds', 'decompression_time_seconds', 'roundtrip_time_seconds']]

        decompression_runtime_df.to_csv(os.path.join(output_dir, 'decompression_runtimes.csv'))

