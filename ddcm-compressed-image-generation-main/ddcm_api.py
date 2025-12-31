import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import numpy as np

from ddcm.latent_runners import get_loader, compress
from ddcm.latent_models import load_model
from ddcm.util.img_utils import clear_color
from ddcm.util.file import get_args_from_filename, save_as_binary_bitwise, load_from_binary_bitwise


# Model cache: keyed by (model_id, T, device_str, float16)
_model_cache = {}


def _get_cached_model(model_id, T, device_str, float16):
    """Get or load cached model."""
    global _model_cache
    
    device = torch.device(device_str)
    cache_key = (model_id, T, device_str, float16)
    
    if cache_key not in _model_cache:
        model, _ = load_model(model_id, T, device, float16, compile=False)
        _model_cache[cache_key] = model
        print(f"Loaded DDCM model: {cache_key}")
    
    return _model_cache[cache_key]


def main_programmatic(mode, input_dir, output_dir, gpu=0, float16=False,
                      model_id=None, K=1024, T=1000, M=1, C=1, t_range=None):
    if t_range is None:
        t_range = [999, 0]
    
    # Device setting
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    if mode == 'decompress':
        # get the first indices file in the input directory
        indices_file = glob(os.path.join(input_dir, '**', '*.bin'), recursive=True)[0]
        T, K, M, C, t_range, model_id = get_args_from_filename(indices_file)

    # Load model (with caching)
    model = _get_cached_model(model_id, T, device_str, float16)
    imsize = model.get_image_size()

    optimized_Ts = len([t for t in model.timesteps if (t_range[0] >= t >= t_range[1])])
    if mode == 'compress' or mode == 'roundtrip':
        out_prefix = f"T={T}_in{t_range[0]}-{t_range[1]}_K={K}_M={M}_C={C}_model={model_id.split('/')[1]}"
        bpp = (optimized_Ts - 1) * (M * np.log2(K) + (M - 1) * C) / (imsize ** 2)
        print(f'The BPP will be: {bpp:.4f}')
    os.makedirs(output_dir, exist_ok=True)

    if mode == 'compress' or mode == 'roundtrip':
        loader = get_loader(input_dir, resize_to=imsize)

        # Do Inference
        for orig_img, orig_path in loader:
            respath = os.path.join(output_dir, out_prefix, os.path.dirname(orig_path[0]))
            os.makedirs(respath, exist_ok=True)  # keep dir structure
            fname = os.path.basename(orig_path[0]).split(os.extsep)[0]

            orig_img = orig_img.to(device)
            with torch.no_grad():
                compressed_im, noise_indices, coeff_indices = compress(model, orig_img, K, device, M, C, t_range)
            plt.imsave(os.path.join(respath, f'{fname}_comp.png'), clear_color(compressed_im, normalize=False))
            save_as_binary_bitwise(noise_indices.numpy(), coeff_indices.numpy(),
                                   K, M, C,
                                   os.path.join(respath, f'{fname}_noise_indices.bin'))

    if mode == 'roundtrip':
        input_dir = os.path.join(output_dir, out_prefix)
        output_dir = input_dir

    if mode == 'decompress' or mode == 'roundtrip':
        indices_files = glob(os.path.join(input_dir, '**', '*.bin'), recursive=True)
        for indices_file in tqdm(indices_files, desc="Decompressing files"):
            indices = load_from_binary_bitwise(indices_file, K, M, C, optimized_Ts)
            respath = os.path.join(output_dir, Path(os.path.dirname(indices_file)).relative_to(input_dir))
            os.makedirs(respath, exist_ok=True)
            fname = os.path.basename(indices_file).split(os.extsep)[0].split('_noise_indices')[0]

            # Decompress
            with torch.no_grad():
                decompressed_im, _, _ = compress(model, None, K, device, M, C, t_range, indices)
            plt.imsave(os.path.join(respath, f'{fname}_decomp.png'), clear_color(decompressed_im, normalize=False))

