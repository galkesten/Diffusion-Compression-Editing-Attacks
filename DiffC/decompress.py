import argparse
from pathlib import Path
import yaml
from easydict import EasyDict as edict
import zlib
import struct

from lib import image_utils
from lib.diffc.denoise import denoise
from lib.diffc.decode import decode
from lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator

import time
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompress DiffC-compressed images"
    )
    parser.add_argument(
        "--config",
        help="Path to the compression config file",
        required=True
    )
    parser.add_argument(
        "--input_path",
        default=None,
        help="Path to a single .diffc file to decompress"
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Path to a directory containing .diffc files to decompress"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to output the decompressed images to"
    )
    return parser.parse_args()

def get_noise_prediction_model(model_name, config):
    if model_name == "SD1.5":
        from lib.models.SD15 import SD15Model
        return SD15Model()
    elif model_name == "SD2.1":
        from lib.models.SD21 import SD21Model
        return SD21Model()
    elif config.model == "SD2.1Base":
        from lib.models.SD21Base import SD21BaseModel
        return SD21BaseModel()
    elif config.model == "SD3":
        from lib.models.SD3 import SD3Model
        return SD3Model()
    elif model_name == "SDXL":
        from lib.models.SDXL import SDXLModel
        use_refiner = config.get("use_refiner", False)
        return SDXLModel(use_refiner=use_refiner)
    elif model_name == 'Flux':
        from lib.models.Flux import FluxModel
        return FluxModel()
    else:
        raise ValueError(f"Unrecognized model: {model_name}")

def read_diffc_file(file_path):
    with open(file_path, 'rb') as f:
        # Read caption length (4 bytes)
        caption_length = struct.unpack('<I', f.read(4))[0]
        
        # Read width, height, and step_idx (2 bytes each)
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        step_idx = struct.unpack('<H', f.read(2))[0]
        
        # Read and decompress caption
        compressed_caption = f.read(caption_length)
        caption = zlib.decompress(compressed_caption).decode('utf-8')
        
        # Read remaining bytes for image data
        image_bytes = list(f.read())
    
    return caption, width, height, step_idx, image_bytes

def decompress_file(input_path, output_path, noise_prediction_model, 
                   gaussian_channel_simulator, config, output_dir):
    # Read compressed data
    caption, width, height, step_idx, compressed_bytes = read_diffc_file(input_path)

    decompression_start = time.process_time()
    # Decompress the representation
    chunk_seeds_per_step = gaussian_channel_simulator.decompress_chunk_seeds(
        compressed_bytes, config.manual_dkl_per_step[:step_idx+1]
    )

    timestep = config.encoding_timesteps[step_idx]
    
    # Configure model with caption
    noise_prediction_model.configure(
        caption, 
        config.denoising_guidance_scale,
        width,
        height
    )
    
    # Get the noisy reconstruction    
    noisy_recon = decode(
        width,
        height,
        config.encoding_timesteps,
        noise_prediction_model,
        gaussian_channel_simulator,
        chunk_seeds_per_step,
        config.manual_dkl_per_step,
        seed=0)
    
    # Denoise
    recon_latent = denoise(
        noisy_recon,
        timestep,
        config.denoising_timesteps,
        noise_prediction_model
    )
    
    # Convert to image and save
    recon_img_pt = noise_prediction_model.latent_to_image(recon_latent)

    decompression_end = time.process_time()
    # update_stats(os.path.basename(output_path) ,round(decompression_end - decompression_start, 3), output_dir)
    
    image_utils.torch_to_pil_img(recon_img_pt).save(output_path)


def update_stats(file_name: str, decompression_time: float, output_dir: str, csv_name='stats.csv'):
    # If the file exists, read it; otherwise, create a new DataFrame
    csv_path = os.path.join(output_dir, csv_name)
    
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['file_name', 'decompression_time'])

    # Check if file_name already exists
    if file_name in df['file_name'].values:
        # Update existing row
        # df.loc[df['file_name'] == file_name, 'compression_time'] = compression_time
        assert False
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([{'file_name': file_name, 'decompression_time': decompression_time}])], ignore_index=True)

    # Save back to CSV
    df.to_csv(csv_path, index=False)

# --config configs/SD-2.1Base-base.yaml --input_dir results/SD-2.1-base/Kodak24/compressed --output_dir results/SD-2.1-base/Kodak24/reconstructions
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = edict(yaml.safe_load(f))

    assert config.manual_dkl_per_step is not None, "Config must specify a manual_dkl_per_step to perform decompression."
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get input paths
    if not bool(args.input_path) ^ bool(args.input_dir):
        raise ValueError("Must specify exactly one of --input_path or --input_dir")

    input_paths = []
    if args.input_path:
        input_paths.append(Path(args.input_path))
    else:
        input_dir = Path(args.input_dir)
        input_paths = list(input_dir.glob("*.diffc"))

    # Initialize models
    gaussian_channel_simulator = GaussianChannelSimulator(
        config.max_chunk_size, 
        config.chunk_padding
    )

    noise_prediction_model = get_noise_prediction_model(config.model, config)

    # Process each file
    for input_path in input_paths:
        # Create output path: {original_name}_decompressed.png
        output_path = output_dir / f"{input_path.stem}.png"
        
        decompress_file(
            input_path,
            output_path,
            noise_prediction_model,
            gaussian_channel_simulator,
            config,
            output_dir
        )

    # --- join both stat files ---
    # reconstruction_timestep = os.path.split(output_dir)[-1]
    # csv_name = 'stats.csv'
    
    # compressed_file_df = pd.read_csv(f'{output_dir}/../../compressed/{reconstruction_timestep}/stats.csv')
    # decompressed_file_df = pd.read_csv(os.path.join(output_dir, csv_name))

    # no duplicates in file_name
    # assert len(compressed_file_df) == len(compressed_file_df['file_name'].unique())
    # assert len(decompressed_file_df) == len(decompressed_file_df['file_name'].unique())

    # assert same files
    # assert all(compressed_file_df['file_name'].unique() == decompressed_file_df['file_name'].unique())

    # final file
    # merged_df = compressed_file_df.merge(decompressed_file_df, on='file_name')
    # merged_df.to_csv(os.path.join(output_dir, csv_name), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)