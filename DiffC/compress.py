import argparse
from PIL import Image
from pathlib import Path
import yaml
import zlib
import struct
from easydict import EasyDict as edict

from lib import image_utils
from lib.diffc.encode import encode
from lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from lib.blip import BlipCaptioner

import time
import os
import pandas as pd
import warnings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress an image or folder of images using the DiffC algorithm."
    )
    parser.add_argument(
        "--config",
        help="Path to the compression config file",
        required=True
    )
    parser.add_argument(
        "--image_path",
        default=None,
        help="Path to a single image to compress"
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Path to a directory containing one or more images to compress"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to output the compressed files to"
    )
    parser.add_argument(
        "--recon_timestep",
        type=int,
        required=True,
        help="Timestep at which to save the encoded version"
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
    elif model_name == "SDXL":
        from lib.models.SDXL import SDXLModel
        use_refiner = config.get("use_refiner", False)
        return SDXLModel(use_refiner=use_refiner)
    elif model_name == 'Flux':
        from lib.models.Flux import FluxModel
        return FluxModel()
    elif model_name == 'SD3':
        from lib.models.SD3 import SD3Model
        return SD3Model()
    else:
        raise ValueError(f"Unrecognized model: {model_name}")

def write_diffc_file(caption, image_bytes, width, height, step_idx, output_path):
    # Compress caption with zlib
    compressed_caption = zlib.compress(caption.encode('utf-8'))
    caption_length = len(compressed_caption)
    
    # Write caption length (4 bytes), width (2 bytes), height (2 bytes), step_idx (2 bytes), 
    # compressed caption, then image data
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', caption_length))  # Write length as 4-byte little-endian uint
        f.write(struct.pack('<H', width))          # Write width as 2-byte little-endian uint
        f.write(struct.pack('<H', height))         # Write height as 2-byte little-endian uint
        f.write(struct.pack('<H', step_idx))       # Write step_idx as 2-byte little-endian uint
        f.write(compressed_caption)
        f.write(bytes(image_bytes))


def compress_image(image_path, output_path, noise_prediction_model, 
                  gaussian_channel_simulator, config, caption="", output_dir=""):
    # Load and preprocess image
    img_pil = Image.open(image_path)
    img_width, img_height = img_pil.size
    gt_pt = image_utils.pil_to_torch_img(img_pil)

    compression_start = time.process_time()

    gt_latent = noise_prediction_model.image_to_latent(gt_pt)
    
    # Configure model
    noise_prediction_model.configure(
        caption, config.encoding_guidance_scale, img_width, img_height
    )

    # Encode image
    # print("FAKED DKL")
    chunk_seeds_per_step, Dkl_per_step, _, recon_step_indices = encode(
        gt_latent,
        config.encoding_timesteps,
        noise_prediction_model,
        gaussian_channel_simulator,
        config.manual_dkl_per_step,
        [config.recon_timestep]  # Only encode for the specified timestep
    )
    
    # Get the compressed representation
    step_idx = recon_step_indices[0]  # Only one step since we specified one timestep
    bytes_data = gaussian_channel_simulator.compress_chunk_seeds(
        chunk_seeds_per_step[: step_idx + 1], 
        Dkl_per_step[: step_idx + 1]
    )

    compression_end = time.process_time()
    update_stats(os.path.basename(image_path) ,round(compression_end - compression_start, 3), output_dir)

    write_diffc_file(
        caption,
        bytes_data,
        img_width,
        img_height,
        step_idx,
        output_path)


def update_stats(file_name: str, compression_time: float, output_path: str, csv_name='stats.csv'):
    # If the file exists, read it; otherwise, create a new DataFrame
    csv_path = os.path.join(output_path, csv_name)
    
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['file_name', 'compression_time'])

    # Check if file_name already exists
    if file_name in df['file_name'].values:
        # Update existing row
        df.loc[df['file_name'] == file_name, 'compression_time'] = compression_time
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([{'file_name': file_name, 'compression_time': compression_time}])], ignore_index=True)

    # Save back to CSV
    df.to_csv(csv_path, index=False)


# --config configs/SD-2.1Base-base.yaml --image_dir ../../datasets/Kodak24 --output_dir results/SD-2.1-base/Kodak24/compressed --recon_timestep 200
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = edict(yaml.safe_load(f))
    config.recon_timestep = args.recon_timestep

    assert config.manual_dkl_per_step is not None, "Config must specify manual_dkl_per_step for encoding (used by both compress and decompress)."

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get image paths
    if not bool(args.image_path) ^ bool(args.image_dir):
        raise ValueError("Must specify exactly one of --image_path or --image_dir")

    image_paths = []
    if args.image_path:
        image_paths.append(Path(args.image_path))
    else:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_paths = list(map(Path, image_paths))

    # Initialize models
    gaussian_channel_simulator = GaussianChannelSimulator(
        config.max_chunk_size, 
        config.chunk_padding
    )
    noise_prediction_model = get_noise_prediction_model(config.model, config)

    # Get captions if needed
    captions = {}
    if config.encoding_guidance_scale or config.denoising_guidance_scale:
        captioner = BlipCaptioner()
        captions = captioner.process_and_save(image_paths, output_dir)
        del captioner

    # Process each image
    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}.diffc"
        caption = captions.get(str(image_path), "")
        compress_image(
            image_path, 
            output_path,
            noise_prediction_model, 
            gaussian_channel_simulator,
            config,
            caption,
            output_dir
        )

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    args = parse_args()
    main()
