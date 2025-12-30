import os
import argparse
import torch
from tqdm import tqdm
import json

from turbo_ddcm.turbo_ddcm import TurboDDCM
from turbo_ddcm import utils

def main(args):
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    input_dir = args.input_dir
    compressed_dir = args.compressed_dir
    decompressed_dir = args.decompressed_dir
    
    os.makedirs(compressed_dir, exist_ok=True)
    os.makedirs(decompressed_dir, exist_ok=True)
    
    files = sorted(os.listdir(input_dir))
    target_files = [f for f in files if f.endswith('.png')]
    
    if not target_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}.")
    
    print(f"Found {len(target_files)} images to process")
    
    test_img = utils.load_image(os.path.join(input_dir, target_files[0]), None, device_str)
    print(test_img.shape)
    if max(test_img.shape[2], test_img.shape[3]) < 512:
        raise ValueError(f"Too small images. Minimum size is 512x512.")
    elif min(test_img.shape[2], test_img.shape[3]) >= 768:
        print(f"Images are too large, will be resized to 768x768")
        resize_to = (768, 768)
        model_id = "stabilityai/stable-diffusion-2-1"
    else:
        print(f"Images are small enough, will be resized to 512x512")
        resize_to = (512, 512)
        model_id = "Manojb/stable-diffusion-2-1-base"  # Using alternative since official was deprecated by Stability AI
    
    if test_img.shape[2] != resize_to[0] or test_img.shape[3] != resize_to[1]:
        print(f"Images will be resized to {resize_to}")
    else:
        print(f"Images are already {resize_to}, no resizing needed")
    
    turbo_ddcm = TurboDDCM(model_id, args.T, args.K, args.M, args.seed, args.float32, device_str)
    
    config = {
        'model_id': model_id,
        'T': args.T,
        'K': args.K,
        'M': args.M,
        'seed': args.seed,
        'float32': args.float32,
        'resize_to': resize_to
    }
    print(config)
    with open(os.path.join(compressed_dir, 'compression_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    for file_name in tqdm(target_files, desc="Processing images"):
        print(f"Processing {file_name}", flush=True)
        img = utils.load_image(os.path.join(input_dir, file_name), resize_to, device_str)
        
        (reconstruction, encoding, step_info), _ = turbo_ddcm.compress(img, None)
        
        base_name = os.path.splitext(file_name)[0]
        binary_path = os.path.join(compressed_dir, base_name + utils.BIN_SUFFIX)
        utils.save_as_binary(encoding, binary_path)
        
        # Save step information as JSON
        step_info_path = os.path.join(compressed_dir, base_name + '_step_info.json')
        with open(step_info_path, 'w') as f:
            json.dump(step_info, f, indent=2)

        reconstruction = turbo_ddcm.decompress(encoding)
        
        decompressed_path = os.path.join(decompressed_dir, file_name)
        utils.save_decoded_img(decompressed_path, reconstruction)
        
        print(f"Processed {file_name}: compressed -> {binary_path}, decompressed -> {decompressed_path}")
    
    print(f"\nDone! Processed {len(target_files)} images")
    print(f"Compressed files saved to: {compressed_dir}")
    print(f"Decompressed images saved to: {decompressed_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress and decompress images from dataset')
    
    parser.add_argument('--input_dir', type=str, default='dataset_Kodack24', 
                       help='Directory containing images to process')
    parser.add_argument('--compressed_dir', type=str, default='dataset_Kodack24_compressed',
                       help='Directory to save compressed bitstreams')
    parser.add_argument('--decompressed_dir', type=str, default='dataset_Kodack24_decompress',
                       help='Directory to save decompressed images')
    parser.add_argument('--M', type=int, default=30, 
                       help='Atoms to be chosen from the codebook in each diffusion step')
    
    # Optional parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--float32', action='store_true', help='Use float32 precision for model inference')
    parser.add_argument('--seed', type=int, default=88888888, help='Random seed')
    parser.add_argument('--T', type=int, default=20, help='Compress using T diffusion steps')
    parser.add_argument('--K', type=int, default=16384, help="Codebook size")
    
    args = parser.parse_args()
    
    main(args)

