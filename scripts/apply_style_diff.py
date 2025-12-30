#!/usr/bin/env python3
"""
Apply style difference from one image pair to another image.

Given:
- X: original image (e.g., dataset_Kodack24/4.png)
- X_style: styled version of X (e.g., dataset_Kodack24_Pop_Art/4_Pop_Art.png)
- Y: target image to apply style to (e.g., dataset_Kodack24/15.png)

Computes:
1. diff = encode(X) - encode(X_style)  (style difference in VAE latent space)
2. new_latent = encode(Y) + diff
3. new_image = decode(new_latent)

Saves the resulting image.
"""

import os
import argparse
import torch
from turbo_ddcm.ddpm import DDPM
import turbo_ddcm.utils as utils


def main(args):
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    torch_dtype = torch.float32 if args.float32 else torch.float16
    
    print(f"Using device: {device_str}, dtype: {torch_dtype}")
    
    # Initialize DDPM for VAE encoding/decoding
    model_id = "Manojb/stable-diffusion-2-1-base"
    print(f"Loading model: {model_id}")
    ddpm = DDPM(model_id, torch_dtype, T=20, device=device_str, seed=args.seed)
    
    # Load images
    print(f"\nLoading images:")
    print(f"  X (original): {args.x_path}")
    print(f"  X_style (styled): {args.x_style_path}")
    print(f"  Y (target): {args.y_path}")
    
    img_x = utils.load_image(args.x_path, resize_to=512, device=device_str).to(dtype=torch_dtype)
    img_x_style = utils.load_image(args.x_style_path, resize_to=512, device=device_str).to(dtype=torch_dtype)
    img_y = utils.load_image(args.y_path, resize_to=512, device=device_str).to(dtype=torch_dtype)
    
    print(f"  Image shapes: X={img_x.shape}, X_style={img_x_style.shape}, Y={img_y.shape}")
    print(f"  Image dtypes: X={img_x.dtype}, X_style={img_x_style.dtype}, Y={img_y.dtype}")
    
    # Encode to VAE latent space
    print(f"\nEncoding images to VAE latent space...")
    with torch.no_grad():
        latent_x = ddpm.encode_image(img_x)
        latent_x_style = ddpm.encode_image(img_x_style)
        latent_y = ddpm.encode_image(img_y)
    
    print(f"  Latent shapes: X={latent_x.shape}, X_style={latent_x_style.shape}, Y={latent_y.shape}")
    
    # Compute style difference
    print(f"\nComputing style difference: diff = encode(X) - encode(X_style)")
    diff = latent_x - latent_x_style
    print(f"  Diff shape: {diff.shape}")
    print(f"  Diff stats: min={diff.min().item():.4f}, max={diff.max().item():.4f}, mean={diff.mean().item():.4f}, std={diff.std().item():.4f}")
    
    # Apply difference to Y
    print(f"\nApplying style difference to Y: new_latent = encode(Y) + diff")
    new_latent = latent_y + diff
    print(f"  New latent shape: {new_latent.shape}")
    print(f"  New latent stats: min={new_latent.min().item():.4f}, max={new_latent.max().item():.4f}, mean={new_latent.mean().item():.4f}, std={new_latent.std().item():.4f}")
    
    # Decode back to pixel space
    print(f"\nDecoding new latent to pixel space...")
    with torch.no_grad():
        new_image = ddpm.decode_img(new_latent)
    
    print(f"  Decoded image shape: {new_image.shape}")
    
    # Save result
    output_path = args.output_path
    if output_path is None:
        # Auto-generate output path
        y_base = os.path.splitext(os.path.basename(args.y_path))[0]
        x_base = os.path.splitext(os.path.basename(args.x_path))[0]
        style_name = os.path.splitext(os.path.basename(args.x_style_path))[0].replace(x_base, '').strip('_')
        output_path = f"{y_base}_with_{style_name}_style_from_{x_base}.png"
    
    print(f"\nSaving result to: {output_path}")
    # Save image directly (workaround for utils.save_decoded_img issue with empty dirname)
    import matplotlib.pyplot as plt
    from turbo_ddcm.utils import clear_color
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there's a directory path
        os.makedirs(output_dir, exist_ok=True)
    plt.imsave(output_path, clear_color(new_image))
    
    print(f"\nDone! Result saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply style difference from one image pair to another image')
    parser.add_argument('--x_path', type=str, required=True,
                        help='Path to original image X (e.g., dataset_Kodack24/4.png)')
    parser.add_argument('--x_style_path', type=str, required=True,
                        help='Path to styled version of X (e.g., dataset_Kodack24_Pop_Art/4_Pop_Art.png)')
    parser.add_argument('--y_path', type=str, required=True,
                        help='Path to target image Y to apply style to (e.g., dataset_Kodack24/15.png)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for result image (default: auto-generated)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--seed', type=int, default=88888888, help='Random seed')
    parser.add_argument('--float32', action='store_true', help='Use float32 instead of float16')
    
    args = parser.parse_args()
    main(args)

