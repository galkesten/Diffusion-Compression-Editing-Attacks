import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from diffusers.utils import load_image


try:
    from diffusers import FluxKontextPipeline
except ImportError:
    raise ImportError(
        "FluxKontextPipeline not available in your diffusers version.\n"
        "Please upgrade diffusers: pip install --upgrade diffusers\n"
        "Or install the latest: pip install diffusers>=0.33.0"
    )

def apply_style_to_dataset(input_dir, output_dir, style_name, lora_path, device_str, image_size=512, num_inference_steps=24):
    os.makedirs(output_dir, exist_ok=True)
    

    print(f"Loading FLUX.1 Kontext pipeline on {device_str}...")
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    ).to(device_str)
    

    print(f"Loading {style_name} LoRA from {lora_path}...")
    weight_name = f"{style_name}_lora_weights.safetensors"
    pipeline.load_lora_weights(lora_path, weight_name=weight_name, adapter_name="lora")
    pipeline.set_adapters(["lora"], adapter_weights=[1])
    

    files = sorted(os.listdir(input_dir))
    target_files = [f for f in files if f.endswith('.png')]
    
    if not target_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")
    
    print(f"\nFound {len(target_files)} images to process")
    print(f"Applying {style_name} style transformation...")
    print(f"Output images will be exactly {image_size}x{image_size} pixels")
    

    for file_name in tqdm(target_files, desc=f"Processing {style_name} style"):

        image_path = os.path.join(input_dir, file_name)
        image = load_image(image_path).resize((image_size, image_size))
        
        prompt = f"Turn this image into the {style_name} style."
        

        result_image = pipeline(
            image=image,
            prompt=prompt,
            height=image_size,
            width=image_size,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        if result_image.size != (image_size, image_size):
            result_image = result_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_{style_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        result_image.save(output_path)
    
    print(f"\nDone! Processed {len(target_files)} images")
    print(f"Styled images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Apply style LoRA transformations to dataset')
    parser.add_argument('--input_dir', type=str, default='dataset_Kodack24',
                       help='Input directory containing images')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device index to use')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for processing - output will be exactly this size (default: 512)')
    parser.add_argument('--num_inference_steps', type=int, default=24,
                       help='Number of diffusion inference steps (default: 24)')
    parser.add_argument('--styles', type=str, nargs='+', 
                       default=['LEGO', 'Pop_Art'],
                       help='Styles to apply (default: LEGO Pop_Art)')
    
    args = parser.parse_args()
    
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Style configurations
    style_configs = {
        'LEGO': {
            'lora_path': 'Kontext-Style/LEGO_lora',
            'output_suffix': 'LEGO'
        },
        'Pop_Art': {
            'lora_path': 'Kontext-Style/Pop_Art_lora',
            'output_suffix': 'Pop_Art'
        }
    }
    
    # Process each style
    for style_name in args.styles:
        if style_name not in style_configs:
            print(f"Warning: Unknown style '{style_name}', skipping...")
            continue
        
        config = style_configs[style_name]
        output_dir = f"{args.input_dir}_{config['output_suffix']}"
        
        print(f"\n{'='*60}")
        print(f"Processing style: {style_name}")
        print(f"{'='*60}")
        
        apply_style_to_dataset(
            input_dir=args.input_dir,
            output_dir=output_dir,
            style_name=style_name,
            lora_path=config['lora_path'],
            device_str=device_str,
            image_size=args.image_size,
            num_inference_steps=args.num_inference_steps
        )
    
    print(f"\n{'='*60}")
    print("All styles processed successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

