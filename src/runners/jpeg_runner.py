import os
from typing import Any, Dict

from PIL import Image

from .base import BaseModelRunner, list_png_sorted, list_files_sorted


class JpegModelRunner(BaseModelRunner):
    name = "jpeg"

    def get_model_params(self) -> Dict[str, object]:
        return {}

    def run_compression(
        self,
        input_dir: str,
        output_dir: str,
        *,
        img_height: int,
        img_width: int,
        params: Dict[str, Any],
    ) -> Dict[str, str]:
    
        os.makedirs(output_dir, exist_ok=True)
        image_files = list_png_sorted(input_dir)
        quality_by_image: Dict[str, int] = params.get("quality_by_image", {})
        errors: Dict[str, str] = {}
        resize_to = (img_width, img_height)

        for image_file in image_files:
            quality = quality_by_image.get(image_file)
            if quality is None:
                errors[image_file] = "missing JPEG quality for image"
                continue
            try:
                src = os.path.join(input_dir, image_file)
                base = os.path.splitext(image_file)[0]
                out = os.path.join(output_dir, f"{base}_compressed.jpg")
                img = Image.open(src).convert("RGB")
                if img.size != resize_to:
                    img = img.resize(resize_to)
                img.save(out, format="JPEG", quality=int(quality))
            except Exception as exc:
                errors[image_file] = str(exc)

        return errors

    def run_decompression(
        self,
        compressed_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        os.makedirs(compressed_dir, exist_ok=True)
        errors: Dict[str, str] = {}
        expected_size = (img_width, img_height)
        # Compression writes {base}_compressed.jpg into compressed_dir; decompress in place
        jpg_files = list_files_sorted(compressed_dir, ".jpg")

        for jpg_file in jpg_files:
            base = os.path.splitext(jpg_file)[0].replace("_compressed", "")
            image_file = f"{base}.png"
            try:
                src = os.path.join(compressed_dir, jpg_file)
                img = Image.open(src).convert("RGB")
                if img.size != expected_size:
                    raise ValueError(f"JPEG size mismatch: {img.size} != {expected_size}")
                out = os.path.join(compressed_dir, f"{base}_decompressed.png")
                img.save(out, format="PNG")
            except Exception as exc:
                errors[image_file] = str(exc)

        return errors
