import os
import shutil
from typing import Any, Dict, Optional

import pandas as pd
from PIL import Image

from .base import BaseModelRunner, list_png_sorted, list_files_sorted


def _load_jpeg_quality_csv(csv_path: str) -> Dict[str, int]:
    """Load quality per image from CSV with columns image_file and quality."""
    df = pd.read_csv(csv_path)
    if "quality" not in df.columns:
        raise ValueError(f"JPEG quality CSV must have a 'quality' column: {csv_path}")
    image_col = "image_file" if "image_file" in df.columns else df.columns[0]
    return dict(zip(df[image_col].astype(str), df["quality"].astype(int)))


class JpegModelRunner(BaseModelRunner):
    name = "jpeg"

    def get_model_params(self) -> Dict[str, object]:
        return {}

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, f"{base}_compressed.jpg")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"{base}_compressed.jpg")
        shutil.copy(compressed_path, dest)
        return dest

    def find_decompressed_png(self, temp_dir: str, base: str) -> Optional[str]:
        p = os.path.join(temp_dir, f"{base}_decompressed.png")
        return p if os.path.isfile(p) else None

    def get_baseline_params(self, jpeg_quality_csv: Optional[str] = None) -> Dict[str, Any]:
        if not jpeg_quality_csv or not os.path.isfile(jpeg_quality_csv):
            raise ValueError("JPEG requires jpeg_quality_csv (CSV with columns image_file, quality)")
        return {"quality_by_image": _load_jpeg_quality_csv(jpeg_quality_csv)}

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
