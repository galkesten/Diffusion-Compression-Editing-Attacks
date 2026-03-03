"""BPG (Better Portable Graphics) compression runner using bpgenc/bpgdec."""

import os
import shutil
import subprocess
from typing import Any, Dict, Optional

from PIL import Image

from .base import BaseModelRunner, list_png_sorted, list_files_sorted


class BpgModelRunner(BaseModelRunner):
    """Runner for BPG compression via bpgenc/bpgdec. Quality q in 0-51 (lower=better, larger)."""

    name = "bpg"

    def __init__(self, quality_by_image: Dict[str, int]):
        self.quality_by_image = dict(quality_by_image)

    def get_model_params(self) -> Dict[str, Any]:
        return {"quality_by_image": self.quality_by_image}

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, f"{base}_compressed.bpg")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"{base}_compressed.bpg")
        shutil.copy(compressed_path, dest)
        return dest

    def find_decompressed_png(self, temp_dir: str, base: str) -> Optional[str]:
        p = os.path.join(temp_dir, f"{base}_decompressed.png")
        return p if os.path.isfile(p) else None

    def _run_bpgenc(self, png_path: str, bpg_path: str, quality: int) -> None:
        subprocess.run(
            ["bpgenc", "-q", str(quality), "-o", bpg_path, png_path],
            check=True,
            capture_output=True,
            text=True,
        )

    def _run_bpgdec(self, bpg_path: str, png_path: str) -> None:
        subprocess.run(
            ["bpgdec", "-o", png_path, bpg_path],
            check=True,
            capture_output=True,
            text=True,
        )

    def run_compression(
        self,
        input_dir: str,
        output_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        params = self.get_model_params()
        os.makedirs(output_dir, exist_ok=True)
        image_files = list_png_sorted(input_dir)
        quality_by_image: Dict[str, int] = params.get("quality_by_image", {})
        errors: Dict[str, str] = {}
        expected_size = (img_width, img_height)

        for image_file in image_files:
            quality = quality_by_image.get(image_file)
            if quality is None:
                errors[image_file] = "missing BPG quality for image"
                continue
            try:
                src = os.path.join(input_dir, image_file)
                base = os.path.splitext(image_file)[0]
                out_bpg = os.path.join(output_dir, f"{base}_compressed.bpg")
                img = Image.open(src).convert("RGB")
                if img.size != expected_size:
                    raise ValueError(f"image size {img.size} != expected {expected_size}")
                self._run_bpgenc(src, out_bpg, int(quality))
            except subprocess.CalledProcessError as exc:
                errors[image_file] = f"bpgenc failed: {exc.stderr or exc}"
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
        bpg_files = list_files_sorted(compressed_dir, ".bpg")

        for bpg_file in bpg_files:
            base = os.path.splitext(bpg_file)[0].replace("_compressed", "")
            image_file = f"{base}.png"
            try:
                src = os.path.join(compressed_dir, bpg_file)
                out = os.path.join(compressed_dir, f"{base}_decompressed.png")
                self._run_bpgdec(src, out)
                # Optionally verify size
                img = Image.open(out)
                expected = (img_width, img_height)
                if img.size != expected:
                    raise ValueError(f"BPG decode size mismatch: {img.size} != {expected}")
            except subprocess.CalledProcessError as exc:
                errors[image_file] = f"bpgdec failed: {exc.stderr or exc}"
            except Exception as exc:
                errors[image_file] = str(exc)

        return errors
