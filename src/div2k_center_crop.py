"""
Open DIV2K_valid_HR.zip and center-crop all images to 512x512.
Saves cropped images to an output directory (or a new zip if requested).
Uses torchvision.transforms.functional.center_crop for cropping.
"""
import argparse
import os
import zipfile
from pathlib import Path
from typing import cast

from PIL import Image
from torchvision.transforms.functional import center_crop


def process_div2k_zip(
    zip_path: str | Path,
    out_dir: str | Path,
    size: int = 512,
    output_zip: str | Path | None = None,
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> None:
    """
    Read images from zip_path, center-crop to size x size, save to out_dir and optionally to output_zip.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in sorted(names):
            base = os.path.basename(name)
            ext = os.path.splitext(base)[1].lower()
            if ext not in image_extensions:
                print(f"Skip {name}: {ext} not in {image_extensions}")
                continue
            try:
                with zf.open(name) as f:
                    img = Image.open(f).convert("RGB")
            except Exception as e:
                print(f"Skip {name}: {e}")
                continue
            # center_crop accepts PIL and returns PIL; stubs are Tensor-only
            cropped = cast(Image.Image, center_crop(img, [size, size]))  # type: ignore[arg-type]
            out_name = base
            out_path = out_dir / out_name
            cropped.save(out_path, quality=95)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} images...")

    print(f"Done. Saved {count} images to {out_dir}")

    if output_zip:
        output_zip = Path(output_zip)
        print(f"Writing {output_zip}...")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.iterdir()):
                if p.is_file():
                    zf.write(p, p.name)
        print(f"Created {output_zip}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Center-crop DIV2K_valid_HR.zip images to 512x512."
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("DIV2K_valid_HR.zip"),
        help="Path to DIV2K_valid_HR.zip (default: DIV2K_valid_HR.zip in cwd)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("DIV2K_valid_HR_512"),
        help="Output directory for cropped images (default: DIV2K_valid_HR_512)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Crop to size x size (default: 512)",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="If set, also pack cropped images into this zip file",
    )
    args = parser.parse_args()

    process_div2k_zip(
        zip_path=args.zip,
        out_dir=args.out_dir,
        size=args.size,
        output_zip=args.output_zip,
    )


if __name__ == "__main__":
    main()
