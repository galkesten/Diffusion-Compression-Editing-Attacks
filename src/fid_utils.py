import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Per-dataset FID patch size (Amit's config). Fallback 64 for unknown datasets.
FID_PATCH_SIZES: Dict[str, int] = {
    "Kodak24": 64,
    "Kodak24_512": 64,
    "CLIC2020_512": 64,
    "DIV2K_valid_HR_512": 128,
    "DIV2K_valid_HR_768": 64,
    "CLIC2020_768": 64,
}

DEFAULT_FID_PATCH_SIZE = 64


def get_fid_patch_size(dataset_name: str) -> int:
    """Return FID patch size for dataset (from Amit's config). Unknown datasets use DEFAULT_FID_PATCH_SIZE."""
    return FID_PATCH_SIZES.get(dataset_name, DEFAULT_FID_PATCH_SIZE)


def compute_patch_fid(
    ref_dir: str,
    gen_dir: str,
    image_files: List[str],
    img_height: int,
    img_width: int,
    device: Optional[torch.device] = None,
    *,
    dataset_name: Optional[str] = None,
) -> float:
    """Patch FID between ref_dir and gen_dir. Patch size is derived from dataset_name via FID_PATCH_SIZES, or DEFAULT_FID_PATCH_SIZE if dataset_name is None."""
    patch_size = get_fid_patch_size(dataset_name) if dataset_name is not None else DEFAULT_FID_PATCH_SIZE
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        from neuralcompression.metrics import update_patch_fid
    except ImportError as e:
        raise ImportError(
            "neuralcompression is required for FID. Install with: pip install neuralcompression"
        ) from e
    try:
        from torchmetrics.image import FrechetInceptionDistance
    except ImportError as e:
        raise ImportError("torchmetrics is required for FID. Install with: pip install torchmetrics") from e

    FID = FrechetInceptionDistance(normalize=True).to(device)
    n_pairs = 0

    for img_file in image_files:
        if not any(img_file.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
            continue
        ref_path = os.path.join(ref_dir, img_file)
        gen_path = os.path.join(gen_dir, img_file)
        if not os.path.isfile(ref_path) or not os.path.isfile(gen_path):
            continue
        gt_img = to_tensor(Image.open(ref_path).convert("RGB")).to(device).unsqueeze(0)
        rec_img = to_tensor(Image.open(gen_path).convert("RGB")).to(device).unsqueeze(0)
        assert gt_img.shape == rec_img.shape, (
            f"FID requires ref and rec same shape (no interpolation). Got {gt_img.shape} vs {rec_img.shape} for {img_file}"
        )
        update_patch_fid(gt_img, rec_img, fid_metric=FID, patch_size=patch_size)
        n_pairs += 1

    if n_pairs == 0:
        return float("nan")
    return float(FID.compute().item())
