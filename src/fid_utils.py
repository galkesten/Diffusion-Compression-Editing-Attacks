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

    patch_size = get_fid_patch_size(dataset_name) if dataset_name is not None else DEFAULT_FID_PATCH_SIZE
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[FID] Dataset: {dataset_name}")
    print(f"[FID] Patch size: {patch_size}")
    print(f"[FID] Device: {device}")

    from neuralcompression.metrics import update_patch_fid
    from torchmetrics.image import FrechetInceptionDistance

    FID = FrechetInceptionDistance(normalize=True).to(device)

    n_pairs = 0
    total_patches = 0
    missing: List[str] = []
    resized: List[str] = []

    for img_file in image_files:
        if not any(img_file.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
            continue

        ref_path = os.path.abspath(os.path.join(ref_dir, img_file))
        gen_path = os.path.abspath(os.path.join(gen_dir, img_file))
        if not os.path.isfile(ref_path) or not os.path.isfile(gen_path):
            missing.append(img_file)
            continue

        ref_pil = Image.open(ref_path).convert("RGB")
        rec_pil = Image.open(gen_path).convert("RGB")
        if ref_pil.size != rec_pil.size:
            old_w, old_h = ref_pil.size
            ref_pil = ref_pil.resize(rec_pil.size)
            resized.append(img_file)
            print("[FID] Resized %s ref to match decomp: %dx%d -> %dx%d" % (img_file, old_w, old_h, rec_pil.size[0], rec_pil.size[1]))
        gt_img = to_tensor(ref_pil).to(device).unsqueeze(0)
        rec_img = to_tensor(rec_pil).to(device).unsqueeze(0)

        # count patches
        h, w = gt_img.shape[-2:]
        patches = (h // patch_size) * (w // patch_size)
        total_patches += patches

        update_patch_fid(gt_img, rec_img, fid_metric=FID, patch_size=patch_size)
        n_pairs += 1

    if missing:
        print("[FID] Skipping %d missing pair(s): %s" % (len(missing), missing if len(missing) <= 10 else missing[:10]))
    if resized:
        print("[FID] Resized %d ref image(s) to match decomp: %s" % (len(resized), resized if len(resized) <= 10 else resized[:10]))
    if not missing and not resized:
        print("[FID] All %d image pairs found" % n_pairs)
    print("[FID] Images used: %d, total patches: %d" % (n_pairs, total_patches))

    if n_pairs == 0:
        print("[FID] No valid image pairs — returning NaN")
        return float("nan")

    # inspect internal feature tensors
    # try:
    #     real = getattr(FID, "_features_real", None)
    #     fake = getattr(FID, "_features_fake", None)
    #
    #     if real is not None and fake is not None:
    #         print("[FID] real feature vectors:", real.shape)
    #         print("[FID] fake feature vectors:", fake.shape)
    #     else:
    #         print("[FID] Feature buffers not populated yet")
    # except Exception as e:
    #     print("[FID] Could not access internal feature tensors:", e)


    try:
        fid_value = float(FID.compute().item())
        if fid_value != fid_value:  # NaN check
            print("[FID] FID.compute() returned NaN (e.g. too few patches or numerical issue)")
        print(f"[FID] Final FID: {fid_value}")
        return fid_value
    except Exception as e:
        print("[FID] FID.compute() failed: %s" % e)
        import traceback
        traceback.print_exc()
        return float("nan")
