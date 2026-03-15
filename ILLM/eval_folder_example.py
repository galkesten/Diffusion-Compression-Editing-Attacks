# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
from argparse import ArgumentParser
from pathlib import Path
import time
import pickle
from torchvision.utils import save_image
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.transforms import ToTensor
from tqdm import tqdm

from neuralcompression.metrics import (
    MultiscaleStructuralSimilarity,
    calc_psnr,
    pickle_size_of,
    update_patch_fid,
)


def rescale_image(image: Tensor, back_to_float: bool = False) -> Tensor:
    dtype = image.dtype
    image = (image * 255 + 0.5).to(torch.uint8)

    if back_to_float:
        image = image.to(dtype)

    return image


def compress(args):
    clic_path = Path(args.path)
    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/NeuralCompression", args.model)
    model = model.to(device)
    model = model.eval()
    model.update()
    model.update_tensor_devices("compress")
    totensor = ToTensor()

    for image_path in list(clic_path.glob("*.png")):
        with open(image_path, "rb") as f:
            image_pil = Image.open(f)
            image_pil = image_pil.convert("RGB")

        image = totensor(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            compressed = model.compress(image, force_cpu=False)

            with open(os.path.join(args.output_path, f"{os.path.basename(image_path).split('.')[0]}.bin"), "wb") as f:
                pickle.dump(compressed, f)

            # decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)

        # orig_image = rescale_image(image)
        # pred_image = rescale_image(decompressed)

        # with torch.no_grad():
        #     update_patch_fid(image, decompressed, fid_metric)
        #     orig_image = rescale_image(image)
        #     pred_image = rescale_image(decompressed)

def decompress(args):
    bins_path = Path(args.path)
    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/NeuralCompression", args.model)
    model = model.to(device)
    model = model.eval()
    model.update()
    model.update_tensor_devices("compress")

    for bin_path in list(bins_path.glob("*.bin")):
        with open(bin_path, "rb") as f:
            compressed = pickle.load(f)

        with torch.no_grad():
            decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)
            # pred_image = rescale_image(decompressed).detach().cpu().squeeze(0)
            save_image(decompressed, os.path.join(args.output_path, f"{os.path.basename(bin_path).split('.')[0]}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="path to CLIC2020 / BINS directory")
    parser.add_argument("output_path", type=str, help="path to CLIC2020 directory")
    parser.add_argument("model", type=str, help="path to CLIC2020 directory")
    parser.add_argument("com_decom", type=str, help="path to CLIC2020 directory")
    args = parser.parse_args()

    if args.com_decom == 'compress':
        compress(args)
    elif args.com_decom == 'decompress':
        decompress(args)
    else:
        raise ValueError(f"Unknown com_decom {args.com_decom}")

