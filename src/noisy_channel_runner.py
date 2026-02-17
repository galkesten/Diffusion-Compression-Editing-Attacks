import argparse
import csv
import os
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from experiment_utils import flip_bits, get_images, parse_subset, resolve_subset
from fid_utils import compute_patch_fid

from runners import (
    BaseModelRunner,
    DdcmModelRunner,
    JpegModelRunner,
    TurboModelRunner,
)


def _runner_factory(project_root: str, algorithm: str) -> BaseModelRunner:
    if algorithm == "jpeg":
        return JpegModelRunner()
    if algorithm == "ddcm":
        return DdcmModelRunner(project_root)
    if algorithm == "turbo_ddcm":
        return TurboModelRunner(project_root, robust=False)
    if algorithm == "robust_turbo_ddcm":
        return TurboModelRunner(project_root, robust=True)
    raise ValueError(f"Unknown algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# MetricsComputer
# ---------------------------------------------------------------------------
def _img_to_tensor(img: Image.Image) -> Any:
    import torch
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class MetricsComputer:
    """Computes per-image metrics (PSNR, NIQE, LPIPS) and patch FID via fid_utils."""

    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self._niqe_metric = None
        self._lpips_metric = None

    def _get_niqe(self):
        if self._niqe_metric is None:
            try:
                from pyiqa import create_metric
                self._niqe_metric = create_metric("niqe", device="cpu")
            except Exception:
                self._niqe_metric = False
        return self._niqe_metric

    def _get_lpips(self):
        if self._lpips_metric is None:
            try:
                from pyiqa import create_metric
                self._lpips_metric = create_metric("lpips", device="cpu")
            except Exception:
                self._lpips_metric = False
        return self._lpips_metric

    def psnr(self, ref: Image.Image, decomp: Image.Image) -> float:
        if ref.size != decomp.size:
            ref = ref.resize(decomp.size)
        a = np.array(ref)
        b = np.array(decomp)
        return float(psnr(a, b, data_range=255))

    def niqe(self, decomp: Image.Image) -> float:
        m = self._get_niqe()
        if m is False:
            return float("nan")
        t = _img_to_tensor(decomp)
        return float(cast(Any, m)(t))

    def lpips(self, ref: Image.Image, decomp: Image.Image) -> float:
        """LPIPS (lower is better). Returns NaN if metric unavailable."""
        m = self._get_lpips()
        if m is False:
            return float("nan")
        if ref.size != decomp.size:
            ref = ref.resize(decomp.size)
        t_ref = _img_to_tensor(ref)
        t_decomp = _img_to_tensor(decomp)
        return float(cast(Any, m)(t_ref, t_decomp))

    def fid(
        self,
        ref_dir: str,
        decomp_dir: str,
        image_files: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ) -> Optional[float]:
        """Compute patch FID between ref_dir and decomp_dir (same filenames). Uses fid_utils.compute_patch_fid. Patch size from dataset_name when set."""
        allowed = (".png", ".jpg", ".jpeg")
        files = image_files or [f for f in os.listdir(ref_dir) if f.lower().endswith(allowed)]
        if not files:
            return None
        try:
            return float(
                compute_patch_fid(
                    ref_dir,
                    decomp_dir,
                    files,
                    self.resolution,
                    self.resolution,
                    dataset_name=dataset_name,
                )
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# NoisyChannelExperiment
# ---------------------------------------------------------------------------
class NoisyChannelExperiment:
    """Runs BER -> trials -> images; creates temp per image, noising, decompress via runner, move to trial folder, remove temps."""

    def __init__(
        self,
        project_root: str,
        algorithm: str,
        compressed_dir: str,
        dataset_path: str,
        image_files: List[str],
        target_bpp: float,
        resolution: int,
        jpeg_quality_csv: Optional[str] = None,
    ):
        self.project_root = os.path.abspath(project_root)
        self.algorithm = algorithm
        self.compressed_dir = os.path.abspath(compressed_dir)
        self.dataset_path = dataset_path
        self.image_files = list(image_files)
        self.target_bpp = target_bpp
        self.resolution = resolution
        self.metrics = MetricsComputer(resolution)
        self.runner: BaseModelRunner = _runner_factory(self.project_root, algorithm)

    def _prepare_temp_and_flip(
        self,
        compressed_path: str,
        temp_dir: str,
        base: str,
        ber: float,
        trial: int,
    ) -> Tuple[Optional[str], int]:
        """Copy compressed file(s) to temp_dir via runner, apply BER flip. Returns (path_to_file_we_flipped, num_flips)."""
        dest = self.runner.prepare_temp_for_noisy_channel(compressed_path, temp_dir, base)
        n_flips = flip_bits(dest, ber, dest)
        return dest, n_flips

    def _find_decompressed_png(self, temp_dir: str, base: str) -> Optional[str]:
        return self.runner.find_decompressed_png(temp_dir, base)

    def _decompress_temp(self, temp_dir: str) -> Dict[str, str]:
        return self.runner.run_decompression(
            temp_dir,
            img_height=self.resolution,
            img_width=self.resolution,
        )

    def run_one(
        self,
        img_file: str,
        ber: float,
        trial: int,
        temp_base: str,
        trial_folder: str,
    ) -> Tuple[Optional[str], int, Optional[float], Optional[float], Optional[float]]:
        """Run noising + decompress for one image. Returns (decomp_path_in_trial, num_flips, psnr, niqe, lpips)."""
        base = os.path.splitext(img_file)[0]
        compressed_path = self.runner.path_for_compressed(self.compressed_dir, base)
        if not compressed_path:
            return None, 0, None, None, None
        temp_dir = os.path.join(temp_base, f"ber{ber}_trial{trial}_{base}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            _, num_flips = self._prepare_temp_and_flip(compressed_path, temp_dir, base, ber, trial)
            errs = self._decompress_temp(temp_dir)
            if img_file in errs:
                return None, num_flips, None, None, None
            decomp_path = self._find_decompressed_png(temp_dir, base)
            if not decomp_path or not os.path.isfile(decomp_path):
                return None, num_flips, None, None, None
            dest_in_trial = os.path.join(trial_folder, img_file)
            os.makedirs(os.path.dirname(dest_in_trial) or ".", exist_ok=True)
            shutil.move(decomp_path, dest_in_trial)
            ref = Image.open(os.path.join(self.dataset_path, img_file)).convert("RGB").resize((self.resolution, self.resolution))
            decomp = Image.open(dest_in_trial).convert("RGB")
            psnr_val = self.metrics.psnr(ref, decomp)
            niqe_val = self.metrics.niqe(decomp)
            lpips_val = self.metrics.lpips(ref, decomp)
            return dest_in_trial, num_flips, psnr_val, niqe_val, lpips_val
        finally:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _write_baseline_rows(
        self,
        csv_path: str,
        output_dir: Optional[str] = None,
        sample_image_files: Optional[List[str]] = None,
    ) -> None:
        """Write one CSV row per image for baseline (BER=0, no bit flips). If output_dir and sample_image_files are set, save one baseline decompressed image per selected image to samples/baseline/."""
        for img_file in self.image_files:
            base = os.path.splitext(img_file)[0]
            decomp_path = self.runner.find_decompressed_png(self.compressed_dir, base)
            if decomp_path and os.path.isfile(decomp_path):
                ref = Image.open(os.path.join(self.dataset_path, img_file)).convert("RGB").resize((self.resolution, self.resolution))
                decomp = Image.open(decomp_path).convert("RGB")
                psnr_val = self.metrics.psnr(ref, decomp)
                niqe_val = self.metrics.niqe(decomp)
                lpips_val = self.metrics.lpips(ref, decomp)
                if output_dir and sample_image_files and img_file in sample_image_files:
                    sample_dir = os.path.join(output_dir, "samples", "baseline")
                    os.makedirs(sample_dir, exist_ok=True)
                    shutil.copy(decomp_path, os.path.join(sample_dir, f"{base}.png"))
            else:
                psnr_val = niqe_val = lpips_val = "N/A"
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([img_file, 0, 0, 0, psnr_val, niqe_val, lpips_val])

    def run(
        self,
        ber_values: List[float],
        num_trials: int,
        output_dir: str,
        num_samples_per_ber: int = 3,
        sample_image_files: Optional[List[str]] = None,
    ) -> None:
        """sample_image_files: which images to save samples for (e.g. indices 0-5). If None, all images. FID patch size from dataset name (fid_utils.FID_PATCH_SIZES)."""
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        temp_base = os.path.join(output_dir, "temp_work")
        os.makedirs(temp_base, exist_ok=True)
        dataset_name = os.path.basename(os.path.normpath(self.dataset_path))
        csv_path = os.path.join(output_dir, f"noisy_channel_{self.runner.name}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_file", "ber", "trial", "num_bit_flips", "psnr", "niqe", "lpips"])
        images_to_sample = sample_image_files if sample_image_files is not None else self.image_files
        self._write_baseline_rows(csv_path, output_dir=output_dir, sample_image_files=images_to_sample)
        # samples/ has one subfolder per BER (ber0.1, ber0.01, ...) plus baseline/
        fid_rows: List[Tuple[float, int, float]] = []
        for ber in ber_values:
            samples_saved = {img_file: 0 for img_file in images_to_sample}
            sample_ber_dir = os.path.join(output_dir, "samples", f"ber{ber}")
            os.makedirs(sample_ber_dir, exist_ok=True)
            for trial in range(num_trials):
                trial_folder = os.path.join(temp_base, f"ber{ber}_trial{trial}")
                os.makedirs(trial_folder, exist_ok=True)
                for img_file in self.image_files:
                    decomp_path, num_flips, psnr_val, niqe_val, lpips_val = self.run_one(
                        img_file, ber, trial, temp_base, trial_folder
                    )
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            img_file, ber, trial,
                            num_flips if decomp_path else "",
                            psnr_val if psnr_val is not None else "N/A",
                            niqe_val if niqe_val is not None else "N/A",
                            lpips_val if lpips_val is not None else "N/A",
                        ])
                    if decomp_path and num_flips > 0 and img_file in images_to_sample and samples_saved[img_file] < num_samples_per_ber:
                        base = os.path.splitext(img_file)[0]
                        shutil.copy(decomp_path, os.path.join(sample_ber_dir, f"{base}_trial{trial}.png"))
                        samples_saved[img_file] += 1
                fid_val = self.metrics.fid(
                    self.dataset_path,
                    trial_folder,
                    image_files=self.image_files,
                    dataset_name=dataset_name,
                )
                fid_rows.append((ber, trial, fid_val if fid_val is not None else float("nan")))
        if os.path.isdir(temp_base):
            shutil.rmtree(temp_base, ignore_errors=True)
        fid_path = os.path.join(output_dir, f"fid_{self.runner.name}.csv")
        with open(fid_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ber", "trial", "fid"])
            for ber, trial, fid_val in fid_rows:
                w.writerow([ber, trial, fid_val])


def _output_dir_name(dataset_name: str, target_bpp: float, algorithm: str) -> str:
    return f"{dataset_name}_{algorithm}_bpp{target_bpp}"


# ---------------------------------------------------------------------------
# Baseline: run compression for all images (optional; can assume precomputed)
# ---------------------------------------------------------------------------
def run_baseline(
    project_root: str,
    algorithm: str,
    dataset_path: str,
    image_files: List[str],
    target_bpp: float,
    resolution: int,
    jpeg_quality_csv: Optional[str] = None,
) -> str:
    """Run compression (and decompression) for all images; return compressed_dir.
    DDCM/Turbo: use runner default params. JPEG: quality per image from jpeg_quality_csv (required).
    """
    from experiment_utils import prepare_input_dir

    runner = _runner_factory(project_root, algorithm)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    out_name = _output_dir_name(dataset_name, target_bpp, algorithm)
    compressed_dir = os.path.join(project_root, "results", "noisy_channel", algorithm, "compressed", out_name)
    if not os.path.isdir(compressed_dir):
        staging = os.path.join(compressed_dir, "temp_input")
        os.makedirs(staging, exist_ok=True)
        prepare_input_dir(dataset_path, image_files, staging)
        params = runner.get_baseline_params(jpeg_quality_csv=jpeg_quality_csv)
        runner.run_compression(
            staging, compressed_dir, img_height=resolution, img_width=resolution, params=params
        )
        runner.run_decompression(compressed_dir, img_height=resolution, img_width=resolution)
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return compressed_dir


def main():
    parser = argparse.ArgumentParser(description="Noisy-channel experiment (runners + experiment_utils)")
    parser.add_argument("--algorithm", required=True, choices=["jpeg", "ddcm", "turbo_ddcm", "robust_turbo_ddcm"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--bpp", type=float, default=0.1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ber", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--num_samples_per_ber", type=int, default=3, help="Max samples to save per image per BER (only for images in --sample_images)")
    parser.add_argument("--sample_images", type=str, default=None, help="Image indices to save samples for, e.g. 0-5 or 0,2,5. Default: all.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--compressed_dir", type=str, default=None, help="Precomputed compressed dir; if set, skip baseline")
    parser.add_argument("--jpeg_quality_csv", type=str, default=None, help="For JPEG: CSV with image_file, quality (required for baseline if algorithm=jpeg)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    project_root = os.path.dirname(_script_dir)
    dataset_path = args.dataset or os.path.join(project_root, "dataset")
    image_files = get_images(dataset_path)
    sample_image_files = resolve_subset(image_files, parse_subset(args.sample_images)) if args.sample_images else None
    ber_values = [float(x.strip()) for x in args.ber.split(",")]
    output_dir = args.output_dir or os.path.join(project_root, "results", "noisy_channel", args.algorithm)
    compressed_dir = args.compressed_dir
    if not compressed_dir:
        compressed_dir = run_baseline(
            project_root,
            args.algorithm,
            dataset_path,
            image_files,
            args.bpp,
            args.resolution,
            jpeg_quality_csv=args.jpeg_quality_csv,
        )
    exp = NoisyChannelExperiment(
        project_root,
        args.algorithm,
        compressed_dir,
        dataset_path,
        image_files,
        args.bpp,
        args.resolution,
        jpeg_quality_csv=args.jpeg_quality_csv,
    )
    exp.run(
        ber_values,
        args.num_trials,
        output_dir,
        args.num_samples_per_ber,
        sample_image_files=sample_image_files,
    )
    print("Done. CSV, trial folders, samples, and FID placeholder written to", output_dir)


if __name__ == "__main__":
    main()
