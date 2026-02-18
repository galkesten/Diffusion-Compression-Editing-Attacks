import argparse
import csv
import os
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

_UNAVAILABLE = object()  # Sentinel: metric loaded but failed

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Local packages (same as noisy_channel_experiment.py): DDCM and Turbo-DDCM must be on path before importing runners
_project_root = os.path.dirname(_script_dir)
_ddcm_path = os.path.join(_project_root, "ddcm-compressed-image-generation-main")
_turbo_ddcm_path = os.path.join(_project_root, "Turbo-DDCM-master")
if _ddcm_path not in sys.path:
    sys.path.insert(0, _ddcm_path)
if _turbo_ddcm_path not in sys.path:
    sys.path.insert(0, _turbo_ddcm_path)

from experiment_utils import calculate_actual_bpp, flip_bits, get_images, load_jpeg_quality_csv, parse_subset, resolve_subset
from fid_utils import compute_patch_fid

from runners import (
    BaseModelRunner,
    DdcmModelRunner,
    JpegModelRunner,
    TurboModelRunner,
)


def _csv_val(x: Any) -> str:
    """Format value for CSV: None, NaN -> empty string."""
    if x is None:
        return ""
    if isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf")):
        return ""
    return str(x)


@dataclass
class RunOneResult:
    """Result of run_one: noising + decompress for one image."""

    decomp_path: Optional[str]
    num_flips: int
    psnr: Optional[float]
    niqe: Optional[float]
    lpips: Optional[float]
    bpp: Optional[float]
    error: Optional[str]


def _runner_factory(project_root: str, algorithm: str, *, quality_by_image: Optional[Dict[str, int]] = None) -> BaseModelRunner:
    if algorithm == "jpeg":
        return JpegModelRunner(quality_by_image=quality_by_image or {})
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
def _img_to_tensor(img: Image.Image, device: Any = None) -> Any:
    import torch
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def _metrics_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class MetricsComputer:
    """Computes per-image metrics (PSNR, NIQE, LPIPS) and patch FID via fid_utils."""

    def __init__(self, resolution: int = 512, device: Optional[str] = None):
        self.resolution = resolution
        self.device = device or _metrics_device()
        self._niqe_metric = None
        self._lpips_metric = None

    def _get_pyiqa_metric(self, name: str):
        """Create a pyiqa metric. Load pkg_resources first so pyiqa's imports see it."""
        try:
            import pkg_resources  # noqa: F401 - pyiqa depends on it; load first
        except ImportError:
            return None, "pkg_resources not found (install setuptools)"
        try:
            from pyiqa import create_metric
            return create_metric(name, device=self.device), None
        except Exception as e:
            return None, str(e)

    def _get_niqe(self):
        if self._niqe_metric is None:
            m, err = self._get_pyiqa_metric("niqe")
            if err:
                print("[Metrics] NIQE metric unavailable: %s" % err)
                self._niqe_metric = _UNAVAILABLE
            else:
                self._niqe_metric = m
        return self._niqe_metric

    def _get_lpips(self):
        if self._lpips_metric is None:
            m, err = self._get_pyiqa_metric("lpips")
            if err:
                print("[Metrics] LPIPS metric unavailable: %s" % err)
                self._lpips_metric = _UNAVAILABLE
            else:
                self._lpips_metric = m
        return self._lpips_metric

    def psnr(self, ref: Image.Image, decomp: Image.Image) -> float:
        if ref.size != decomp.size:
            ref = ref.resize(decomp.size)
        a = np.array(ref)
        b = np.array(decomp)
        return float(psnr(a, b, data_range=255))

    def niqe(self, decomp: Image.Image) -> float:
        """NIQE (lower is better). Returns NaN if metric unavailable or fails (e.g. degenerate/corrupted image)."""
        m = self._get_niqe()
        if m is _UNAVAILABLE:
            return float("nan")
        try:
            t = _img_to_tensor(decomp, self.device)
            out = cast(Any, m)(t)
            if hasattr(out, "item"):
                return float(out.item())
            return float(out)
        except Exception as e:
            print("NIQE calculation error: %s" % e)
            return float("nan")

    def lpips(self, ref: Image.Image, decomp: Image.Image) -> float:
        """LPIPS (lower is better). Returns NaN if metric unavailable or fails (e.g. corrupted image)."""
        m = self._get_lpips()
        if m is _UNAVAILABLE:
            return float("nan")
        try:
            if ref.size != decomp.size:
                ref = ref.resize(decomp.size)
            t_ref = _img_to_tensor(ref, self.device)
            t_decomp = _img_to_tensor(decomp, self.device)
            out = cast(Any, m)(t_ref, t_decomp)
            if hasattr(out, "item"):
                return float(out.item())
            return float(out)
        except Exception as e:
            print("LPIPS calculation error: %s" % e)
            return float("nan")

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
            import torch
            fid_device = torch.device(self.device)
            return float(
                compute_patch_fid(
                    ref_dir,
                    decomp_dir,
                    files,
                    self.resolution,
                    self.resolution,
                    device=fid_device,
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

    def _bpp_for_image(self, img_file: str) -> Optional[float]:
        """Compute BPP for one image (compressed file size). Same across all trials."""
        base = os.path.splitext(img_file)[0]
        compressed_path = self.runner.path_for_compressed(self.compressed_dir, base)
        if compressed_path and os.path.isfile(compressed_path):
            return calculate_actual_bpp(compressed_path, self.resolution, self.resolution)
        return None

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
        bpp_val: Optional[float],
    ) -> RunOneResult:
        """Run noising + decompress for one image."""
        base = os.path.splitext(img_file)[0]
        compressed_path = self.runner.path_for_compressed(self.compressed_dir, base)
        if not compressed_path:
            return RunOneResult(None, 0, None, None, None, bpp_val, None)
        temp_dir = os.path.join(temp_base, f"ber{ber}_trial{trial}_{base}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            _, num_flips = self._prepare_temp_and_flip(compressed_path, temp_dir, base, ber, trial)
            errs = self._decompress_temp(temp_dir)
            if img_file in errs:
                return RunOneResult(None, num_flips, None, None, None, bpp_val, errs[img_file])
            decomp_path = self._find_decompressed_png(temp_dir, base)
            if not decomp_path or not os.path.isfile(decomp_path):
                return RunOneResult(None, num_flips, None, None, None, bpp_val, errs.get(img_file, "no PNG produced"))
            dest_in_trial = os.path.join(trial_folder, img_file)
            os.makedirs(os.path.dirname(dest_in_trial) or ".", exist_ok=True)
            shutil.move(decomp_path, dest_in_trial)
            ref = Image.open(os.path.join(self.dataset_path, img_file)).convert("RGB").resize((self.resolution, self.resolution))
            decomp = Image.open(dest_in_trial).convert("RGB")
            psnr_val = self.metrics.psnr(ref, decomp)
            niqe_val = self.metrics.niqe(decomp)
            lpips_val = self.metrics.lpips(ref, decomp)
            return RunOneResult(dest_in_trial, num_flips, psnr_val, niqe_val, lpips_val, bpp_val, None)
        finally:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _write_baseline_rows(
        self,
        writer: Any,
        bpp_by_image: Dict[str, Optional[float]],
        output_dir: Optional[str] = None,
        sample_image_files: Optional[List[str]] = None,
    ) -> None:
        """Write one CSV row per image for baseline (BER=0, no bit flips). If output_dir and sample_image_files are set, save one baseline decompressed image per selected image to samples/baseline/."""
        baseline_rows: List[List[Any]] = []
        for img_file in self.image_files:
            base = os.path.splitext(img_file)[0]
            bpp_val = bpp_by_image.get(img_file)
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
                psnr_val = niqe_val = lpips_val = None
            baseline_rows.append([
                img_file, 0, 0, 0,
                bpp_val,
                psnr_val, niqe_val, lpips_val,
                "",
            ])
        for row in baseline_rows:
            writer.writerow([
                row[0], row[1], row[2], row[3],
                _csv_val(row[4]), _csv_val(row[5]), _csv_val(row[6]), _csv_val(row[7]),
                row[8],
            ])

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
        images_to_sample = sample_image_files if sample_image_files is not None else self.image_files
        bpp_by_image = {img_file: self._bpp_for_image(img_file) for img_file in self.image_files}

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_file", "ber", "trial", "num_bit_flips", "bpp", "psnr", "niqe", "lpips", "error"])
            self._write_baseline_rows(
                w, bpp_by_image, output_dir=output_dir, sample_image_files=images_to_sample
            )
            # samples/ has one subfolder per BER (ber0.1, ber0.01, ...) plus baseline/
            fid_rows: List[Tuple[float, int, float]] = []
            for ber in ber_values:
                print("  BER %s" % ber)
                samples_saved = {img_file: 0 for img_file in images_to_sample}
                sample_ber_dir = os.path.join(output_dir, "samples", f"ber{ber}")
                os.makedirs(sample_ber_dir, exist_ok=True)
                for trial in range(num_trials):
                    print("    trial %d/%d" % (trial + 1, num_trials))
                    trial_folder = os.path.join(temp_base, f"ber{ber}_trial{trial}")
                    os.makedirs(trial_folder, exist_ok=True)
                    first_decomp_error: Optional[str] = None
                    trial_rows: List[List[Any]] = []
                    for img_file in self.image_files:
                        r = self.run_one(
                            img_file, ber, trial, temp_base, trial_folder,
                            bpp_val=bpp_by_image.get(img_file),
                        )
                        if r.error and first_decomp_error is None:
                            first_decomp_error = r.error
                        trial_rows.append([
                            img_file, ber, trial,
                            r.num_flips,
                            r.bpp, r.psnr, r.niqe, r.lpips,
                            r.error or "",
                        ])
                        if r.decomp_path and r.num_flips > 0 and img_file in images_to_sample and samples_saved[img_file] < num_samples_per_ber:
                            base = os.path.splitext(img_file)[0]
                            shutil.copy(r.decomp_path, os.path.join(sample_ber_dir, f"{base}_trial{trial}.png"))
                            samples_saved[img_file] += 1
                    for row in trial_rows:
                        w.writerow([
                            row[0], row[1], row[2], row[3],
                            _csv_val(row[4]), _csv_val(row[5]), _csv_val(row[6]), _csv_val(row[7]),
                            row[8],
                        ])
                    n_decompressed = sum(1 for f in os.listdir(trial_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))) if os.path.isdir(trial_folder) else 0
                    if n_decompressed == 0:
                        msg = "[Noisy channel] Decompression failed for all %d images at BER=%s trial=%s. Bit-flipped streams may be undecodable by this codec." % (len(self.image_files), ber, trial)
                        if first_decomp_error:
                            msg += " First error: %s" % first_decomp_error
                        print(msg)
                    fid_val = self.metrics.fid(
                        os.path.abspath(self.dataset_path),
                        os.path.abspath(trial_folder),
                        image_files=self.image_files,
                        dataset_name=dataset_name,
                    )
                    fid_rows.append((ber, trial, fid_val if fid_val is not None else float("nan")))
                    if os.path.isdir(trial_folder):
                        shutil.rmtree(trial_folder, ignore_errors=True)
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

    quality_by_image = load_jpeg_quality_csv(jpeg_quality_csv, target_bpp) if algorithm == "jpeg" and jpeg_quality_csv and os.path.isfile(jpeg_quality_csv) else None
    runner = _runner_factory(project_root, algorithm, quality_by_image=quality_by_image)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    out_name = _output_dir_name(dataset_name, target_bpp, algorithm)
    compressed_dir = os.path.join(project_root, "results", "noisy_channel", algorithm, "compressed", out_name)
    has_content = os.path.isdir(compressed_dir) and any(
        f for f in os.listdir(compressed_dir) if not f.startswith(".")
    )
    if not has_content:
        if algorithm == "jpeg" and not quality_by_image:
            raise ValueError("JPEG requires jpeg_quality_csv (CSV with columns image_file, quality)")
        staging = os.path.join(compressed_dir, "temp_input")
        os.makedirs(staging, exist_ok=True)
        prepare_input_dir(dataset_path, image_files, staging)
        comp_errors = runner.run_compression(
            staging, compressed_dir, img_height=resolution, img_width=resolution
        )
        if comp_errors:
            raise RuntimeError("Baseline compression had errors: %s" % comp_errors)
        decomp_errors = runner.run_decompression(compressed_dir, img_height=resolution, img_width=resolution)
        if decomp_errors:
            raise RuntimeError("Baseline decompression had errors: %s" % decomp_errors)
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
        print("Baseline done. compressed_dir=%s" % compressed_dir)
    return compressed_dir


def main():
    parser = argparse.ArgumentParser(description="Noisy-channel experiment (runners + experiment_utils)")
    parser.add_argument("--algorithm", required=True, choices=["jpeg", "ddcm", "turbo_ddcm", "robust_turbo_ddcm"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--bpp", type=float, default=0.1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--ber", type=float, nargs="*", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], help="BER values (list, e.g. --ber 1e-6 1e-5 1e-4)")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--num_samples_per_ber", type=int, default=5, help="Max samples to save per image per BER (only for images in --sample_images)")
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
    dataset_path = os.path.abspath(args.dataset or os.path.join(project_root, "dataset"))
    image_files = get_images(dataset_path)
    sample_image_files = resolve_subset(image_files, parse_subset(args.sample_images)) if args.sample_images else None
    ber_values = list(args.ber)
    output_dir = args.output_dir or os.path.join(project_root, "results", "noisy_channel", args.algorithm)
    compressed_dir = args.compressed_dir

    import torch
    gpu_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "N/A"
    print("GPU: %s (%s)" % ("cuda:0" if gpu_avail else "cpu", gpu_name))
    print("noisy_channel_runner params: algorithm=%s dataset=%s bpp=%s resolution=%s num_images=%d" % (args.algorithm, dataset_path, args.bpp, args.resolution, len(image_files)))
    print("  ber=%s num_trials=%d num_samples_per_ber=%d output_dir=%s" % (ber_values, args.num_trials, args.num_samples_per_ber, output_dir))
    print("  compressed_dir=%s" % (compressed_dir or "(will run baseline)"))

    if not compressed_dir:
        default_compressed_dir = os.path.join(
            project_root, "results", "noisy_channel", args.algorithm, "compressed",
            _output_dir_name(os.path.basename(os.path.normpath(dataset_path)), args.bpp, args.algorithm),
        )
        need_baseline = not os.path.isdir(default_compressed_dir) or not any(
            f for f in os.listdir(default_compressed_dir) if not f.startswith(".")
        )
        if need_baseline:
            print("Running baseline compression...")
            compressed_dir = run_baseline(
                project_root,
                args.algorithm,
                dataset_path,
                image_files,
                args.bpp,
                args.resolution,
                jpeg_quality_csv=args.jpeg_quality_csv,
            )
        else:
            compressed_dir = default_compressed_dir
            print("Using existing compressed_dir: %s" % compressed_dir)
    exp = NoisyChannelExperiment(
        project_root,
        args.algorithm,
        compressed_dir,
        dataset_path,
        image_files,
        args.bpp,
        args.resolution,
    )
    print("Starting noisy-channel run: %d BER values x %d trials x %d images" % (len(ber_values), args.num_trials, len(image_files)))
    exp.run(
        ber_values,
        args.num_trials,
        output_dir,
        args.num_samples_per_ber,
        sample_image_files=sample_image_files,
    )
    print("Done. CSV, samples, and FID written to %s" % output_dir)


if __name__ == "__main__":
    main()
