"""
Find compression parameters per algorithm. All algorithm-specific behavior is
defined in ALGO_CONFIG; run_experiment is a single config-driven loop.
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
for _path in (
    os.path.join(_project_root, "ddcm-compressed-image-generation-main"),
    os.path.join(_project_root, "Turbo-DDCM-master"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from PIL import Image
import torch

import turbo_ddcm.utils as turbo_utils
from experiment_utils import (
    append_csv_row,
    avg_or_zero,
    binary_search_best_int,
    collect_actual_bpp_by_glob,
    collect_actual_bpp_from_output,
    get_images,
    parse_subset,
    prepare_input_dir,
    resolve_subset,
)
from runners import BaseModelRunner, BpgModelRunner, DdcmModelRunner, JpegModelRunner, TurboModelRunner


AlgoConfigItem = Dict[str, Any]  # runner_factory, prepare_params, output_dir_name, path_for_compressed, csv_*, write_csv_rows


def find_bpg_quality_for_bpp(img: Image.Image, target_bpp: float, verbose: bool = True) -> Tuple[int, float]:
    w, h = img.size
    pixels = w * h

    low, high = 0, 51
    best_q, best_bpp, best_diff = 0, 0.0, float("inf")
    iteration = 0

    for _ in range(20):
        iteration += 1
        mid = (low + high) // 2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            tmp_png = fp.name
        with tempfile.NamedTemporaryFile(suffix=".bpg", delete=False) as fq:
            tmp_bpg = fq.name

        try:
            img.save(tmp_png, format="PNG")
            subprocess.run(
                ["bpgenc", "-q", str(mid), "-o", tmp_bpg, tmp_png],
                check=True,
                capture_output=True,
            )
            size_bytes = os.path.getsize(tmp_bpg)
            bpp = (size_bytes * 8) / pixels
        finally:
            for p in (tmp_png, tmp_bpg):
                if os.path.exists(p):
                    os.unlink(p)

        diff = abs(bpp - target_bpp)
        if diff < best_diff:
            best_q, best_bpp, best_diff = mid, bpp, diff
        if verbose:
            print(f"    BPG binary search iter {iteration}: q={mid} bpp={bpp:.4f} (target={target_bpp})")

        if bpp > target_bpp:
            # file too big → increase q (worse quality, smaller file)
            low = mid + 1
        else:
            # file too small → decrease q (better quality, larger file)
            high = mid - 1

        if low > high:
            break

    if verbose:
        print(f"    BPG best: q={best_q} bpp={best_bpp:.4f}")
    return best_q, best_bpp


def find_jpeg_quality_for_bpp(img, target_bpp, resize_to=(512, 512)):
    low, high = 1, 100
    best_quality, best_bpp, best_diff = 1, 0, float("inf")
    for _ in range(20):
        mid = (low + high) // 2
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=mid)
        bpp = (len(buffer.getvalue()) * 8) / (resize_to[0] * resize_to[1])
        diff = abs(bpp - target_bpp)
        if diff < best_diff:
            best_quality, best_bpp, best_diff = mid, bpp, diff
        if bpp < target_bpp:
            low = mid + 1
        else:
            high = mid - 1
    return best_quality, best_bpp


def _turbo_bpp_with_manual_list_ind(T, K, M, C, B, img_height, img_width):
    bpp = turbo_utils.turbo_ddcm_bpp(T, K, M, C, B, 0, img_height, img_width)
    bins = torch.logspace(start=torch.log10(torch.tensor(0.01)), end=torch.log10(torch.tensor(0.15)), steps=70)
    nbs = torch.max(torch.tensor(0), torch.min(torch.tensor(T - 2), 70 - torch.bucketize(torch.tensor(bpp), bins) - 1)).item()
    return turbo_utils.turbo_ddcm_bpp(T, K, M, C, B, nbs, img_height, img_width)


def ddcm_bpp(T, K, M, C, img_height, img_width, optimized_Ts):
    return (optimized_Ts - 1) * (M * np.log2(K) + (M - 1) * C) / (img_height * img_width)


def find_turbo_ddcm_M_for_bpp(target_bpp, T, K, C, B, img_height, img_width):
    return binary_search_best_int(target_bpp, 1, min(K, 10000), lambda m: _turbo_bpp_with_manual_list_ind(T, K, m, C, B, img_height, img_width))


def find_ddcm_M_for_bpp(target_bpp, T, K, C, img_height, img_width):
    return binary_search_best_int(target_bpp, 1, 100, lambda m: ddcm_bpp(T, K, m, C, img_height, img_width, T))


# ---------------------------------------------------------------------------
# Per-algorithm behavior (used inside ALGO_CONFIG)
# ---------------------------------------------------------------------------
def _bpg_runner_factory(_root: str, params_info: Optional[Dict[str, Any]]) -> BpgModelRunner:
    params = params_info["params"] if params_info else {}
    quality_by_image = params.get("quality_by_image", {})
    return BpgModelRunner(quality_by_image=quality_by_image)


def _prepare_params_bpg(target_bpp: float, dataset_path: str, images_to_use: List[str], img_height: int, img_width: int, _resize_to: Tuple[int, int], _params: Dict[str, Any]) -> Dict[str, Any]:
    quality_by_image = {}
    n = len(images_to_use)
    for i, img_file in enumerate(images_to_use):
        print(f"  BPG prepare_params: image {i+1}/{n} {img_file} (target_bpp={target_bpp})")
        img = Image.open(os.path.join(dataset_path, img_file)).convert("RGB")
        if img.size != (img_width, img_height):
            raise ValueError(f"image {img_file} size {img.size} != expected {img_width}x{img_height}")
        q, _ = find_bpg_quality_for_bpp(img, target_bpp)
        quality_by_image[img_file] = q
    return {"params": {"quality_by_image": quality_by_image}}


def _write_csv_bpg(csv_path: str, target_bpp: float, params_info: Dict[str, Any], pairs: List[Tuple[str, float]], images_to_use: List[str], errors: Dict[str, str], dataset_name: str) -> None:
    quality_by_image = params_info["params"].get("quality_by_image", {})
    pair_dict = dict(pairs)
    for img_file in images_to_use:
        q = quality_by_image.get(img_file, "")
        actual = errors.get(img_file, "") if errors else pair_dict.get(img_file, 0.0)
        append_csv_row(csv_path, [dataset_name, img_file, target_bpp, q, actual])


def _jpeg_runner_factory(_root: str, params_info: Optional[Dict[str, Any]]) -> JpegModelRunner:
    params = params_info["params"] if params_info else {}
    quality_by_image = params.get("quality_by_image", {})
    return JpegModelRunner(quality_by_image=quality_by_image)


def _prepare_params_jpeg(target_bpp: float, dataset_path: str, images_to_use: List[str], img_height: int, img_width: int, resize_to: Tuple[int, int], _params: Dict[str, Any]) -> Dict[str, Any]:
    quality_by_image = {}
    for img_file in images_to_use:
        img = Image.open(os.path.join(dataset_path, img_file)).convert("RGB")
        if img.size != resize_to:
            img = img.resize(resize_to)
        q, _ = find_jpeg_quality_for_bpp(img, target_bpp, resize_to)
        quality_by_image[img_file] = q
    return {"params": {"quality_by_image": quality_by_image}}


def _prepare_params_ddcm(target_bpp: float, _dataset_path: str, _images_to_use: List[str], img_height: int, img_width: int, _resize_to: Tuple[int, int], params: Dict[str, Any]) -> Dict[str, Any]:
    p = params
    T, K, C = int(p["T"]), int(p["K"]), int(p["C"])
    model_id = str(p["model_id"])
    M = find_ddcm_M_for_bpp(target_bpp, T, K, C, img_height, img_width)
    theoretical = ddcm_bpp(T, K, M, C, img_height, img_width, T)
    t0, t1 = T - 1, 0
    model_name = model_id.split("/")[1] if "/" in model_id else model_id
    out_prefix = f"T={T}_in{t0}-{t1}_K={K}_M={M}_C={C}_model={model_name}"
    return {"params": {"T": T, "K": K, "M": M, "C": C, "model_id": model_id}, "out_prefix": out_prefix, "theoretical": theoretical}


def _prepare_params_turbo(target_bpp: float, _dataset_path: str, _images_to_use: List[str], img_height: int, img_width: int, _resize_to: Tuple[int, int], params: Dict[str, Any]) -> Dict[str, Any]:
    p = params
    T, K, C = int(p["T"]), int(p["K"]), int(p["C"])
    B_param = p["B"]
    if B_param == "M":
        # robust_turbo_ddcm: B equals chosen M — search with B=m for each candidate m
        M = binary_search_best_int(
            target_bpp, 1, min(K, 10000),
            lambda m: _turbo_bpp_with_manual_list_ind(T, K, m, C, m, img_height, img_width),
        )
        B = M
    else:
        B = int(B_param)
        M = find_turbo_ddcm_M_for_bpp(target_bpp, T, K, C, B, img_height, img_width)
    print(f"M: {M}")
    theoretical = _turbo_bpp_with_manual_list_ind(T, K, M, C, B, img_height, img_width)
    return {"params": {**p, "M": M, "B": B}, "theoretical": theoretical}


def _write_csv_jpeg(csv_path: str, target_bpp: float, params_info: Dict[str, Any], pairs: List[Tuple[str, float]], images_to_use: List[str], errors: Dict[str, str], dataset_name: str) -> None:
    quality_by_image = params_info["params"].get("quality_by_image", {})
    pair_dict = dict(pairs)
    for img_file in images_to_use:
        q = quality_by_image.get(img_file, "")
        actual = errors.get(img_file, "") if errors else pair_dict.get(img_file, 0.0)
        append_csv_row(csv_path, [dataset_name, img_file, target_bpp, q, actual])


def _write_csv_combined(csv_path: str, target_bpp: float, params_info: Dict[str, Any], pairs: List[Tuple[str, float]], _images_to_use: List[str], errors: Dict[str, str], dataset_name: str) -> None:
    avg_actual = avg_or_zero([bpp for _, bpp in pairs]) if not errors else 0.0
    p = params_info["params"]
    row = [dataset_name, target_bpp, p["M"], params_info["theoretical"], avg_actual, p["T"], p["K"], p["C"]]
    if "B" in p:
        row.append(p["B"])
    append_csv_row(csv_path, row)


def _make_config(
    *,
    params: Dict[str, Any],
    runner_factory: Callable[..., BaseModelRunner],
    prepare_params: Callable[..., Dict[str, Any]],
    output_dir_name: Callable[[str, float, str], str],
    path_for_compressed: Callable[[str, str, Dict[str, Any]], str],
    csv_key: str,
    csv_filename: str,
    csv_headers: List[str],
    write_csv_rows: Callable[..., None],
    bpp_search_glob: Optional[str] = None,
) -> AlgoConfigItem:
    out = {
        "params": params,
        "runner_factory": runner_factory,
        "prepare_params": prepare_params,
        "output_dir_name": output_dir_name,
        "path_for_compressed": path_for_compressed,
        "csv_key": csv_key,
        "csv_filename": csv_filename,
        "csv_headers": csv_headers,
        "write_csv_rows": write_csv_rows,
    }
    if bpp_search_glob is not None:
        out["bpp_search_glob"] = bpp_search_glob
    return out


# Algorithm config: one place for all algo-specific behavior
ALGO_CONFIG: Dict[str, AlgoConfigItem] = {
    "bpg": _make_config(
        params={},
        runner_factory=_bpg_runner_factory,
        prepare_params=_prepare_params_bpg,
        output_dir_name=lambda dataset_name, target_bpp, _algo: f"{dataset_name}_bpg_bpp{target_bpp}",
        path_for_compressed=lambda out_dir, base, _pi: os.path.join(out_dir, f"{base}_compressed.bpg"),
        csv_key="bpg",
        csv_filename="bpg_compression_params.csv",
        csv_headers=["dataset", "image_file", "target_bpp", "quality", "actual_bpp"],
        write_csv_rows=_write_csv_bpg,
    ),
    "jpeg": _make_config(
        params={},
        runner_factory=_jpeg_runner_factory,
        prepare_params=_prepare_params_jpeg,
        output_dir_name=lambda dataset_name, target_bpp, _algo: f"{dataset_name}_jpeg_bpp{target_bpp}",
        path_for_compressed=lambda out_dir, base, _pi: os.path.join(out_dir, f"{base}_compressed.jpg"),
        csv_key="jpeg",
        csv_filename="jpeg_compression_params.csv",
        csv_headers=["dataset", "image_file", "target_bpp", "quality", "actual_bpp"],
        write_csv_rows=_write_csv_jpeg,
    ),
    "ddcm": _make_config(
        params={"T": 1000, "K": 8192, "C": 3, "t_range": (999, 0), "model_id": "Manojb/stable-diffusion-2-1-base"},
        runner_factory=lambda root, params_info=None: DdcmModelRunner(root, model_params=params_info["params"] if params_info else None),
        prepare_params=_prepare_params_ddcm,
        output_dir_name=lambda dataset_name, target_bpp, _algo: f"{dataset_name}_ddcm_bpp{target_bpp}_compressed",
        path_for_compressed=lambda out_dir, base, pi: os.path.join(out_dir, pi["out_prefix"], f"{base}_noise_indices.bin"),
        csv_key="combined",
        csv_filename="ddcm_bpp.csv",
        csv_headers=["dataset", "target_bpp", "M", "theoretical_bpp", "actual_bpp", "T", "K", "C"],
        write_csv_rows=_write_csv_combined,
    ),
    "turbo_ddcm": _make_config(
        params={"T": 30, "K": 16384, "C": 1, "B": 0, "seed": 88888888, "manual_list_ind": True},
        runner_factory=lambda root, params_info=None: TurboModelRunner(root, robust=False, model_params=params_info["params"] if params_info else None),
        prepare_params=_prepare_params_turbo,
        output_dir_name=lambda dataset_name, target_bpp, algo: f"{dataset_name}_{algo}_bpp{target_bpp}_compressed",
        path_for_compressed=lambda out_dir, base, _pi: os.path.join(out_dir, f"{base}{turbo_utils.BIN_SUFFIX}"),
        csv_key="combined",
        csv_filename="turbo_ddcm_bpp.csv",
        csv_headers=["dataset", "target_bpp", "M", "theoretical_bpp", "actual_bpp", "T", "K", "C", "B"],
        write_csv_rows=_write_csv_combined,
    ),
    "robust_turbo_ddcm": _make_config(
        params={"T": 30, "K": 16384, "C": 1, "B": "M", "seed": 88888888, "manual_list_ind": True},
        runner_factory=lambda root, params_info=None: TurboModelRunner(root, robust=True, model_params=params_info["params"] if params_info else None),
        prepare_params=_prepare_params_turbo,
        output_dir_name=lambda dataset_name, target_bpp, algo: f"{dataset_name}_{algo}_bpp{target_bpp}_compressed",
        path_for_compressed=lambda out_dir, base, _pi: os.path.join(out_dir, f"{base}{turbo_utils.BIN_SUFFIX}"),
        csv_key="combined",
        csv_filename="robust_turbo_ddcm_bpp.csv",
        csv_headers=["dataset", "target_bpp", "M", "theoretical_bpp", "actual_bpp", "T", "K", "C", "B"],
        write_csv_rows=_write_csv_combined,
    ),
}
ALL_ALGOS = list(ALGO_CONFIG.keys())


def run_experiment(
    algorithm: str,
    dataset_path: str,
    all_images: List[str],
    target_bpps: List[float],
    project_root: str,
    csv_path: str,
    img_height: int,
    img_width: int,
    subset: Any = None,
) -> None:
    cfg = ALGO_CONFIG[algorithm]
    prepare_params = cfg["prepare_params"]
    output_dir_name = cfg["output_dir_name"]
    path_for_compressed = cfg["path_for_compressed"]
    write_csv_rows = cfg["write_csv_rows"]

    images_to_use = resolve_subset(all_images, subset)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    results_dir = os.path.join(project_root, "results", "compression_ratio_estimate", algorithm)
    os.makedirs(results_dir, exist_ok=True)
    resize_to = (img_width, img_height)

    for target_bpp in target_bpps:
        fixed_params = cfg.get("params") or {}
        params_info = prepare_params(target_bpp, dataset_path, images_to_use, img_height, img_width, resize_to, fixed_params)
        out_name = output_dir_name(dataset_name, target_bpp, algorithm)
        output_dir = os.path.join(results_dir, out_name)
        os.makedirs(output_dir, exist_ok=True)
        staging_dir = os.path.join(output_dir, "temp_input")
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)

        prepare_input_dir(dataset_path, all_images, staging_dir, subset=subset)
        runner = cfg["runner_factory"](project_root, params_info)
        errors = runner.run_compression(staging_dir, output_dir, img_height=img_height, img_width=img_width)
        decomp_errors = runner.run_decompression(output_dir, img_height=img_height, img_width=img_width)
        errors = {**errors, **decomp_errors}

        if cfg.get("bpp_search_glob"):
            pairs = collect_actual_bpp_by_glob(images_to_use, output_dir, img_height, img_width, cfg["bpp_search_glob"])
        else:
            path_fn = lambda out_dir, base, pi=params_info: path_for_compressed(out_dir, base, pi)
            pairs = collect_actual_bpp_from_output(images_to_use, output_dir, img_height, img_width, path_fn)
        write_csv_rows(csv_path, target_bpp, params_info, pairs, images_to_use, errors, dataset_name)

        shutil.rmtree(staging_dir, ignore_errors=True)


def init_csv_for_algorithm(project_root: str, algorithm: str, dataset_name: str) -> str:
    cfg = ALGO_CONFIG[algorithm]
    results_dir = os.path.join(project_root, "results", "compression_ratio_estimate", algorithm)
    os.makedirs(results_dir, exist_ok=True)
    csv_basename = f"dataset_{dataset_name}_{cfg['csv_filename']}"
    path = os.path.join(results_dir, csv_basename)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(cfg["csv_headers"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find compression parameters for different algorithms")
    parser.add_argument("--algorithms", nargs="+", choices=ALL_ALGOS + ["all"], default=["all"])
    parser.add_argument("--bpp", nargs="+", type=float, choices=[0.1, 0.2, 0.5, 1.0], default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None, help="'0-4' or '0,2,5'")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    project_root = os.path.dirname(_script_dir)
    dataset_path = args.dataset or os.path.join(project_root, "dataset")
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    algorithms_to_run = ALL_ALGOS if "all" in args.algorithms else args.algorithms
    all_images = get_images(dataset_path)
    subset = parse_subset(args.subset) if args.subset else None

    print(f"\n{'=' * 60}\nFinding compression parameters\nAlgorithms: {', '.join(algorithms_to_run)}\nTarget BPPs: {args.bpp}\nDataset: {dataset_path}\nResolution: {args.height}x{args.width}\nSubset: {args.subset or 'all'}\n{'=' * 60}\n")

    for algo in algorithms_to_run:
        csv_path = init_csv_for_algorithm(project_root, algo, dataset_name)
        print(f"Running {algo}...")
        run_experiment(algo, dataset_path, all_images, args.bpp, project_root, csv_path, args.height, args.width, subset)
        print(f"✓ {algo} done")

    print(f"\n{'=' * 60}\nAll complete\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
