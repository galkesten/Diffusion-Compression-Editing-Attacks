#!/usr/bin/env python3
"""Merge 6 DDCM run folders (DIV2K_valid_HR_512_run0..run5) into a single DIV2K_valid_HR_512 folder."""

import os
import shutil
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    ddcm_root = project_root / "results" / "noisy_channel" / "ddcm"
    target = ddcm_root / "DIV2K_valid_HR_512"
    runs = [ddcm_root / f"DIV2K_valid_HR_512_run{i}" for i in range(6)]

    for r in runs:
        if not r.is_dir():
            raise FileNotFoundError(f"Run folder not found: {r}")

    target.mkdir(parents=True, exist_ok=True)

    # 1. Compressed: copy from run0, rename inner folder (strip _run0)
    src_compressed = runs[0] / "compressed" / "DIV2K_valid_HR_512_run0_ddcm_bpp0.1"
    dst_compressed = target / "compressed" / "DIV2K_valid_HR_512_ddcm_bpp0.1"
    if src_compressed.exists():
        dst_compressed.parent.mkdir(parents=True, exist_ok=True)
        if dst_compressed.exists():
            shutil.rmtree(dst_compressed)
        shutil.copytree(src_compressed, dst_compressed)
        print(f"Copied compressed: {dst_compressed}")
    else:
        print(f"Warning: compressed source not found: {src_compressed}")

    # 2. Samples: baseline once (from run0) + all ber folders from all runs
    samples_dir = target / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Baseline from run0
    src_baseline = runs[0] / "samples" / "baseline"
    if src_baseline.exists():
        dst_baseline = samples_dir / "baseline"
        if dst_baseline.exists():
            shutil.rmtree(dst_baseline)
        shutil.copytree(src_baseline, dst_baseline)
        print(f"Copied baseline: {dst_baseline}")

    # BER folders from each run
    ber_folders = [
        ("ber1e-06", 0),
        ("ber1e-05", 1),
        ("ber0.0001", 2),
        ("ber0.001", 3),
        ("ber0.01", 4),
        ("ber0.1", 5),
    ]
    for ber_name, run_idx in ber_folders:
        src_ber = runs[run_idx] / "samples" / ber_name
        if src_ber.exists():
            dst_ber = samples_dir / ber_name
            if dst_ber.exists():
                shutil.rmtree(dst_ber)
            shutil.copytree(src_ber, dst_ber)
            print(f"Copied {ber_name} from run{run_idx}")

    # 3. Merge fid_ddcm.csv
    fid_dfs = []
    for i, r in enumerate(runs):
        p = r / "fid_ddcm.csv"
        if p.exists():
            fid_dfs.append(pd.read_csv(p))
    if fid_dfs:
        fid_merged = pd.concat(fid_dfs, ignore_index=True)
        fid_out = target / "fid_ddcm.csv"
        fid_merged.to_csv(fid_out, index=False)
        print(f"Written {fid_out} ({len(fid_merged)} rows)")
    else:
        print("Warning: no fid_ddcm.csv files found")

    # 4. Merge noisy_channel_ddcm.csv (ber=0 only once, rest from all runs)
    nc_dfs = [pd.read_csv(r / "noisy_channel_ddcm.csv") for r in runs]
    baseline_rows = nc_dfs[0][nc_dfs[0]["ber"] == 0]
    non_baseline = pd.concat([df[df["ber"] != 0] for df in nc_dfs], ignore_index=True)
    nc_merged = pd.concat([baseline_rows, non_baseline], ignore_index=True)
    nc_merged = nc_merged.sort_values(["ber", "trial", "image_file"])
    nc_out = target / "noisy_channel_ddcm.csv"
    nc_merged.to_csv(nc_out, index=False)
    print(f"Written {nc_out} ({len(nc_merged)} rows)")

    print("Done. Merged folder:", target)


if __name__ == "__main__":
    main()
