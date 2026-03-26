#!/usr/bin/env python3
"""Merge 6 DiffC run folders (DIV2K_valid_HR_512_run0..run5) into a single DIV2K_valid_HR_512 folder."""

import shutil
import csv
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    diffc_root = project_root / "results2" / "noisy_channel" / "diffc"
    target = diffc_root / "DIV2K_valid_HR_512"
    runs = [diffc_root / f"DIV2K_valid_HR_512_run{i}" for i in range(6)]

    for r in runs:
        if not r.is_dir():
            raise FileNotFoundError(f"Run folder not found: {r}")

    target.mkdir(parents=True, exist_ok=True)

    # 1. Compressed: copy from run0, rename inner folder (strip _run0)
    src_compressed = runs[0] / "compressed" / "DIV2K_valid_HR_512_run0_diffc_bpp0.1"
    dst_compressed = target / "compressed" / "DIV2K_valid_HR_512_diffc_bpp0.1"
    if src_compressed.exists():
        dst_compressed.parent.mkdir(parents=True, exist_ok=True)
        if dst_compressed.exists():
            shutil.rmtree(dst_compressed)
        shutil.copytree(src_compressed, dst_compressed)
        print(f"Copied compressed: {dst_compressed}")
    else:
        print(f"Warning: compressed source not found: {src_compressed}")

    # 2. Merge fid_DiffC.csv
    fid_rows = []
    fid_header = None
    for r in runs:
        p = r / "fid_DiffC.csv"
        if not p.exists():
            continue
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if fid_header is None:
                fid_header = reader.fieldnames
            fid_rows.extend(reader)

    if fid_header is not None:
        fid_out = target / "fid_DiffC.csv"
        with fid_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fid_header)
            writer.writeheader()
            writer.writerows(fid_rows)
        print(f"Written {fid_out} ({len(fid_rows)} rows)")
    else:
        print("Warning: no fid_DiffC.csv files found")

    # 3. Merge noisy_channel_DiffC.csv (ber=0 only once, rest from all runs)
    nc_header = None
    baseline_rows = []
    non_baseline_rows = []

    for i, r in enumerate(runs):
        p = r / "noisy_channel_DiffC.csv"
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if nc_header is None:
                nc_header = reader.fieldnames
            for row in reader:
                ber_val = float(row["ber"])
                if ber_val == 0.0:
                    if i == 0:
                        baseline_rows.append(row)
                else:
                    non_baseline_rows.append(row)

    if nc_header is None:
        raise FileNotFoundError("No noisy_channel_DiffC.csv files found in run folders")

    nc_rows = baseline_rows + non_baseline_rows
    nc_rows.sort(key=lambda row: (float(row["ber"]), int(row["trial"]), row["image_file"]))

    nc_out = target / "noisy_channel_DiffC.csv"
    with nc_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=nc_header)
        writer.writeheader()
        writer.writerows(nc_rows)
    print(f"Written {nc_out} ({len(nc_rows)} rows)")

    print("Done. Merged folder:", target)


if __name__ == "__main__":
    main()
