import csv
import glob
import os
import shutil
import subprocess


def append_csv_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def init_csv(project_root, method_subdir, filename, header):
    results_dir = os.path.join(project_root, "results", "compression_ratio_estimate", method_subdir)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)
    return path


def binary_search_best_int(target_value, low, high, eval_fn, iterations=30):
    best_value, best_diff = low, float("inf")
    for _ in range(iterations):
        mid = (low + high) // 2
        try:
            value = eval_fn(mid)
            diff = abs(value - target_value)
            if diff < best_diff:
                best_diff = diff
                best_value = mid
            if value < target_value:
                low = mid + 1
            else:
                high = mid - 1
        except Exception:
            high = mid - 1
    return best_value


def get_images(dataset_path, num_samples=None):
    images = sorted([f for f in os.listdir(dataset_path) if f.endswith(".png")], key=lambda x: int(x.split(".")[0]))
    if num_samples is None:
        return images
    return images[:num_samples]


def calculate_actual_bpp(binary_file_path, img_height=512, img_width=512):
    return (os.path.getsize(binary_file_path) * 8) / (img_height * img_width)


def collect_actual_bpp_from_output(
    image_files,
    output_dir,
    img_height,
    img_width,
    path_for_base_fn,
):
    """Collect actual BPP for each image whose compressed file exists.
    path_for_base_fn(output_dir, base) -> path to compressed file (base = stem of image filename).
    Returns list of (image_file, bpp)."""
    out = []
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        path = path_for_base_fn(output_dir, base)
        if path and os.path.exists(path):
            out.append((img_file, calculate_actual_bpp(path, img_height, img_width)))
    return out


def collect_actual_bpp_by_glob(
    image_files,
    output_dir,
    img_height,
    img_width,
    pattern,
):
    """Find compressed files under output_dir by glob pattern and match by base name.
    pattern: e.g. '**/*_noise_indices.bin'. Each match's stem (before _noise_indices.bin or similar)
    is matched to image file stems. Returns list of (image_file, bpp)."""
    stem_to_img = {os.path.splitext(f)[0]: f for f in image_files}
    out = []
    for path in glob.glob(os.path.join(output_dir, pattern), recursive=True):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        if "_noise_indices.bin" in name:
            base = name.split("_noise_indices.bin")[0]
        else:
            base = os.path.splitext(name)[0]
        if base in stem_to_img:
            out.append((stem_to_img[base], calculate_actual_bpp(path, img_height, img_width)))
    return out


def copy_images(dataset_path, image_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_file in image_files:
        shutil.copy(os.path.join(dataset_path, img_file), os.path.join(output_dir, img_file))


def resolve_subset(image_files, subset):
    """Return the list of image files to use. subset: None (all), (start, end) inclusive, or list[int] indices."""
    if subset is None:
        return list(image_files)
    if isinstance(subset, (list, tuple)) and len(subset) == 2 and all(isinstance(x, int) for x in subset):
        start, end = subset[0], subset[1]
        if start > end:
            start, end = end, start
        indices = range(start, end + 1)
    elif isinstance(subset, list) and all(isinstance(x, int) for x in subset):
        indices = subset
    else:
        raise ValueError("subset must be None, (start, end), or list[int] indices")
    out = []
    for i in indices:
        if 0 <= i < len(image_files):
            out.append(image_files[i])
    return out


def parse_subset(s):
    """Parse a subset string: '0-4' -> (0, 4) inclusive range, '0,2,5' -> [0, 2, 5] indices. Returns None if s is falsy."""
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return (int(a.strip()), int(b.strip()))
    return [int(p.strip()) for p in s.split(",")]


def prepare_input_dir(dataset_path, image_files, input_dir, subset=None):
    """Create input_dir and copy the given image files from dataset_path into it (unified staging for runners).
    subset: None (copy all), (start, end) inclusive range of indices, or list[int] indices."""
    os.makedirs(input_dir, exist_ok=True)
    to_copy = resolve_subset(image_files, subset)
    copy_images(dataset_path, to_copy, input_dir)


def avg_or_zero(values):
    return sum(values) / len(values) if values else 0


def run_cmd(cmd, cwd):
    subprocess.run(cmd, check=True, cwd=cwd)
