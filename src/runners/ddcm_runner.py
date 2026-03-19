"""
DDCM runner: same usage as noisy_channel_experiment.py.
DDCM's FoldersDataset does root.strip(os.sep), so absolute paths make it return
paths like "/2.png" and ddcm_api then writes to "/". We must pass paths relative
to project_root and run with cwd=project_root.
"""
import glob
import os
import shutil
import sys
from typing import Any, Dict, Optional
import numpy as np

from .base import BaseModelRunner, list_png_sorted

DEFAULT_DDCM_PARAMS: Dict[str, Any] = {
    "T": 1000,
    "K": 8192,
    "M": 2,
    "C": 3,
    "model_id": "Manojb/stable-diffusion-2-1-base",
}


class DdcmModelRunner(BaseModelRunner):
    name = "ddcm"

    def __init__(self, project_root: str, model_params: Optional[Dict[str, Any]] = None):
        self.project_root = os.path.abspath(project_root)
        self.ddcm_root = os.path.join(self.project_root, "ddcm-compressed-image-generation-main")
        self.model_params = dict(model_params) if model_params else dict(DEFAULT_DDCM_PARAMS)

    def _out_prefix(self) -> str:
        T = int(self.model_params["T"])
        K = int(self.model_params["K"])
        M = int(self.model_params["M"])
        C = int(self.model_params["C"])
        model_id = str(self.model_params["model_id"])
        t0, t1 = T - 1, 0
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        return f"T={T}_in{t0}-{t1}_K={K}_M={M}_C={C}_model={model_name}"

    def get_params_info(self) -> Dict[str, Any]:
        return {"out_prefix": self._out_prefix()}

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, self._out_prefix(), f"{base}_noise_indices.bin")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        out_prefix = self._out_prefix()
        subdir = os.path.join(temp_dir, out_prefix)
        os.makedirs(subdir, exist_ok=True)
        dest = os.path.join(subdir, f"{base}_noise_indices.bin")
        shutil.copy(compressed_path, dest)
        return dest

    def find_decompressed_png(self, temp_dir: str, base: str) -> Optional[str]:
        candidates = glob.glob(os.path.join(temp_dir, "**", "*_decomp.png"), recursive=True)
        for c in candidates:
            if os.path.splitext(os.path.basename(c))[0].replace("_decomp", "") == base:
                return c
        return None

    def _get_api(self):
        if self.ddcm_root not in sys.path:
            sys.path.insert(0, self.ddcm_root)
        from ddcm_api import main_programmatic as ddcm_main_programmatic  # type: ignore
        return ddcm_main_programmatic

    def get_model_params(self) -> Dict[str, Any]:
        return dict(self.model_params)

    def _resolve_params(self, runtime_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**self.model_params, **runtime_params}
        for k in ["T", "K", "M", "C", "model_id"]:
            if k not in params:
                raise ValueError(f"Missing DDCM param: {k}")
        return params

    def run_compression(
        self,
        input_dir: str,
        output_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        params = self.get_model_params()
        print(f"[DdcmModelRunner] Model params: {params}")
        resolved = self._resolve_params(params)
        input_abs = os.path.abspath(input_dir)
        output_abs = os.path.abspath(output_dir)
        os.makedirs(output_abs, exist_ok=True)
        image_files = list_png_sorted(input_abs)
        errors: Dict[str, str] = {}
        cwd = os.getcwd()
        try:
            os.chdir(self.project_root)
            input_rel = os.path.relpath(input_abs, self.project_root)
            output_rel = os.path.relpath(output_abs, self.project_root)
            ddcm_main = self._get_api()
            ddcm_main(
                mode="compress",
                input_dir=input_rel,
                output_dir=output_rel,
                gpu=0,
                float16=True,
                model_id=str(resolved["model_id"]),
                K=int(resolved["K"]),
                T=int(resolved["T"]),
                M=int(resolved["M"]),
                C=int(resolved["C"]),
                t_range=[int(resolved["T"]) - 1, 0],
            )
            T, K, M, C = int(resolved["T"]), int(resolved["K"]), int(resolved["M"]), int(resolved["C"])
            model_name = str(resolved["model_id"]).split("/")[1] if "/" in str(resolved["model_id"]) else str(resolved["model_id"])
            out_prefix = self._out_prefix()
            for image_file in image_files:
                base = os.path.splitext(image_file)[0]
                bin_file = os.path.join(output_abs, out_prefix, f"{base}_noise_indices.bin")
                if not os.path.exists(bin_file):
                    errors[image_file] = "missing DDCM binary output"
        except Exception as exc:
            for image_file in image_files:
                errors[image_file] = str(exc)
        finally:
            os.chdir(cwd)
        return errors

    def run_decompression(
        self,
        compressed_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        compressed_abs = os.path.abspath(compressed_dir)
        os.makedirs(compressed_abs, exist_ok=True)
        bin_files = glob.glob(os.path.join(compressed_abs, "**", "*_noise_indices.bin"), recursive=True)
        image_names = [os.path.basename(p).split("_noise_indices")[0] + ".png" for p in bin_files]
        errors: Dict[str, str] = {}
        cwd = os.getcwd()
        try:
            np_state_before_ddcm = np.random.get_state()
            os.chdir(self.project_root)
            compressed_rel = os.path.relpath(compressed_abs, self.project_root)
            ddcm_main = self._get_api()
            ddcm_main(
                mode="decompress",
                input_dir=compressed_rel,
                output_dir=compressed_rel,
                gpu=0,
                float16=True,
            )
            np.random.set_state(np_state_before_ddcm)
        except Exception as exc:
            for image_file in image_names:
                errors[image_file] = str(exc)
        finally:
            os.chdir(cwd)
        return errors
