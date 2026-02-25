import os
import shutil
import sys
from typing import Any, Dict, Optional

import turbo_ddcm.utils as turbo_utils
from .base import BaseModelRunner, list_png_sorted

# Default params when robust=False (non-robust / standard turbo)
DEFAULT_TURBO_PARAMS: Dict[str, Any] = {
    "T": 30,
    "K": 16384,
    "M": 114,
    "B": 0,
    "C": 1,
    "seed": 88888888,
    "manual_list_ind": True,
}

# Default params when robust=True (robust turbo with B>0)
DEFAULT_ROBUST_TURBO_PARAMS: Dict[str, Any] = {
    "T": 30,
    "K": 16384,
    "M": 106,
    "B": 10,
    "C": 1,
    "seed": 88888888,
    "manual_list_ind": True,
}


class TurboModelRunner(BaseModelRunner):
    name = "turbo_ddcm"

    def __init__(
        self,
        project_root: str,
        *,
        robust: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.project_root = project_root
        self.turbo_root = os.path.join(project_root, "Turbo-DDCM-master")
        mp = model_params if model_params else None
        if mp is not None:
            self.model_params = dict(mp)
        else:
            default = DEFAULT_ROBUST_TURBO_PARAMS if robust else DEFAULT_TURBO_PARAMS
            self.model_params = dict(default)
        print(self.model_params)
        self.name = "robust_turbo_ddcm" if robust else "turbo_ddcm"

    def get_model_params(self) -> Dict[str, Any]:
        return dict(self.model_params)

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, f"{base}{turbo_utils.BIN_SUFFIX}")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"{base}{turbo_utils.BIN_SUFFIX}")
        shutil.copy(compressed_path, dest)
        config_src = os.path.join(os.path.dirname(compressed_path), "compression_config.json")
        if os.path.isfile(config_src):
            shutil.copy(config_src, os.path.join(temp_dir, "compression_config.json"))
        return dest

    def find_decompressed_png(self, temp_dir: str, base: str) -> Optional[str]:
        p = os.path.join(temp_dir, f"{base}.png")
        return p if os.path.isfile(p) else None

    def _resolve_params(self, runtime_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**self.model_params, **runtime_params}
        required = ["T", "K", "M", "B"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing Turbo params: {missing}")
        return params

    def _get_api(self):
        if self.turbo_root not in sys.path:
            sys.path.insert(0, self.turbo_root)
        from turbo_ddcm_api import compress_main_programmatic as turbo_compress_main_programmatic  # type: ignore
        from turbo_ddcm_api import decompress_main_programmatic as turbo_decompress_main_programmatic  # type: ignore

        return turbo_compress_main_programmatic, turbo_decompress_main_programmatic

    def run_compression(
        self,
        input_dir: str,
        output_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        params = self.get_model_params()
        print(f"[TurboModelRunner] Model params: {params}")
        resolved_params = self._resolve_params(params)
        os.makedirs(output_dir, exist_ok=True)
        image_files = list_png_sorted(input_dir)
        errors: Dict[str, str] = {}
        input_abs = os.path.abspath(input_dir)
        output_abs = os.path.abspath(output_dir)
        cwd = os.getcwd()
        try:
            os.chdir(self.turbo_root)
            turbo_compress, _ = self._get_api()
            turbo_compress(
                input_dir=input_abs,
                output_dir=output_abs,
                M=int(resolved_params["M"]),
                gpu=0,
                float32=False,
                seed=int(resolved_params.get("seed", 88888888)),
                T=int(resolved_params["T"]),
                K=int(resolved_params["K"]),
                B=int(resolved_params["B"]),
                weights_dir=None,
                save_reconstructions=False,
                save_runtimes=False,
                manual_list_ind=bool(resolved_params.get("manual_list_ind", True)),
            )
            for image_file in image_files:
                base = os.path.splitext(image_file)[0]
                out = os.path.join(output_abs, f"{base}{turbo_utils.BIN_SUFFIX}")
                if not os.path.exists(out):
                    errors[image_file] = "missing Turbo binary output"
                    continue
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
        os.makedirs(compressed_dir, exist_ok=True)
        errors: Dict[str, str] = {}
        # Compression writes {base}.turbo_ddcm and compression_config.json into compressed_dir; decompress in place
        bin_suffix = turbo_utils.BIN_SUFFIX
        bin_files = [f for f in os.listdir(compressed_dir) if f.endswith(bin_suffix)]
        image_names = [os.path.splitext(f)[0] + ".png" for f in bin_files]

        config_path = os.path.join(compressed_dir, "compression_config.json")
        if not os.path.exists(config_path):
            for image_file in image_names:
                errors[image_file] = "compression_config.json not found in compressed directory"
            return errors

        try:
            _, turbo_decompress = self._get_api()
            turbo_decompress(input_dir=compressed_dir, output_dir=compressed_dir, gpu=0, save_runtimes=False)
        except Exception as exc:
            for image_file in image_names:
                errors[image_file] = str(exc)

        return errors
