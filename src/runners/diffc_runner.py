import os
import shutil
import sys
from typing import Any, Dict, Optional
from types import SimpleNamespace

from .base import BaseModelRunner, list_png_sorted

DEFAULT_DIFFC_PARAMS = {
    'config_dir': 'DiffC/configs/SD-2.1Base-base.yaml',
    'recon_timestep': 20
}


class DiffCRunner(BaseModelRunner):
    name = "DiffC"
    bin_suffix = ".diffc"

    def __init__(
            self,
            project_root: str,
            *,
            model_params: Optional[Dict[str, Any]] = None,
    ):
        self.project_root = project_root
        self.diffc_root = os.path.join(project_root, "DiffC")
        mp = model_params if model_params else None
        if mp is not None:
            self.model_params = dict(mp)
        else:
            default = DEFAULT_DIFFC_PARAMS
            self.model_params = dict(default)
        print(self.model_params)
        self.name = DiffCRunner.name

    def get_model_params(self) -> Dict[str, Any]:
        return dict(self.model_params)

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, f"{base}{DiffCRunner.bin_suffix}")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"{base}{DiffCRunner.bin_suffix}")
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
        required = ["config_dir", "recon_timestep"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing DiffC param: {missing}")
        return params

    def _get_api(self):
        if self.diffc_root not in sys.path:
            sys.path.insert(0, self.diffc_root)
        from compress import main as diffc_compress
        from decompress import main as diffc_decompress

        return diffc_compress, diffc_decompress

    def run_compression(
            self,
            input_dir: str,
            output_dir: str,
            *,
            img_height: int,
            img_width: int,
    ) -> Dict[str, str]:
        params = self.get_model_params()
        print(f"[DiffCRunner] Model params: {params}")
        resolved_params = self._resolve_params(params)
        os.makedirs(output_dir, exist_ok=True)
        image_files = list_png_sorted(input_dir)
        errors: Dict[str, str] = {}
        input_abs = os.path.abspath(input_dir)
        output_abs = os.path.abspath(output_dir)
        config_abs = os.path.abspath(resolved_params['config_dir'])
        cwd = os.getcwd()

        try:
            os.chdir(self.diffc_root)
            diffc_compress, _ = self._get_api()
            args = SimpleNamespace(image_dir=input_abs, output_dir=output_abs, config=config_abs,
                                   recon_timestep=resolved_params['recon_timestep'],
                                   image_path=None)
            diffc_compress(args)
            for image_file in image_files:
                base = os.path.splitext(image_file)[0]
                out = os.path.join(output_abs, f"{base}{DiffCRunner.bin_suffix}")
                if not os.path.exists(out):
                    errors[image_file] = "missing DiffC binary output"
                    continue
        except Exception as exc:
            print(exc)
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
        params = self.get_model_params()
        resolved_params = self._resolve_params(params)
        bin_suffix = DiffCRunner.bin_suffix
        bin_files = [f for f in os.listdir(compressed_dir) if f.endswith(bin_suffix)]
        image_names = [os.path.splitext(f)[0] + ".png" for f in bin_files]

        try:
            config_dir = os.path.abspath(resolved_params['config_dir'])
            _, diffc_decompress = self._get_api()
            args = SimpleNamespace(input_dir=compressed_dir, output_dir=compressed_dir, config=config_dir,
                                   input_path=None)
            diffc_decompress(args)
        except Exception as exc:
            for image_file in image_names:
                errors[image_file] = str(exc)

        return errors
