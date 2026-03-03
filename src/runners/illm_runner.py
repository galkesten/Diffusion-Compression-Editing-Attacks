import os
import shutil
import sys
from typing import Any, Dict, Optional
from types import SimpleNamespace

from .base import BaseModelRunner, list_png_sorted

DEFAULT_ILLM_PARAMS = {
    'model': 'msillm_quality_2'
}


class ILLMRunner(BaseModelRunner):
    name = "ILLM"
    bin_suffix = ".bin"

    def __init__(
            self,
            project_root: str,
            *,
            model_params: Optional[Dict[str, Any]] = None,
    ):
        self.project_root = project_root
        self.illm_root = os.path.join(project_root, "ILLM")
        mp = model_params if model_params else None
        if mp is not None:
            self.model_params = dict(mp)
        else:
            default = DEFAULT_ILLM_PARAMS
            self.model_params = dict(default)
        print(self.model_params)
        self.name = ILLMRunner.name

    def get_model_params(self) -> Dict[str, Any]:
        return dict(self.model_params)

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, f"{base}{ILLMRunner.bin_suffix}")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"{base}{ILLMRunner.bin_suffix}")
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
        required = ["model"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing ILLM param: {missing}")
        return params

    def _get_api(self):
        if self.illm_root not in sys.path:
            sys.path.insert(0, self.illm_root)
        from eval_folder_example import compress as compress
        from eval_folder_example import decompress as decompress

        return compress, decompress

    def run_compression(
            self,
            input_dir: str,
            output_dir: str,
            *,
            img_height: int,
            img_width: int,
    ) -> Dict[str, str]:
        params = self.get_model_params()
        print(f"[ILLMRunner] Model params: {params}")
        params = self.get_model_params()
        resolved_params = self._resolve_params(params)
        os.makedirs(output_dir, exist_ok=True)
        image_files = list_png_sorted(input_dir)
        errors: Dict[str, str] = {}
        input_abs = os.path.abspath(input_dir)
        output_abs = os.path.abspath(output_dir)
        cwd = os.getcwd()

        try:
            os.chdir(self.illm_root)
            illm_compress, _ = self._get_api()
            args = SimpleNamespace(path=input_abs, output_path=output_abs, model=resolved_params['model'], com_decom='compression')
            illm_compress(args)
            for image_file in image_files:
                base = os.path.splitext(image_file)[0]
                out = os.path.join(output_abs, f"{base}{ILLMRunner.bin_suffix}")
                if not os.path.exists(out):
                    errors[image_file] = "missing ILLM binary output"
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
        params = self.get_model_params()
        resolved_params = self._resolve_params(params)
        bin_suffix = ILLMRunner.bin_suffix
        bin_files = [f for f in os.listdir(compressed_dir) if f.endswith(bin_suffix)]
        image_names = [os.path.splitext(f)[0] + ".png" for f in bin_files]

        try:
            _, illm_decompress = self._get_api()
            args = SimpleNamespace(path=compressed_dir, output_path=compressed_dir, model=resolved_params['model'], com_decom='decompression')
            illm_decompress(args)
        except Exception as exc:
            for image_file in image_names:
                errors[image_file] = str(exc)

        return errors
