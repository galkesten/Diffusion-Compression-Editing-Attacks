import multiprocessing
import os
import shutil
import sys
from typing import Any, Dict, Optional
from types import SimpleNamespace
import concurrent.futures

from .base import BaseModelRunner, list_png_sorted

DEFAULT_SC_PARAMS: Dict[str, Any] = {
    "elic_path":"../ckpts/elic_official.pth",
    "codec_path":"../ckpts/stablecodec_ft2.pkl"
}


class StableCodecRunner(BaseModelRunner):
    name = "StableCodec"
    BIN_SUFFIX = ""

    def __init__(
            self,
            project_root: str,
            *,
            model_params: Optional[Dict[str, Any]] = None,
    ):
        self.project_root = project_root
        self.stable_codec_root = os.path.join(project_root, "StableCodec", "src")
        assert model_params is None
        self.model_params = DEFAULT_SC_PARAMS
        self.name = StableCodecRunner.name

        # args = SimpleNamespace(seed=None,
        #                        elic_path=DEFAULT_SC_PARAMS["elic_path"], codec_path=DEFAULT_SC_PARAMS["codec_path"],
        #                        lora_rank_unet=32, lora_rank_vae=16, vae_decoder_tiled_size=160,
        #                        vae_encoder_tiled_size=1024,latent_tiled_size=96,
        #                        latent_tiled_overlap=32,
        #                        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        #                        enable_xformers_memory_efficient_attention=False, set_grads_to_none=False, world_size=1, local_rank=-1,
        #                        dist_url='env://', color_fix=False)
        #
        # os.chdir(self.stable_codec_root)
        # if self.stable_codec_root not in sys.path:
        #     sys.path.insert(0, self.stable_codec_root)
        # from StableCodec import StableCodec
        # from huggingface_hub import snapshot_download
        # sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
        # net = StableCodec(sd_path=sd_path, args=args)
        # net.cuda().eval()
        # net.codec.update(force=True)
        #
        # if args.enable_xformers_memory_efficient_attention:
        #     from diffusers.utils.import_utils import is_xformers_available
        #     if is_xformers_available():
        #         net.unet.enable_xformers_memory_efficient_attention()
        #     else:
        #         raise ValueError("xformers is not available, please install it by running `pip install xformers`")
        # self.net = net

    def get_model_params(self) -> Dict[str, Any]:
        return dict(self.model_params)

    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        p = os.path.join(compressed_dir, f"{base}{StableCodecRunner.BIN_SUFFIX}")
        return p if os.path.isfile(p) else None

    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"{base}{StableCodecRunner.BIN_SUFFIX}")
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
        assert runtime_params is None
        return params

    def _get_api(self):
        if self.stable_codec_root not in sys.path:
            sys.path.insert(0, self.stable_codec_root)
        print(self.stable_codec_root, '\n')

        from compress import compress
        from compress import decompress

        return compress, decompress

    def run_compression(
            self,
            input_dir: str,
            output_dir: str,
            *,
            img_height: int,
            img_width: int,
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        image_files = list_png_sorted(input_dir)
        errors: Dict[str, str] = {}
        input_abs = os.path.abspath(input_dir)
        output_abs = os.path.abspath(output_dir)
        cwd = os.getcwd()

        os.chdir(self.stable_codec_root)
        compress, _ = self._get_api()

        args = SimpleNamespace(img_path=input_abs, bin_path=output_abs, sd_path=None, seed=None,
                               elic_path=DEFAULT_SC_PARAMS["elic_path"], codec_path=DEFAULT_SC_PARAMS["codec_path"],
                               lora_rank_unet=32, lora_rank_vae=16, vae_decoder_tiled_size=160,
                               vae_encoder_tiled_size=1024,latent_tiled_size=96,
                               latent_tiled_overlap=32,
                               pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
                               enable_xformers_memory_efficient_attention=False, set_grads_to_none=False, world_size=1, local_rank=-1,
                               dist_url='env://', color_fix=False)

        try:
            compress(args)
            for image_file in image_files:
                base = os.path.splitext(image_file)[0]
                out = os.path.join(output_abs, f"{base}{StableCodecRunner.BIN_SUFFIX}")
                if not os.path.exists(out):
                    errors[image_file] = "missing SC binary output"
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
        bin_suffix = StableCodecRunner.BIN_SUFFIX
        bin_files = [f for f in os.listdir(compressed_dir) if f.endswith(bin_suffix)]
        image_names = [os.path.splitext(f)[0] + ".png" for f in bin_files]

        import subprocess
        import sys

        try:
            # os.chdir(self.stable_codec_root)
            # _, decompress = self._get_api()
            # args = SimpleNamespace(input_path=compressed_dir, rec_path=compressed_dir, sd_path=None, seed=None, elic_path=DEFAULT_SC_PARAMS["elic_path"], codec_path=DEFAULT_SC_PARAMS["codec_path"], lora_rank_unet=32, lora_rank_vae=16, vae_decoder_tiled_size=160, vae_encoder_tiled_size=1024,latent_tiled_size=96, latent_tiled_overlap=32, pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.", enable_xformers_memory_efficient_attention=False, set_grads_to_none=False, world_size=1, local_rank=-1, dist_url='env://', color_fix=False, ori_h=512, ori_w=512, net=self.net)
            # decompress(args)

            cmd = [
                sys.executable,
                "-c",
                f"""
from types import SimpleNamespace
import os
os.chdir('{self.project_root}')
from src.runners.stable_codec_runner import StableCodecRunner


obj = StableCodecRunner('{self.project_root}', model_params=None)

args = SimpleNamespace(input_path=r'{compressed_dir}', rec_path=r'{compressed_dir}', sd_path=None, seed=None, elic_path=r'{DEFAULT_SC_PARAMS["elic_path"]}', codec_path=r'{DEFAULT_SC_PARAMS["codec_path"]}', lora_rank_unet=32, lora_rank_vae=16, vae_decoder_tiled_size=160, vae_encoder_tiled_size=1024,latent_tiled_size=96, latent_tiled_overlap=32, pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.", enable_xformers_memory_efficient_attention=False, set_grads_to_none=False, world_size=1, local_rank=-1, dist_url='env://', color_fix=False, ori_h=512, ori_w=512)

os.chdir('{self.stable_codec_root}')
_, decompress = obj._get_api()
decompress(args)

"""
            ]

            timeout_seconds = len(bin_files) * 30
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or f"Subprocess failed (code {result.returncode})")


        # try:
        #     timeout_seconds = 100000
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #         future = executor.submit(decompress, args)
        #         future.result(timeout=timeout_seconds)
        except Exception as exc:
            for image_file in image_names:
                errors[image_file] = str(exc)

        return errors
