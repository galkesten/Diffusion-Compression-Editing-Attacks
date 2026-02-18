from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional

__all__ = [
    "BaseModelRunner",
    "list_png_sorted",
    "list_files_sorted",
]


class BaseModelRunner(ABC):
    name: str

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Return compression params (set in __init__)."""
        raise NotImplementedError

    @abstractmethod
    def run_compression(
        self,
        input_dir: str,
        output_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        """Return errors dict: image_file -> error message. Empty dict if no errors. Uses params set in __init__."""
        raise NotImplementedError

    @abstractmethod
    def run_decompression(
        self,
        compressed_dir: str,
        *,
        img_height: int,
        img_width: int,
    ) -> Dict[str, str]:
        """Decompress in place: read compressed data from compressed_dir and write decompressed output to the same dir. Return errors dict: image_file -> error message."""
        raise NotImplementedError

    # Noisy-channel experiment: path/layout logic lives in the runner (no algorithm switch in caller).

    def get_params_info(self) -> Dict[str, Any]:
        """Extra info for path/layout (e.g. DDCM out_prefix). Default: empty."""
        return {}

    @abstractmethod
    def path_for_compressed(self, compressed_dir: str, base: str) -> Optional[str]:
        """Path to the compressed file for image base name. None if not applicable."""
        raise NotImplementedError

    @abstractmethod
    def prepare_temp_for_noisy_channel(self, compressed_path: str, temp_dir: str, base: str) -> str:
        """Copy compressed file(s) into temp_dir in the layout expected by run_decompression. Returns path to the file to apply bit flips (same layout)."""
        raise NotImplementedError

    @abstractmethod
    def find_decompressed_png(self, temp_dir: str, base: str) -> Optional[str]:
        """Path to the decompressed PNG in temp_dir after run_decompression. None if not found."""
        raise NotImplementedError


def _stem_key(name: str):
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else stem


def list_png_sorted(input_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(input_dir) if f.endswith(".png")], key=_stem_key)


def list_files_sorted(input_dir: str, suffix: str) -> List[str]:
    return sorted([f for f in os.listdir(input_dir) if f.endswith(suffix)], key=_stem_key)

