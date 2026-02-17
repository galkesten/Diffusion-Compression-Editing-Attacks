from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List

__all__ = [
    "BaseModelRunner",
    "list_png_sorted",
    "list_files_sorted",
]


class BaseModelRunner(ABC):
    name: str

    @abstractmethod
    def get_model_params(self) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def run_compression(
        self,
        input_dir: str,
        output_dir: str,
        *,
        img_height: int,
        img_width: int,
        params: Dict[str, Any],
    ) -> Dict[str, str]:
        """Return errors dict: image_file -> error message. Empty dict if no errors."""
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


def _stem_key(name: str):
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else stem


def list_png_sorted(input_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(input_dir) if f.endswith(".png")], key=_stem_key)


def list_files_sorted(input_dir: str, suffix: str) -> List[str]:
    return sorted([f for f in os.listdir(input_dir) if f.endswith(suffix)], key=_stem_key)

