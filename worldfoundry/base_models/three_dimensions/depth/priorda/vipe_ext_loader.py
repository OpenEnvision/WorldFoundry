from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import torch.utils.cpp_extension as cpp_extension


def _additional_include_flags() -> list[str]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_include_path = Path(conda_prefix) / "include"
        if conda_include_path.exists():
            return ["-isystem", str(conda_include_path)]
    return []


def _configure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
        return
    candidates: list[Path] = []
    for root in (Path(sys.prefix), Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"), reverse=True)):
        candidates.append(root)
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidates.insert(0, Path(nvcc_path).resolve().parents[1])
    for root in candidates:
        if (root / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(root)
            os.environ["PATH"] = f"{root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            cpp_extension.CUDA_HOME = str(root)
            return


def load_vipe_ext() -> ModuleType:
    try:
        import vipe_ext as compiled_ext

        return compiled_ext
    except ImportError:
        pass

    csrc_dir = Path(__file__).resolve().parent / "csrc"
    env_bin = Path(sys.prefix) / "bin"
    if env_bin.exists():
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{env_bin}{os.pathsep}{current_path}"
    _configure_cuda_home()
    sources = [
        csrc_dir / "bind.cpp",
        csrc_dir / "utils_ext" / "utils_bind.cpp",
        csrc_dir / "utils_ext" / "knn.cu",
        csrc_dir / "utils_ext" / "cuda_kdtree.cu",
    ]
    flags = _additional_include_flags()
    return cpp_extension.load(
        name="worldfoundry_priorda_vipe_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3", "-DWITH_CUDA", *flags],
        extra_cuda_cflags=["-O3", "-DWITH_CUDA", "--use_fast_math", *flags],
        verbose=os.environ.get("WORLDFOUNDRY_VERBOSE_EXT_BUILD", "0") == "1",
    )
