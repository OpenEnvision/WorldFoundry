"""Build and load the pinned ViPE CUDA extension.

The WorldFoundry wheel ships the official native sources but does not compile
them at package-install time. This keeps the main package installable on CPU
hosts while making the GPU requirement explicit at ViPE preflight/inference.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.specs import (
    UPSTREAM_REVISION,
    get_cpp_flags,
    get_cuda_flags,
    get_sources,
    resolve_cuda_home,
    resolve_eigen_include,
)
from worldfoundry.core.io.paths import cache_root_path


class NativeExtensionUnavailable(RuntimeError):
    """Raised when the real ViPE CUDA extension cannot be loaded or built."""


@dataclass(frozen=True, slots=True)
class NativeExtensionStatus:
    """Serializable ViPE native-runtime preflight result."""

    ready: bool
    module_file: str | None
    build_dir: str
    source_count: int
    upstream_revision: str
    torch_version: str
    torch_cuda: str | None
    cuda_home: str | None
    nvcc_version: str | None
    eigen_include: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _torch_runtime():
    import torch

    return torch


def _nvcc_release(cuda_home: Path) -> str:
    output = subprocess.check_output(
        [str(cuda_home / "bin" / "nvcc"), "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        raise NativeExtensionUnavailable(f"Could not parse nvcc version from: {output.strip()}")
    return match.group(1)


def _validate_toolkit(torch_cuda: str | None, cuda_home: Path) -> str:
    if torch_cuda is None:
        raise NativeExtensionUnavailable("The active PyTorch build has no CUDA runtime; ViPE requires CUDA PyTorch.")
    nvcc_version = _nvcc_release(cuda_home)
    if nvcc_version != ".".join(torch_cuda.split(".")[:2]):
        raise NativeExtensionUnavailable(
            f"ViPE CUDA toolkit mismatch: torch was built for CUDA {torch_cuda}, but "
            f"{cuda_home / 'bin' / 'nvcc'} reports CUDA {nvcc_version}. Set CUDA_HOME "
            "to a matching toolkit before building."
        )
    return nvcc_version


def _source_digest() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent.parent
    for source in get_sources():
        path = Path(source)
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _architecture_key(torch) -> str:
    explicit = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        capabilities = sorted(
            {
                ".".join(str(value) for value in torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            }
        )
        return ";".join(capabilities)
    return "unset"


def _native_build_root() -> Path:
    root_override = os.environ.get("WORLDFOUNDRY_VIPE_BUILD_DIR")
    return Path(root_override).expanduser() if root_override else cache_root_path() / "native" / "vipe"


def native_build_dir() -> Path:
    """Return the ABI- and source-keyed cache directory for ``vipe_ext``."""
    torch = _torch_runtime()
    payload = {
        "arch": _architecture_key(torch),
        "machine": platform.machine(),
        "python": sys.implementation.cache_tag,
        "source": _source_digest(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "upstream": UPSTREAM_REVISION,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return _native_build_root() / key


def _candidate_shared_objects(build_dir: Path) -> list[Path]:
    return sorted(build_dir.glob("vipe_ext*.so"), key=lambda path: path.stat().st_mtime, reverse=True)


def _validate_module(module: ModuleType) -> ModuleType:
    required = (
        "corr_ext",
        "droid_net_ext",
        "grounding_dino_ext",
        "lietorch_ext",
        "scatter_ext",
        "slam_ext",
        "utils_ext",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise NativeExtensionUnavailable(
            f"Loaded {getattr(module, '__file__', module)!s}, but it lacks ViPE native submodules: {missing}. "
            "Remove the stale build directory and rebuild from the pinned WorldFoundry sources."
        )
    return module


def _load_cached(build_dir: Path) -> ModuleType | None:
    for library in _candidate_shared_objects(build_dir):
        loader = importlib.machinery.ExtensionFileLoader("vipe_ext", str(library))
        spec = importlib.util.spec_from_loader("vipe_ext", loader)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["vipe_ext"] = module
        try:
            spec.loader.exec_module(module)
            return _validate_module(module)
        except Exception:
            sys.modules.pop("vipe_ext", None)
    return None


def _configure_build_environment(cuda_home: Path) -> None:
    os.environ["CUDA_HOME"] = str(cuda_home)
    bin_candidates = [cuda_home / "bin"]
    for variable in ("CONDA_PREFIX", "WORLDFOUNDRY_UNIFIED_ENV_PREFIX", "WORLDFOUNDRY_CONDA_ENV_PREFIX"):
        if value := os.environ.get(variable):
            bin_candidates.append(Path(value).expanduser() / "bin")
    current_path = os.environ.get("PATH", "")
    prefixes = [str(path) for path in bin_candidates if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join([*prefixes, current_path])

    import torch.utils.cpp_extension as cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_home)
    if not cpp_extension.is_ninja_available():
        raise NativeExtensionUnavailable(
            "The `ninja` executable is required to build ViPE. Install ninja in the active conda environment."
        )


def load_native_extension(*, build_if_missing: bool = False, verbose: bool = False) -> ModuleType:
    """Load a prebuilt/cached extension, optionally compiling the pinned sources."""
    existing = sys.modules.get("vipe_ext")
    if isinstance(existing, ModuleType):
        return _validate_module(existing)

    try:
        return _validate_module(importlib.import_module("vipe_ext"))
    except (ImportError, NativeExtensionUnavailable):
        pass

    build_dir = native_build_dir()
    if cached := _load_cached(build_dir):
        return cached
    if not build_if_missing:
        raise NativeExtensionUnavailable(
            "vipe_ext is not built for this Python/Torch/CUDA/source combination. Run "
            "`python -m worldfoundry.base_models.three_dimensions.general_3d.vipe build-native` "
            "or set VIPE_EXT_JIT=1 for an explicit build-on-import."
        )

    torch = _torch_runtime()
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise NativeExtensionUnavailable(
            "Building ViPE without a visible GPU requires TORCH_CUDA_ARCH_LIST (for example `8.0` for A100)."
        )
    cuda_home = resolve_cuda_home()
    _validate_toolkit(torch.version.cuda, cuda_home)
    resolve_eigen_include()
    _configure_build_environment(cuda_home)
    build_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.cpp_extension import load

    try:
        module = load(
            name="vipe_ext",
            sources=get_sources(),
            extra_cflags=get_cpp_flags(),
            extra_cuda_cflags=get_cuda_flags(),
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=verbose,
        )
    except Exception as exc:
        raise NativeExtensionUnavailable(
            f"Failed to build ViPE native extension in {build_dir}: {type(exc).__name__}: {exc}"
        ) from exc
    return _validate_module(module)


def native_extension_status(*, build_if_missing: bool = False, verbose: bool = False) -> NativeExtensionStatus:
    """Inspect the complete native toolchain and extension import path."""
    torch = _torch_runtime()
    build_dir: Path | None = None
    source_count = 0
    cuda_home: Path | None = None
    eigen_include: Path | None = None
    nvcc_version: str | None = None
    reason: str | None = None
    module: ModuleType | None = None
    try:
        sources = get_sources()
        source_count = len(sources)
        if source_count == 0:
            raise NativeExtensionUnavailable("No ViPE native sources are available in this installation.")
        build_dir = native_build_dir()
        cuda_home = resolve_cuda_home()
        nvcc_version = _validate_toolkit(torch.version.cuda, cuda_home)
        eigen_include = resolve_eigen_include()
        if shutil.which("c++") is None:
            raise NativeExtensionUnavailable("A C++ compiler is required to build ViPE (set CXX if needed).")
        module = load_native_extension(build_if_missing=build_if_missing, verbose=verbose)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"

    return NativeExtensionStatus(
        ready=module is not None,
        module_file=str(getattr(module, "__file__", "")) if module is not None else None,
        build_dir=str(build_dir if build_dir is not None else _native_build_root()),
        source_count=source_count,
        upstream_revision=UPSTREAM_REVISION,
        torch_version=str(torch.__version__),
        torch_cuda=torch.version.cuda,
        cuda_home=str(cuda_home) if cuda_home is not None else None,
        nvcc_version=nvcc_version,
        eigen_include=str(eigen_include) if eigen_include is not None else None,
        reason=reason,
    )


__all__ = [
    "NativeExtensionStatus",
    "NativeExtensionUnavailable",
    "load_native_extension",
    "native_build_dir",
    "native_extension_status",
]
