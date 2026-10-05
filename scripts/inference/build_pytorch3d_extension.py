#!/usr/bin/env python3
"""Rebuild the installed PyTorch3D extension for the active Torch ABI."""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import shutil
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-arch-list", default="9.0")
    parser.add_argument(
        "--with-pulsar",
        action="store_true",
        help="Build the complete extension, including the renderer's Pulsar bindings.",
    )
    return parser.parse_args()


def _source_package(configured: Path | None) -> Path:
    if configured is not None:
        package = configured.expanduser().resolve()
    else:
        import pytorch3d

        package = Path(pytorch3d.__file__).resolve().parent
    if not (package / "csrc" / "ext.cpp").is_file():
        raise FileNotFoundError(f"PyTorch3D C++ sources are missing under {package}")
    return package


def _write_extension_without_pulsar(package: Path, build_dir: Path) -> Path:
    """Create an extension entrypoint when wheel installs omit Pulsar headers."""

    source = (package / "csrc" / "ext.cpp").read_text(encoding="utf-8")
    source = source.replace(
        '#include "pulsar/global.h" // Include before <torch/extension.h>.\n', ""
    )
    source = source.replace('#include "pulsar/pytorch/renderer.h"\n', "")
    source = source.replace('#include "pulsar/pytorch/tensor_util.h"\n', "")
    marker = "  // Pulsar.\n"
    if marker not in source:
        raise RuntimeError("Could not locate the Pulsar registration block in ext.cpp")
    source = source.split(marker, 1)[0] + "}\n"
    destination = build_dir / "ext_without_pulsar.cpp"
    destination.write_text(source, encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    package = _source_package(args.source_package)
    build_dir = args.build_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.with_pulsar:
        cpp_sources = sorted((package / "csrc").rglob("*.cpp"))
        cuda_sources = sorted((package / "csrc").rglob("*.cu"))
    else:
        cpp_sources = sorted(
            path
            for path in (package / "csrc").rglob("*.cpp")
            if path.name != "ext.cpp" and "pulsar" not in path.parts
        )
        cuda_sources = sorted(
            path
            for path in (package / "csrc").rglob("*.cu")
            if "pulsar" not in path.parts
        )
        cpp_sources.append(_write_extension_without_pulsar(package, build_dir))
    if not cpp_sources or not cuda_sources:
        raise RuntimeError(
            f"Expected both C++ and CUDA PyTorch3D sources, found "
            f"{len(cpp_sources)} C++ and {len(cuda_sources)} CUDA files"
        )

    os.environ["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch_list
    module = load(
        name="_C",
        sources=[str(path) for path in (*cpp_sources, *cuda_sources)],
        extra_include_paths=[str(package / "csrc")],
        extra_cflags=["-O2", "-std=c++17", "-DWITH_CUDA"],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++17",
            "-DWITH_CUDA",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )

    built = Path(module.__file__).resolve()
    # ``torch.utils.cpp_extension.load`` normally emits ``_C.so``.  Installing
    # that name alongside an older ABI-tagged extension does not replace the
    # latter: Python prefers ``_C.cpython-...so`` and silently keeps loading it.
    # Always install using this interpreter's canonical extension suffix.
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    destination = output_dir / f"_C{suffix}"
    shutil.copy2(built, destination)
    print(destination)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")


if __name__ == "__main__":
    main()
