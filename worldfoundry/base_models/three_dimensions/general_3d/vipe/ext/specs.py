# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import sys
from pathlib import Path

UPSTREAM_REVISION = "157494a2aca56c9f5adbd36977d892e88401b4e2"


def _csrc_path() -> Path:
    return Path(__file__).resolve().parent.parent / "csrc"


def get_sources() -> list[str]:
    csrc_path = _csrc_path()
    sources = sorted(p for p in csrc_path.rglob("*") if p.suffix in {".cpp", ".cu"})
    if not sources:
        raise RuntimeError(
            f"No ViPE native sources were found under {csrc_path}. "
            "Reinstall WorldFoundry from an sdist/wheel that includes the ViPE csrc tree."
        )
    return [str(path) for path in sources]


def _environment_prefixes() -> list[Path]:
    """Return likely environment/toolkit roots in stable priority order."""
    values = [
        os.environ.get("CONDA_PREFIX"),
        os.environ.get("WORLDFOUNDRY_UNIFIED_ENV_PREFIX"),
        os.environ.get("WORLDFOUNDRY_CONDA_ENV_PREFIX"),
        str(Path(sys.executable).resolve().parent.parent),
    ]
    prefixes: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        if path not in prefixes:
            prefixes.append(path)
    return prefixes


def cuda_home_candidates() -> list[Path]:
    """Return CUDA toolkit roots without assuming the active shell is conda-activated."""
    values = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    candidates = [Path(value).expanduser() for value in values if value]
    candidates.extend(_environment_prefixes())
    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.extend([Path("/usr/local/cuda"), Path("/usr/local/cuda-12")])

    unique: list[Path] = []
    for path in candidates:
        if path not in unique and (path / "bin" / "nvcc").is_file():
            unique.append(path)
    return unique


def resolve_cuda_home() -> Path:
    """Resolve a CUDA toolkit with ``nvcc`` or raise an actionable error."""
    candidates = cuda_home_candidates()
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "ViPE native kernels require a CUDA toolkit with nvcc. Activate the WorldFoundry "
        "environment or set CUDA_HOME to a toolkit matching torch.version.cuda."
    )


def eigen_include_candidates() -> list[Path]:
    """Return candidate include roots whose direct child is ``Eigen``."""
    candidates: list[Path] = []
    for variable in ("EIGEN3_INCLUDE_DIR", "EIGEN_INCLUDE_DIR"):
        if value := os.environ.get(variable):
            candidates.append(Path(value).expanduser())
    candidates.append(_csrc_path() / "include" / "eigen3")
    for prefix in _environment_prefixes():
        candidates.extend([prefix / "include" / "eigen3", prefix / "include"])
    candidates.extend([Path("/usr/include/eigen3"), Path("/usr/local/include/eigen3")])
    return candidates


def resolve_eigen_include() -> Path:
    """Resolve Eigen headers or raise an actionable environment error."""
    for include_path in eigen_include_candidates():
        if (include_path / "Eigen" / "Core").is_file():
            return include_path.resolve()
    raise RuntimeError(
        "Eigen 3 headers are required to build ViPE. Install `eigen=3.4` in the active "
        "conda environment or set EIGEN3_INCLUDE_DIR to the directory containing Eigen/Core."
    )


def _eigen_include_flags() -> list[str]:
    if os.environ.get("USE_SYSTEM_EIGEN", "0") == "1":
        return []
    return ["-isystem", str(resolve_eigen_include())]


def _additional_include_flags() -> list[str]:
    flags = _eigen_include_flags()
    seen_paths = {flags[index + 1] for index in range(0, len(flags), 2)}
    for conda_prefix in _environment_prefixes():
        include_paths = [
            conda_prefix / "include",
            conda_prefix / "nvvm" / "include",
            *sorted((conda_prefix / "targets").glob("*/include")),
        ]
        for include_path in include_paths:
            include_path_str = str(include_path)
            if include_path.exists() and include_path_str not in seen_paths:
                flags += ["-isystem", include_path_str]
                seen_paths.add(include_path_str)

    # CUDA-enabled PyTorch wheels install library development headers in
    # namespace-package directories such as ``nvidia/cusparse/include`` rather
    # than below CUDA_HOME.  PyTorch's own CUDA headers include these files, so
    # expose the matching wheel headers to nvcc when they are present.  Ignore
    # aggregate versioned trees (for example ``nvidia/cu13``) because an
    # environment may also contain tooling for a CUDA major other than the one
    # used by the active torch build.
    torch_cuda_header_components = ("cublas", "cusolver", "cusparse")
    for python_path in map(Path, sys.path):
        nvidia_root = python_path / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for component in torch_cuda_header_components:
            include_path = nvidia_root / component / "include"
            if not include_path.is_dir():
                continue
            include_path_str = str(include_path)
            if include_path_str not in seen_paths:
                flags += ["-isystem", include_path_str]
                seen_paths.add(include_path_str)
    return flags


def get_cpp_flags() -> list[str]:
    return ["-O3", "-DWITH_CUDA"] + _additional_include_flags()  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.


def get_cuda_flags() -> list[str]:
    return ["-O3", "-DWITH_CUDA", "--use_fast_math"] + _additional_include_flags()  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
