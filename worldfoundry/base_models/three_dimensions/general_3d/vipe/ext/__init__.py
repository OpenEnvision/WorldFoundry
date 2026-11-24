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
import sys

from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.build import (
    NativeExtensionUnavailable,
    load_native_extension,
)


class _MissingVipeExtModule:
    def __init__(self, module_name: str, reason: str) -> None:
        self._module_name = module_name
        self._reason = reason

    def __getattr__(self, op_name: str):
        raise RuntimeError(
            f"VIPE extension module {self._module_name!r} is unavailable; "
            f"operation {op_name!r} cannot run. {self._reason}"
        )


class _MissingVipeExt:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def __getattr__(self, module_name: str) -> _MissingVipeExtModule:
        return _MissingVipeExtModule(module_name, self._reason)


try:
    _C = load_native_extension(
        build_if_missing=os.environ.get("VIPE_EXT_JIT", "0") == "1",
        verbose=os.environ.get("VIPE_EXT_VERBOSE", "0") == "1",
    )
    vipe_ext_not_found = False
except NativeExtensionUnavailable as exc:
    vipe_ext_not_found = True
    _C = _MissingVipeExt(str(exc))

# Reference to submodules
droid_net_ext = _C.droid_net_ext
grounding_dino_ext = _C.grounding_dino_ext
utils_ext = _C.utils_ext
slam_ext = _C.slam_ext
scatter_ext = _C.scatter_ext
lietorch_ext = _C.lietorch_ext
corr_ext = _C.corr_ext

if not vipe_ext_not_found:
    # Some upstream consumers import the pybind submodule directly. The
    # extension still has one canonical binary; this is only an import alias.
    sys.modules.setdefault("lietorch_ext", lietorch_ext)
