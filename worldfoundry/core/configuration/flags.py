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

"""Fixed feature switches for the inference-only runtime.

The flag values are snapshots taken when this module is first imported;
changing ``os.environ`` afterwards has no effect (tests that need different
flags must reload the module). Each flag accepts the framework-standard
``WORLDFOUNDRY_*`` variable and falls back to the legacy ``COSMOS_*``
spelling that vendored Cosmos code documents.
"""

import os
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────
# Import-time snapshots — WORLDFOUNDRY_* first, then legacy COSMOS_*
# ──────────────────────────────────────────────────────────────────────────


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean flag from WORLDFOUNDRY_<name>, falling back to COSMOS_<name>."""

    value = os.environ.get(f"WORLDFOUNDRY_{name}")
    if value is None:
        value = os.environ.get(f"COSMOS_{name}")
    return default if value is None else value.lower() in {"1", "true", "yes", "y", "on"}


INTERNAL = _env_bool("INTERNAL")
VALIDATION = _env_bool("VALIDATION")
VERBOSE = _env_bool("VERBOSE")
EXPERIMENTAL_CHECKPOINTS = _env_bool("EXPERIMENTAL_CHECKPOINTS")


@dataclass(frozen=True)
class Flags:
    """Frozen bundle of the import-time snapshots for structured config trees."""

    internal: bool = INTERNAL
    validation: bool = VALIDATION
    verbose: bool = VERBOSE
    experimental_checkpoints: bool = EXPERIMENTAL_CHECKPOINTS


FLAGS = Flags()


__all__ = [
    "EXPERIMENTAL_CHECKPOINTS",
    "FLAGS",
    "INTERNAL",
    "VALIDATION",
    "VERBOSE",
    "Flags",
]
