# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Default location for explicitly staged WorldMemBench metric assets."""

import os
from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path

DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get("WORLDFOUNDRY_WORLDMEMBENCH_WEIGHTS_DIR", str(checkpoint_root_path("worldmembench")))
).expanduser()
