# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise serial and multiprocessing maps through the public process helper."""

from __future__ import annotations

import pytest

from worldfoundry.core.execution.parallel_execution import parallel_processes

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_processes_preserves_results_and_order(workers: int) -> None:
    assert parallel_processes(abs, [-3, 1, -2, 0], workers=workers, disable=True) == [3, 1, 2, 0]
