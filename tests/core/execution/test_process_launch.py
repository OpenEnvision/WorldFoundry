# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""In-tree module launch commands used by model runtimes."""

from __future__ import annotations

import subprocess
import sys

import pytest

from worldfoundry.core.execution.process import package_module_command

pytestmark = pytest.mark.unit


def test_plain_module_command_runs_the_package_module() -> None:
    command = package_module_command("worldfoundry.demo.worker", ["--steps", 4], python_executable="/env/bin/python")

    assert command == ["/env/bin/python", "-m", "worldfoundry.demo.worker", "--steps", "4"]


def test_torchrun_command_places_launcher_options_before_the_module() -> None:
    command = package_module_command(
        "worldfoundry.demo.worker",
        ["--seed", "0"],
        python_executable="/env/bin/python",
        nproc_per_node=2,
        launcher_options=("--nnodes=1", "--master_port", "29500"),
    )

    assert command == [
        "/env/bin/python",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--master_port",
        "29500",
        "--nproc_per_node",
        "2",
        "-m",
        "worldfoundry.demo.worker",
        "--seed",
        "0",
    ]


def test_explicit_torchrun_executable_replaces_the_interpreter_prefix() -> None:
    command = package_module_command("worldfoundry.demo.worker", nproc_per_node=1, launcher=["/env/bin/torchrun"])

    assert command == ["/env/bin/torchrun", "--standalone", "--nproc_per_node", "1", "-m", "worldfoundry.demo.worker"]


@pytest.mark.parametrize("module", ["wan.generate", "worldfoundry/demo.py", "worldfoundry..demo", "solarwm"])
def test_modules_outside_the_package_are_rejected(module: str) -> None:
    with pytest.raises(ValueError, match="in-tree worldfoundry"):
        package_module_command(module)


def test_process_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        package_module_command("worldfoundry.demo.worker", nproc_per_node=0)


def test_command_runs_an_in_tree_module_with_the_inherited_environment() -> None:
    """The command adds no import-path entries; the child resolves the package as the caller does."""
    command = package_module_command("worldfoundry.core.execution.process")

    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)

    assert command[0] == sys.executable
    assert completed.returncode == 0, completed.stderr
