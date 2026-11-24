# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from torch.distributed.run import parse_args

from worldfoundry.synthesis.action_generation.lingbot_va import runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("torchrun", ["", "/env/torchrun", "/env/python"])
@pytest.mark.parametrize("workers", [1, 2])
def test_server_launch_preserves_rendezvous_and_executable_selection(torchrun: str, workers: int) -> None:
    config = runtime.LingBotVARuntimeConfig(
        checkpoint_dir=Path("/assets/lingbot"),
        config_name="franka-base",
        host="localhost",
        port=8080,
        nproc_per_node=workers,
        master_port=29711,
    )
    command = runtime.build_server_command(python="/env/python", torchrun=torchrun, config=config, save_root="/out")
    prefix = ["/env/torchrun"] if torchrun == "/env/torchrun" else ["/env/python", "-m", "torch.distributed.run"]
    assert command[: len(prefix)] == prefix
    parsed = parse_args(command[len(prefix) :])
    assert parsed.module is True
    assert parsed.standalone is False
    assert str(parsed.master_port) == "29711"
    assert str(parsed.nproc_per_node) == str(workers)
    assert parsed.training_script == "worldfoundry.synthesis.action_generation.wan_va.runtime"
    assert parsed.training_script_args == [
        "--config-name",
        "franka-base",
        "--port",
        "8080",
        "--save_root",
        "/out",
        "--checkpoint-dir",
        "/assets/lingbot",
    ]
