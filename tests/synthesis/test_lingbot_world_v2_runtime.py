# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.lingbot_world_v2.runtime import (
    MODEL_ID,
    RUNNER_MODULE,
    SHARED_ASSET_PATHS,
    SMALL_MODEL_ID,
    LingBotWorldV2Runtime,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model_id,nproc", [(MODEL_ID, 1), (MODEL_ID, 8), (SMALL_MODEL_ID, 1), (SMALL_MODEL_ID, 4)])
def test_lingbot_world_v2_launches_each_topology_as_a_package_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, nproc: int
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    shared = tmp_path / "shared"
    for name in SHARED_ASSET_PATHS:
        path = shared / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    runtime = LingBotWorldV2Runtime(
        tmp_path / "checkpoint",
        model_id=model_id,
        defaults={
            "nproc_per_node": nproc,
            "t5_fsdp": nproc > 1,
            "dit_fsdp": nproc > 1,
            "shared_assets_path": str(shared),
            "attention_backend": "sdpa",
            "vae_streaming_compile": True,
        },
    )

    command = runtime.build_command(
        checkpoint_dir=tmp_path / "checkpoint",
        image_path=tmp_path / "image.png",
        action_path=tmp_path / "actions",
        prompt="camera moves forward",
        output_path=tmp_path / "output.mp4",
        options=runtime.defaults,
    )

    assert command[:8] == [
        str(Path(sys.executable).resolve()),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "-m",
        RUNNER_MODULE,
    ]
    assert command[command.index("--attention-backend") + 1] == "sdpa"
    assert "--vae-streaming-compile" in command
    assert "--no-offload-model" in command
    assert ("--dit-fsdp" in command) is (nproc > 1)
    assert ("--t5-fsdp" in command) is (nproc > 1)
    assert ("--shared-assets-path" in command) is (model_id == SMALL_MODEL_ID)


@pytest.mark.parametrize("pythonpath", [None, "/operator/site"])
def test_lingbot_world_v2_preserves_the_operator_pythonpath(
    monkeypatch: pytest.MonkeyPatch, pythonpath: str | None
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    if pythonpath is not None:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    runtime = LingBotWorldV2Runtime(device="cuda:2,3")

    env = runtime._subprocess_env()

    assert env.get("PYTHONPATH") == pythonpath
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
