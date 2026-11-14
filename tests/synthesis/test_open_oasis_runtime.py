# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.open_oasis import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("plan_fps", [None, 12])
@pytest.mark.parametrize("linked_assets", [False, True])
def test_module_command_preserves_asset_paths_and_request_overrides(
    tmp_path: Path, plan_fps: int | None, linked_assets: bool
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    asset_input = assets
    if linked_assets:
        asset_input = tmp_path / "asset-link"
        asset_input.symlink_to(assets, target_is_directory=True)
    output_path = str(tmp_path / "result.mp4")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"fps": plan_fps, "extra": {"num_frames": 49, "ddim_steps": 8}}))
    options = {
        "checkpoint_path": str(asset_input / "oasis.safetensors"),
        "vae_checkpoint_path": str(asset_input / "vae.safetensors"),
        "input_path": str(asset_input / "prompt.png"),
        "action_path": str(asset_input / "actions.pt"),
        "num_frames": 32,
        "fps": 20,
        "ddim_steps": 10,
    }
    original_options = dict(options)
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_path": output_path,
            "plan_path": str(plan),
            "options": options,
        }
    )
    assert command[:3] == ["/env/python", "-m", "worldfoundry.synthesis.visual_generation.open_oasis.generate"]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--oasis-ckpt": str((assets / "oasis.safetensors").resolve()),
        "--vae-ckpt": str((assets / "vae.safetensors").resolve()),
        "--prompt-path": str((assets / "prompt.png").resolve()),
        "--actions-path": str((assets / "actions.pt").resolve()),
        "--output-path": output_path,
        "--num-frames": "49",
        "--fps": str(plan_fps or 20),
        "--ddim-steps": "8",
    }
    assert options == original_options


def test_generate_uses_the_same_packaged_components() -> None:
    from worldfoundry.synthesis.visual_generation.open_oasis import (
        attention,
        dit,
        generate,
        rotary_embedding_torch,
        utils,
        vae,
    )

    assert generate.DiT_models is dit.DiT_models
    assert generate.VAE_models is vae.VAE_models
    assert generate.load_actions is utils.load_actions
    assert generate.load_prompt is utils.load_prompt
    assert (
        dit.RotaryEmbedding
        is attention.RotaryEmbedding
        is vae.RotaryEmbedding
        is rotary_embedding_torch.RotaryEmbedding
    )
    assert vae.PatchEmbed is dit.PatchEmbed
