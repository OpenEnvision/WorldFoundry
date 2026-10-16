"""The 1.3B model has independent loading, topology and reference sampling."""

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.lingbot_world_v2 import (
    LINGBOT_WORLD_V2_1P3B_CONFIG,
    LINGBOT_WORLD_V2_CONFIG,
)
from worldfoundry.core.execution.realtime.flow_sampling import reference_flow_schedule, reference_flow_to_clean
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import LingBotWorldV2SmallRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.runtime import (
    REQUIRED_CHECKPOINT_PATHS,
    SHARED_ASSET_PATHS,
    SMALL_CHECKPOINT,
    SMALL_CHECKPOINT_PATHS,
    SMALL_MODEL_ID,
    LingBotWorldV2Runtime,
)


def test_small_model_is_not_an_alias_for_the_14b_architecture(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    runtime = LingBotWorldV2Runtime.from_pretrained(model_id=SMALL_MODEL_ID)
    assert runtime.model_id == SMALL_MODEL_ID and runtime.checkpoint_source == SMALL_CHECKPOINT
    assert runtime.num_heads == 12 and runtime.required_checkpoint_paths == SMALL_CHECKPOINT_PATHS
    assert runtime.defaults["sample_shift"] == 5 and runtime.defaults["nproc_per_node"] == 1
    assert runtime.defaults["t5_fsdp"] is runtime.defaults["dit_fsdp"] is False
    assert (
        LINGBOT_WORLD_V2_1P3B_CONFIG.dim,
        LINGBOT_WORLD_V2_1P3B_CONFIG.ffn_dim,
        LINGBOT_WORLD_V2_1P3B_CONFIG.num_layers,
    ) == (1536, 8960, 30)
    assert LINGBOT_WORLD_V2_CONFIG.dim == 5120 and LINGBOT_WORLD_V2_CONFIG.num_heads == 40
    assert LingBotWorldV2Runtime().required_checkpoint_paths == REQUIRED_CHECKPOINT_PATHS
    assert runtime._validated_options({**runtime.defaults, "nproc_per_node": 3})["nproc_per_node"] == 3
    with pytest.raises(ValueError, match="12 attention heads"):
        runtime._validated_options({**runtime.defaults, "nproc_per_node": 8})


def test_small_checkpoint_and_shared_assets_are_resolved_separately(tmp_path):
    checkpoint, shared = tmp_path / "dit", tmp_path / "wan"
    for root, names in ((checkpoint, SMALL_CHECKPOINT_PATHS), (shared, SHARED_ASSET_PATHS)):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "google/umt5-xxl":
                path.mkdir()
            else:
                path.touch()
    runtime = LingBotWorldV2Runtime.from_pretrained(
        {"model_id": SMALL_MODEL_ID, "checkpoint_source": str(checkpoint), "shared_assets_path": str(shared)}
    )
    assert runtime.runtime_plan()["status"] == "ready"
    assert runtime.inference_options()["shared_assets_path"] == shared.resolve()
    command = runtime.build_command(
        checkpoint_dir=checkpoint,
        image_path=Path("image.png"),
        action_path=Path("poses"),
        prompt="scene",
        output_path=Path("out.mp4"),
        options=runtime.defaults,
    )
    assert command[command.index("--model-id") + 1] == SMALL_MODEL_ID
    assert command[command.index("--shared-assets-path") + 1] == str(shared.resolve())
    assert "--dit-fsdp" not in command and "--t5-fsdp" not in command
    (shared / "Wan2.1_VAE.pth").unlink()
    assert runtime.runtime_plan()["status"] == "checkpoint_missing"
    with pytest.raises(FileNotFoundError, match="shared_assets_path"):
        runtime.inference_options()


def test_small_studio_discovery_and_topology_remain_independent(monkeypatch):
    import json

    from worldfoundry.studio.inference.catalog import find_entry
    from worldfoundry.studio.inference.dispatch import _validate_model_torchrun_nproc, _with_lingbot_world_parallelism
    from worldfoundry.studio.ui.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout

    entry = find_entry(SMALL_MODEL_ID)
    assert entry.model_id == SMALL_MODEL_ID
    assert _validate_model_torchrun_nproc(SMALL_MODEL_ID, 2) == 2
    with pytest.raises(ValueError, match="12 heads"):
        _validate_model_torchrun_nproc(SMALL_MODEL_ID, 8)
    options = _with_lingbot_world_parallelism(SMALL_MODEL_ID, {"call_kwargs_text": "{}"}, world_size=2)
    assert json.loads(options["call_kwargs_text"])["nproc_per_node"] == 2
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert launch_uses_lingbot_torchrun_rollout(StudioLaunchConfig(model_id=SMALL_MODEL_ID))
    with pytest.raises(ValueError, match="four or eight"):
        launch_uses_lingbot_torchrun_rollout(StudioLaunchConfig(model_id="lingbot-world-v2"))


def test_reference_schedule_keeps_fp32_sigmas_and_integer_timesteps():
    timesteps, sigmas = reference_flow_schedule(indices=(0, 250, 500, 750), shift=5.0)
    full = torch.linspace(0.999, 0.0, 1001)[:-1]
    full = 5.0 * full / (1.0 + 4.0 * full)
    torch.testing.assert_close(sigmas, full[[0, 250, 500, 750]], rtol=0, atol=0)
    assert timesteps.dtype == torch.int64 and sigmas.dtype == torch.float32
    assert timesteps.tolist() == (sigmas * 1000).long().tolist()
    noisy = torch.ones(2, dtype=torch.float32)
    flow = torch.ones(2, dtype=torch.bfloat16)
    assert reference_flow_to_clean(noisy, flow, sigmas[0]).dtype == torch.bfloat16


def test_reference_noise_bank_has_channel_first_rng_layout_and_continues():
    session = object.__new__(LingBotWorldV2SmallRealtimeSession)
    session.device = torch.device("cpu")
    session.latent_height, session.latent_width = 2, 3
    session._generator = torch.Generator().manual_seed(42)
    session.autoregressive_index = 0
    expected = torch.randn(16, 80, 2, 3, generator=torch.Generator().manual_seed(42))
    first = session._draw_initial_noise()
    torch.testing.assert_close(first, expected[:, :4], rtol=0, atol=0)
    bank = session._reference_noise_bank
    session.autoregressive_index = 1
    second = session._draw_initial_noise()
    assert session._reference_noise_bank is bank
    torch.testing.assert_close(second, expected[:, 4:8], rtol=0, atol=0)
    session.autoregressive_index = 20
    continuation = session._draw_initial_noise()
    assert session._reference_noise_bank is not bank and continuation.shape == first.shape
