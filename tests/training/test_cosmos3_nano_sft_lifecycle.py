from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")

from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos3 import Cosmos3JointDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.cosmos3.model import Cosmos3OmniTransformer
from worldfoundry.training.data.video_bucketing import VideoLatentGeometry
from worldfoundry.training.data.video_cache import VideoCachedDataset, VideoCacheProvenance, VideoCacheStore
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.ema import PowerEMA, power_ema_beta, power_ema_exponent
from worldfoundry.training.engine.cosmos.sft import (
    build_cosmos_fsdp2_session,
    build_cosmos_single_device_session,
)
from worldfoundry.training.models.cosmos import Cosmos3TrainAdapter
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning import load_full_model


def _tiny_model() -> Cosmos3OmniTransformer:
    return Cosmos3OmniTransformer(
        hidden_size=12,
        head_dim=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=24,
        num_hidden_layers=1,
        latent_channel=2,
        latent_patch_size=1,
        patch_latent_dim=2,
        vocab_size=32,
        rope_scaling={"mrope_section": [1, 1, 1]},
    )


def _recipe(*, backend: str = "single") -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": "tiny-cosmos3-nano", "output_dir": "unused"},
            "model": {
                "recipe": "cosmos3-nano",
                "checkpoint": "unused",
                "options": {"use_system_prompt": False},
            },
            "tuning": {"mode": "partial", "preset": "cosmos3-nano-vision-sft"},
            "data": {
                "manifest": "unused",
                "cache": "unused",
                "max_latent_tokens_per_microbatch": 2,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"num_workers": 0, "pin_memory": False},
            },
            "objective": {
                "type": "flow-matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "waver",
                "conditioning_dropout": 0.1,
                "options": {
                    "flow_shift": 3.0,
                    "conditioning_config": {0: 0.7, 1: 0.2, 2: 0.1},
                },
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "epsilon": 1.0e-6,
                "max_grad_norm": 0.1,
                "gradient_accumulation_steps": 2,
            },
            "runtime": {
                "param_dtype": "float32",
                "reduce_dtype": "float32",
                "activation_checkpoint": "full",
                "compile": False,
            },
            "distributed": {"backend": backend, "dp_shard": "auto"},
            "checkpoint": {"save_every_steps": 1, "async": False},
        }
    )


def _cache(root: Path) -> VideoCachedDataset:
    geometry = VideoLatentGeometry(16, 16, 4, "first-frame")
    provenance = VideoCacheProvenance(
        media_uri="tiny.mp4",
        prompt="a moving square",
        model_recipe="cosmos3-nano",
        codec={"name": "tiny"},
        conditioner={"name": "cosmos3-tokenizer"},
        tokenizer={"name": "cosmos3-tokenizer"},
        conditioning_inputs={"use_system_prompt": False},
        safety_audit={"safe": True},
        frame_sampling={"mode": "tiny"},
        spatial_transform={"mode": "tiny"},
        latent_normalization={"mode": "identity"},
        task="t2v",
        conditioning_layout="cosmos3-token-sequence",
        aspect_bin="1:1",
        source_num_frames=5,
        source_height=16,
        source_width=16,
        source_fps=24.0,
        target_num_frames=5,
        target_height=16,
        target_width=16,
        target_fps=24.0,
        latent_geometry=geometry,
    )
    store = VideoCacheStore(root)
    entry = store.write_sample(
        sample_id="tiny",
        provenance=provenance,
        clean_latents=torch.randn(2, 2, 1, 1, generator=torch.Generator().manual_seed(7)),
        conditioning={
            "input_ids": torch.tensor([1, 2, 3]),
            "empty_input_ids": torch.tensor([4, 5]),
        },
        conditioning_layouts={
            "input_ids": "tokens",
            "empty_input_ids": "tokens",
        },
        latent_loss_mask=torch.ones(1, 2, 1, 1),
        valid_latent_mask=torch.ones(1, 2, 1, 1, dtype=torch.bool),
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def _session(cache: VideoCachedDataset, output_dir: Path):
    torch.manual_seed(101)
    model = _tiny_model()
    adapter = Cosmos3TrainAdapter(
        Cosmos3JointDenoiser(model),
        expected_latent_channels=2,
        gradient_checkpointing=True,
    )
    return build_cosmos_single_device_session(
        recipe=_recipe(),
        adapter=adapter,
        dataset=cache,
        output_dir=output_dir,
        fused_adamw=False,
    )


def _tensor_state(component: object) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in component.state_dict().items()}


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def test_power_ema_matches_cosmos_author_formula_and_resume_state() -> None:
    expected_exponent = float(__import__("numpy").roots([1.0, 7.0, 16.0 - 0.1**-2, 12.0 - 0.1**-2]).real.max())
    assert power_ema_exponent(0.1) == expected_exponent
    assert power_ema_beta(0, rate=0.1, iteration_shift=0) == 0.0
    expected_second_beta = (1.0 - 1.0 / 2.0) ** (expected_exponent + 1.0)
    assert power_ema_beta(1, rate=0.1, iteration_shift=0) == expected_second_beta

    model = torch.nn.Linear(2, 2, bias=False)
    ema = PowerEMA(model, rate=0.1, iteration_shift=0)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema(model)
    torch.testing.assert_close(ema.shadow_000000, model.weight, rtol=0, atol=0)
    assert ema.num_updates.item() == 1
    assert ema.last_beta.item() == 0.0

    restored = PowerEMA(model, rate=0.1, iteration_shift=0)
    restored.load_state_dict(ema.state_dict(), strict=True)
    with torch.no_grad():
        model.weight.fill_(4.0)
    restored(model)
    expected = torch.full_like(restored.shadow_000000, 2.0 * expected_second_beta + 4.0 * (1.0 - expected_second_beta))
    torch.testing.assert_close(restored.shadow_000000, expected, rtol=0, atol=1.0e-7)
    assert restored.num_updates.item() == 2
    assert restored.last_beta.item() == expected_second_beta


def test_cosmos3_nano_real_step_power_ema_dcp_resume_and_export(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    continuous = _session(cache, tmp_path / "continuous")
    continuous.run(max_steps=2, seed=31)
    expected_model = _tensor_state(continuous.engine.adapter.trainable_module)
    expected_ema = _tensor_state(continuous.ema)

    assert continuous.ema.num_updates.item() == 2
    assert continuous.progress.microbatches_seen == 4
    assert continuous._checkpoint_state is not None
    assert continuous._checkpoint_state.optional_state_presence["ema"] is True

    resumed = _session(cache, tmp_path / "resumed")
    resumed.run(
        max_steps=1,
        seed=31,
        resume_checkpoint=continuous.output_dir / "checkpoints" / "step-00000001",
    )
    assert resumed.ema.num_updates.item() == 2
    for name, expected in expected_model.items():
        torch.testing.assert_close(
            resumed.engine.adapter.trainable_module.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    for name, expected in expected_ema.items():
        torch.testing.assert_close(resumed.ema.state_dict()[name], expected, rtol=0, atol=0)

    module = continuous.engine.adapter.trainable_module
    online = _tensor_state(module)
    expected_export = _tiny_model()
    expected_export.load_state_dict(online, strict=True)
    continuous.ema.copy_to(expected_export)
    artifact = continuous.export_full_model(max_shard_size_bytes=4096)

    exported = _tiny_model()
    load_full_model(exported, artifact.path)
    for name, expected in expected_export.state_dict().items():
        torch.testing.assert_close(exported.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in online.items():
        torch.testing.assert_close(module.state_dict()[name], expected, rtol=0, atol=0)
    manifest = json.loads(continuous.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["full_model"]["weights"] == "ema"
    assert math.isfinite(continuous.summary.final_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cosmos3_nano_power_ema_fsdp2_checkpoint_and_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    context = DistributedTrainingContext(device_type="cuda")
    try:
        cache = _cache(tmp_path / "cache")
        torch.manual_seed(107)
        model = _tiny_model().to(context.device)
        adapter = Cosmos3TrainAdapter(
            Cosmos3JointDenoiser(model),
            expected_latent_channels=2,
            gradient_checkpointing=True,
        )
        session = build_cosmos_fsdp2_session(
            recipe=_recipe(backend="fsdp2"),
            adapter=adapter,
            dataset=cache,
            distributed_context=context,
            output_dir=tmp_path / "run",
            fused_adamw=False,
        )
        session.run(max_steps=1, seed=43)
        assert session.ema.num_updates.item() == 1
        assert session.ema.last_beta.item() == 0.0
        checkpoint = session.output_dir / "checkpoints" / "step-00000001"
        assert checkpoint.is_dir()

        torch.manual_seed(107)
        resumed_model = _tiny_model().to(context.device)
        resumed_adapter = Cosmos3TrainAdapter(
            Cosmos3JointDenoiser(resumed_model),
            expected_latent_channels=2,
            gradient_checkpointing=True,
        )
        resumed = build_cosmos_fsdp2_session(
            recipe=_recipe(backend="fsdp2"),
            adapter=resumed_adapter,
            dataset=cache,
            distributed_context=context,
            output_dir=tmp_path / "resumed",
            fused_adamw=False,
        )
        resumed.run(max_steps=1, seed=43, resume_checkpoint=checkpoint)
        assert resumed.ema.num_updates.item() == 2
        assert resumed.ema.last_beta.item() == power_ema_beta(1)
        artifact = resumed.export_full_model(max_shard_size_bytes=4096)
    finally:
        context.close()

    exported = _tiny_model()
    load_full_model(exported, artifact.path)
