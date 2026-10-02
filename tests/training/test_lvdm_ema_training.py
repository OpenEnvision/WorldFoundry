from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.core.nn.ema import LitEma  # noqa: E402
from worldfoundry.training.data.video_bucketing import VideoLatentGeometry  # noqa: E402
from worldfoundry.training.data.video_cache import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
)
from worldfoundry.training.distributed.parallel import DistributedTrainingContext  # noqa: E402
from worldfoundry.training.engine.lvdm.sft import (  # noqa: E402
    _build_lvdm_ema,
    _lvdm_ema_checkpoint_state,
    build_lvdm_short_fsdp2_session,
    build_lvdm_short_single_device_session,
)
from worldfoundry.training.models.lvdm import LVDMUnconditionalTrainAdapter  # noqa: E402
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402
from worldfoundry.training.tuning import load_full_model  # noqa: E402


class _TinyDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv3d(4, 4, 1, bias=False)

    def forward(self, noisy, timesteps):
        return self.projection(noisy) + timesteps.reshape(-1, 1, 1, 1, 1) * 0.0


def _cache(root: Path) -> VideoCachedDataset:
    store = VideoCacheStore(root)
    provenance = VideoCacheProvenance(
        media_uri="tiny.mp4",
        prompt="a moving square",
        model_recipe="lvdm-short-unconditional",
        codec={"name": "tiny"},
        conditioner={"name": "none"},
        tokenizer={"name": "none"},
        conditioning_inputs={},
        safety_audit={"safe": True},
        frame_sampling={"mode": "tiny"},
        spatial_transform={"mode": "tiny"},
        latent_normalization={
            "posterior": "sample",
            "operation": "scale*(sample+shift)",
            "scale": 0.220142075,
            "shift": 0.5837740898,
        },
        task="t2v",
        conditioning_layout="none",
        aspect_bin="1:1",
        source_num_frames=8,
        source_height=16,
        source_width=16,
        source_fps=8.0,
        target_num_frames=8,
        target_height=16,
        target_width=16,
        target_fps=8.0,
        latent_geometry=VideoLatentGeometry(8, 8, 4, "uniform"),
    )
    entry = store.write_sample(
        sample_id="tiny",
        provenance=provenance,
        clean_latents=torch.randn(4, 2, 2, 2, generator=torch.Generator().manual_seed(5)),
        latent_loss_mask=torch.ones(1, 2, 2, 2),
        valid_latent_mask=torch.ones(1, 2, 2, 2, dtype=torch.bool),
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def _recipe(*, backend: str = "single") -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": "tiny-lvdm-ema", "output_dir": "unused"},
            "model": {"recipe": "lvdm-short-unconditional", "checkpoint": "unused.ckpt"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused",
                "cache": "unused",
                "max_latent_tokens_per_microbatch": 8,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"num_workers": 0, "pin_memory": False},
            },
            "objective": {
                "type": "classic-diffusion",
                "prediction_type": "epsilon",
                "timestep_sampler": "uniform",
                "conditioning_dropout": 0.0,
                "options": {
                    "num_train_timesteps": 1000,
                    "beta_start": 0.0015,
                    "beta_end": 0.0155,
                    "loss_type": "l1",
                },
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.00006,
                "weight_decay": 0.01,
                "max_grad_norm": None,
                "gradient_accumulation_steps": 2,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": backend, "dp_shard": "auto"},
            "checkpoint": {"save_every_steps": 1, "async": False},
        }
    )


def _session(cache: VideoCachedDataset, output: Path):
    torch.manual_seed(11)
    return build_lvdm_short_single_device_session(
        recipe=_recipe(),
        adapter=LVDMUnconditionalTrainAdapter(_TinyDenoiser(), codec=None),
        dataset=cache,
        output_dir=output,
        fused_adamw=False,
    )


def _tensor_state(component: object) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in component.state_dict().items()}


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def test_lvdm_ema_updates_per_train_batch_and_resumes_exactly(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    continuous = _session(cache, tmp_path / "continuous")
    before = continuous.engine.adapter.trainable_module.projection.weight.detach().clone()
    continuous.run(max_steps=2, seed=31)
    expected_model = _tensor_state(continuous.engine.adapter.trainable_module)
    expected_ema = _tensor_state(continuous.ema)
    assert continuous.ema.num_updates.item() == 4

    shadow_name = continuous.ema.m_name2s_name["projection.weight"]
    shadow_after_two_steps = dict(continuous.ema.named_buffers())[shadow_name]
    assert not torch.equal(shadow_after_two_steps, before)

    resumed = _session(cache, tmp_path / "resumed")
    resumed.run(
        max_steps=1,
        seed=31,
        resume_checkpoint=continuous.output_dir / "checkpoints" / "step-00000001",
    )
    assert resumed.ema.num_updates.item() == 4
    for name, expected in expected_model.items():
        torch.testing.assert_close(resumed.engine.adapter.trainable_module.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in expected_ema.items():
        torch.testing.assert_close(resumed.ema.state_dict()[name], expected, rtol=0, atol=0)


def test_lvdm_full_export_uses_ema_and_restores_online_weights(tmp_path: Path) -> None:
    session = _session(_cache(tmp_path / "cache"), tmp_path / "run")
    initial = session.engine.adapter.trainable_module.projection.weight.detach().clone()
    session.run(max_steps=1, seed=37)
    module = session.engine.adapter.trainable_module
    online = _tensor_state(module)
    shadow_name = session.ema.m_name2s_name["projection.weight"]
    expected_ema = dict(session.ema.named_buffers())[shadow_name].detach().clone()
    torch.testing.assert_close(expected_ema, 0.25 * initial + 0.75 * online["projection.weight"])

    artifact = session.export_full_model(max_shard_size_bytes=256)

    restored = _TinyDenoiser()
    load_full_model(restored, artifact.path)
    torch.testing.assert_close(restored.projection.weight, expected_ema, rtol=0, atol=0)
    for name, expected in online.items():
        torch.testing.assert_close(module.state_dict()[name], expected, rtol=0, atol=0)
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["full_model"]["weights"] == "ema"


def test_lvdm_official_checkpoint_ema_prefix_loads_strictly() -> None:
    module = _TinyDenoiser()
    source = LitEma(module, decay=0.9999, use_num_upates=True)
    source(module)
    checkpoint = {
        "state_dict": {
            (
                f"model_ema.{name}" if name in {"decay", "num_updates"} else f"model_ema.diffusion_model{name}"
            ): value.clone()
            for name, value in source.state_dict().items()
        },
        "global_step": 1,
    }

    restored = _build_lvdm_ema(module, _lvdm_ema_checkpoint_state(checkpoint))

    for name, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_lvdm_ema_keeps_released_decay_precision() -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    ema = LitEma(module, decay=0.9999, use_num_upates=False)

    assert ema.decay.dtype is torch.float32
    assert float(ema.decay) < 1.0
    with torch.no_grad():
        module.weight.zero_()
    ema(module)

    assert float(ema.weight) < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lvdm_fsdp2_ema_checkpoint_and_full_export(
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
        adapter = LVDMUnconditionalTrainAdapter(_TinyDenoiser().to(context.device), codec=None)
        session = build_lvdm_short_fsdp2_session(
            recipe=_recipe(backend="fsdp2"),
            adapter=adapter,
            dataset=_cache(tmp_path / "cache"),
            distributed_context=context,
            output_dir=tmp_path / "run",
            fused_adamw=False,
        )
        session.run(max_steps=1, seed=41)
        assert session.ema.num_updates.item() == 2
        assert (session.output_dir / "checkpoints" / "step-00000001").is_dir()
        artifact = session.export_full_model(max_shard_size_bytes=256)
    finally:
        context.close()

    restored = _TinyDenoiser()
    load_full_model(restored, artifact.path)
