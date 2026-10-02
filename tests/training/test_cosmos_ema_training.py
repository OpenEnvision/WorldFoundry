from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
pytest.importorskip("peft")

from worldfoundry.base_models.diffusion_model.models.denoisers.cosmos2p5 import (  # noqa: E402
    Cosmos25Denoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.model import (  # noqa: E402
    Cosmos25Transformer3DModel,
)
from worldfoundry.training.data.video_bucketing import VideoLatentGeometry  # noqa: E402
from worldfoundry.training.data.video_cache import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
)
from worldfoundry.training.ema import PowerEMA, power_ema_beta  # noqa: E402
from worldfoundry.training.engine.cosmos.sft import (  # noqa: E402
    build_cosmos_single_device_session,
)
from worldfoundry.training.models.cosmos import CosmosPredict25TrainAdapter  # noqa: E402
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402

FORMAL_LORA_PROFILE = Path(__file__).parents[2] / "configs/training/cosmos_predict2p5_2b_video_lora.yaml"


def _model() -> Cosmos25Transformer3DModel:
    return Cosmos25Transformer3DModel(
        in_channels=3,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=1,
        mlp_ratio=2.0,
        text_in_channels=4,
        text_embed_dim=24,
        adaln_lora_dim=4,
        max_size=(2, 4, 4),
        patch_size=(1, 1, 1),
        rope_scale=(1.0, 1.0, 1.0),
        use_crossattn_projection=True,
        concat_padding_mask=True,
    )


def _recipe() -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": "cosmos-predict25-power-ema", "output_dir": "unused"},
            "model": {
                "recipe": "cosmos-predict2.5-2b",
                "options": {"checkpoint_variant": "pretrained"},
            },
            "tuning": {
                "mode": "lora",
                "preset": "cosmos-predict-attention-mlp",
                "rank": 2,
                "alpha": 2,
            },
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "max_latent_tokens_per_microbatch": 4,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {"num_workers": 0, "pin_memory": False},
            },
            "objective": {
                "type": "flow-matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "logit-normal",
                "conditioning_dropout": 0.2,
                "options": {"flow_shift": 5.0},
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2 ** (-14.5),
                "weight_decay": 0.001,
                "betas": [0.9, 0.99],
            },
            "scheduler": {
                "type": "warmup-cosine",
                "total_steps": 100_000,
                "warmup_steps": 2_000,
                "start_factor": 0.0,
                "peak_factor": 0.5,
                "end_factor": 0.2,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "checkpoint": {"save_every_steps": 1, "async": False},
        }
    )


def _cache(root: Path) -> VideoCachedDataset:
    store = VideoCacheStore(root)
    provenance = VideoCacheProvenance(
        media_uri="tiny.mp4",
        prompt="a tiny moving square",
        model_recipe="cosmos-predict2.5-2b",
        codec={"name": "tiny"},
        conditioner={"name": "tiny"},
        tokenizer={"name": "tiny"},
        conditioning_inputs={},
        safety_audit={"safe": True},
        frame_sampling={"mode": "tiny"},
        spatial_transform={"mode": "tiny"},
        latent_normalization={"mode": "identity"},
        task="t2v",
        conditioning_layout="cosmos-predict-context",
        aspect_bin="1:1",
        source_num_frames=1,
        source_height=2,
        source_width=2,
        source_fps=16.0,
        target_num_frames=1,
        target_height=2,
        target_width=2,
        target_fps=16.0,
        latent_geometry=VideoLatentGeometry(1, 1, 1, "uniform"),
    )
    generator = torch.Generator().manual_seed(17)
    entry = store.write_sample(
        sample_id="tiny",
        provenance=provenance,
        clean_latents=torch.randn(2, 1, 2, 2, generator=generator),
        conditioning={
            "context": torch.randn(3, 4, generator=generator),
            "negative_context": torch.randn(3, 4, generator=generator),
        },
        conditioning_layouts={
            "context": "sequence-features",
            "negative_context": "sequence-features",
        },
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def _session(cache: VideoCachedDataset, output_dir: Path):
    torch.manual_seed(23)
    adapter = CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(_model()),
        None,
        None,
        expected_latent_channels=2,
        temporal_compression=1,
        spatial_compression=1,
    )
    return build_cosmos_single_device_session(
        recipe=_recipe(),
        adapter=adapter,
        dataset=cache,
        output_dir=output_dir,
        fused_adamw=False,
        initialization_seed=29,
    )


def test_predict25_lora_keeps_fp32_master_parameters_and_adamw_state_under_bf16_autocast(
    tmp_path: Path,
) -> None:
    formal_recipe = TrainingRecipe.from_file(FORMAL_LORA_PROFILE)
    assert formal_recipe.runtime.param_dtype == "bfloat16"
    assert formal_recipe.tuning.mode == "lora"
    recipe = replace(
        formal_recipe,
        data=replace(
            formal_recipe.data,
            shuffle=False,
            tail_policy="pad",
            options={"num_workers": 0, "pin_memory": False},
        ),
        distributed=replace(formal_recipe.distributed, backend="single"),
    )
    adapter = CosmosPredict25TrainAdapter(
        Cosmos25Denoiser(_model().to(dtype=torch.bfloat16)),
        None,
        None,
        expected_latent_channels=2,
        temporal_compression=1,
        spatial_compression=1,
    )
    session = build_cosmos_single_device_session(
        recipe=recipe,
        adapter=adapter,
        dataset=_cache(tmp_path / "cache"),
        output_dir=tmp_path / "run",
        fused_adamw=False,
        initialization_seed=29,
    )
    parameters = [parameter for group in session.engine.optimizer.param_groups for parameter in group["params"]]
    assert parameters
    assert all(parameter.dtype is torch.float32 for parameter in parameters)
    assert session.engine.autocast_dtype is torch.bfloat16

    session.run(max_steps=1, seed=31)
    for parameter in parameters:
        state = session.engine.optimizer.state[parameter]
        assert state["exp_avg"].dtype is torch.float32
        assert state["exp_avg_sq"].dtype is torch.float32


def _tensor_state(component: object) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in component.state_dict().items()}


def test_power_ema_uses_the_released_zero_based_iteration_sequence() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    ema = PowerEMA(model, rate=0.1, iteration_shift=0)

    model.weight.data.fill_(1.0)
    ema(model)
    shadow = getattr(ema, ema._shadow_names["weight"])  # noqa: SLF001 - formula contract.
    torch.testing.assert_close(shadow, torch.ones_like(shadow), rtol=0, atol=0)
    assert ema.last_beta.item() == 0.0

    model.weight.data.fill_(3.0)
    ema(model)
    beta = power_ema_beta(1, rate=0.1, iteration_shift=0)
    torch.testing.assert_close(
        shadow,
        torch.full_like(shadow, beta + (1.0 - beta) * 3.0),
    )
    assert ema.last_beta.item() == pytest.approx(beta)
    assert ema.num_updates.item() == 2


def test_predict25_power_ema_updates_once_per_optimizer_step_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path / "cache")
    uninterrupted = _session(cache, tmp_path / "uninterrupted")
    assert isinstance(uninterrupted.ema, PowerEMA)
    assert uninterrupted.export_ema is True
    uninterrupted.run(max_steps=3, seed=31)
    expected_model = _tensor_state(uninterrupted.engine.adapter.trainable_module)
    expected_ema = _tensor_state(uninterrupted.ema)
    assert uninterrupted.ema.num_updates.item() == 3

    interrupted = _session(cache, tmp_path / "interrupted")
    interrupted.run(max_steps=2, seed=31)
    checkpoint = interrupted.output_dir / "checkpoints" / "step-00000002"
    resumed = _session(cache, tmp_path / "resumed")
    resumed.run(max_steps=1, seed=31, resume_checkpoint=checkpoint)

    assert resumed.ema.num_updates.item() == 3
    assert resumed.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
    for name, expected in expected_model.items():
        torch.testing.assert_close(
            resumed.engine.adapter.trainable_module.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    for name, expected in expected_ema.items():
        torch.testing.assert_close(resumed.ema.state_dict()[name], expected, rtol=0, atol=0)


def test_predict25_peft_export_uses_power_ema_and_restores_online_weights(tmp_path: Path) -> None:
    session = _session(_cache(tmp_path / "cache"), tmp_path / "run")
    session.run(max_steps=2, seed=37)
    module = session.engine.adapter.trainable_module
    online = {
        name: parameter.detach().clone() for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    for shadow_name in session.ema._shadow_names.values():  # noqa: SLF001 - exported EMA state contract.
        getattr(session.ema, shadow_name).fill_(0.125)

    artifact = session.export_peft()

    exported = safetensors.load_file(artifact.path / "adapter_model.safetensors")
    assert exported
    assert all(torch.equal(value, torch.full_like(value, 0.125)) for value in exported.values())
    for name, expected in online.items():
        torch.testing.assert_close(dict(module.named_parameters())[name], expected, rtol=0, atol=0)
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["peft_adapter"]["weights"] == "ema"
