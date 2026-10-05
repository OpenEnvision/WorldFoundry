from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (  # noqa: E402
    WanDenoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (  # noqa: E402
    WanModel,
)
from worldfoundry.training.data import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
    VideoLatentGeometry,
)
from worldfoundry.training.data.wan.contracts import (  # noqa: E402
    wan_latent_normalization,
)
from worldfoundry.training.distributed import DistributedTrainingContext  # noqa: E402
from worldfoundry.training.engine import (  # noqa: E402
    build_wan_fsdp2_session,
    build_wan_single_device_session,
)
from worldfoundry.training.models import WanTrainAdapter  # noqa: E402
from worldfoundry.training.recipes import TrainingRecipe  # noqa: E402


def _recipe(*, backend: str = "single", save_every_steps: int = 1) -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-training",
            "execution_owner": "worldfoundry-native",
            "run": {"id": "wan-session-test", "output_dir": "unused"},
            "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
            "tuning": {
                "mode": "lora",
                "preset": "wan-attention",
                "rank": 2,
                "alpha": 2,
                "dropout": 0.0,
            },
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "max_latent_tokens_per_microbatch": 8,
                "split": "train",
                "shuffle": False,
                "shuffle_seed": 17,
                "tail_policy": "pad",
                "options": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "snapshot_every_n_steps": 1,
                },
            },
            "objective": {
                "type": "flow_matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "logit_normal",
                "conditioning_dropout": 0.0,
                "options": {"num_train_timesteps": 1000, "flow_shift": 1.0},
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.99],
                "epsilon": 1.0e-8,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 1,
            },
            "runtime": {
                "param_dtype": "float32",
                "reduce_dtype": "float32",
                "activation_checkpoint": "none",
                "compile": False,
            },
            "distributed": {"backend": backend},
            "checkpoint": {
                "save_every_steps": save_every_steps,
                "async": False,
            },
        }
    )


def _adapter(seed: int, *, device: torch.device | str = "cpu") -> WanTrainAdapter:
    torch.manual_seed(seed)
    model = WanModel(
        dim=24,
        in_dim=16,
        ffn_dim=48,
        out_dim=16,
        text_dim=4096,
        freq_dim=16,
        eps=1.0e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
    )
    model.to(device)
    return WanTrainAdapter(
        WanDenoiser(model, compute_dtype=torch.float32),
        codec=None,
        conditioner=None,
    )


def _cache(tmp_path: Path) -> VideoCachedDataset:
    root = tmp_path / "cache"
    store = VideoCacheStore(root)
    provenance = VideoCacheProvenance(
        media_uri="cached-video.mp4",
        prompt="cached video prompt",
        model_recipe="wan2.1-t2v-1.3b",
        codec={"repo_id": "vae", "revision": "main"},
        conditioner={"repo_id": "umt5", "revision": "main"},
        tokenizer={"repo_id": "tokenizer", "revision": "main"},
        conditioning_inputs={"task": "t2v", "conditions": {}},
        safety_audit={"prompt": "cached video prompt", "safe": True},
        frame_sampling={"mode": "head", "selected_frame_indices": [0, 1, 2, 3, 4]},
        spatial_transform={"mode": "identity"},
        latent_normalization=wan_latent_normalization(),
        task="t2v",
        conditioning_layout="umt5-sequence",
        aspect_bin="1:1",
        source_num_frames=5,
        source_height=16,
        source_width=16,
        source_fps=5.0,
        target_num_frames=5,
        target_height=16,
        target_width=16,
        target_fps=5.0,
        latent_geometry=VideoLatentGeometry(8, 8, 4, "first-frame"),
    )
    generator = torch.Generator().manual_seed(41)
    entry = store.write_sample(
        sample_id="cached-video",
        provenance=provenance,
        clean_latents=torch.randn(16, 2, 2, 2, generator=generator),
        conditioning={
            "context": torch.randn(512, 4096, generator=generator),
        },
        conditioning_layouts={"context": "sequence-features"},
        latent_loss_mask=torch.ones(1, 2, 2, 2),
        valid_latent_mask=torch.ones(1, 2, 2, 2, dtype=torch.bool),
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def _trainable_state(session) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in session.engine.adapter.trainable_module.named_parameters()
        if parameter.requires_grad
    }


def test_wan_session_exact_resume_export_and_only_lora_gradients(tmp_path: Path) -> None:
    recipe = _recipe()
    cache = _cache(tmp_path)
    continuous = build_wan_single_device_session(
        recipe=recipe,
        adapter=_adapter(43),
        dataset=cache,
        output_dir=tmp_path / "continuous",
        fused_adamw=False,
        initialization_seed=47,
    )
    continuous_summary = continuous.run(max_steps=2, seed=53)
    continuous_state = _trainable_state(continuous)
    continuous_artifact = continuous.export_peft()
    metrics = [json.loads(line) for line in continuous.metrics_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(continuous.manifest_path.read_text(encoding="utf-8"))
    first_checkpoint = Path(manifest["checkpoints"][0]["path"])

    resumed = build_wan_single_device_session(
        recipe=recipe,
        adapter=_adapter(43),
        dataset=cache,
        output_dir=tmp_path / "resumed",
        fused_adamw=False,
        initialization_seed=47,
    )
    resumed_summary = resumed.run(
        max_steps=1,
        seed=53,
        resume_checkpoint=first_checkpoint,
    )
    resumed_state = _trainable_state(resumed)
    resumed_artifact = resumed.export_peft()

    assert continuous_summary.changed_parameter_tensors > 0
    assert continuous.peft_application is not None
    assert continuous.peft_application.target_audit.block_count == 2
    assert len(continuous.peft_application.targeted_module_names) == 16
    assert all("lora_" in name for name in continuous_state)
    assert resumed_summary.final_loss == metrics[1]["loss"]
    assert set(resumed_state) == set(continuous_state)
    for name, parameter in resumed_state.items():
        torch.testing.assert_close(parameter, continuous_state[name], rtol=0, atol=0)
    assert (
        resumed_artifact.file_size_bytes["adapter_model.safetensors"]
        == continuous_artifact.file_size_bytes["adapter_model.safetensors"]
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["initial_global_step"] == 1
    assert resumed_manifest["resumed_from"]["global_step"] == 1
    assert resumed_manifest["cumulative_progress"]["optimizer_steps"] == 2


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wan_fsdp2_world_one_step_and_adapter_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_local_port()))
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    context = DistributedTrainingContext(device_type="cuda")
    session = None
    try:
        session = build_wan_fsdp2_session(
            recipe=_recipe(backend="fsdp2", save_every_steps=0),
            adapter=_adapter(61, device=context.device),
            dataset=_cache(tmp_path),
            distributed_context=context,
            output_dir=tmp_path / "fsdp2",
            fused_adamw=False,
            initialization_seed=67,
        )
        summary = session.run(max_steps=1, seed=71)
        artifact = session.export_peft()

        assert summary.changed_parameter_tensors > 0
        assert artifact.path.is_dir()
        assert "adapter_model.safetensors" in artifact.file_size_bytes
        assert session.engine.application.parallel_plan.world_size == 1
    finally:
        if session is not None:
            session.close()
        else:
            context.close()
