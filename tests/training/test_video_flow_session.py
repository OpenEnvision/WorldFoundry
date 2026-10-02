from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from worldfoundry.training.api.contracts import (  # noqa: E402
    ObjectiveBatch,
    PreparedBatch,
    TrainingBatch,
)
from worldfoundry.training.data.video_bucketing import VideoLatentGeometry  # noqa: E402
from worldfoundry.training.data.video_cache import (  # noqa: E402
    VideoCachedDataset,
    VideoCacheProvenance,
    VideoCacheStore,
)
from worldfoundry.training.engine.video_flow import (  # noqa: E402
    build_cached_video_flow_single_device_session,
)
from worldfoundry.training.objectives.flow_matching import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingObjective,
)
from worldfoundry.training.recipes.spec import TrainingRecipe  # noqa: E402


class _CachedVideoAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = None
    fsdp_block_classes = (torch.nn.Conv3d,)

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Conv3d(2, 2, kernel_size=1, bias=False)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        latents = batch.conditions["clean_latents"]
        assert isinstance(latents, torch.Tensor)
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=latents,
            loss_mask=batch.conditions.get("latent_loss_mask"),
            sample_weights=batch.sample_weights,
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        assert isinstance(batch.model_input, torch.Tensor)
        return self.trainable_module(batch.model_input)


def _recipe() -> TrainingRecipe:
    return TrainingRecipe.from_mapping(
        {
            "run": {"id": "shared-video-flow", "output_dir": "unused"},
            "model": {"recipe": "tiny-video-flow"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "max_latent_tokens_per_microbatch": 8,
                "shuffle": False,
                "tail_policy": "pad",
                "options": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "video_buckets": [{"num_frames": 2, "height": 2, "width": 2}],
                },
            },
            "objective": {
                "type": "flow-matching",
                "prediction_type": "flow_velocity",
                "timestep_sampler": "uniform",
            },
            "optimizer": {"type": "adamw", "learning_rate": 1.0e-2},
            "scheduler": {
                "type": "warmup-cosine",
                "total_steps": 10,
                "warmup_steps": 2,
                "start_factor": 0.0,
                "peak_factor": 0.5,
                "end_factor": 0.2,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
        }
    )


def _cache(root: Path) -> VideoCachedDataset:
    store = VideoCacheStore(root)
    source = VideoCacheProvenance(
        media_uri="tiny.mp4",
        prompt="a moving square",
        model_recipe="tiny-video-flow",
        codec={"name": "tiny"},
        conditioner={"name": "none"},
        tokenizer={"name": "none"},
        conditioning_inputs={},
        safety_audit={"safe": True},
        frame_sampling={"mode": "head"},
        spatial_transform={"mode": "identity"},
        latent_normalization={"mode": "identity"},
        task="t2v",
        conditioning_layout="none",
        aspect_bin="1:1",
        source_num_frames=2,
        source_height=2,
        source_width=2,
        source_fps=2.0,
        target_num_frames=2,
        target_height=2,
        target_width=2,
        target_fps=2.0,
        latent_geometry=VideoLatentGeometry(1, 1, 1, "uniform"),
    )
    entry = store.write_sample(
        sample_id="tiny",
        provenance=source,
        clean_latents=torch.randn(2, 2, 2, 2),
        latent_loss_mask=torch.ones(1, 2, 2, 2),
    )
    store.write_index(entries=(entry,))
    return VideoCachedDataset(root)


def test_shared_cached_video_flow_session_updates_and_checkpoints(tmp_path: Path) -> None:
    adapter = _CachedVideoAdapter()
    before = adapter.trainable_module.weight.detach().clone()
    session = build_cached_video_flow_single_device_session(
        recipe=_recipe(),
        adapter=adapter,
        dataset=_cache(tmp_path / "cache"),
        objective=FlowMatchingObjective(FlowMatchingConfig(timestep_sampler="uniform")),
        cache_contract={"prediction_type": "flow_velocity", "latent_channels": 2},
        output_dir=tmp_path / "run",
        fused_adamw=False,
    )

    summary = session.run(max_steps=2, seed=19)

    assert summary.changed_parameter_tensors == 1
    assert not torch.equal(adapter.trainable_module.weight.detach(), before)
    assert session.lr_scheduler is not None
    assert session.lr_scheduler.last_epoch == 2
    assert session.engine.optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-3)
    assert session.data_identity["cache_contract"] == {
        "prediction_type": "flow_velocity",
        "latent_channels": 2,
    }
    assert session.data_identity["token_sampler"]["world_size"] == 1
