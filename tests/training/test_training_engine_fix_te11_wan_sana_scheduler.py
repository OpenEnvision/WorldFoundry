"""TE-11 regression: wan/sana sessions must honor ``recipe.scheduler``.

``build_wan_*_session``/``build_sana_*_session`` used to ignore the public
``scheduler`` recipe field entirely: recipes configured with warmup/cosine or
linear schedules silently trained at a constant learning rate.  This test
builds a real (tiny) Wan single-device session and checks that a configured
linear schedule actually drives the optimizer learning rate, is exposed on
the session (so it enters checkpoint state), and that scheduler-free recipes
keep the previous constant-LR behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

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
from worldfoundry.training.engine import build_wan_single_device_session  # noqa: E402
from worldfoundry.training.models import WanTrainAdapter  # noqa: E402
from worldfoundry.training.recipes import TrainingRecipe  # noqa: E402

_BASE_LR = 1.0e-3


@pytest.fixture(autouse=True)
def _plain_torch_dataloader(monkeypatch: pytest.MonkeyPatch):
    """Replace the torchdata-backed loader with a plain DataLoader.

    The scheduler wiring under test lives in the session builder; the
    checkpointable-loader dependency (torchdata) is unavailable on pure-CPU
    CI hosts and irrelevant to this regression, so checkpointing stays off
    (``save_every_steps=0``) and iteration uses torch's stock DataLoader.
    """

    import worldfoundry.training.engine.wan.cache as wan_cache

    def _loader(dataset, *, batch_sampler, collate_fn, **_ignored):
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

    monkeypatch.setattr(wan_cache, "build_stateful_dataloader", _loader)


def _recipe(*, scheduler: dict[str, object] | None) -> TrainingRecipe:
    mapping: dict[str, object] = {
        "schema": "worldfoundry-training",
        "execution_owner": "worldfoundry-native",
        "run": {"id": "wan-scheduler-test", "output_dir": "unused"},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
        "tuning": {"mode": "full"},
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
            "learning_rate": _BASE_LR,
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
        "distributed": {"backend": "single"},
        "checkpoint": {"save_every_steps": 0, "async": False},
    }
    if scheduler is not None:
        mapping["scheduler"] = scheduler
    return TrainingRecipe.from_mapping(mapping)


def _adapter(seed: int) -> WanTrainAdapter:
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
    model.to("cpu")
    return WanTrainAdapter(
        WanDenoiser(model, compute_dtype=torch.float32),
        codec=None,
        conditioner=None,
    )


def _cache(root: Path) -> VideoCachedDataset:
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


def _learning_rate(session) -> float:
    (group,) = session.engine.optimizer.param_groups
    return float(group["lr"])


def test_wan_session_applies_configured_linear_lr_schedule(tmp_path: Path) -> None:
    scheduler = {
        "type": "linear",
        "total_steps": 4,
        "start_factor": 1.0,
        "end_factor": 0.5,
    }
    session = build_wan_single_device_session(
        recipe=_recipe(scheduler=scheduler),
        adapter=_adapter(23),
        dataset=_cache(tmp_path / "cache-scheduled"),
        output_dir=tmp_path / "run-scheduled",
        fused_adamw=False,
    )

    assert session.lr_scheduler is not None
    assert _learning_rate(session) == pytest.approx(_BASE_LR)

    session.run(max_steps=2, seed=7)

    # LinearLR(start=1.0, end=0.5, total_iters=4) after two scheduler steps.
    assert _learning_rate(session) == pytest.approx(_BASE_LR * 0.75)


def test_wan_session_without_scheduler_keeps_constant_lr(tmp_path: Path) -> None:
    session = build_wan_single_device_session(
        recipe=_recipe(scheduler=None),
        adapter=_adapter(23),
        dataset=_cache(tmp_path / "cache-constant"),
        output_dir=tmp_path / "run-constant",
        fused_adamw=False,
    )

    assert session.lr_scheduler is None

    session.run(max_steps=2, seed=7)

    assert _learning_rate(session) == pytest.approx(_BASE_LR)
