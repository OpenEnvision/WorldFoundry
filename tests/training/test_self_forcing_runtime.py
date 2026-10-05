from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.training.data.rollout_cache import RolloutConditionedPrompt  # noqa: E402
from worldfoundry.training.data.rollout_manifest import RolloutPromptRecord  # noqa: E402
from worldfoundry.training.data.sana_cache import CacheTensorDescriptor  # noqa: E402
from worldfoundry.training.data.shared_conditioning import (  # noqa: E402
    SharedConditioningArtifact,
    SharedConditioningIdentity,
)
from worldfoundry.training.distributed.parallel import ParallelPlan  # noqa: E402
from worldfoundry.training.engine.wan.self_forcing import (  # noqa: E402
    materialize_wan_self_forcing_training_run,
)
from worldfoundry.training.engine.wan.self_forcing_recipe import (  # noqa: E402
    validate_wan_self_forcing_recipe,
)
from worldfoundry.training.models.causal_wan import (  # noqa: E402
    SELF_FORCING_ODE_CHECKPOINT,
    convert_self_forcing_causal_state_dict,
)
from worldfoundry.training.post_training.distillation.self_forcing import (  # noqa: E402
    NativeSelfForcingDataLoader,
    WanSelfForcingChunkAdapter,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402
from worldfoundry.training.safety.shieldgemma import (  # noqa: E402
    SHIELDGEMMA_PROMPT_POLICIES,
    PromptSafetyAudit,
)


def _recipe_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "run": {"id": "self-forcing", "output_dir": str(tmp_path / "run")},
        "model": {"recipe": "wan2.1-t2v-1.3b", "checkpoint": "default"},
        "tuning": {"mode": "full"},
        "data": {
            "manifest": "prompts.jsonl",
            "cache": "conditioning",
            "tail_policy": "pad",
            "options": {
                "prompt_batch_size": 1,
                "generation": {"height": 32, "width": 32, "num_frames": 9},
            },
        },
        "algorithm": {
            "type": "self-forcing",
            "denoising_timesteps": [1000, 750, 500, 250],
            "frames_per_block": 3,
            "real_score_model_recipe": "wan2.1-t2v-14b",
            "real_score_checkpoint": "default",
            "fake_score_model_recipe": "wan2.1-t2v-1.3b",
            "fake_score_checkpoint": "default",
            "ema_start_step": 2,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 2e-6,
            "betas": [0.0, 0.999],
        },
        "fake_score_optimizer": {
            "type": "adamw",
            "learning_rate": 4e-7,
            "betas": [0.0, 0.999],
        },
        "runtime": {
            "param_dtype": "float32",
            "reduce_dtype": "float32",
            "activation_checkpoint": "full",
        },
        "distributed": {"backend": "single"},
        "export": {"format": "distributed-checkpoint"},
    }


def _conditioned(prompt_id: str, value: float) -> RolloutConditionedPrompt:
    prompt = f"prompt {prompt_id}"
    record = RolloutPromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        safety_audit=PromptSafetyAudit(
            prompt=prompt,
            unsafe_probabilities={name: 0.01 for name in SHIELDGEMMA_PROMPT_POLICIES},
            threshold=0.5,
        ),
    )
    identity = SharedConditioningIdentity(
        branch=f"rollout-{prompt_id}",
        prompt=prompt,
        model_recipe="wan2.1-t2v-1.3b",
        conditioner={"repository": "test/conditioner", "revision": "test"},
        tokenizer={"repository": "test/tokenizer", "revision": "test"},
        tensors={
            "context": CacheTensorDescriptor(
                dtype="float32",
                shape=(3, 4),
                layout="sequence-features",
            )
        },
    )
    artifact = SharedConditioningArtifact(
        identity=identity,
        object_size_bytes=100,
        object_path=f"shared-objects/{identity.branch}.safetensors",
    )
    return RolloutConditionedPrompt(
        record=record,
        conditioning={"context": torch.full((3, 4), value)},
        artifact=artifact,
    )


class _PromptSource:
    def __init__(self) -> None:
        self.position = 0

    def __iter__(self) -> Iterator[tuple[RolloutConditionedPrompt, ...]]:
        self.position += 1
        yield (_conditioned("first", 1.0), _conditioned("second", 2.0))

    def state_dict(self) -> dict[str, object]:
        return {"position": self.position}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.position = int(state_dict["position"])


class _TinyCausalWan(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.25))
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.dim = 4
        self.local_attn_size = -1
        self.model_type = "t2v"
        self.num_heads = 1
        self.num_layers = 2
        self.patch_size = (1, 1, 1)
        self.text_len = 3
        self.events: list[tuple[int, float]] = []

    def forward(self, *, x, t, kv_cache, current_start, **kwargs):
        del kwargs
        self.events.append((int(current_start), float(t[0, 0].item())))
        for cache in kv_cache:
            end = current_start + x.shape[2] * x.shape[3] * x.shape[4]
            cache["global_end_index"].fill_(end)
            cache["local_end_index"].fill_(end)
        return x * self.gain


class _TinyWanScoreAdapter:
    prediction_type = "flow_velocity"
    lora_target_preset = "wan-attention"
    expected_latent_channels = 16
    temporal_compression = 4
    spatial_compression = 8
    expected_text_length = 512
    expected_context_features = 4096

    def __init__(self) -> None:
        self.trainable_module = torch.nn.Sequential(torch.nn.Linear(1, 1))
        self.fsdp_block_classes = (torch.nn.Linear,)

    def forward_model(self, batch, *, training, branch):
        del training, branch
        return batch.model_input * self.trainable_module[0].weight.reshape(())


def test_official_causal_checkpoint_converter_is_strict() -> None:
    weight = torch.ones(2, 3)
    converted = convert_self_forcing_causal_state_dict({"generator": {"model.patch_embedding.weight": weight}})
    assert converted == {"patch_embedding.weight": weight}
    assert SELF_FORCING_ODE_CHECKPOINT.revision == "47f4d3cf430cf000fcad587ba02c83ed971bba69"

    with pytest.raises(ValueError, match="mixes wrapped"):
        convert_self_forcing_causal_state_dict({"model.weight": weight, "bias": torch.zeros(2)})
    with pytest.raises(TypeError, match="non-tensor"):
        convert_self_forcing_causal_state_dict({"generator": {"model.weight": "bad"}})


def test_self_forcing_recipe_and_prompt_loader_consume_every_active_field(tmp_path) -> None:
    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path))
    restored = PostTrainingRecipe.from_mapping(recipe.to_dict())
    algorithm, plan = validate_wan_self_forcing_recipe(recipe)
    assert restored == recipe
    assert restored == recipe
    assert algorithm.real_score_model_recipe == "wan2.1-t2v-14b"
    assert plan.generation == {"height": 32, "width": 32, "num_frames": 9}

    payload = _recipe_mapping(tmp_path)
    payload["data"]["options"]["unused"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown Wan Self-Forcing"):
        validate_wan_self_forcing_recipe(PostTrainingRecipe.from_mapping(payload))

    source = _PromptSource()
    loader = NativeSelfForcingDataLoader(
        source,
        latent_shape=(16, 3, 4, 4),
        device="cpu",
        dtype=torch.float32,
        shared_unconditional_conditioning={"context": torch.full((3, 4), -1.0)},
    )
    batch = next(iter(loader))
    assert batch.sample_ids == ("first", "second")
    assert batch.clean_latents.shape == (2, 16, 3, 4, 4)
    torch.testing.assert_close(batch.conditioning["context"][0], torch.ones(3, 4))
    torch.testing.assert_close(
        batch.unconditional_conditioning["context"],
        torch.full((2, 3, 4), -1.0),
    )
    state = loader.state_dict()
    source.position = 99
    loader.load_state_dict(state)
    assert source.position == 1


def test_wan_chunk_bridge_reuses_and_commits_live_cache() -> None:
    graph = _TinyCausalWan()
    adapter = WanSelfForcingChunkAdapter(graph, frames_per_block=2)
    reference = torch.ones(1, 1, 4, 1, 1)
    conditioning = {"context": torch.ones(1, 3, 4)}
    cache = adapter.initialize_cache(
        reference,
        sample_ids=("sample",),
        conditioning=conditioning,
    )
    noisy = torch.ones(1, 1, 2, 1, 1, requires_grad=True)
    first = adapter.predict_clean_chunk(
        noisy,
        torch.tensor([1000.0]),
        torch.tensor([1.0]),
        block_index=0,
        start_frame=0,
        sample_ids=("sample",),
        conditioning=conditioning,
        cache=cache,
        training=True,
    )
    second = adapter.predict_clean_chunk(
        noisy,
        torch.tensor([750.0]),
        torch.tensor([0.75]),
        block_index=0,
        start_frame=0,
        sample_ids=("sample",),
        conditioning=conditioning,
        cache=cache,
        training=True,
    )
    adapter.commit_clean_chunk(
        second,
        block_index=0,
        start_frame=0,
        sample_ids=("sample",),
        conditioning=conditioning,
        cache=cache,
    )
    assert graph.events == [(0, 1000.0), (0, 750.0), (0, 0.0)]
    assert cache["committed_blocks"] == 1
    second.sum().backward()
    assert graph.gain.grad is not None
    assert first.grad_fn is not None


def test_materializer_resolves_14b_teacher_and_1p3b_fake_as_distinct_roles(
    tmp_path,
    monkeypatch,
) -> None:
    import worldfoundry.training.engine.wan.self_forcing as module

    recipe = PostTrainingRecipe.from_mapping(_recipe_mapping(tmp_path))

    class _Conditioning:
        index = SimpleNamespace(to_dict=lambda: {"entries": []})

        def __len__(self):
            return 2

    prompt_records = tuple(_conditioned(name, value).record for name, value in (("first", 1.0), ("second", 2.0)))

    assets = SimpleNamespace(
        root=tmp_path,
        output_dir=tmp_path / "run",
        cache_path=tmp_path / "conditioning",
        manifest_path=tmp_path / "prompts.jsonl",
        device=torch.device("cpu"),
        reward_device=torch.device("cpu"),
        distributed_context=None,
        parallel_plan=ParallelPlan.resolve(recipe.distributed, world_size=1),
        world_size=1,
        rank=0,
        prompts=prompt_records,
        conditioning=_Conditioning(),
        generation_geometry=(32, 32, 9),
        assembler=object(),
        native_recipe=object(),
        model_contract={"model_recipe": "wan2.1-t2v-1.3b"},
        conditioner={"repository": "test/conditioner", "revision": "test"},
        tokenizer={"repository": "test/tokenizer", "revision": "test"},
        dtype=torch.float32,
        base_seed=42,
    )
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(module, "prepare_wan_rollout_assets", lambda *args, **kwargs: assets)
    monkeypatch.setattr(module, "load_causal_wan_1p3b", lambda *args, **kwargs: _TinyCausalWan())
    loaded_recipes: list[str] = []

    def load_score(*, native_recipe, **kwargs):
        del kwargs
        loaded_recipes.append(native_recipe.model_id)
        return _TinyWanScoreAdapter()

    monkeypatch.setattr(module, "load_wan_role_adapter", load_score)
    source = _PromptSource()
    unconditional = SimpleNamespace(
        tensors={"context": torch.zeros(512, 4096)},
        artifact=SimpleNamespace(to_dict=lambda: {"branch": "unconditional"}),
    )
    monkeypatch.setattr(
        module,
        "build_wan_rollout_source",
        lambda *args, **kwargs: SimpleNamespace(
            loader=source,
            generator=torch.Generator().manual_seed(42),
            unconditional=unconditional,
        ),
    )

    run = materialize_wan_self_forcing_training_run(recipe, device="cpu")
    try:
        assert loaded_recipes == ["wan2.1-t2v-14b", "wan2.1-t2v-1.3b"]
        assert run.roles.student_checkpoint.checkpoint == SELF_FORCING_ODE_CHECKPOINT
        assert run.roles.real_score_checkpoint.checkpoint.repo_id == "Wan-AI/Wan2.1-T2V-14B"
        assert run.roles.fake_score_checkpoint.checkpoint.repo_id == "Wan-AI/Wan2.1-T2V-1.3B"
        assert run.run_schema == "worldfoundry-wan-self-forcing-run"
    finally:
        run.close()
