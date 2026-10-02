from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.assembly import (  # noqa: E402
    NativeDiffusionAssembler,
)
from worldfoundry.base_models.diffusion_model.components import (  # noqa: E402
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserOutput,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.hunyuan_video import (  # noqa: E402
    HunyuanVideo15Denoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.hunyuanvideo_1_5_transformer import (  # noqa: E402
    HunyuanVideo_1_5_DiffusionTransformer,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original import (  # noqa: E402
    HunyuanVideoDiT,
)
from worldfoundry.training.distributed.parallel import ParallelPlan  # noqa: E402
from worldfoundry.training.engine.hunyuan_video import (  # noqa: E402
    apply_hunyuan_video_activation_checkpointing,
    apply_hunyuan_video_tuning,
    audit_hunyuan_video_lora_targets,
    build_hunyuan_video_diffusion_nft_stack,
    build_hunyuan_video_flow_policy_materialization,
    hunyuan_video_rl_profile,
    load_hunyuan_video_role_adapter,
    validate_hunyuan_video_flow_policy_recipe,
)
from worldfoundry.training.models.hunyuan_video import (  # noqa: E402
    HunyuanVideoTrainAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.dance_grpo import (  # noqa: E402
    NativeDanceGRPOEngine,
)
from worldfoundry.training.post_training.rl.algorithms.flow_dppo import (  # noqa: E402
    NativeFlowDPPOEngine,
)
from worldfoundry.training.post_training.rl.algorithms.flow_grpo import (  # noqa: E402
    NativeFlowGRPOEngine,
)
from worldfoundry.training.post_training.rl.algorithms.mix_grpo import (  # noqa: E402
    NativeMixGRPOEngine,
)
from worldfoundry.training.post_training.rl.batching import (  # noqa: E402
    flow_rollout_batch_from_prompts,
)
from worldfoundry.training.post_training.rl.contracts import RolloutPrompt  # noqa: E402
from worldfoundry.training.post_training.shared.prediction import (  # noqa: E402
    NativeFlowPredictionAdapter,
)
from worldfoundry.training.recipes.post_training.recipe import (  # noqa: E402
    PostTrainingRecipe,
)


class _SignatureGraph(nn.Module):
    def __init__(self, gain: float, *, use_meanflow: bool = False) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))
        self.config = SimpleNamespace(use_meanflow=use_meanflow)


class _SignatureDenoiser:
    def __init__(self, gain: float, *, use_meanflow: bool = False) -> None:
        self.model = _SignatureGraph(gain, use_meanflow=use_meanflow)
        self.calls = []

    def __call__(self, model_input):
        self.calls.append(model_input)
        return DenoiserOutput(sample=model_input.latents * self.model.gain)


def _adapter(model_recipe: str, gain: float = 0.2) -> HunyuanVideoTrainAdapter:
    return HunyuanVideoTrainAdapter(
        _SignatureDenoiser(gain),
        model_recipe=model_recipe,
    )


def _algorithm(algorithm_type: str) -> dict[str, object]:
    value: dict[str, object] = {
        "type": algorithm_type,
        "sigmas": [1.0, 0.6, 0.0],
        "guidance_scale": 1.0,
        "eta": 0.2,
        "group_size": 2,
        "trajectory_dtype": "float32",
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        },
        "reward_model": {"type": "videoalign"},
    }
    if algorithm_type in {"flow-grpo", "flow-dppo"}:
        value["sde_step_indices"] = [0]
    elif algorithm_type == "dance-grpo":
        value.update(
            {
                "sde_step_indices": [0, 1],
                "transition_strategy": "constant-diffusion",
                "init_same_noise": True,
                "old_log_prob_source": "rollout",
                "advantage_normalization": "group-sample-std",
            }
        )
    elif algorithm_type == "mix-grpo":
        value.update(
            {
                "sde_window": {"window_size": 1, "iterations_per_window": 1},
                "init_same_noise": True,
                "old_log_prob_source": "rollout",
                "advantage_normalization": "group-sample-std",
            }
        )
    return value


def _recipe(
    model_recipe: str = "hunyuanvideo-t2v",
    algorithm_type: str = "flow-grpo",
) -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-post-training",
            "execution_owner": "worldfoundry-native",
            "run": {"id": "tiny-hunyuan-video", "output_dir": "unused"},
            "model": {"recipe": model_recipe, "checkpoint": "tiny-policy"},
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "shuffle": False,
                "options": {
                    "generation": {"height": 16, "width": 16, "num_frames": 1},
                    "rollout_forward_batch_size": 1,
                    "replay_microbatch_size": 1,
                },
            },
            "algorithm": _algorithm(algorithm_type),
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
            "runtime": {
                "param_dtype": "float32",
                "reduce_dtype": "float32",
                "activation_checkpoint": "none",
            },
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


@pytest.mark.parametrize(
    ("model_recipe", "channels", "packed_channels"),
    (
        ("hunyuanvideo-t2v", 16, None),
        ("hunyuanvideo-1.5-t2v", 32, 33),
    ),
)
def test_hunyuan_video_native_signature_uses_scaled_timestep_and_exact_packing(
    model_recipe: str,
    channels: int,
    packed_channels: int | None,
) -> None:
    adapter = _adapter(model_recipe)
    prediction = NativeFlowPredictionAdapter(adapter)
    latents = torch.randn(2, channels, 1, 1, 1)
    output = prediction.predict_velocity(
        latents,
        torch.tensor([0.75, 0.25]),
        sample_ids=("sample-a", "sample-b"),
        conditioning={},
        training=True,
    )
    torch.testing.assert_close(output, latents * adapter.trainable_module.gain)
    call = adapter.denoiser.calls[-1]
    torch.testing.assert_close(call.timestep, torch.tensor([750.0, 250.0]))
    if packed_channels is None:
        assert "condition_latents" not in call.conditioning
    else:
        condition = call.conditioning["condition_latents"]
        assert condition.shape == (2, packed_channels, 1, 1, 1)
        assert torch.count_nonzero(condition) == 0


def _tiny_hunyuan_video_1p5_graph(
    *,
    text_projection: str = "linear",
) -> HunyuanVideo_1_5_DiffusionTransformer:
    return HunyuanVideo_1_5_DiffusionTransformer(
        patch_size=[1, 1, 1],
        in_channels=32,
        out_channels=32,
        hidden_size=16,
        heads_num=2,
        mlp_width_ratio=2.0,
        mm_double_blocks_depth=1,
        mm_single_blocks_depth=1,
        rope_dim_list=[2, 2, 4],
        qk_norm=True,
        guidance_embed=False,
        use_meanflow=False,
        text_projection=text_projection,
        use_attention_mask=True,
        text_states_dim=8,
        text_states_dim_2=4,
        text_pool_type=None,
        attn_mode="torch",
        glyph_byT5_v2=False,
        vision_projection="none",
        vision_states_dim=3,
    )


def test_hunyuan_video_1p5_real_native_graph_runs_65_to_32_and_backpropagates() -> None:
    model = _tiny_hunyuan_video_1p5_graph()
    adapter = HunyuanVideoTrainAdapter(
        HunyuanVideo15Denoiser(model, image_to_video=False),
        model_recipe="hunyuanvideo-1.5-t2v",
    )
    prediction = NativeFlowPredictionAdapter(adapter)
    latents = torch.randn(2, 32, 1, 1, 1)
    output = prediction.predict_velocity(
        latents,
        torch.tensor([0.8, 0.3]),
        sample_ids=("sample-a", "sample-b"),
        conditioning={
            "text_states": torch.randn(2, 2, 8),
            "text_mask": torch.ones(2, 2, dtype=torch.long),
            "byt5_text_states": torch.zeros(2, 1, 1472),
            "byt5_text_mask": torch.zeros(2, 1, dtype=torch.long),
            "vision_states": torch.zeros(2, 1, 3),
        },
        training=True,
    )
    assert model.img_in.proj.in_channels == 65
    assert output.shape == latents.shape
    output.sum().backward()
    assert model.final_layer.linear.weight.grad is not None


def test_hunyuan_video_1p5_lora_checkpointing_reenters_the_real_graph() -> None:
    mapping = _recipe("hunyuanvideo-1.5-t2v").to_dict()
    mapping["tuning"] = {
        "mode": "lora",
        "preset": "hunyuanvideo-1.5-attention-mlp",
        "rank": 2,
        "alpha": 2,
    }
    mapping["export"] = {"format": "peft"}
    recipe = PostTrainingRecipe.from_mapping(mapping)
    model = _tiny_hunyuan_video_1p5_graph()
    adapter = HunyuanVideoTrainAdapter(
        HunyuanVideo15Denoiser(model, image_to_video=False),
        model_recipe="hunyuanvideo-1.5-t2v",
    )
    application = apply_hunyuan_video_tuning(recipe, adapter)
    assert application is not None
    assert len(application.targeted_module_names) == 17
    apply_hunyuan_video_activation_checkpointing(adapter)
    assert type(model.double_blocks[0]) in adapter.fsdp_block_classes

    prediction = NativeFlowPredictionAdapter(adapter)
    latents = torch.randn(2, 32, 1, 1, 1)
    output = prediction.predict_velocity(
        latents,
        0.5,
        sample_ids=("sample-a", "sample-b"),
        conditioning={
            "text_states": torch.randn(2, 2, 8),
            "text_mask": torch.ones(2, 2, dtype=torch.long),
            "byt5_text_states": torch.zeros(2, 1, 1472),
            "byt5_text_mask": torch.zeros(2, 1, dtype=torch.long),
            "vision_states": torch.zeros(2, 1, 3),
        },
        training=True,
    )
    output.sum().backward()
    assert all(
        parameter.grad is not None for parameter in adapter.trainable_module.parameters() if parameter.requires_grad
    )


@pytest.mark.parametrize(
    ("algorithm_type", "engine_type"),
    (
        ("flow-grpo", NativeFlowGRPOEngine),
        ("flow-dppo", NativeFlowDPPOEngine),
        ("dance-grpo", NativeDanceGRPOEngine),
        ("mix-grpo", NativeMixGRPOEngine),
    ),
)
def test_hunyuan_video_routes_every_shared_flow_policy_algorithm(
    algorithm_type: str,
    engine_type: type,
) -> None:
    runtime = build_hunyuan_video_flow_policy_materialization(
        _recipe(algorithm_type=algorithm_type),
        policy=_adapter("hunyuanvideo-t2v"),
        fused_adamw=False,
    )
    assert isinstance(runtime.stack.engine, engine_type)


@pytest.mark.parametrize(
    ("model_recipe", "channels"),
    (
        ("hunyuanvideo-t2v", 16),
        ("hunyuanvideo-1.5-t2v", 32),
    ),
)
def test_hunyuan_video_rollout_replay_and_backward_reenter_each_native_contract(
    model_recipe: str,
    channels: int,
) -> None:
    adapter = _adapter(model_recipe)
    runtime = build_hunyuan_video_flow_policy_materialization(
        _recipe(model_recipe),
        policy=adapter,
        fused_adamw=False,
    )
    trajectory = runtime.stack.sampler.sample(
        torch.randn(2, channels, 1, 1, 1),
        runtime.stack.sigmas,
        sample_ids=("sample-a", "sample-b"),
        group_ids=("prompt", "prompt"),
        conditioning={},
        policy_revision="tiny-policy",
        sde_step_indices=runtime.stack.sde_step_indices,
        generator=torch.Generator().manual_seed(19),
    )
    replay = runtime.stack.replay.replay(trajectory, training=True)
    (-replay.log_probs.mean()).backward()
    assert adapter.trainable_module.gain.grad is not None
    assert torch.isfinite(adapter.trainable_module.gain.grad)


class _Reward:
    reward_ids = ("video_quality", "motion_quality", "text_alignment")

    def score(self, terminal_latents):
        return {reward_id: torch.ones(terminal_latents.batch_size) for reward_id in self.reward_ids}


def _diffusion_nft_recipe() -> PostTrainingRecipe:
    mapping = _recipe("hunyuanvideo-1.5-t2v").to_dict()
    mapping["data"]["options"].pop("replay_microbatch_size")
    mapping["algorithm"] = {
        "type": "diffusion-nft",
        "collection": {
            "sigmas": [1.0, 0.6, 0.0],
            "group_size": 2,
            "guidance_scale": 1.0,
            "latent_dtype": "float32",
        },
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        },
        "reward_model": {"type": "videoalign"},
    }
    return PostTrainingRecipe.from_mapping(mapping)


def test_hunyuan_video_diffusion_nft_collects_with_independent_native_roles() -> None:
    policy = _adapter("hunyuanvideo-1.5-t2v", gain=0.15)
    old_policy = copy.deepcopy(policy)
    runtime = build_hunyuan_video_diffusion_nft_stack(
        _diffusion_nft_recipe(),
        policy=policy,
        old_policy=old_policy,
        reward_adapter=_Reward(),
        fused_adamw=False,
    )
    batch = flow_rollout_batch_from_prompts(
        (
            RolloutPrompt(
                prompt_id="prompt",
                prompt="a moving object",
                conditions={"text_states": torch.zeros(1, 8)},
                generation={"height": 16, "width": 16, "num_frames": 1},
            ),
        ),
        group_size=2,
        policy_revision="tiny-policy",
        latent_shape=runtime.latent_shape,
        sigmas=runtime.stack.sigmas,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(23),
    )
    terminal = runtime.stack.collector.collect(batch, collection_id="collection")
    assert terminal.clean_latents.shape == (2, 32, 1, 1, 1)
    assert policy.trainable_module.gain.requires_grad
    assert not old_policy.trainable_module.gain.requires_grad


def test_hunyuan_video_lora_targets_follow_each_actual_native_graph() -> None:
    original_denoiser = _SignatureDenoiser(0.2)
    original_denoiser.model = HunyuanVideoDiT(
        hidden_size=24,
        text_dim=8,
        num_double_blocks=1,
        num_single_blocks=1,
    )
    original = HunyuanVideoTrainAdapter(
        original_denoiser,
        model_recipe="hunyuanvideo-t2v",
    )
    refined = HunyuanVideoTrainAdapter(
        HunyuanVideo15Denoiser(_tiny_hunyuan_video_1p5_graph(), image_to_video=False),
        model_recipe="hunyuanvideo-1.5-t2v",
    )
    original_audit = audit_hunyuan_video_lora_targets(original)
    refined_audit = audit_hunyuan_video_lora_targets(refined)
    assert len(original_audit.module_names) == 12
    assert "double_blocks.0.component_a.to_qkv" in original_audit.module_names
    assert len(refined_audit.module_names) == 17
    assert "double_blocks.0.img_attn_q" in refined_audit.module_names
    assert "single_blocks.0.linear2.fc" in refined_audit.module_names

    refined_with_token_refiner = HunyuanVideoTrainAdapter(
        HunyuanVideo15Denoiser(
            _tiny_hunyuan_video_1p5_graph(text_projection="single_refiner"),
            image_to_video=False,
        ),
        model_recipe="hunyuanvideo-1.5-t2v",
    )
    refiner_audit = audit_hunyuan_video_lora_targets(refined_with_token_refiner)
    assert len(refiner_audit.module_names) == 25
    assert "txt_in.individual_token_refiner.blocks.0.self_attn_qkv" in refiner_audit.module_names


@pytest.mark.parametrize(
    ("filename", "model_recipe"),
    (
        ("hunyuan_video_flow_grpo.yaml", "hunyuanvideo-t2v"),
        ("hunyuan_video_1p5_flow_grpo.yaml", "hunyuanvideo-1.5-t2v"),
    ),
)
def test_hunyuan_video_configs_match_profiles_and_scale_with_world_size(
    filename: str,
    model_recipe: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training" / filename)
    algorithm, plan = validate_hunyuan_video_flow_policy_recipe(recipe)
    profile = hunyuan_video_rl_profile(model_recipe)
    assert recipe.model.recipe == profile.model_recipe
    assert algorithm.sigmas == profile.sigmas
    assert algorithm.group_size == profile.group_size
    assert algorithm.sigma_max == profile.sigma_max == algorithm.sigmas[1]
    assert algorithm.advantage_normalization == "group-mean-global-population-std"
    assert algorithm.guidance_scale == profile.guidance_scale
    assert algorithm.old_log_prob_source == profile.old_log_prob_source == "rollout"
    assert algorithm.reward_model.type == "remote"
    assert algorithm.reward_model.reward_ids == ("videopickscore",)
    assert dict(algorithm.reward_weights) == {"videopickscore": 1.0}
    assert tuple(plan.generation.values()) == profile.generation
    assert plan.global_prompt_batch_size == profile.global_prompt_batch_size
    assert ParallelPlan.resolve(recipe.distributed, world_size=3).dp_shard == 3


def test_hunyuan_video_1p5_zero_guidance_keeps_the_single_conditional_policy() -> None:
    payload = _recipe("hunyuanvideo-1.5-t2v").to_dict()
    payload["algorithm"]["guidance_scale"] = 0.0
    runtime = build_hunyuan_video_flow_policy_materialization(
        PostTrainingRecipe.from_mapping(payload),
        policy=_adapter("hunyuanvideo-1.5-t2v"),
        fused_adamw=False,
    )
    assert runtime.stack.sampler.policy is runtime.prediction
    assert runtime.stack.replay.policy is runtime.prediction


def test_hunyuan_video_1p5_rl_loads_the_official_480p_training_graph(monkeypatch) -> None:
    captured = {}
    denoiser_key = ComponentKey(ComponentKind.DENOISER)

    def build_components(self, recipe, **kwargs):
        del self, recipe
        captured.update(kwargs)
        return {denoiser_key: _SignatureDenoiser(0.2)}

    monkeypatch.setattr(NativeDiffusionAssembler, "build_components", build_components)
    adapter = load_hunyuan_video_role_adapter(
        model_recipe="hunyuanvideo-1.5-t2v",
        checkpoint=None,
        device="cpu",
        dtype=torch.float32,
    )
    assert adapter.model_recipe == "hunyuanvideo-1.5-t2v"
    checkpoint = captured["checkpoint_overrides"]["transformer"]
    assert checkpoint.repo_id == "tencent/HunyuanVideo-1.5"
    assert checkpoint.files == ("transformer/480p_t2v/diffusion_pytorch_model.safetensors",)
    assert captured["component_options"][denoiser_key]["config_path"] == ("transformer/480p_t2v/config.json")


def test_hunyuan_video_graphs_cannot_be_renamed_or_use_meanflow_replay() -> None:
    with pytest.raises(ValueError, match="unsupported HunyuanVideo"):
        HunyuanVideoTrainAdapter(
            _SignatureDenoiser(0.2),
            model_recipe="hunyuanvideo-1.5-i2v",
        )
    with pytest.raises(ValueError, match="current and next timesteps"):
        HunyuanVideoTrainAdapter(
            _SignatureDenoiser(0.2, use_meanflow=True),
            model_recipe="hunyuanvideo-1.5-t2v",
        )
