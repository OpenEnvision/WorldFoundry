from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from torch import nn  # noqa: E402

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import (  # noqa: E402
    LTXModel,
    LTXModelType,
)
from worldfoundry.training.engine.ltx.flow_policy import (  # noqa: E402
    ltx_flow_policy_profile,
    ltx_flow_policy_sigmas,
    materialize_ltx_diffusion_nft_stack,
    materialize_ltx_flow_policy_stack,
    validate_ltx_flow_policy_recipe,
)
from worldfoundry.training.engine.ltx.flow_policy_roles import (  # noqa: E402
    audit_ltx_policy_lora_targets,
    load_ltx_policy_adapter,
)
from worldfoundry.training.engine.ltx.trajectory import (  # noqa: E402
    LTX_AUDIO_TRAJECTORY,
    LTX_AUDIO_TRANSITION_MEANS,
    LTX_AUDIO_TRANSITION_SCALES,
    LTXAudioConditionedTrajectorySampler,
)
from worldfoundry.training.engine.video_policy import (  # noqa: E402
    materialize_video_flow_policy_roles,
)
from worldfoundry.training.models.ltx import LTXTrainAdapter  # noqa: E402
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
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe  # noqa: E402


class _TinyLTXModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            cross_attention_dim=16,
            positional_embedding_max_pos=[20, 64, 64],
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_cross_attention_dim=8,
            audio_positional_embedding_max_pos=[20],
        )
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(mean=0.0, std=0.02)
        self.last_video = None
        self.last_audio = None

    def forward(self, *args, **kwargs):
        self.last_video = kwargs["video"]
        self.last_audio = kwargs["audio"]
        return self.velocity_model(*args, **kwargs)


def _adapter(seed: int) -> LTXTrainAdapter:
    torch.manual_seed(seed)
    return LTXTrainAdapter(
        SimpleNamespace(model=_TinyLTXModule()),
        expected_latent_channels=4,
        temporal_compression=8,
        spatial_compression=32,
        default_fps=24.0,
        first_frame_conditioning_probability=0.0,
        causal_positions=True,
    )


def _algorithm(algorithm_type: str) -> dict[str, object]:
    sigmas = list(ltx_flow_policy_sigmas(2))
    value: dict[str, object] = {
        "type": algorithm_type,
        "sigmas": sigmas,
        "guidance_scale": 1.0,
        "eta": 0.7,
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


def _recipe(algorithm_type: str = "flow-grpo", *, model: str = "ltx-2-i2v") -> PostTrainingRecipe:
    algorithm = _algorithm(algorithm_type)
    num_frames = 9
    rollout_forward_batch_size = None
    if model == "ltx-2.3-i2v":
        num_frames = 33
        rollout_forward_batch_size = 2
        algorithm.update(
            {
                "trajectory_dtype": "float16",
                "group_size": 4,
            }
        )
        if algorithm_type not in {"dance-grpo", "mix-grpo"}:
            algorithm["old_log_prob_source"] = "replay"
    return PostTrainingRecipe.from_mapping(
        {
            "run": {"id": f"ltx-{algorithm_type}", "output_dir": "unused"},
            "model": {"recipe": model, "checkpoint": "default"},
            "tuning": {"mode": "full"},
            "export": {"format": "safetensors"},
            "data": {
                "manifest": "unused.jsonl",
                "cache": "unused-cache",
                "shuffle": False,
                "options": {
                    "generation": {"height": 64, "width": 64, "num_frames": num_frames},
                    "target_fps": 24.0,
                    "audio_joint_sde": model == "ltx-2.3-i2v",
                    "rollout_forward_batch_size": rollout_forward_batch_size,
                },
            },
            "algorithm": algorithm,
            "optimizer": {"type": "adamw", "learning_rate": 1.0e-4, "max_grad_norm": 1.0},
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
        }
    )


def _conditions() -> dict[str, torch.Tensor]:
    return {
        "video_context": torch.randn(2, 3, 16),
        "audio_context": torch.randn(2, 3, 8),
        "context_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
    }


def test_ltx_policy_schedule_matches_diffusers() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
        shift_terminal=0.1,
    )
    scheduler.set_timesteps(
        num_inference_steps=10,
        sigmas=np.linspace(1.0, 0.1, 10),
        mu=2.05,
    )
    torch.testing.assert_close(
        torch.tensor(ltx_flow_policy_sigmas()),
        scheduler.sigmas.cpu(),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_ltx_policy_profiles_match_the_released_recipes() -> None:
    ltx2 = ltx_flow_policy_profile("ltx-2-i2v")
    assert ltx2.generation == {"height": 512, "width": 768, "num_frames": 9}
    assert (ltx2.group_size, ltx2.lora_alpha) == (16, 256)
    assert (ltx2.old_log_prob_source, ltx2.trajectory_dtype) == ("rollout", "float32")
    assert ltx2.rollout_forward_batch_size == 1
    assert not ltx2.audio_joint_sde

    ltx23 = ltx_flow_policy_profile("ltx-2.3-i2v")
    assert ltx23.generation == {"height": 512, "width": 768, "num_frames": 33}
    assert (ltx23.group_size, ltx23.lora_alpha) == (4, 64)
    assert (ltx23.old_log_prob_source, ltx23.trajectory_dtype) == ("replay", "float16")
    assert ltx23.rollout_forward_batch_size == 2
    assert ltx23.audio_joint_sde


@pytest.mark.parametrize(
    ("algorithm_type", "engine_type"),
    (
        ("flow-grpo", NativeFlowGRPOEngine),
        ("flow-dppo", NativeFlowDPPOEngine),
        ("dance-grpo", NativeDanceGRPOEngine),
        ("mix-grpo", NativeMixGRPOEngine),
    ),
)
def test_ltx_materializer_routes_shared_flow_algorithms(
    algorithm_type: str,
    engine_type: type,
) -> None:
    runtime = materialize_ltx_flow_policy_stack(
        _recipe(algorithm_type),
        policy=_adapter(11),
        initial_policy_revision="ltx-policy",
        fused_adamw=False,
    )
    assert runtime.latent_shape == (4, 2, 2, 2)
    assert isinstance(runtime.stack.engine, engine_type)


def test_ltx_rollout_replay_reenters_native_model_and_backpropagates() -> None:
    adapter = _adapter(17)
    runtime = materialize_ltx_flow_policy_stack(
        _recipe(),
        policy=adapter,
        initial_policy_revision="ltx-policy",
        fused_adamw=False,
    )
    trajectory = runtime.stack.sampler.sample(
        torch.randn(2, *runtime.latent_shape),
        runtime.stack.sigmas,
        sample_ids=("sample-0", "sample-1"),
        group_ids=("prompt", "prompt"),
        conditioning=_conditions(),
        policy_revision="ltx-policy",
        sde_step_indices=runtime.stack.sde_step_indices,
        generator=torch.Generator().manual_seed(23),
    )
    replay = runtime.stack.replay.replay(trajectory, training=True)
    torch.testing.assert_close(replay.log_probs.detach(), trajectory.old_log_probs, atol=2.0e-5, rtol=2.0e-5)
    replay.log_probs.mean().backward()
    assert any(parameter.grad is not None for parameter in adapter.trainable_module.parameters())
    assert adapter.denoiser.model.last_video.positions.shape[-2] == 8
    assert adapter.denoiser.model.last_audio.positions.shape[-2] == 9
    torch.testing.assert_close(adapter.denoiser.model.last_video.sigma, torch.full((2,), 1.0))
    torch.testing.assert_close(adapter.denoiser.model.last_audio.sigma, torch.full((2,), 1.0))
    assert trajectory.conditioning[LTX_AUDIO_TRAJECTORY].shape == (2, 3, 9, 4)


def test_ltx_video_velocity_depends_on_the_current_audio_state() -> None:
    runtime = materialize_ltx_flow_policy_stack(
        _recipe(),
        policy=_adapter(29),
        initial_policy_revision="ltx-policy",
        fused_adamw=False,
    )
    video = torch.randn(2, *runtime.latent_shape)
    audio = torch.randn(2, 9, 4)
    sigma = torch.full((2,), 0.6)
    conditioning = _conditions()
    first, _ = runtime.prediction.predict_joint_velocity(
        video,
        audio,
        sigma,
        sample_ids=("sample-0", "sample-1"),
        conditioning=conditioning,
        training=False,
    )
    second, _ = runtime.prediction.predict_joint_velocity(
        video,
        audio + 1.0,
        sigma,
        sample_ids=("sample-0", "sample-1"),
        conditioning=conditioning,
        training=False,
    )
    assert not torch.equal(first, second)


def test_ltx23_joint_av_policy_replays_and_updates_from_joint_log_prob() -> None:
    runtime = materialize_ltx_flow_policy_stack(
        _recipe(model="ltx-2.3-i2v"),
        policy=_adapter(30),
        initial_policy_revision="ltx-policy",
        fused_adamw=False,
    )
    initial = torch.randn(2, *runtime.latent_shape)
    generator = torch.Generator().manual_seed(31)
    trajectory = runtime.stack.sampler.sample(
        initial,
        runtime.stack.sigmas,
        sample_ids=("sample-0", "sample-1"),
        group_ids=("prompt", "prompt"),
        conditioning=_conditions(),
        policy_revision="ltx-policy",
        sde_step_indices=runtime.stack.sde_step_indices,
        generator=generator,
    )
    replay = runtime.stack.replay.replay(trajectory, training=False)
    torch.testing.assert_close(replay.log_probs, trajectory.old_log_probs, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        replay.audio_transition_means,
        trajectory.conditioning[LTX_AUDIO_TRANSITION_MEANS],
    )
    torch.testing.assert_close(
        replay.audio_transition_scales,
        trajectory.conditioning[LTX_AUDIO_TRANSITION_SCALES],
    )
    video = trajectory.latents[:, 0]
    next_video = trajectory.latents[:, 1]
    audio = trajectory.conditioning[LTX_AUDIO_TRAJECTORY][:, 0]
    next_audio = trajectory.conditioning[LTX_AUDIO_TRAJECTORY][:, 1]
    sigma = trajectory.sigmas[0].expand(trajectory.batch_size)
    sigma_next = trajectory.sigmas[1].expand(trajectory.batch_size)
    video_velocity, audio_velocity = runtime.prediction.predict_joint_velocity(
        video,
        audio,
        sigma,
        sample_ids=trajectory.sample_ids,
        conditioning=trajectory.conditioning,
        training=False,
    )
    video_transition = runtime.stack.transition_strategy.step(
        video_velocity,
        video,
        sigma,
        sigma_next,
        next_sample=next_video,
        trajectory_dtype=trajectory.latents.dtype,
    )
    audio_transition = runtime.stack.transition_strategy.step(
        audio_velocity,
        audio,
        sigma,
        sigma_next,
        next_sample=next_audio,
        trajectory_dtype=trajectory.latents.dtype,
    )
    video_elements = video[0].numel()
    audio_elements = audio[0].numel()
    expected_joint = (video_transition.log_prob * video_elements + audio_transition.log_prob * audio_elements) / (
        video_elements + audio_elements
    )
    torch.testing.assert_close(trajectory.old_log_probs[:, 0], expected_joint)
    assert replay.transition_means.shape == (2, 1, *runtime.latent_shape)
    before = {name: parameter.detach().clone() for name, parameter in runtime.prediction.module.named_parameters()}
    anchor = runtime.stack.engine.prepare_trajectory(
        trajectory,
        torch.tensor([1.0, -1.0]),
        old_log_prob_source=runtime.stack.old_log_prob_source,
        advantage_epsilon=runtime.stack.advantage_epsilon,
        advantage_normalization=runtime.stack.advantage_normalization,
    )
    result = runtime.stack.engine.train_step(anchor_id=anchor)
    assert result.trajectory_complete
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in runtime.prediction.module.named_parameters()
    )


def test_integrated_video_policy_roles_keep_ltx23_joint_av_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.training.engine.ltx import flow_policy_roles

    monkeypatch.setattr(
        flow_policy_roles,
        "load_ltx_policy_adapter",
        lambda recipe, **kwargs: _adapter(41),
    )
    materialized = materialize_video_flow_policy_roles(
        _recipe(model="ltx-2.3-i2v"),
        device="cpu",
        fused_adamw=False,
    )
    assert isinstance(materialized.stack.sampler, LTXAudioConditionedTrajectorySampler)
    assert materialized.stack.sampler.audio_joint_sde


def test_integrated_ltx_reference_does_not_reuse_policy_checkpoint_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.training.engine.ltx import flow_policy_roles

    payload = _recipe().to_dict()
    payload["algorithm"]["reference_kl_weight"] = 0.1
    payload["algorithm"]["reference_checkpoint"] = "default"
    calls: list[dict[str, object]] = []

    def load_role(recipe, **kwargs):
        del recipe
        calls.append(dict(kwargs["checkpoint_overrides"]))
        return _adapter(50 + len(calls))

    monkeypatch.setattr(flow_policy_roles, "load_ltx_policy_adapter", load_role)
    materialized = materialize_video_flow_policy_roles(
        PostTrainingRecipe.from_mapping(payload),
        device="cpu",
        checkpoint_overrides={"model": "policy.safetensors"},
        fused_adamw=False,
    )

    assert materialized.reference_policy is not None
    assert calls == [{"model": "policy.safetensors"}, {}]


def test_ltx_models_reject_each_others_audio_transition_semantics() -> None:
    mapping = _recipe().to_dict()
    mapping["data"]["options"]["audio_joint_sde"] = True
    with pytest.raises(ValueError, match="video SDE with audio ODE"):
        validate_ltx_flow_policy_recipe(PostTrainingRecipe.from_mapping(mapping))

    mapping = _recipe(model="ltx-2.3-i2v").to_dict()
    mapping["data"]["options"]["audio_joint_sde"] = False
    recipe = PostTrainingRecipe.from_mapping(mapping)
    with pytest.raises(ValueError, match="joint audio-video SDE"):
        validate_ltx_flow_policy_recipe(recipe)


@pytest.mark.parametrize("algorithm_type", ("flow-dppo", "dance-grpo", "mix-grpo"))
def test_ltx23_rejects_objectives_that_need_an_unrepresented_joint_mean(
    algorithm_type: str,
) -> None:
    with pytest.raises(NotImplementedError, match="joint AV transition means"):
        validate_ltx_flow_policy_recipe(_recipe(algorithm_type, model="ltx-2.3-i2v"))


def test_ltx23_rejects_reference_kl_until_joint_transition_means_are_typed() -> None:
    mapping = _recipe(model="ltx-2.3-i2v").to_dict()
    mapping["algorithm"]["reference_kl_weight"] = 0.1
    mapping["algorithm"]["reference_checkpoint"] = "default"
    with pytest.raises(NotImplementedError, match="joint AV transition-mean"):
        validate_ltx_flow_policy_recipe(PostTrainingRecipe.from_mapping(mapping))


def test_ltx_policy_lora_includes_attention_and_ffn() -> None:
    audit = audit_ltx_policy_lora_targets(_adapter(31).trainable_module)
    suffixes = {name.split("transformer_blocks.0.", 1)[1] for name in audit.module_names}
    assert suffixes == {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }


class _Reward:
    reward_ids = ("video_quality", "motion_quality", "text_alignment")

    def score(self, terminal_latents):
        return {reward_id: torch.ones(terminal_latents.batch_size) for reward_id in self.reward_ids}


def test_ltx_diffusion_nft_rejects_single_state_collection() -> None:
    mapping = _recipe().to_dict()
    mapping["run"] = {"id": "ltx-diffusion-nft", "output_dir": "unused"}
    mapping["algorithm"] = {
        "type": "diffusion-nft",
        "collection": {
            "sigmas": list(ltx_flow_policy_sigmas(2)),
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
    recipe = PostTrainingRecipe.from_mapping(mapping)
    with pytest.raises(NotImplementedError, match="audio trajectory"):
        materialize_ltx_diffusion_nft_stack(
            recipe,
            policy=_adapter(37),
            old_policy=_adapter(38),
            initial_old_policy_revision="ltx-old-policy",
            reward_adapter=_Reward(),
            fused_adamw=False,
        )


@pytest.mark.parametrize(
    "filename",
    ("ltx_2_video_flow_grpo.yaml", "ltx_2p3_video_flow_grpo.yaml"),
)
def test_ltx_policy_configs_parse(filename: str) -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training" / filename)
    algorithm, plan = validate_ltx_flow_policy_recipe(recipe)
    profile = ltx_flow_policy_profile(recipe.model.recipe)
    assert algorithm.sigmas == ltx_flow_policy_sigmas()
    assert plan.generation == profile.generation
    assert plan.audio_joint_sde == profile.audio_joint_sde
    assert algorithm.group_size == profile.group_size
    assert algorithm.old_log_prob_source == profile.old_log_prob_source
    assert algorithm.trajectory_dtype == profile.trajectory_dtype
    assert algorithm.advantage_normalization == "group-population-std"
    assert algorithm.reward_model.type == "remote"
    assert algorithm.reward_model.reward_ids == ("videopickscore",)
    assert dict(algorithm.reward_weights) == {"videopickscore": 1.0}
    assert plan.rollout_forward_batch_size == profile.rollout_forward_batch_size
    assert plan.global_prompt_batch_size == profile.global_prompt_batch_size == 8
    assert recipe.tuning.rank == profile.lora_rank
    assert recipe.tuning.alpha == profile.lora_alpha


@pytest.mark.parametrize(
    ("model_recipe", "filename"),
    (
        ("ltx-2-i2v", "ltx-2-19b-dev.safetensors"),
        ("ltx-2.3-i2v", "ltx-2.3-22b-dev.safetensors"),
    ),
)
def test_ltx_policy_loader_uses_dev_checkpoint_by_default(
    model_recipe: str,
    filename: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler

    captured = None

    class _StopLoad(RuntimeError):
        pass

    def capture_build(self, native_recipe, **kwargs):
        nonlocal captured
        captured = kwargs["checkpoint_overrides"]["model"]
        raise _StopLoad

    monkeypatch.setattr(NativeDiffusionAssembler, "build_components", capture_build)
    with pytest.raises(_StopLoad):
        load_ltx_policy_adapter(
            _recipe(model=model_recipe),
            device="cpu",
            dtype=torch.float32,
        )
    assert captured.files == (filename,)
