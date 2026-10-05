from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.diffusion_model.contracts import (  # noqa: E402
    DenoiserOutput,
)
from worldfoundry.training.api.contracts import ObjectiveBatch  # noqa: E402
from worldfoundry.training.engine.video_policy import (  # noqa: E402
    materialize_video_flow_policy_roles,
)
from worldfoundry.training.engine.wan22 import (  # noqa: E402
    load_wan22_role_adapter,
    materialize_wan22_diffusion_nft_stack,
    materialize_wan22_flow_policy_stack,
    validate_wan22_flow_policy_recipe,
    wan22_flow_policy_profile,
    wan22_flow_policy_sigmas,
    wan22_role_checkpoints,
)
from worldfoundry.training.engine.wan22 import roles as wan22_roles  # noqa: E402
from worldfoundry.training.engine.wan22.tuning import (  # noqa: E402
    audit_wan22_lora_targets,
)
from worldfoundry.training.models import (  # noqa: E402
    Wan22TrainAdapter,
    WanTrainAdapter,
)
from worldfoundry.training.post_training.rl.algorithms.diffusion_nft.builder import (  # noqa: E402
    NativeDiffusionNFTTrainingStack,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (  # noqa: E402
    NativeFlowPolicyTrainingStack,
)
from worldfoundry.training.post_training.rl.transitions.flow_sde import (  # noqa: E402
    flow_match_sigma_schedule,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(2, 2, bias=False)
        self.k = torch.nn.Linear(2, 2, bias=False)
        self.v = torch.nn.Linear(2, 2, bias=False)
        self.o = torch.nn.Linear(2, 2, bias=False)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.cross_attn = _Attention()


class _Expert(torch.nn.Module):
    patch_size = (1, 1, 1)

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])
        self.gain = torch.nn.Parameter(torch.tensor(gain))
        self.attention_compatibility_mode = False

    def set_attention_compatibility_mode(self, enabled: bool) -> None:
        self.attention_compatibility_mode = enabled

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.gain


class _Denoiser:
    def __init__(self, gain: float) -> None:
        self.model = _Expert(gain)
        self.manage_autocast = False
        self.calls = []

    def __call__(self, model_input) -> DenoiserOutput:
        self.calls.append(model_input)
        return DenoiserOutput(sample=self.model(model_input.latents))


def _branch(gain: float) -> WanTrainAdapter:
    return WanTrainAdapter(
        _Denoiser(gain),
        codec=None,
        conditioner=None,
        expected_latent_channels=16,
        expected_text_length=2,
        expected_context_features=3,
        temporal_compression=4,
        spatial_compression=8,
    )


def _adapter() -> Wan22TrainAdapter:
    return Wan22TrainAdapter(_branch(2.0), _branch(3.0), boundary_ratio=0.875)


def _recipe() -> PostTrainingRecipe:
    return PostTrainingRecipe.from_mapping(
        {
            "schema": "worldfoundry-post-training",
            "execution_owner": "worldfoundry-native",
            "run": {"id": "tiny-wan22", "output_dir": "unused"},
            "model": {
                "recipe": "wan2.2-t2v-a14b",
                "checkpoint": "tiny-a14b",
                "options": {"boundary_ratio": 0.875},
            },
            "tuning": {"mode": "full"},
            "data": {
                "manifest": "unused.jsonl",
                "shuffle": False,
                "options": {
                    "generation": {"height": 16, "width": 16, "num_frames": 1},
                    "rollout_forward_batch_size": 1,
                    "replay_microbatch_size": 1,
                },
            },
            "algorithm": {
                "type": "flow-grpo",
                "sigmas": [1.0, 0.8, 0.0],
                "sde_step_indices": [0, 1],
                "transition_strategy": "constant-diffusion",
                "eta": 0.2,
                "updates_per_trajectory": 1,
                "group_size": 2,
                "old_log_prob_source": "replay",
                "trajectory_dtype": "float32",
                "reward_weights": {
                    "video_quality": 1.0,
                    "motion_quality": 1.0,
                    "text_alignment": 1.0,
                },
                "reward_model": {"type": "videoalign"},
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            },
            "runtime": {"param_dtype": "float32", "reduce_dtype": "float32"},
            "distributed": {"backend": "single"},
            "export": {"format": "safetensors"},
        }
    )


def _objective(adapter: Wan22TrainAdapter) -> ObjectiveBatch:
    latents = torch.ones(3, 16, 1, 2, 2)
    sigmas = torch.tensor([0.875, 0.2, 0.95])
    return ObjectiveBatch(
        sample_ids=("boundary", "low", "high"),
        model_input=latents,
        target=latents,
        sigmas=sigmas,
        timesteps=sigmas,
        conditioning={"context": torch.zeros(3, 2, 3)},
        metadata={"prediction_type": adapter.prediction_type},
    )


class _TerminalReward:
    reward_ids = ("video_quality", "motion_quality", "text_alignment")

    def score(self, terminal_latents):
        values = terminal_latents.clean_latents.flatten(1).mean(dim=1)
        return {name: values for name in self.reward_ids}


def test_wan22_routes_boundary_and_mixed_batch_without_reordering() -> None:
    adapter = _adapter()
    prediction = adapter.forward_model(_objective(adapter), training=True)
    assert torch.equal(prediction[:, 0, 0, 0, 0], torch.tensor([2.0, 3.0, 2.0]))
    torch.testing.assert_close(
        adapter.high_noise.denoiser.calls[-1].timestep,
        torch.tensor([875.0, 950.0]),
    )
    torch.testing.assert_close(
        adapter.low_noise.denoiser.calls[-1].timestep,
        torch.tensor([200.0]),
    )
    prediction.sum().backward()
    assert adapter.high_noise.trainable_module.gain.grad is not None
    assert adapter.low_noise.trainable_module.gain.grad is not None


def test_wan22_lora_audit_covers_every_attention_role_in_both_experts() -> None:
    audit = audit_wan22_lora_targets(_adapter().trainable_module)
    assert audit.preset == "wan22-dual-attention"
    assert audit.block_count == 2
    assert len(audit.module_names) == 16
    assert any(name.startswith("high_noise.") for name in audit.module_names)
    assert any(name.startswith("low_noise.") for name in audit.module_names)


def test_wan22_role_loader_materializes_both_official_expert_subtrees(monkeypatch) -> None:
    checkpoints = wan22_role_checkpoints()
    assert checkpoints.high_noise.files == ("high_noise_model/diffusion_pytorch_model.safetensors.index.json",)
    assert checkpoints.low_noise.files == ("low_noise_model/diffusion_pytorch_model.safetensors.index.json",)
    seen = []

    def build(context):
        seen.append((context.key.name, context.require_checkpoint("weights")))
        return _Denoiser(2.0 if context.key.name == "high-noise" else 3.0)

    monkeypatch.setattr(wan22_roles, "build_wan22_t2v_a14b_denoiser", build)
    adapter = load_wan22_role_adapter(
        checkpoints=checkpoints,
        device="cpu",
        dtype=torch.float32,
    )
    assert isinstance(adapter, Wan22TrainAdapter)
    assert [name for name, _ in seen] == ["high-noise", "low-noise"]
    assert seen[0][1] is checkpoints.high_noise
    assert seen[1][1] is checkpoints.low_noise


def test_wan22_rollout_replay_and_backward_cross_the_expert_boundary() -> None:
    adapter = _adapter()
    runtime = materialize_wan22_flow_policy_stack(
        _recipe(),
        policy=adapter,
        initial_policy_revision="tiny-policy",
        fused_adamw=False,
    )
    assert isinstance(runtime.stack, NativeFlowPolicyTrainingStack)
    assert runtime.latent_shape == (16, 1, 2, 2)

    initial = torch.randn(2, *runtime.latent_shape)
    trajectory = runtime.stack.sampler.sample(
        initial,
        runtime.stack.sigmas,
        sample_ids=("sample-a", "sample-b"),
        group_ids=("prompt", "prompt"),
        conditioning={"context": torch.zeros(2, 2, 3)},
        policy_revision="tiny-policy",
        sde_step_indices=runtime.stack.sde_step_indices,
        generator=torch.Generator().manual_seed(12),
    )
    replay = runtime.stack.replay.replay(trajectory, training=True)
    assert torch.equal(replay.log_probs.detach(), trajectory.old_log_probs)
    (-replay.log_probs.sum()).backward()
    assert adapter.high_noise.trainable_module.gain.grad is not None
    assert adapter.low_noise.trainable_module.gain.grad is not None


def test_wan22_lora_materialization_keeps_dual_routing_callable() -> None:
    pytest.importorskip("peft")
    payload = _recipe().to_dict()
    payload["tuning"] = {
        "mode": "lora",
        "preset": "wan22-dual-attention",
        "rank": 2,
        "alpha": 4,
    }
    payload["export"] = {"format": "peft"}
    adapter = _adapter()
    runtime = materialize_wan22_flow_policy_stack(
        PostTrainingRecipe.from_mapping(payload),
        policy=adapter,
        initial_policy_revision="tiny-policy",
        fused_adamw=False,
    )
    assert runtime.policy_tuning is not None
    targets = runtime.policy_tuning.targeted_module_names
    assert any(name.startswith("high_noise.") for name in targets)
    assert any(name.startswith("low_noise.") for name in targets)
    assert adapter.forward_model(_objective(adapter), training=True).shape == (
        3,
        16,
        1,
        2,
        2,
    )


def test_wan22_diffusion_nft_uses_the_same_dual_expert_prediction_surface() -> None:
    payload = _recipe().to_dict()
    payload["algorithm"] = {
        "type": "diffusion-nft",
        "collection": {
            "sigmas": [1.0, 0.8, 0.0],
            "group_size": 2,
            "guidance_scale": 1.0,
            "latent_dtype": "float32",
        },
        "old_policy_refresh": {"decay": "copy", "interval": 1},
        "reward_weights": {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        },
        "reward_model": {"type": "videoalign"},
    }
    runtime = materialize_wan22_diffusion_nft_stack(
        PostTrainingRecipe.from_mapping(payload),
        policy=_adapter(),
        old_policy=_adapter(),
        initial_old_policy_revision="tiny-old-policy",
        reward_adapter=_TerminalReward(),
        fused_adamw=False,
    )
    assert isinstance(runtime.stack, NativeDiffusionNFTTrainingStack)
    assert runtime.stack.sigmas == (1.0, 0.8, 0.0)
    assert runtime.latent_shape == (16, 1, 2, 2)


@pytest.mark.parametrize("algorithm_type", ["flow-dppo", "dance-grpo", "mix-grpo"])
def test_wan22_materializes_all_shared_flow_policy_algorithms(algorithm_type: str) -> None:
    payload = _recipe().to_dict()
    algorithm = payload["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm["type"] = algorithm_type
    if algorithm_type == "flow-dppo":
        for field in ("clip_range", "clip_schedule", "clip_schedule_steps"):
            algorithm.pop(field)
        algorithm.update({"kl_mask_threshold": 0.01, "add_kl_coefficient": True})
    elif algorithm_type == "dance-grpo":
        algorithm.update(
            {
                "init_same_noise": True,
                "old_log_prob_source": "rollout",
                "advantage_normalization": "group-sample-std",
                "update_timestep_fraction": 1.0,
            }
        )
    else:
        algorithm.update(
            {
                "sde_step_indices": None,
                "sde_window": {
                    "window_size": 1,
                    "iterations_per_window": 1,
                    "stride": 1,
                    "initial_index": 0,
                    "rollback": True,
                },
                "transition_strategy": "variance-preserving",
                "sigma_max": 0.8,
                "init_same_noise": True,
                "old_log_prob_source": "rollout",
                "advantage_normalization": "group-sample-std",
            }
        )
    runtime = materialize_wan22_flow_policy_stack(
        PostTrainingRecipe.from_mapping(payload),
        policy=_adapter(),
        initial_policy_revision=f"tiny-{algorithm_type}",
        fused_adamw=False,
    )
    assert isinstance(runtime.stack, NativeFlowPolicyTrainingStack)
    assert runtime.stack.engine.algorithm.name == algorithm_type


def test_wan22_profile_config_and_recipe_keep_a14b_distinct_from_ti2v() -> None:
    root = Path(__file__).resolve().parents[2]
    recipe = PostTrainingRecipe.from_file(root / "configs/post_training/wan22_t2v_a14b_flow_grpo.yaml")
    algorithm, plan = validate_wan22_flow_policy_recipe(recipe)
    profile = wan22_flow_policy_profile()
    assert recipe.model.recipe == profile.model_recipe == "wan2.2-t2v-a14b"
    assert plan.boundary_ratio == profile.boundary_ratio == 0.875
    assert algorithm.sigmas == profile.sigmas
    assert algorithm.sigmas == wan22_flow_policy_sigmas()
    assert algorithm.sigmas == flow_match_sigma_schedule(20, shift=5.0)
    assert algorithm.sde_step_indices == profile.sde_step_indices
    assert algorithm.sigma_max == profile.sigma_max == algorithm.sigmas[1]
    assert algorithm.advantage_normalization == "group-mean-global-population-std"
    assert algorithm.old_log_prob_source == profile.old_log_prob_source == "rollout"
    assert algorithm.reward_model.type == "remote"
    assert algorithm.reward_model.reward_ids == ("videopickscore",)
    assert dict(algorithm.reward_weights) == {"videopickscore": 1.0}
    assert recipe.runtime.activation_checkpoint == "none"
    assert dict(plan.generation) == {"height": 480, "width": 640, "num_frames": 1}
    assert plan.global_prompt_batch_size == profile.global_prompt_batch_size == 48

    payload = recipe.to_dict()
    payload["model"] = {"recipe": "wan2.2-ti2v-5b", "checkpoint": "default"}
    with pytest.raises(ValueError, match="TI2V-5B"):
        validate_wan22_flow_policy_recipe(PostTrainingRecipe.from_mapping(payload))


def test_integrated_wan22_reference_uses_its_own_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _recipe().to_dict()
    payload["algorithm"]["reference_kl_weight"] = 0.1
    payload["algorithm"]["reference_checkpoint"] = "reference-repository"
    loaded = []

    def load_role(*, checkpoints, **kwargs):
        del kwargs
        loaded.append(checkpoints)
        return _adapter()

    monkeypatch.setattr(wan22_roles, "load_wan22_role_adapter", load_role)
    materialized = materialize_video_flow_policy_roles(
        PostTrainingRecipe.from_mapping(payload),
        device="cpu",
        checkpoint_overrides={
            "high-noise": "policy-high.safetensors",
            "low-noise": "policy-low.safetensors",
        },
        fused_adamw=False,
    )

    assert materialized.reference_policy is not None
    assert loaded[0].high_noise.source == ("policy-high.safetensors",)
    assert loaded[0].low_noise.source == ("policy-low.safetensors",)
    assert loaded[1].high_noise.repo_id == "reference-repository"
    assert loaded[1].low_noise.repo_id == "reference-repository"
