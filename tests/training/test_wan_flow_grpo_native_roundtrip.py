from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

import gc
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (  # noqa: E402
    WanDenoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (  # noqa: E402
    WanModel,
)
from worldfoundry.training.checkpoint import (  # noqa: E402
    TrainingCheckpointer,
    TrainingProgress,
    TrainingState,
)
from worldfoundry.training.models import WanTrainAdapter  # noqa: E402
from worldfoundry.training.post_training.rewards.videoalign import (  # noqa: E402
    VideoAlignRewardEvaluator,
)
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (  # noqa: E402
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.batching import (  # noqa: E402
    RolloutPrompt,
    flow_rollout_batch_from_prompts,
)
from worldfoundry.training.post_training.rl.contracts import FlowRolloutBatch  # noqa: E402
from worldfoundry.training.post_training.rl.trajectory_rewards import (  # noqa: E402
    DecodedTerminalRewardAdapter,
)
from worldfoundry.training.post_training.shared.prediction import (  # noqa: E402
    NativeFlowPredictionAdapter,
)
from worldfoundry.training.recipes import PostTrainingRecipe  # noqa: E402
from worldfoundry.training.recipes.post_training.rewards.videoalign import (  # noqa: E402
    VIDEOALIGN_REWARD_IDS,
    VideoAlignRewardSpec,
)


def _recipe() -> PostTrainingRecipe:
    root = Path(__file__).resolve().parents[2]
    payload = PostTrainingRecipe.from_file(
        root / "configs/post_training/wan_1p3b_flow_grpo.yaml"
    ).to_dict()
    payload["run"] = {"id": "tiny-wan-flow-grpo", "output_dir": "unused"}
    payload["tuning"] = {"mode": "full"}
    payload["export"] = {"format": "safetensors"}
    payload["runtime"] = {
        "param_dtype": "float32",
        "reduce_dtype": "float32",
        "activation_checkpoint": "none",
        "compile": False,
    }
    payload["distributed"] = {
        "backend": "single",
        "dp_replicate": 1,
        "dp_shard": "auto",
        "cp": 1,
        "tp": 1,
    }
    payload["optimizer"] = {
        "type": "adamw",
        "learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
    }
    algorithm = payload["algorithm"]
    assert isinstance(algorithm, dict)
    algorithm.update(
        {
            "sigmas": [1.0, 0.6, 0.0],
            "sde_step_indices": [0, 1],
            "sde_timestep_fraction": None,
            "num_sde_steps": None,
            "sde_window": None,
            "guidance_scale": 1.0,
            "eta": 0.25,
            "sigma_max": 0.6,
            "group_size": 2,
            "updates_per_trajectory": 1,
            "old_log_prob_source": "replay",
            "trajectory_dtype": "float32",
        }
    )
    return PostTrainingRecipe.from_mapping(payload)


def _adapter(seed: int, *, dtype: torch.dtype = torch.float32) -> WanTrainAdapter:
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
        num_layers=1,
        has_image_input=False,
    ).cuda().to(dtype=dtype)
    return WanTrainAdapter(
        WanDenoiser(model, compute_dtype=dtype),
        codec=None,
        conditioner=None,
        expected_text_length=4,
    )


class _TinyWanDecoder:
    def decode(self, latents, request):
        values = latents[:, :3].float()
        return torch.nn.functional.interpolate(
            values,
            size=(request.num_frames, request.height, request.width),
            mode="trilinear",
            align_corners=False,
        ).clamp(-1, 1)


class _TinyVideoAlignProcessor:
    def apply_chat_template(self, chats, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return [str(chat) for chat in chats]

    def __call__(
        self,
        *,
        text,
        images,
        videos,
        padding,
        return_tensors,
        videos_kwargs,
    ):
        del text
        assert images is None
        assert padding is True
        assert return_tensors == "pt"
        assert videos_kwargs == {"do_rescale": False}
        return {"video_signal": torch.stack([video.float().mean() for video in videos])}


class _TinyVideoAlignModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, video_signal, return_dict):
        assert return_dict is True
        signal = video_signal.float()
        return {
            "logits": torch.stack(
                (signal, signal.square(), -signal),
                dim=1,
            )
        }


class _StatefulCursor:
    def __init__(self) -> None:
        self.position = 0

    def next(self) -> int:
        value = self.position
        self.position += 1
        return value

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state_dict) -> None:
        if set(state_dict) != {"position"}:
            raise ValueError("tiny rollout cursor state differs")
        self.position = int(state_dict["position"])


def _reward_adapter() -> DecodedTerminalRewardAdapter:
    evaluator = VideoAlignRewardEvaluator(
        _TinyVideoAlignModel(),
        _TinyVideoAlignProcessor(),
        VideoAlignRewardSpec(batch_size=2),
        device="cuda",
    )
    return DecodedTerminalRewardAdapter(
        _TinyWanDecoder(),
        evaluator,
        reward_ids=VIDEOALIGN_REWARD_IDS,
        evaluator_identity=evaluator.identity,
    )


def _runtime(seed: int, checkpoint_root: Path):
    recipe = _recipe()
    adapter = _adapter(seed)
    prediction = NativeFlowPredictionAdapter(
        adapter,
        checkpoint_identity=recipe.model.checkpoint,
    )
    stack = build_native_flow_policy_training_stack(
        recipe,
        policy=prediction,
        initial_policy_revision="a" * 64,
        fused_adamw=False,
    )
    progress = TrainingProgress()
    cursor = _StatefulCursor()
    generator = torch.Generator(device="cuda").manual_seed(907)
    state = TrainingState(
        model=adapter.trainable_module,
        optimizer=stack.optimizer,
        engine=stack.engine,
        dataloader=cursor,
        objective_generator=generator,
        progress=progress,
        identity={"gate": "tiny-native-wan-flow-grpo"},
        algorithm_state=stack.scalarizer,
    )
    checkpointer = TrainingCheckpointer(checkpoint_root)
    session = stack.session_type(
        sampler=stack.sampler,
        reward_adapter=_reward_adapter(),
        scalarizer=stack.scalarizer,
        engine=stack.engine,
        progress=progress,
        sde_index_schedule=stack.sde_index_schedule,
        old_log_prob_source=stack.old_log_prob_source,
        advantage_epsilon=stack.advantage_epsilon,
        advantage_normalization=stack.advantage_normalization,
        advantage_clip_max=stack.advantage_clip_max,
    )
    return adapter, stack, session, cursor, generator, state, checkpointer


def _iteration(stack, session, cursor, generator):
    index = cursor.next()
    group_id = f"group-{index}"
    batch = FlowRolloutBatch(
        sample_ids=(f"sample-{index}-a", f"sample-{index}-b"),
        group_ids=(group_id, group_id),
        policy_revision=stack.engine.current_policy_revision,
        initial_latents=torch.randn(
            (2, 16, 1, 2, 2),
            device="cuda",
            generator=generator,
        ),
        sigmas=torch.tensor(stack.sigmas, device="cuda"),
        conditioning={"context": torch.zeros(2, 4, 4096, device="cuda")},
        metadata={
            "prompt_by_group": {group_id: "a colorful square moves"},
            "generation_by_group": {
                group_id: {"height": 16, "width": 16, "num_frames": 5}
            },
        },
    )
    batch.conditioning["context"][:, 0, 0] = torch.tensor(
        [float(index + 1), float(index + 2)],
        device="cuda",
    )
    return session.train_iteration(batch, generator=generator)


def _state(adapter: WanTrainAdapter) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in adapter.trainable_module.named_parameters()
    }


def _release(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_wan_accepts_bfloat16_context_with_float32_flow_trajectory() -> None:
    adapter = _adapter(797, dtype=torch.bfloat16)
    prediction = NativeFlowPredictionAdapter(
        adapter,
        autocast_dtype=torch.bfloat16,
        checkpoint_identity="default",
    )
    rollout = flow_rollout_batch_from_prompts(
        (
            RolloutPrompt(
                prompt_id="group",
                prompt="a colorful square moves",
                conditions={
                    "context": torch.zeros(
                        4,
                        4096,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                },
            ),
        ),
        group_size=2,
        policy_revision="a" * 64,
        latent_shape=(16, 1, 2, 2),
        sigmas=(1.0, 0.6, 0.0),
        device="cuda",
        dtype=torch.float32,
        generator=torch.Generator(device="cuda").manual_seed(809),
        shared_negative_conditioning={
            "context": torch.ones(4, 4096, device="cuda", dtype=torch.float32)
        },
    )

    assert rollout.initial_latents.dtype is torch.float32
    assert rollout.conditioning["context"].dtype is torch.bfloat16
    assert rollout.conditioning["negative_context"].dtype is torch.bfloat16
    with torch.no_grad():
        velocity = prediction.predict_velocity(
            rollout.initial_latents,
            rollout.sigmas[0],
            sample_ids=rollout.sample_ids,
            conditioning=rollout.conditioning,
            training=False,
        )

    assert velocity.dtype is torch.float32
    assert velocity.shape == rollout.initial_latents.shape
    assert bool(torch.isfinite(velocity).all())
    _release(adapter, prediction, rollout, velocity)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_wan_flow_grpo_cuda_update_and_dcp_continuation_are_exact(
    tmp_path: Path,
) -> None:
    continuous = _runtime(811, tmp_path / "continuous")
    continuous_adapter, continuous_stack, continuous_session, cursor, generator, _, _ = continuous
    initial = _state(continuous_adapter)
    _iteration(continuous_stack, continuous_session, cursor, generator)
    final_result = _iteration(continuous_stack, continuous_session, cursor, generator)
    expected = _state(continuous_adapter)
    expected_generator = generator.get_state().cpu().clone()
    assert any(not torch.equal(initial[name], expected[name]) for name in initial)
    assert final_result.updates[-1].trajectory_complete is True
    _release(continuous)

    split = _runtime(811, tmp_path / "split")
    split_adapter, split_stack, split_session, split_cursor, split_generator, split_state, checkpointer = split
    _iteration(split_stack, split_session, split_cursor, split_generator)
    at_checkpoint = _state(split_adapter)
    checkpoint = checkpointer.save(split_state, asynchronous=False)
    _release(split)

    resumed = _runtime(811, tmp_path / "resumed")
    resumed_adapter, resumed_stack, resumed_session, resumed_cursor, resumed_generator, resumed_state, _ = resumed
    checkpointer.load(resumed_state, checkpoint.path)
    restored = _state(resumed_adapter)
    assert set(restored) == set(at_checkpoint)
    assert all(torch.equal(restored[name], at_checkpoint[name]) for name in restored)
    assert resumed_cursor.position == 1
    _iteration(resumed_stack, resumed_session, resumed_cursor, resumed_generator)
    actual = _state(resumed_adapter)

    assert resumed_stack.engine.global_step == 2
    assert resumed_state.progress.optimizer_steps == 2
    assert resumed_cursor.position == 2
    assert torch.equal(resumed_generator.get_state().cpu(), expected_generator)
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[name], expected[name]) for name in actual)
