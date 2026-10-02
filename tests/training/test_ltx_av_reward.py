from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.training.engine.ltx.rewards import LTXAVTerminalRewardAdapter
from worldfoundry.training.engine.ltx.trajectory import LTX_AUDIO_TRAJECTORY
from worldfoundry.training.post_training.rewards.contracts import RewardRequest, RewardResult
from worldfoundry.training.post_training.rl.algorithms.flow_policy.builder import (
    build_native_flow_policy_training_stack,
)
from worldfoundry.training.post_training.rl.contracts import FlowTrajectory
from worldfoundry.training.post_training.rl.remote_rewards import HTTPTerminalRewardAdapter
from worldfoundry.training.recipes import PostTrainingRecipe, RemoteRewardSpec


class _MediaDecoder:
    def __init__(self) -> None:
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def decode(self, latents, request):
        del request
        return latents

    def decode_modalities(self, states, request):
        video = states["video"].latent
        audio = states["audio"].latent
        self.inputs.append((video.clone(), audio.clone()))
        return {
            "video": torch.full(
                (request.num_frames, request.height, request.width, 3),
                float(video.mean()),
            ),
            "audio": torch.full((1, 16), float(audio.mean())),
            "audio_sampling_rate": 24_000,
        }


class _Evaluator:
    identity = {"transport": "fake"}

    def __init__(self) -> None:
        self.requests: tuple[RewardRequest, ...] = ()

    def evaluate(self, requests: tuple[RewardRequest, ...]) -> tuple[RewardResult, ...]:
        self.requests = requests
        return tuple(
            RewardResult(
                request_id=request.request_id,
                rollout_id=request.rollout_id,
                values={
                    "videopickscore": float(request.artifacts["video"].mean()),
                    "clap": float(request.artifacts["audio"].mean()),
                },
                valid={"videopickscore": True, "clap": True},
                diagnostics={},
                latency_ms=0.0,
            )
            for request in requests
        )


def _trajectory() -> FlowTrajectory:
    video = torch.zeros(2, 2, 128, 2, 1, 1)
    video[0, -1].fill_(1.0)
    video[1, -1].fill_(2.0)
    audio = torch.zeros(2, 2, 9, 128)
    audio[0, -1].fill_(10.0)
    audio[1, -1].fill_(20.0)
    return FlowTrajectory(
        sample_ids=("sample-a", "sample-b"),
        group_ids=("prompt", "prompt"),
        policy_revision="policy-3",
        latents=video,
        sigmas=torch.tensor([1.0, 0.0]),
        step_indices=(0,),
        old_log_probs=torch.zeros(2, 1),
        transition_means=torch.zeros(2, 1, 128, 2, 1, 1),
        transition_scales=torch.ones(2, 1, 1, 1, 1, 1),
        conditioning={LTX_AUDIO_TRAJECTORY: audio},
        metadata={
            "prompt_by_group": {"prompt": "a drummer on stage"},
            "generation_by_group": {"prompt": {"height": 32, "width": 32, "num_frames": 9}},
        },
    )


def test_ltx_av_terminal_reward_preserves_artifact_shape_and_sample_order() -> None:
    decoder = _MediaDecoder()
    evaluator = _Evaluator()
    adapter = LTXAVTerminalRewardAdapter(
        decoder,
        evaluator,
        reward_ids=("videopickscore", "clap"),
    )

    rewards = adapter.score(_trajectory())

    assert [request.request_id for request in evaluator.requests] == ["sample-a", "sample-b"]
    assert [tuple(request.artifacts) for request in evaluator.requests] == [
        ("video", "audio"),
        ("video", "audio"),
    ]
    assert [tuple(request.artifacts["video"].shape) for request in evaluator.requests] == [
        (3, 9, 32, 32),
        (3, 9, 32, 32),
    ]
    assert [tuple(request.artifacts["audio"].shape) for request in evaluator.requests] == [
        (1, 16),
        (1, 16),
    ]
    assert [request.metadata["audio_sampling_rate"] for request in evaluator.requests] == [
        24_000,
        24_000,
    ]
    assert [tuple(video.shape) for video, _ in decoder.inputs] == [(1, 2, 128), (1, 2, 128)]
    assert [tuple(audio.shape) for _, audio in decoder.inputs] == [(1, 9, 128), (1, 9, 128)]
    torch.testing.assert_close(rewards["videopickscore"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(rewards["clap"], torch.tensor([10.0, 20.0]))


def _av_recipe() -> PostTrainingRecipe:
    root = Path(__file__).resolve().parents[2]
    return PostTrainingRecipe.from_file(root / "configs/post_training/ltx_2p3_av_flow_grpo.yaml")


def test_remote_reward_recipe_is_strict_and_round_trippable() -> None:
    recipe = _av_recipe()
    assert isinstance(recipe.algorithm.reward_model, RemoteRewardSpec)
    assert recipe.algorithm.reward_model.reward_ids == ("videopickscore", "clap")
    assert recipe.algorithm.advantage_normalization == "group-population-std"
    assert recipe.algorithm.reward_weights == {"videopickscore": 0.5, "clap": 0.5}
    assert PostTrainingRecipe.from_mapping(recipe.to_dict()) == recipe

    mapping = recipe.to_dict()
    mapping["algorithm"]["reward_model"]["batch_size"] = 4
    with pytest.raises(ValueError, match="unknown fields"):
        PostTrainingRecipe.from_mapping(mapping)


class _TinyFlowPrediction:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)

    def predict_velocity(
        self,
        noisy_latents,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
        branch="positive",
    ):
        del sigmas, sample_ids, conditioning, branch
        self.module.train(training)
        return noisy_latents * self.module.weight.reshape(1, 1, 1, 1, 1)

    def predict_clean(self, noisy_latents, sigmas, **kwargs):
        return noisy_latents - self.predict_velocity(noisy_latents, sigmas, **kwargs)


def test_remote_reward_recipe_builds_identity_scalarization() -> None:
    mapping = _av_recipe().to_dict()
    mapping["distributed"] = {"backend": "single"}
    mapping["tuning"] = {"mode": "full"}
    mapping["export"] = {"format": "safetensors"}
    stack = build_native_flow_policy_training_stack(
        PostTrainingRecipe.from_mapping(mapping),
        policy=_TinyFlowPrediction(),
        initial_policy_revision="policy",
        fused_adamw=False,
    )
    assert stack.scalarizer.weights == {"videopickscore": 0.5, "clap": 0.5}
    assert stack.scalarizer.calibration_mean == {"videopickscore": 0.0, "clap": 0.0}
    assert stack.scalarizer.calibration_std == {"videopickscore": 1.0, "clap": 1.0}


class _HTTPRewardEvaluator(_Evaluator):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("remote_reward", "adapter_type", "joint_av"),
    (
        (True, LTXAVTerminalRewardAdapter, True),
        (False, HTTPTerminalRewardAdapter, False),
    ),
)
def test_video_policy_routes_only_remote_ltx23_rewards_through_av_decoder(
    remote_reward: bool,
    adapter_type: type,
    joint_av: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from worldfoundry.training.engine import video_policy

    mapping = _av_recipe().to_dict()
    mapping["run"] = {"id": "ltx-av-route", "output_dir": str(tmp_path / "run")}
    mapping["distributed"] = {"backend": "single"}
    if not remote_reward:
        mapping["algorithm"]["reward_weights"] = {
            "video_quality": 1.0,
            "motion_quality": 1.0,
            "text_alignment": 1.0,
        }
        mapping["algorithm"]["reward_model"] = {"type": "videoalign"}
    recipe = PostTrainingRecipe.from_mapping(mapping)

    policy_module = torch.nn.Linear(1, 1)

    class _Materialized:
        def __init__(self) -> None:
            self.stack = object()
            self.policy_tuning = None
            self.policy_module = policy_module

        def build_rollout_loader(self, *args, **kwargs):
            del args, kwargs
            return object()

    class _Conditioning:
        index = SimpleNamespace(to_dict=lambda: {"model_recipe": "ltx-2.3-i2v"})

        def __len__(self) -> int:
            return 8

    conditioning = _Conditioning()
    captured: dict[str, object] = {}

    monkeypatch.setattr(video_policy, "create_run_directory", lambda *args: None)
    monkeypatch.setattr(
        video_policy,
        "_load_conditioning_dataset",
        lambda *args, **kwargs: (object(), conditioning),
    )
    monkeypatch.setattr(
        video_policy,
        "materialize_video_flow_policy_roles",
        lambda *args, **kwargs: _Materialized(),
    )
    monkeypatch.setattr(
        video_policy,
        "_build_conditioned_source",
        lambda *args, **kwargs: (
            object(),
            torch.Generator().manual_seed(5),
            conditioning,
        ),
    )

    def build_decoder(*args, **kwargs):
        captured["joint_av"] = kwargs["joint_av"]
        return _MediaDecoder()

    monkeypatch.setattr(video_policy, "_build_native_decoder", build_decoder)
    monkeypatch.setattr(video_policy, "HTTPRewardEvaluator", _HTTPRewardEvaluator)

    def build_run(*args, **kwargs):
        captured["adapter"] = kwargs["reward_adapter"]
        captured["closeables"] = kwargs["closeables"]
        return "run"

    monkeypatch.setattr(video_policy, "build_native_flow_policy_training_run", build_run)

    result = video_policy.materialize_video_flow_policy_training_run(
        recipe,
        base_dir=tmp_path,
        device="cpu",
        reward_url="http://reward.test",
        fused_adamw=False,
    )

    assert result == "run"
    assert captured["joint_av"] is joint_av
    assert isinstance(captured["adapter"], adapter_type)
    assert len(captured["closeables"]) == 1
