from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from worldfoundry.training.post_training import (  # noqa: E402
    VIDEOALIGN_SPECIAL_TOKEN_IDS,
    VideoAlignRewardEvaluator,
    build_videoalign_prompt,
    pool_videoalign_special_tokens,
    prepare_videoalign_video,
    remap_videoalign_checkpoint_keys,
    videoalign_frame_indices,
    videoalign_frame_size,
)
from worldfoundry.training.post_training.rewards import RewardRequest  # noqa: E402
from worldfoundry.training.recipes import (  # noqa: E402
    VIDEOALIGN_REWARD_IDS,
    VideoAlignRewardSpec,
)


def test_videoalign_uniform_sampling_and_qwen_grid_resize_match_official_contract() -> None:
    assert videoalign_frame_indices(17, source_fps=24.0) == (0, 5, 11, 16)
    assert videoalign_frame_size(480, 480) == (448, 448)

    spec = VideoAlignRewardSpec()
    source = torch.linspace(-1.0, 1.0, 3 * 17 * 480 * 480).reshape(3, 17, 480, 480)
    prepared = prepare_videoalign_video(source, spec)

    assert prepared.frames.shape == (4, 3, 448, 448)
    assert prepared.frame_indices == (0, 5, 11, 16)
    assert prepared.source_shape == (17, 480, 480)
    assert prepared.resized_shape == (4, 448, 448)
    assert prepared.frames.device.type == "cpu"
    assert float(prepared.frames.min()) >= 0.0
    assert float(prepared.frames.max()) <= 1.0
    torch.testing.assert_close(
        prepared.frames.mul(255).round(),
        prepared.frames.mul(255),
        rtol=0,
        atol=1.0e-5,
    )


def test_videoalign_prompt_preserves_literal_user_braces_and_special_token_order() -> None:
    prompt = build_videoalign_prompt("a {literal} red cube")

    assert "a {literal} red cube" in prompt
    assert prompt.count("<|VQ_reward|>") == 1
    assert prompt.count("<|MQ_reward|>") == 1
    assert prompt.count("<|TA_reward|>") == 1
    assert prompt.index("<|VQ_reward|>") < prompt.index("<|MQ_reward|>")
    assert prompt.index("<|MQ_reward|>") < prompt.index("<|TA_reward|>")


def test_videoalign_pooling_requires_exactly_one_of_each_reward_token() -> None:
    input_ids = torch.tensor(
        [
            [
                0,
                VIDEOALIGN_SPECIAL_TOKEN_IDS[0],
                2,
                VIDEOALIGN_SPECIAL_TOKEN_IDS[1],
                4,
                VIDEOALIGN_SPECIAL_TOKEN_IDS[2],
            ],
            [
                VIDEOALIGN_SPECIAL_TOKEN_IDS[0],
                1,
                VIDEOALIGN_SPECIAL_TOKEN_IDS[1],
                3,
                VIDEOALIGN_SPECIAL_TOKEN_IDS[2],
                5,
            ],
        ]
    )
    scores = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1)

    pooled = pool_videoalign_special_tokens(scores, input_ids)

    torch.testing.assert_close(pooled, torch.tensor([[1.0, 3.0, 5.0], [6.0, 8.0, 10.0]]))
    duplicate = input_ids.clone()
    duplicate[0, 0] = VIDEOALIGN_SPECIAL_TOKEN_IDS[0]
    with pytest.raises(ValueError, match="exactly once"):
        pool_videoalign_special_tokens(scores, duplicate)


def test_videoalign_checkpoint_remap_accepts_only_the_audited_transformers_layout_change() -> None:
    state = {
        "base_model.model.visual.patch.weight": torch.tensor([1.0]),
        "base_model.model.model.layers.0.weight": torch.tensor([2.0]),
        "base_model.model.rm_head.weight": torch.tensor([3.0]),
    }
    targets = (
        "base_model.model.model.visual.patch.weight",
        "base_model.model.model.language_model.layers.0.weight",
        "base_model.model.rm_head.weight",
    )

    remapped = remap_videoalign_checkpoint_keys(state, targets)

    assert set(remapped) == set(targets)
    with pytest.raises(ValueError, match="missing=.*unexpected"):
        remap_videoalign_checkpoint_keys(
            {**state, "unknown.weight": torch.tensor([4.0])},
            targets,
        )


class _FakeProcessor:
    def __init__(self) -> None:
        self.videos: list[torch.Tensor] = []
        self.videos_kwargs: Mapping[str, object] = {}

    def apply_chat_template(self, chats, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return [str(value) for value in chats]

    def __call__(self, *, text, images, videos, padding, return_tensors, videos_kwargs):
        del text
        assert images is None
        assert padding is True
        assert return_tensors == "pt"
        self.videos = list(videos)
        self.videos_kwargs = dict(videos_kwargs)
        return {
            "input_ids": torch.tensor(
                [
                    [
                        VIDEOALIGN_SPECIAL_TOKEN_IDS[0],
                        VIDEOALIGN_SPECIAL_TOKEN_IDS[1],
                        VIDEOALIGN_SPECIAL_TOKEN_IDS[2],
                    ]
                    for _ in videos
                ]
            )
        }


class _FakeRewardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids, return_dict):
        assert return_dict is True
        batch = input_ids.shape[0]
        logits = torch.tensor([3.0, 1.0, 2.0], device=input_ids.device).repeat(batch, 1)
        return {"logits": logits}


def test_videoalign_evaluator_returns_raw_components_without_mp4_or_hidden_normalization() -> None:
    processor = _FakeProcessor()
    evaluator = VideoAlignRewardEvaluator(
        _FakeRewardModel(),
        processor,
        VideoAlignRewardSpec(batch_size=2),
        device="cpu",
    )
    requests = tuple(
        RewardRequest(
            request_id=f"sample-{index}",
            rollout_id=f"rollout-{index}",
            prompt="a red cube rolls",
            conditions={},
            artifacts={"video": torch.zeros(3, 17, 280, 364)},
            reward_ids=VIDEOALIGN_REWARD_IDS,
        )
        for index in range(2)
    )

    results = evaluator.evaluate(requests)

    assert len(results) == 2
    assert results[0].values == {
        "video_quality": 3.0,
        "motion_quality": 1.0,
        "text_alignment": 2.0,
    }
    assert all(results[0].valid.values())
    assert processor.videos_kwargs == {"do_rescale": False}
    assert len(processor.videos) == 2
    assert all(video.shape[0] == 4 for video in processor.videos)
    assert results[0].diagnostics["frame_indices"] == [0, 5, 11, 16]


def test_videoalign_recipe_rejects_unpinned_or_component_mismatched_configuration() -> None:
    assert VideoAlignRewardSpec().normalization_epsilon == 0.0
    with pytest.raises(ValueError, match="immutable 40-hex"):
        VideoAlignRewardSpec(checkpoint_revision="main")
    with pytest.raises(ValueError, match="calibration keys"):
        VideoAlignRewardSpec(calibration_mean={"video_quality": 0.0})


def test_videoalign_builder_pins_the_official_slow_qwen_processor() -> None:
    source = Path("worldfoundry/training/post_training/rewards/videoalign.py").read_text(encoding="utf-8")

    assert "AutoImageProcessor.from_pretrained" in source
    assert "use_fast=False" in source
    assert "chat_template=chat_template" in source
