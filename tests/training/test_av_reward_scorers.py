from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.training.post_training.rewards import (
    AVRewardScorersConfig as ExportedAVRewardScorersConfig,
)
from worldfoundry.training.post_training.rewards import (
    CLAPScorer as ExportedCLAPScorer,
)
from worldfoundry.training.post_training.rewards import (
    VideoPickScoreScorer as ExportedVideoPickScoreScorer,
)
from worldfoundry.training.post_training.rewards.contracts import RewardRequest
from worldfoundry.training.post_training.rewards.http import NativeRewardService, RewardScorerRegistry
from worldfoundry.training.post_training.rewards.scorers import (
    AVRewardScorersConfig,
    CLAPConfig,
    CLAPScorer,
    VideoPickScoreConfig,
    VideoPickScoreScorer,
    build_av_reward_scorer_registry,
)


def _request(
    index: int,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    sampling_rate: int,
) -> RewardRequest:
    return RewardRequest(
        request_id=f"request-{index}",
        rollout_id=f"rollout-{index}",
        prompt=f"prompt {index}",
        conditions={},
        artifacts={"video": video, "audio": audio},
        reward_ids=("videopickscore", "clap"),
        metadata={"audio_sampling_rate": sampling_rate},
    )


class _PickProcessor:
    def __init__(self) -> None:
        self.images: list[object] = []
        self.prompts: list[str] = []

    def __call__(self, *, images=None, text=None, **kwargs):
        assert kwargs == {
            "padding": True,
            "truncation": True,
            "max_length": 77,
            "return_tensors": "pt",
        }
        if images is not None:
            self.images.extend(images)
            return {"pixel_values": torch.ones(len(images), 1)}
        self.prompts.extend(text)
        return {"input_ids": torch.ones(len(text), 1)}


class _PickModel:
    def __init__(self) -> None:
        self.logit_scale = torch.tensor(26.0).log()

    def get_image_features(self, *, pixel_values):
        return torch.tensor([[3.0, 4.0], [0.0, 5.0]])[: pixel_values.shape[0]]

    def get_text_features(self, *, input_ids):
        return torch.tensor([[0.0, 2.0], [5.0, 0.0]])[: input_ids.shape[0]]


class _CLAPProcessor:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.waveforms: list[torch.Tensor] = []
        self.sampling_rate = 0

    def __call__(self, *, text, audios, sampling_rate, return_tensors, padding):
        self.prompts.extend(text)
        self.waveforms.extend(torch.from_numpy(waveform.copy()) for waveform in audios)
        self.sampling_rate = sampling_rate
        assert return_tensors == "pt"
        assert padding is True
        return {"input_features": torch.ones(len(text), 1)}


class _CLAPModel:
    def __call__(self, *, input_features):
        count = input_features.shape[0]
        return SimpleNamespace(
            audio_embeds=torch.tensor([[3.0, 4.0], [1.0, 0.0]])[:count],
            text_embeds=torch.tensor([[0.0, 2.0], [2.0, 0.0]])[:count],
        )


def _videos() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.full((3, 2, 1, 2), 7, dtype=torch.uint8)
    first[:, 0, 0] = torch.tensor([[255, 0], [0, 128], [0, 255]], dtype=torch.uint8)
    second = torch.zeros(3, 1, 1, 1)
    return first, second


def _requests() -> tuple[RewardRequest, RewardRequest]:
    first, second = _videos()
    return (
        _request(
            0,
            video=first,
            audio=torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]),
            sampling_rate=24_000,
        ),
        _request(
            1,
            video=second,
            audio=torch.tensor([2.0, 3.0, 4.0]),
            sampling_rate=48_000,
        ),
    )


def test_video_pickscore_uses_first_frame_and_official_normalization() -> None:
    processor = _PickProcessor()
    scorer = VideoPickScoreScorer(
        VideoPickScoreConfig(batch_size=2, device="cpu"),
        processor=processor,
        model=_PickModel(),
    )

    scores = scorer.score(_requests())

    assert scores == pytest.approx((0.8, 0.0))
    assert processor.prompts == ["prompt 0", "prompt 1"]
    assert processor.images[0].tolist() == [[[255, 0, 0], [0, 128, 255]]]


def test_clap_downmixes_and_resamples_each_waveform_to_48khz() -> None:
    processor = _CLAPProcessor()
    resample_calls: list[tuple[torch.Tensor, int, int]] = []

    def resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
        resample_calls.append((waveform.clone(), source_rate, target_rate))
        return waveform.repeat_interleave(2)

    scorer = CLAPScorer(
        CLAPConfig(batch_size=2, device="cpu"),
        processor=processor,
        model=_CLAPModel(),
        resample=resample,
    )

    scores = scorer.score(_requests())

    assert scores == pytest.approx((0.8, 1.0))
    assert processor.sampling_rate == 48_000
    torch.testing.assert_close(resample_calls[0][0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert resample_calls[0][1:] == (24_000, 48_000)
    torch.testing.assert_close(
        processor.waveforms[0],
        torch.tensor([2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]),
    )
    torch.testing.assert_close(processor.waveforms[1], torch.tensor([2.0, 3.0, 4.0]))


def test_av_scorers_integrate_with_native_reward_service() -> None:
    registry = RewardScorerRegistry()
    registry.register(
        "videopickscore",
        VideoPickScoreScorer(
            VideoPickScoreConfig(batch_size=2, device="cpu"),
            processor=_PickProcessor(),
            model=_PickModel(),
        ),
    )
    registry.register(
        "clap",
        CLAPScorer(
            CLAPConfig(batch_size=2, device="cpu"),
            processor=_CLAPProcessor(),
            model=_CLAPModel(),
            resample=lambda waveform, source_rate, target_rate: waveform.repeat_interleave(target_rate // source_rate),
        ),
    )

    results = NativeRewardService(registry).evaluate(_requests())

    assert [result.values for result in results] == [
        {"videopickscore": pytest.approx(0.8), "clap": pytest.approx(0.8)},
        {"videopickscore": pytest.approx(0.0), "clap": pytest.approx(1.0)},
    ]
    assert all(result.valid == {"videopickscore": True, "clap": True} for result in results)


def test_typed_builder_registers_both_lazy_sidecar_scorers() -> None:
    config = AVRewardScorersConfig(
        videopickscore=VideoPickScoreConfig(device="cpu"),
        clap=CLAPConfig(device="cpu"),
    )

    registry = build_av_reward_scorer_registry(config)

    assert registry.names == ("clap", "videopickscore")
    assert isinstance(registry.scorer("videopickscore"), VideoPickScoreScorer)
    assert isinstance(registry.scorer("clap"), CLAPScorer)
    assert registry.scorer("videopickscore").loaded is False
    assert registry.scorer("clap").loaded is False
    assert ExportedAVRewardScorersConfig is AVRewardScorersConfig
    assert ExportedVideoPickScoreScorer is VideoPickScoreScorer
    assert ExportedCLAPScorer is CLAPScorer


def test_importing_scorer_package_does_not_import_model_or_media_stacks() -> None:
    script = """
import sys
from worldfoundry.training.post_training.rewards import CLAPScorer, VideoPickScoreScorer
assert CLAPScorer.__name__ == 'CLAPScorer'
assert VideoPickScoreScorer.__name__ == 'VideoPickScoreScorer'
for name in ('transformers', 'torchvision', 'torchaudio'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_video_training_extra_declares_lazy_audio_resampler_dependency() -> None:
    project = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    train_video = project.split("train_video = [", 1)[1].split("]", 1)[0]
    assert '"torchaudio>=2.9,<2.12.0"' in train_video


@pytest.mark.parametrize(
    ("scorer", "message"),
    (
        (
            VideoPickScoreScorer(
                VideoPickScoreConfig(device="cpu"),
                processor=_PickProcessor(),
                model=_PickModel(),
            ),
            "\\[C,T,H,W\\]",
        ),
        (
            CLAPScorer(
                CLAPConfig(device="cpu"),
                processor=_CLAPProcessor(),
                model=_CLAPModel(),
            ),
            "audio_sampling_rate",
        ),
    ),
)
def test_av_scorers_enforce_artifact_contracts(scorer, message: str) -> None:
    request = _request(
        0,
        video=torch.zeros(2, 2, 3),
        audio=torch.zeros(8),
        sampling_rate=48_000,
    )
    if isinstance(scorer, CLAPScorer):
        request = RewardRequest(
            request_id=request.request_id,
            rollout_id=request.rollout_id,
            prompt=request.prompt,
            conditions=request.conditions,
            artifacts=request.artifacts,
            reward_ids=request.reward_ids,
            metadata={},
        )
    with pytest.raises(ValueError, match=message):
        scorer.score((request,))
