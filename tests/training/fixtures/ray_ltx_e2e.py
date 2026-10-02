"""Importable joint audio-video toy policy for real Ray rollout tests."""

from __future__ import annotations

import torch


class RayTinyLTXPolicy:
    def __init__(self, value: float = 0.0) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.module.weight, value)

    def _velocity(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.module.weight.reshape(*((1,) * value.ndim))

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
        return self._velocity(noisy_latents)

    def predict_clean(self, noisy_latents, sigmas, **kwargs):
        return noisy_latents - self.predict_velocity(
            noisy_latents,
            sigmas,
            **kwargs,
        )

    def predict_joint_velocity(
        self,
        video,
        audio,
        sigmas,
        *,
        sample_ids,
        conditioning,
        training,
    ):
        del sigmas, sample_ids, conditioning
        self.module.train(training)
        return self._velocity(video), self._velocity(audio)

    def initial_audio_latents(self, video, *, generator, dtype):
        del generator
        sample_value = video.flatten(1).mean(dim=1).to(dtype=dtype)
        return sample_value[:, None, None].expand(-1, 2, 1).clone()


def ray_tiny_ltx_policy_factory(*, context):
    del context
    return RayTinyLTXPolicy()


__all__ = ["RayTinyLTXPolicy", "ray_tiny_ltx_policy_factory"]
