"""Importable tiny video policy factory used by real Ray tests."""

from __future__ import annotations

import torch


class RayTinyVideoPrediction:
    def __init__(self, value: float = 0.0) -> None:
        self.module = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.module.weight, value)

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
        return noisy_latents - self.predict_velocity(
            noisy_latents,
            sigmas,
            **kwargs,
        )


def ray_tiny_video_policy_factory(*, context):
    del context
    return RayTinyVideoPrediction()


__all__ = ["RayTinyVideoPrediction", "ray_tiny_video_policy_factory"]
