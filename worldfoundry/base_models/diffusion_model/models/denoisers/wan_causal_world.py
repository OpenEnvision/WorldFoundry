# SPDX-License-Identifier: Apache-2.0
"""InSpatio's 36-channel causal Wan adapter, using the shared Wan graph."""

from __future__ import annotations

import torch

from worldfoundry.core.model_loading import load_state_dict
from worldfoundry.core.nn import FlowMatchScheduler

from ..networks.wan.variants.causal_camera_21 import CausalWanModel


class WanCausalWorldDenoiser(torch.nn.Module):
    """Preserve checkpoint keys and the released float64 flow-to-x0 math."""

    def __init__(self, model, *, timestep_shift=5.0):
        super().__init__()
        self.model = model
        self.scheduler = FlowMatchScheduler(
            num_inference_steps=1000, shift=timestep_shift,
            sigma_min=0.0, extra_one_step=True,
        )

    @classmethod
    def from_pretrained(cls, checkpoint, config_folder, *, device="cuda", dtype=torch.bfloat16):
        config = CausalWanModel.load_config(config_folder)
        with torch.device("meta"):
            model = CausalWanModel.from_config(config, in_dim=36)
        state = load_state_dict(str(checkpoint), device="cpu")
        if not state or not all(key.startswith("model.") for key in state):
            raise ValueError("Expected an InSpatio causal Wan checkpoint with model.* keys")
        model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=True, assign=True)
        return cls(model).to(device=device, dtype=dtype).eval().requires_grad_(False)

    def get_scheduler(self):
        return self.scheduler

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        times = self.scheduler.timesteps.to(device=flow_pred.device, dtype=torch.float64)
        sigmas = self.scheduler.sigmas.to(device=flow_pred.device, dtype=torch.float64)
        index = (times[None] - timestep[:, None]).abs().argmin(dim=1)
        sigma = sigmas[index].reshape(-1, 1, 1, 1)
        return (xt.double() - sigma * flow_pred.double()).to(flow_pred.dtype)

    def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache,
                kv_size, render_latent_input, freqs_offset=0):
        spatial_tokens = noisy_image_or_video.shape[-2] * noisy_image_or_video.shape[-1] // 4
        flow = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous(),
            t=timestep, context=conditional_dict["prompt_embeds"],
            seq_len=spatial_tokens * 24, kv_cache=kv_cache, kv_size=kv_size,
            render_latent_input=render_latent_input.permute(0, 2, 1, 3, 4).contiguous(),
            freqs_offset=freqs_offset,
        ).permute(0, 2, 1, 3, 4)
        if kv_size[1] < 0:
            return flow
        clean = self._convert_flow_pred_to_x0(
            flow.flatten(0, 1), noisy_image_or_video.flatten(0, 1), timestep.flatten(),
        ).unflatten(0, flow.shape[:2])
        return flow, clean
