"""DreamX-World causal Wan wrappers used by the camera-conditioned recipe.

These classes are **not** the shared :class:`~...contracts.Denoiser`
adapter.  DreamX-World's streaming camera loop owns its own scheduler and
KV-cache protocol; it calls :class:`WanDiffusionCameraWrapper.forward`
with ``noisy_image_or_video``, ``conditional_dict["prompt_embeds"]``, and
optional ``y_camera`` / ``y`` tensors.

:class:`WanTextEncoder` and :class:`WanVAEWrapper` re-export the resident
Wan 2.1 / 2.2 helpers with DreamX's 48-channel Wan2.2 VAE statistics.
The causal DiT is :class:`~..networks.wan.variants.dreamx_world.causal.CausalWanModel`.
"""

from typing import List, Optional
import torch

from worldfoundry.base_models.diffusion_model.schedulers import (
    InferenceFlowMatchScheduler,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.resident import (
    WanVAEWrapper as _WanVAEWrapper,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.resident import (
    WanTextEncoder as _WanTextEncoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import (
    WAN_2P2_VAE_MEAN,
    WAN_2P2_VAE_STD,
    _video_vae as _video_vae_2_2,
)
from worldfoundry.base_models.diffusion_model.loaders.wan_variant import load_wan_config


class WanTextEncoder(_WanTextEncoder):
    """Resident Wan UMT5 encoder with DreamX default bfloat16 weights."""

    def __init__(
        self,
        text_encoder_path: str,
        tokenizer_path: str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
            dtype=dtype,
        )


class WanVAEWrapper(_WanVAEWrapper):
    """Wan2.2 48-channel VAE wrapper with official mean / std statistics."""

    def __init__(self, vae_path: str):
        super().__init__(
            vae_path=vae_path,
            mean=WAN_2P2_VAE_MEAN,
            std=WAN_2P2_VAE_STD,
            z_dim=48,
            temporal_downsample=(False, True, True),
            model_factory=_video_vae_2_2,
        )


class WanDiffusionCameraWrapper(torch.nn.Module):
    """Causal Wan DiT plus flow-match scheduler for DreamX camera streaming.

    ``forward`` returns ``(flow_pred, pred_x0)`` rather than
    :class:`~...contracts.DenoiserOutput`; the DreamX runtime converts
    flow to x0 through :class:`InferenceFlowMatchScheduler`.
    """

    def __init__(
        self,
        model_config_path: str,
        *,
        num_output_frames: int = 21,
        timestep_shift: float = 5.0,
        local_attn_size: int = 12,
        sink_size: int = 3,
        attn_compress: int | None = None,
    ):
        super().__init__()

        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamx_world.causal import (
            CausalWanModel,
        )

        model_overrides = {
            "local_attn_size": local_attn_size,
            "sink_size": sink_size,
        }
        if attn_compress is not None:
            model_overrides["attn_compress"] = int(attn_compress)
        self.model = CausalWanModel.from_config(
            load_wan_config(model_config_path),
            **model_overrides,
        )
        self.model.eval()

        self.scheduler = InferenceFlowMatchScheduler(
            num_inference_steps=1000,
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.seq_len = 880 * num_output_frames

    def get_scheduler(self) -> InferenceFlowMatchScheduler:
        """Return the flow-match scheduler owned by this wrapper."""
        return self.scheduler

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        y_camera,
        timestep: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        cache_update_policy: str = "commit_detached",
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        skip_length = noisy_image_or_video.shape[-1] * noisy_image_or_video.shape[-2] // 4
        original_timestep = timestep[:, ::skip_length]

        y_camera_input = (
            y_camera if (y_camera is None or isinstance(y_camera, dict)) else y_camera.permute(0, 2, 1, 3, 4)
        )

        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=prompt_embeds,
            y=y.permute(0, 2, 1, 3, 4) if y is not None else None,
            y_camera=y_camera_input,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            cache_update_policy=cache_update_policy,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self.scheduler.flow_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            original_timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0
