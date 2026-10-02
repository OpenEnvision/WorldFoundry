"""Inference-only SVI conditioning pipeline.

This module ports the image-conditioning path from the official Stable Video
Infinity ``SVIVideoPipeline`` at revision
``63af54329f38b10ed1a91dc88bc67a765d0c4a18``.  The shared Wan/FlowMatch
implementation is provided by WorldFoundry's native Wan staged runner; this
file owns only SVI-specific multi-frame continuation and reference padding.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.runners import WanStagedPipeline
from worldfoundry.core.vram import AutoTorchModule


class SVIVideoPipeline(WanStagedPipeline):
    """Wan I2V pipeline with Stable Video Infinity continuation conditioning."""

    @staticmethod
    def _cast_managed_model(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
        """Cast a model and keep VRAM-managed lifecycle dtypes in sync.

        ``Module.to`` changes the wrapped parameters but not the lifecycle
        metadata stored by WorldFoundry's VRAM wrappers.  Without updating that
        metadata, the next onload/preparing transition silently casts SVI's VAE
        back to bfloat16 before its official float32 encode/decode path runs.
        """

        for module in model.modules():
            if isinstance(module, AutoTorchModule):
                module.offload_dtype = dtype
                module.onload_dtype = dtype
                module.preparing_dtype = dtype
                module.computation_dtype = dtype
        return model.to(dtype=dtype)

    @staticmethod
    def _coerce_reference_frame(frame: Any) -> Any:
        """Accept the uint8 tensor reference format used by the official CLI."""

        if not torch.is_tensor(frame):
            return frame
        from PIL import Image

        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[-1] not in {1, 3, 4} and tensor.shape[0] in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 0)
        if tensor.ndim != 3:
            raise ValueError(f"SVI reference tensors must be HWC or CHW images, got {tuple(tensor.shape)}.")
        if tensor.dtype.is_floating_point:
            tensor = tensor.float()
            if tensor.numel() and tensor.max().item() <= 1.0:
                tensor = tensor * 255.0
        array = tensor.clamp(0, 255).to(torch.uint8).numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array).convert("RGB")

    @classmethod
    def from_components(
        cls,
        *,
        text_encoder,
        image_encoder,
        dit,
        vae,
        tokenizer_path: str,
        torch_dtype: torch.dtype,
        device: str,
        use_usp: bool = False,
    ) -> "SVIVideoPipeline":
        pipe = cls(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        pipe.text_encoder = text_encoder
        pipe.image_encoder = image_encoder
        pipe.dit = dit
        pipe.vae = vae
        pipe.prompter.fetch_models(text_encoder)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from worldfoundry.core.attention.parallel.patch_xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe

    def _encode_svi_images(
        self,
        first_frames: list[Any],
        random_ref_frame: Any,
        *,
        num_frames: int,
        height: int,
        width: int,
        ref_pad_cfg: bool,
        ref_pad_num: int,
    ) -> dict[str, torch.Tensor]:
        """Encode motion frames and the persistent SVI reference frame."""

        if not first_frames:
            raise ValueError("SVI requires at least one conditioning frame.")
        if len(first_frames) > num_frames:
            raise ValueError(f"SVI received {len(first_frames)} motion frames for a {num_frames}-frame segment.")
        if ref_pad_num < -1:
            raise ValueError("SVI ref_pad_num must be -1, 0, or a positive integer.")

        remaining_frames = num_frames - len(first_frames)
        if ref_pad_num > remaining_frames:
            raise ValueError(f"SVI ref_pad_num={ref_pad_num} exceeds the {remaining_frames} available frames.")

        original_vae_dtype = next(iter(self.vae.parameters())).dtype
        original_image_encoder_dtype = next(iter(self.image_encoder.parameters())).dtype
        try:
            self.vae = self._cast_managed_model(self.vae, torch.float32)
            self.image_encoder = self.image_encoder.to(dtype=torch.float32)

            reference = self.preprocess_image(random_ref_frame.resize((width, height))).to(
                dtype=torch.float32, device=self.device
            )
            first = self.preprocess_image(first_frames[0].resize((width, height))).to(
                dtype=torch.float32,
                device=self.device,
            )
            clip_context = self.image_encoder.encode_image([first])

            mask = torch.ones(
                1,
                num_frames,
                height // 8,
                width // 8,
                device=self.device,
                dtype=torch.float32,
            )
            if ref_pad_cfg:
                mask[:, len(first_frames) :] = 0
            else:
                mask[:, 1:] = 0
            mask = torch.concat(
                [torch.repeat_interleave(mask[:, :1], repeats=4, dim=1), mask[:, 1:]],
                dim=1,
            )
            mask = mask.view(1, mask.shape[1] // 4, 4, height // 8, width // 8)
            mask = mask.transpose(1, 2)[0]

            condition_tensors = [
                self.preprocess_image(frame.resize((width, height))).to(
                    dtype=torch.float32,
                    device=self.device,
                )
                for frame in first_frames
            ]
            condition = torch.cat(condition_tensors, dim=0).permute(1, 0, 2, 3)

            if ref_pad_num == 0:
                padding = torch.zeros(
                    3,
                    remaining_frames,
                    height,
                    width,
                    device=self.device,
                    dtype=torch.float32,
                )
            elif ref_pad_num == -1:
                padding = reference.transpose(0, 1).repeat(1, remaining_frames, 1, 1)
            else:
                reference_padding = reference.transpose(0, 1).repeat(1, ref_pad_num, 1, 1)
                zero_padding = torch.zeros(
                    3,
                    remaining_frames - ref_pad_num,
                    height,
                    width,
                    device=self.device,
                    dtype=torch.float32,
                )
                padding = torch.cat([reference_padding, zero_padding], dim=1)

            vae_input = torch.cat([condition, padding], dim=1)
            encoded = self.vae.encode(
                [vae_input.to(dtype=torch.float32, device=self.device)],
                device=self.device,
            )[0]
            encoded = torch.cat([mask, encoded]).unsqueeze(0)
            return {
                "clip_feature": clip_context.to(dtype=self.torch_dtype, device=self.device),
                "y": encoded.to(dtype=self.torch_dtype, device=self.device),
            }
        finally:
            self.vae = self._cast_managed_model(self.vae, original_vae_dtype)
            self.image_encoder = self.image_encoder.to(dtype=original_image_encoder_dtype)

    def encode_image(
        self,
        image: Any,
        end_image: Any,
        num_frames: int,
        height: int,
        width: int,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Implement the hook used by the shared Wan denoising pipeline."""

        if end_image is not None:
            raise ValueError("SVI does not support end-image conditioning.")
        first_frames = list(image) if isinstance(image, (list, tuple)) else [image]
        reference = getattr(self, "_svi_random_ref_frame", None)
        if reference is None:
            reference = first_frames[0]
        return self._encode_svi_images(
            first_frames,
            reference,
            num_frames=num_frames,
            height=height,
            width=width,
            ref_pad_cfg=bool(getattr(self, "_svi_ref_pad_cfg", False)),
            ref_pad_num=int(getattr(self, "_svi_ref_pad_num", 0)),
        )

    def decode_video(
        self,
        latents: torch.Tensor,
        tiled: bool = True,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Preserve the official SVI float32 VAE decode path."""

        original_vae_dtype = next(iter(self.vae.parameters())).dtype
        try:
            self.vae = self._cast_managed_model(self.vae, torch.float32).to(device=self.device)
            latents = latents.to(device=self.device, dtype=torch.float32)
            return self.vae.decode(
                latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        finally:
            self.vae = self._cast_managed_model(self.vae, original_vae_dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        input_image: Any,
        random_ref_frame: Any | None = None,
        ref_pad_cfg: bool = False,
        ref_pad_num: int = 0,
        seed: int | None = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        tea_cache_l1_thresh: float | None = None,
        tea_cache_model_id: str = "Wan2.1-I2V-14B-480P",
        progress_bar_cmd: Any = None,
    ) -> list[Any]:
        """Generate one official 81-frame SVI segment."""

        if input_image is None:
            raise ValueError("SVI requires an input image or continuation frames.")
        first_frames = list(input_image) if isinstance(input_image, (list, tuple)) else [input_image]
        self._svi_random_ref_frame = self._coerce_reference_frame(
            first_frames[0] if random_ref_frame is None else random_ref_frame
        )
        self._svi_ref_pad_cfg = bool(ref_pad_cfg)
        self._svi_ref_pad_num = int(ref_pad_num)
        kwargs: dict[str, Any] = {}
        if progress_bar_cmd is not None:
            kwargs["progress_bar_cmd"] = progress_bar_cmd
        try:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=first_frames,
                seed=seed,
                rand_device=rand_device,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=float(cfg_scale),
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                tea_cache_l1_thresh=tea_cache_l1_thresh,
                tea_cache_model_id=tea_cache_model_id,
                **kwargs,
            )
        finally:
            self.__dict__.pop("_svi_random_ref_frame", None)
            self.__dict__.pop("_svi_ref_pad_cfg", None)
            self.__dict__.pop("_svi_ref_pad_num", None)


__all__ = ["SVIVideoPipeline"]
