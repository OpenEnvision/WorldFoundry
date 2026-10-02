"""UniAnimate pose conditioning over WorldFoundry's native Wan runner."""

import types

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import WanVideoVAE
from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
    WanImageEncoder,
    WanTextEncoder,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
from worldfoundry.base_models.diffusion_model.loaders import load_wan_inference_components
from worldfoundry.base_models.diffusion_model.runners import (
    TeaCache,
    WanStagedPipeline,
    model_fn_wan_video,
)
from worldfoundry.core.model_loading import GeneralLoRALoader, load_state_dict


class WanUniAnimateVideoPipeline(WanStagedPipeline):
    """Wan2.1 UniAnimate inference pipeline.

    This class keeps the UniAnimate-specific pose conditioning path in-tree
    while reusing the WorldFoundry Wan video pipeline base implementation.
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.model_names = ["text_encoder", "image_encoder", "dit", "vae"]
        self.use_unified_sequence_parallel = False
        self.dwpose_embedding = None
        self.randomref_embedding_pose = None

    @staticmethod
    def _pose_modules(auxiliary_state: dict[str, torch.Tensor]):
        concat_dim = 4
        dwpose_embedding = nn.Sequential(
            nn.Conv3d(3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        randomref_dim = 20
        randomref_embedding_pose = nn.Sequential(
            nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
        )

        dwpose_state = {
            key.split("dwpose_embedding.")[1]: value
            for key, value in auxiliary_state.items()
            if "dwpose_embedding" in key
        }
        dwpose_embedding.load_state_dict(dwpose_state, strict=True)

        randomref_state = {
            key.split("randomref_embedding_pose.")[1]: value
            for key, value in auxiliary_state.items()
            if "randomref_embedding_pose" in key
        }
        randomref_embedding_pose.load_state_dict(randomref_state, strict=True)
        return dwpose_embedding, randomref_embedding_pose

    @classmethod
    def from_components(
        cls,
        *,
        text_encoder: WanTextEncoder,
        dit: WanModel,
        vae: WanVideoVAE,
        image_encoder: WanImageEncoder,
        tokenizer_path: str,
        auxiliary_state: dict[str, torch.Tensor],
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        use_usp: bool = False,
    ):
        pipe = cls(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        pipe.text_encoder = text_encoder
        pipe.dit = dit
        pipe.vae = vae
        pipe.image_encoder = image_encoder
        pipe.prompter.fetch_models(text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_path)
        pipe.dwpose_embedding, pipe.randomref_embedding_pose = cls._pose_modules(auxiliary_state)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from worldfoundry.core.attention.parallel.patch_xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True

        return pipe

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
            dim=1,
        )
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        dwpose_data=None,
        random_ref_dwpose=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 == 1` is acceptable. We round it up to {num_frames}.")
        if dwpose_data is None or random_ref_dwpose is None:
            raise ValueError("UniAnimate requires dwpose_data and random_ref_dwpose inputs.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}

        extra_input = self.prepare_extra_input(latents)
        tea_cache_posi = {
            "tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
            if tea_cache_l1_thresh is not None
            else None
        }
        tea_cache_nega = {
            "tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
            if tea_cache_l1_thresh is not None
            else None
        }

        self.load_models_to_device(["dit"])
        usp_kwargs = self.prepare_unified_sequence_parallel()

        self.dwpose_embedding.to(self.device)
        self.randomref_embedding_pose.to(self.device)
        dwpose_data = dwpose_data.unsqueeze(0)
        dwpose_data = self.dwpose_embedding(
            (torch.cat([dwpose_data[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_data], dim=2) / 255.0).to(self.device)
        ).to(torch.bfloat16)
        random_ref_dwpose_data = self.randomref_embedding_pose(
            (random_ref_dwpose.unsqueeze(0) / 255.0).to(self.device).permute(0, 3, 1, 2)
        ).unsqueeze(2).to(torch.bfloat16)

        image_emb["y"] = image_emb["y"] + random_ref_dwpose_data
        condition = rearrange(dwpose_data, "b c f h w -> b (f h w) c").contiguous()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            model_timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            scheduler_timestep = self.scheduler.timesteps[progress_id]

            noise_pred_posi = model_fn_wan_video(
                self.dit,
                x=latents,
                timestep=model_timestep,
                **prompt_emb_posi,
                **image_emb,
                **extra_input,
                **tea_cache_posi,
                **usp_kwargs,
                add_condition=condition,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit,
                    x=latents,
                    timestep=model_timestep,
                    **prompt_emb_nega,
                    **image_emb,
                    **extra_input,
                    **tea_cache_nega,
                    **usp_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(noise_pred, scheduler_timestep, latents)

        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        return self.tensor2video(frames[0])


def load_unianimate_pipeline(
    wan_checkpoint_root: str,
    unianimate_checkpoint: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    use_usp: bool = False,
) -> WanUniAnimateVideoPipeline:
    """Load the official UniAnimate checkpoint through native Wan roles."""

    components = load_wan_inference_components(
        wan_checkpoint_root,
        image_conditioned=True,
        torch_dtype=torch_dtype,
        device="cpu",
    )
    state = load_state_dict(unianimate_checkpoint, torch_dtype=torch_dtype, device="cpu")
    lora_state = {}
    for name, value in state.items():
        if "pipe.dit." in name:
            lora_state[name.split("pipe.dit.", 1)[1]] = value
        else:
            lora_state[name] = value
    updated = GeneralLoRALoader(device="cpu", torch_dtype=torch_dtype).load(
        components.dit,
        lora_state,
        alpha=1.0,
    )
    if updated == 0:
        raise ValueError("UniAnimate checkpoint did not contain a compatible Wan LoRA")
    if components.image_encoder is None:
        raise RuntimeError("UniAnimate requires the Wan image encoder role")

    pipe = WanUniAnimateVideoPipeline.from_components(
        text_encoder=components.text_encoder,
        dit=components.dit,
        vae=components.vae,
        image_encoder=components.image_encoder,
        tokenizer_path=str(components.tokenizer_path),
        auxiliary_state=state,
        torch_dtype=torch_dtype,
        device=device,
        use_usp=use_usp,
    )
    if str(device) != "cpu":
        pipe.enable_cpu_offload()
    return pipe


__all__ = ["WanUniAnimateVideoPipeline", "load_unianimate_pipeline"]
