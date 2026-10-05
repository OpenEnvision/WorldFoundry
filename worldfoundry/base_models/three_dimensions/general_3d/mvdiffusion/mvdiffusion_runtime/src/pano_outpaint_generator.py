"""Inference-only multi-view panorama outpainting generator."""

from pathlib import Path

import torch
from torch import nn
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from worldfoundry.core.io.paths import resolve_local_hf_model_path

from .models.pano.MVGenModel import MultiViewBaseModel


def _resolve_model_id(configured_model_id):
    """Prefer a complete local HF export while preserving the remote-id fallback."""
    try:
        local_path = resolve_local_hf_model_path(
            configured_model_id,
            required_files=("model_index.json",),
        )
    except FileNotFoundError:
        return configured_model_id
    return str(local_path.resolve())


def _local_fp16_variant_kwargs(model_id, *, subfolder, weight_stem):
    """Select Diffusers' fp16 variant when that is the only local export.

    Hugging Face mirrors often materialize only files such as
    ``model.fp16.safetensors``.  ``from_pretrained(..., use_safetensors=True)``
    does not discover those files unless ``variant=\"fp16\"`` is explicit.
    Keep remote IDs and full-precision local exports on their existing path.
    """

    model_path = Path(model_id).expanduser()
    component = model_path / subfolder
    standard = component / f"{weight_stem}.safetensors"
    fp16 = component / f"{weight_stem}.fp16.safetensors"
    if model_path.is_dir() and fp16.is_file() and not standard.is_file():
        return {"variant": "fp16"}
    return {}


class PanoOutpaintGenerator(nn.Module):
    """Pano outpaint generator implementation."""
    def __init__(self, config):
        """Init.

        Args:
            config: The config.
        """
        super().__init__()

        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']
        configured_model_id = config['model']['model_id']
        model_id = _resolve_model_id(configured_model_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
            **_local_fp16_variant_kwargs(
                model_id,
                subfolder="text_encoder",
                weight_stem="model",
            ),
        )

        self.vae, self.scheduler, unet = self.load_model(model_id)
        self.mv_base_model = MultiViewBaseModel(
            unet, config['model'])

    def load_model(self, model_id):
        """Load model.

        Args:
            model_id: The model id.
        """
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
            **_local_fp16_variant_kwargs(
                model_id,
                subfolder="vae",
                weight_stem="diffusion_pytorch_model",
            ),
        )
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            local_files_only=True,
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            local_files_only=True,
            use_safetensors=True,
            **_local_fp16_variant_kwargs(
                model_id,
                subfolder="unet",
                weight_stem="diffusion_pytorch_model",
            ),
        )
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        """Encode text.

        Args:
            text: The text.
            device: The device.
        """
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        """Encode image.

        Args:
            x_input: The x input.
            vae: The vae.
        """
        b = x_input.shape[0]

        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        """Decode latent.

        Args:
            latents: The latents.
            vae: The vae.
        """
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)

        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width
    ):
        """Prepare mask latents.

        Args:
            mask: The mask.
            masked_image: The masked image.
            batch_size: The batch size.
            height: The height.
            width: The width.
        """

        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        masked_image_latents = self.encode_image(masked_image, self.vae)

        return mask, masked_image_latents

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        """Gen cls free guide pair.

        Args:
            latents: The latents.
            timestep: The timestep.
            prompt_embd: The prompt embd.
            batch: The batch.
        """
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)

        R = torch.cat([batch['R']]*2)
        K = torch.cat([batch['K']]*2)

        meta = {
            'K': K,
            'R': R
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        """Forward cls free.

        Args:
            latents_high_res: The latents high res.
            _timestep: The timestep.
            prompt_embd: The prompt embd.
            batch: The batch.
            model: The model.
        """
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def prepare_mask_image(self, images):
        """Prepare mask image.

        Args:
            images: The images.
        """
        bs, m, _, h, w = images.shape
        mask=torch.ones(bs, m, 1, h, w, device=images.device)
        mask[:,0]=0
        masked_image=images*(mask<0.5)
        mask_latnets=[]
        masked_image_latents=[]
        for i in range(m):
            _mask, _masked_image_latent = self.prepare_mask_latents(
                mask[:,i],
                masked_image[:,i],
                bs,
                h,
                w,
            )
            mask_latnets.append(_mask)
            masked_image_latents.append(_masked_image_latent)
        mask_latnets = torch.stack(mask_latnets, dim=1)
        masked_image_latents = torch.stack(masked_image_latents, dim=1)
        return mask_latnets, masked_image_latents

    @torch.no_grad()
    def inference(self, batch):
        """Inference.

        Args:
            batch: The batch.
        """
        images = batch['images']


        bs, m, h, w, _ = images.shape
        images=rearrange(images, 'bs m h w c -> bs m c h w')
        mask_latnets, masked_image_latents=self.prepare_mask_image(images)

        device = images.device

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps


        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)
            latent_model_input = torch.cat([latents, mask_latnets, masked_image_latents], dim=2)

            noise_pred = self.forward_cls_free(
                latent_model_input, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample

        images_pred = self.decode_latent(
            latents, self.vae)

        return images_pred
