"""Inference-only multi-view panorama generator."""

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from .models.pano.MVGenModel import MultiViewBaseModel


class PanoGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.diff_timestep = config["model"]["diff_timestep"]
        self.guidance_scale = config["model"]["guidance_scale"]

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config["model"]["model_id"], subfolder="tokenizer", torch_dtype=torch.float16
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config["model"]["model_id"], subfolder="text_encoder", torch_dtype=torch.float16
        )

        self.vae, self.scheduler, unet = self.load_model(config["model"]["model_id"])
        self.mv_base_model = MultiViewBaseModel(unet, config["model"])

    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
        x_input = x_input.reshape(-1, x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2], z.shape[-1])  # (bs, 2, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = 1 / vae.config.scaling_factor * latents
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype("uint8")

        return image

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents] * 2)
        timestep = torch.cat([timestep] * 2)

        R = torch.cat([batch["R"]] * 2)
        K = torch.cat([batch["K"]] * 2)

        meta = {
            "K": K,
            "R": R,
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch
        )

        noise_pred = model(latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def inference(self, batch):
        images = batch["images"]
        bs, m, h, w, _ = images.shape
        device = images.device

        latents = torch.randn(bs, m, 4, h // 8, w // 8, device=device)

        prompt_embds = []
        for prompt in batch["prompt"]:
            prompt_embds.append(self.encode_text(prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text("", device)[0]
        prompt_embd = torch.cat([prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for _i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]] * m, dim=1)

            noise_pred = self.forward_cls_free(latents, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        images_pred = self.decode_latent(latents, self.vae)

        return images_pred
