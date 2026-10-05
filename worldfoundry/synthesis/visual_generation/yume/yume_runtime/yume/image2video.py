import gc
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors import safe_open
from safetensors.torch import load_file

from worldfoundry.core.distributed.sharding.block_fsdp import shard_model
from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import CLIPModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_21 import WanModel
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import T5EncoderModel
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import WanVAE

def upsample_conv3d_weights(conv_small: nn.Conv3d, size: Tuple[int, int, int]):
    old_weight = conv_small.weight.data
    new_weight = F.interpolate(
        old_weight,
        size=size,
        mode="trilinear",
        align_corners=False,
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0,
    )
    conv_large.weight.data = new_weight
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large


def load_yume_checkpoint(model: nn.Module, checkpoint_dir: str) -> nn.Module:
    device = next(model.parameters()).device
    index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        unique_shards = set(index_data["weight_map"].values())
        model_weights = {}

        for shard_name in unique_shards:
            shard_path = os.path.join(checkpoint_dir, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    model_weights[key] = f.get_tensor(key).to(device)
    else:
        full_model_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"No model files found in: {checkpoint_dir}")
        model_weights = load_file(full_model_path, device=device)

    model.load_state_dict(model_weights, strict=False)
    del model_weights
    torch.cuda.empty_cache()
    gc.collect()
    return model


class YumeI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        dit_fsdp=False,
    ):
        self.config = config
        self.device = torch.device(f"cuda:{device_id}")
        self.patch_size = config.patch_size
        self.vae_stride = config.vae_stride
        self.sample_neg_prompt = config.sample_neg_prompt
        self.sp_size = 1
        self.t5_cpu = False

        config_wan = {
            "model_type": "i2v",
            "text_len": 512,
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
            "_class_name": "WanModel",
            "_diffusers_version": "0.30.0",
        }

        self.model = WanModel.from_config(config_wan)
        self.model.patch_embedding_2x = upsample_conv3d_weights(
            deepcopy(self.model.patch_embedding), (1, 4, 4)
        )
        self.model.patch_embedding_2x_f = torch.nn.Conv3d(
            36, 36, kernel_size=(1, 4, 4), stride=(1, 4, 4)
        )
        self.model.patch_embedding_4x = upsample_conv3d_weights(
            deepcopy(self.model.patch_embedding), (1, 8, 8)
        )
        self.model.patch_embedding_8x = upsample_conv3d_weights(
            deepcopy(self.model.patch_embedding), (1, 16, 16)
        )
        self.model.patch_embedding_16x = upsample_conv3d_weights(
            deepcopy(self.model.patch_embedding), (1, 32, 32)
        )

        self.model = load_yume_checkpoint(self.model, os.path.join(checkpoint_dir, "Yume-Dit"))
        self.model.eval().requires_grad_(False)
        # The released shards materialize as float32 tensors. Cast on CPU before
        # the device transfer so the 14B DiT does not consume roughly 64 GiB of
        # H100 memory while T5, CLIP, and the VAE are still being initialized.
        self.model.to(dtype=config.param_dtype)

        if dit_fsdp:
            self.model = shard_model(self.model, device_id=device_id)
        else:
            self.model.to(self.device)

        self.text_encoder: Optional[T5EncoderModel] = None
        self.vae: Optional[WanVAE] = None
        self.clip: Optional[CLIPModel] = None

    def init_model(
        self,
        config,
        checkpoint_dir,
        device_id=None,
        t5_cpu=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.t5_cpu = t5_cpu

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device="cpu",
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )
        self.text_encoder.model.to(torch.bfloat16)
        self.text_encoder.model.eval().requires_grad_(False)
        if not t5_cpu:
            self.text_encoder.model.to(self.device)

        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
        )
        self.vae.model.eval().requires_grad_(False)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device="cpu",
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer),
        )
        self.clip.model.to(torch.bfloat16).to(self.device)
        self.clip.model.eval().requires_grad_(False)

    def _check_ready(self):
        if self.text_encoder is None or self.vae is None or self.clip is None:
            raise RuntimeError("Call init_model() before generate()/generate_next().")

    def generate(
        self,
        model_input,
        device,
        input_prompt,
        img,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        rand_num_img=None,
        offload_model=False,
        clip_context=None,
        flag_sample=False,
    ):
        self._check_ready()

        if rand_num_img is None:
            rand_num_img = 0.6

        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        frame_num = model_input.shape[1]

        h, w = img.shape[1:]
        frame_zero = 32 if flag_sample else 33

        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        if rand_num_img < 0.4:
            img = model_input[:, 0, :, :]
        else:
            img = model_input[:, 0:-frame_zero, :, :]

        if rand_num_img >= 0.4:
            model_input = torch.cat(
                [
                    self.vae.encode([model_input[:, 0:-frame_zero]])[0],
                    self.vae.encode([model_input[:, -frame_zero:]])[0],
                ],
                dim=1,
            )
        else:
            model_input = self.vae.encode([model_input])[0]

        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2]
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        noise = torch.randn_like(model_input)

        if rand_num_img < 0.4:
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
        else:
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, -frame_zero:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            context = self.text_encoder([input_prompt], self.device)
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context = [context[0].to(self.device)]

        cache_path_null = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
            "paintings, images, static, overall gray, worst quality, low quality, JPEG "
            "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, "
            "walking backwards"
        )
        if not self.t5_cpu:
            context_null = self.text_encoder([cache_path_null], self.device)
        else:
            context_null = self.text_encoder([cache_path_null], torch.device("cpu"))
            context_null = [context_null[0].to(self.device)]

        self.clip.model.to(self.device)
        if rand_num_img < 0.4:
            if clip_context is None:
                clip_context = self.clip.visual([img[:, None, :, :]])
        else:
            if clip_context is None:
                clip_context = self.clip.visual([img[:, -1, :, :].unsqueeze(1)])

        if rand_num_img < 0.4:
            y = self.vae.encode(
                [
                    torch.concat(
                        [
                            img[None].cpu().transpose(0, 1),
                            torch.zeros(3, frame_num - 1, h, w),
                        ],
                        dim=1,
                    ).to(self.device)
                ]
            )[0]
        else:
            y = self.vae.encode(
                [
                    torch.concat(
                        [
                            img.cpu(),
                            torch.zeros(3, frame_zero, h, w),
                        ],
                        dim=1,
                    ).to(self.device)
                ]
            )[0]

        y = torch.concat([msk, y])

        sigmas = torch.sigmoid(torch.randn((1,), device=self.device))
        timesteps = (sigmas * 1000).view(-1)

        arg_c = {
            "context": [context[0]],
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y],
        }
        arg_null = {
            "context": context_null,
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y],
        }
        latent_model_input = [noise.to(self.device).squeeze()]
        timestep = torch.tensor([timesteps[0]], device=self.device)

        return latent_model_input, timestep, arg_c, noise, model_input, clip_context, arg_null

    def generate_next(
        self,
        model_input,
        model_input_1,
        device,
        input_prompt,
        img,
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        rand_num_img=None,
        offload_model=False,
        clip_context=None,
        flag_sample=False,
    ):
        self._check_ready()

        if rand_num_img is None:
            rand_num_img = 0.6

        frame_zero = 32 if flag_sample else 33
        frame_num = model_input.shape[1]
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        h, w = img.shape[1:]
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        img = model_input
        model_input = model_input_1

        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2]
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        noise = torch.randn(
            16,
            ((frame_num - 1) // self.vae_stride[0] + 1),
            lat_h,
            lat_w,
            dtype=torch.float32,
            device=self.device,
        )

        msk = torch.ones(1, frame_zero + img.shape[1], lat_h, lat_w, device=self.device)
        msk[:, -frame_zero:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            context = self.text_encoder([input_prompt], self.device)
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context = [context[0].to(self.device)]

        cache_path_null = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
            "paintings, images, static, overall gray, worst quality, low quality, JPEG "
            "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, "
            "walking backwards"
        )
        if not self.t5_cpu:
            context_null = self.text_encoder([cache_path_null], self.device)
        else:
            context_null = self.text_encoder([cache_path_null], torch.device("cpu"))
            context_null = [context_null[0].to(self.device)]

        self.clip.model.to(self.device)
        if rand_num_img < 0.4:
            if clip_context is None:
                clip_context = self.clip.visual([img[:, None, :, :]])
        else:
            if clip_context is None:
                clip_context = self.clip.visual([img[:, -1, :, :].unsqueeze(1)])

        y = self.vae.encode(
            [
                torch.concat(
                    [
                        img.cpu(),
                        torch.zeros(3, frame_zero, h, w),
                    ],
                    dim=1,
                ).to(self.device)
            ]
        )[0]
        y = torch.concat([msk, y])

        sigmas = torch.sigmoid(torch.randn((1,), device=self.device))
        timesteps = (sigmas * 1000).view(-1)

        arg_c = {
            "context": [context[0]],
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y],
        }
        arg_null = {
            "context": context_null,
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y],
        }
        timestep = torch.tensor([timesteps[0]], device=self.device)

        return None, timestep, arg_c, noise, model_input, clip_context, arg_null
