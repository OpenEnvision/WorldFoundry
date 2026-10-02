import random
import os
import sys
import torch
from functools import partial
import math
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import logging
from safetensors.torch import load_file

from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import T5EncoderModel
from worldfoundry.core.distributed.sharding.block_fsdp import shard_model
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import Wan2_2_VAE
from worldfoundry.base_models.diffusion_model.models.networks.wan.media_geometry import best_output_size, masks_like

from .modules.model import Yume1p5WanModel, Yume1p5WanAttentionBlock


def upsample_conv3d_weights(conv_small, size):
    old_weight = conv_small.weight.data 
    new_weight = F.interpolate(
        old_weight,                      
        size=size,              
        mode='trilinear',             
        align_corners=False           
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0
    )
    conv_large.weight.data = new_weight
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large


class Yume1p5TI2V():  # adapted from wan2p2.WanTI2V

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        
        
        self.sample_neg_prompt = config.sample_neg_prompt

        # Specific to Yume 1.5
        self.sp_size = 1

        logging.info(f"Creating Yume1p5Model from {checkpoint_dir}")

        config_wan = {
            "_class_name": "WanModel",
            "_diffusers_version": "0.33.0",
            "dim": 3072,
            "eps": 1e-06,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "in_dim": 48,
            "model_type": "ti2v",
            "num_heads": 24,
            "num_layers": 30,
            "out_dim": 48,
            "text_len": 512
            }

        self.model = Yume1p5WanModel.from_config(config_wan)
        self.model.patch_embedding_2x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,4,4))
        self.model.patch_embedding_4x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,8,8))
        self.model.patch_embedding_8x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,16,16))
        self.model.patch_embedding_16x = upsample_conv3d_weights(deepcopy(self.model.patch_embedding),(1,32,32))
        self.model.patch_embedding_2x_f = torch.nn.Conv3d(48, 48, kernel_size=(1,4,4), stride=(1,4,4))
        self.model.sideblock = Yume1p5WanAttentionBlock(self.model.dim, self.model.ffn_dim, self.model.num_heads, self.model.window_size, \
                                                    self.model.qk_norm, self.model.cross_attn_norm, self.model.eps)
        self.model.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, self.model.dim, device=self.model.device)
        )

        self.model = Yume1p5WanModel.from_pretrained(checkpoint_dir)
        state_dict = load_file(checkpoint_dir + "/diffusion_pytorch_model.safetensors")
        self.model.load_state_dict(state_dict)

        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 # specific to Yume 1.5
                 current_latent_num=8):

        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                n_prompt=n_prompt,
                seed=seed,
                # specific to Yume 1.5
                current_latent_num=current_latent_num)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            n_prompt="",
            seed=-1,
            offload_model=True):

        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # specific to Yume 1.5
        noise = torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g)

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}
        
        return arg_c, arg_null, noise

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            n_prompt="",
            seed=-1,
            # specific to Yume 1.5
            current_latent_num=8):

        # preprocess
        ih, iw = img.shape[2:] # specific to Yume 1.5
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)


        # Comment for Yume 1.5

        # scale = max(ow / iw, oh / ih)
        # img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # # center-crop
        # x1 = (img.width - ow) // 2
        # y1 = (img.height - oh) // 2
        # img = img.crop((x1, y1, x1 + ow, y1 + oh))
        # assert img.width == ow and img.height == oh

        # # to tensor
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)


        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            # comment for Yume 1.5
            # if offload_model:
            #    self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]


        # specific to Yume 1.5
        z = img
        C, F_z, H, W = z.shape
        _, F_target, _, _ = noise.shape
        
        padding = F_target - F_z
        z = torch.cat([z, torch.zeros_like(z[:, -1:, :, :]).repeat(1, padding, 1, 1)], dim=1)
        z = [z]
        
        # sample videos
        latent = noise
        mask1, mask2 = masks_like([noise], zero=True, current_latent_num=current_latent_num)
        latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
        
        arg_c = {
            'context': [context[0]],
            'seq_len': seq_len,
        }

        arg_null = {
            'context': context_null,
            'seq_len': seq_len,
        }

        return arg_c, arg_null, noise, mask2, z

