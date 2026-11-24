import torch
from einops import rearrange
from torch import nn

# Standalone: only Wan VAE (used by data/models/runtime/configs/infinite_world/inference.yaml)
from .vae import WanVAE


class WanVAEModelWrapper(nn.Module):
    def __init__(self, vae_pth, dtype=torch.float, device="cuda", patch_size=(4, 8, 8)):
        super().__init__()
        self.module = WanVAE(
            vae_pth=vae_pth,
            dtype=dtype,
            device=device,
        )
        self.dtype = dtype
        self.device = device
        self.out_channels = 16
        self.patch_size = patch_size

    def encode(self, x):
        # input: x: B, C, T, H, W or B, C, H, W
        # return: x: B, C, T, H, W
        if len(x.shape) == 4:
            x = rearrange(x, "B C H W -> B C 1 H W")
        x = self.module.encode_batch(x)
        return x

    def decode(self, x):
        # input: x: B, C, T, H, W or B, C, H, W
        # return: x: B, C, T, H, W
        if len(x.shape) == 4:
            x = rearrange(x, "T C H W -> 1 C T H W")
        x = self.module.decode_batch(x)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if i == 0:
                target_size = 1 + (input_size[i] - 1) // self.patch_size[i]
                latent_size.append(target_size)
            else:
                assert input_size[i] % self.patch_size[i] == 0, "Input spatial size must be divisible by patch size"
                target_size = input_size[i] // self.patch_size[i]
                latent_size.append(target_size)
        return latent_size
