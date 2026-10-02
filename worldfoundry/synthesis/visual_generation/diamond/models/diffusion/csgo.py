"""CS:GO conditioning on the shared DIAMOND network and EDM sampler.

The released CS:GO branch adds history-noise embeddings, multi-hot actions
and a spatial upsampler. UNet, normalization, output quantization and solver
updates remain shared with Atari. Attribution: root THIRD-PARTY-NOTICES.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..blocks import Conv3x3, FourierFeatures
from .denoiser import Conditioners, Denoiser, DenoiserConfig, add_dims
from .diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from .inner_model import InnerModel


class CsgoInnerModel(InnerModel):
    def __init__(self, cfg, *, is_upsampler=False):
        super().__init__(cfg)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)
        if is_upsampler:
            self.act_emb = None
            self.conv_in = Conv3x3((cfg.num_steps_conditioning + 2) * cfg.img_channels, cfg.channels[0])

    def conditioning(self, c_noise: Tensor, act: Tensor | None, *, c_noise_cond: Tensor) -> Tensor:
        if self.act_emb is None:
            if act is not None:
                raise ValueError("DIAMOND upsampler does not accept actions")
            action = 0
        elif act is not None and act.ndim == 2:
            action = self.act_emb(act)
        elif act is not None and act.ndim == 3 and act.shape[-1] == self.act_emb[0].num_embeddings:
            if not bool(((act == 0) | (act == 1)).all()):
                raise ValueError("DIAMOND CS:GO actions must be binary multi-hot vectors")
            action = self.act_emb[1](act.float() @ self.act_emb[0].weight)
        else:
            raise ValueError("DIAMOND CS:GO expects action indices or multi-hot action vectors")
        return self.cond_proj(self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + action)


@dataclass
class CsgoDenoiserConfig(DenoiserConfig):
    noise_previous_obs: bool = False
    upsampling_factor: int | None = None


@dataclass
class CsgoConditioners(Conditioners):
    c_noise_cond: Tensor


class CsgoDenoiser(Denoiser):
    def __init__(self, cfg: CsgoDenoiserConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.is_upsampler = cfg.upsampling_factor is not None
        self.inner_model = CsgoInnerModel(cfg.inner_model, is_upsampler=self.is_upsampler)

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        # Preserve the upstream offset-noise RNG draw even when its scale is zero.
        offset = sigma_offset_noise * torch.randn(*x.shape[:2], 1, 1, device=self.device)
        return x + offset + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Tensor | None = None):
        cs = super().compute_conditioners(sigma)
        noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(cs.c_noise)
        return CsgoConditioners(**vars(cs), c_noise_cond=noise_cond)

    def compute_model_output(self, noisy_next_obs, obs, act, cs):
        return self.inner_model(
            noisy_next_obs * cs.c_in, cs.c_noise, obs / self.cfg.sigma_data, act,
            c_noise_cond=cs.c_noise_cond,
        )

    @torch.no_grad()
    def denoise(self, noisy_next_obs, sigma, obs, act, *, sigma_cond=None):
        cs = self.compute_conditioners(sigma, sigma_cond)
        output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        return self.wrap_model_output(noisy_next_obs, output, cs)


@dataclass
class CsgoSamplerConfig(DiffusionSamplerConfig):
    s_cond: float = 0


class CsgoSampler(DiffusionSampler):
    def prepare_conditioning(self, prev_obs):
        sigma_cond = None
        if self.cfg.s_cond > 0:
            sigma_cond = torch.full((prev_obs.shape[0],), self.cfg.s_cond, device=prev_obs.device)
            prev_obs = self.denoiser.apply_noise(prev_obs, sigma_cond, sigma_offset_noise=0)
        return prev_obs, {"sigma_cond": sigma_cond}
