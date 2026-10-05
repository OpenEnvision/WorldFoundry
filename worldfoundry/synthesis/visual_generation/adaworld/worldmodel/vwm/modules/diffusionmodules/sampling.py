"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import torch
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm
from vwm.modules.diffusionmodules.sampling_utils import to_d
from vwm.util import append_dims, default, instantiate_from_config


class BaseDiffusionSampler:
    def __init__(
            self,
            discretization_config: Union[Dict, ListConfig, OmegaConf],
            num_steps: Union[int, None] = None,
            guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
            verbose: bool = False,
            device: str = "cuda",
            n_context_frames: int = 5
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(guider_config)
        self.verbose = verbose
        self.device = device
        self.num_frames = n_context_frames + 1

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling Setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps"
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EulerEDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super(EulerEDMSampler, self).__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        return euler_step

    def replace_cond(self, x, x_ori):
        x = rearrange(x, "(b t) ... -> b t ...", t=self.num_frames)
        x = torch.cat([x_ori[:, :-1], x[:, [-1]]], dim=1)
        x = rearrange(x, "b t ... -> (b t) ...")
        return x

    def __call__(self, denoiser, x, cond, x_ori, uc=None, num_steps=None):
        x_ori = rearrange(x_ori, "(b t) ... -> b t ...", t=self.num_frames)
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.replace_cond(x, x_ori)

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma
            )

        x = self.replace_cond(x, x_ori)
        return x
