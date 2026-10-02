import math
import time
import numpy as np

from tqdm import tqdm
from typing import Callable
from einops import rearrange
from functools import partial

import torch
from torch.distributions import LogisticNormal

from worldfoundry.core.distributed.model_parallel import context_state as context_parallel_util

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py
# and https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def mean_flat(tensor: torch.Tensor, stoploss_mask=None):
    """
    Take the mean over all non-batch dimensions.
        tensor: [B, C, T, H, W]
        stoploss_mask: [B, T, H, W]
    """
    if stoploss_mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        stoploss_mask = stoploss_mask.unsqueeze(1).expand_as(tensor) # [B, T, H, W] --> [B, C, T, H, W]
        assert tensor.shape == stoploss_mask.shape, f"shape of tensor {tensor.shape} and stoploss_mask {stoploss_mask.shape} should be the same"
        loss_mask = ~stoploss_mask
        masked_loss = tensor * loss_mask
        sum_loss = masked_loss.sum(dim=list(range(1, len(tensor.shape))))
        count_nonzero = loss_mask.sum(dim=list(range(1, len(tensor.shape))))
        mean_loss = sum_loss / count_nonzero.clamp(min=1)

        return mean_loss


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))



def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        shift=5.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        use_reversed_velocity=False,
        cfg_scale=7.0,
        **kwargs,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_reversed_velocity = use_reversed_velocity
        self.cfg_scale = cfg_scale

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        self.shift = shift
        sigmas = torch.linspace(0, 1, num_timesteps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = sigmas * num_timesteps

        y = torch.exp(-2 * ((self.timesteps - num_timesteps/2) / num_timesteps)**2)
        y_shifted = y - y.min()
        self.bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, x_ignore_mask=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                latent_size = x_start.shape[-3:]
                t = timestep_transform(t, shift=self.shift, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape


        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(noise)
            context_parallel_util.cp_broadcast(t)

        x_t = self.add_noise(x_start, noise, t)


        target = x_start - noise
        if self.use_reversed_velocity:
            target = -target

        terms = {}
        model_output = model(x_t, t, x_ignore_mask=x_ignore_mask, **model_kwargs)
        velocity_pred = model_output

        T = target.shape[2]
        loss = mean_flat((velocity_pred[:, :, -T:] - target).pow(2), stoploss_mask=x_ignore_mask[:, -T:])

        # # get loss weight
        # timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - t.unsqueeze(1).to(self.timesteps.device)).abs(), dim=1)
        # weights = self.bsmntw_weighing[timestep_id]
        # loss = weights.to(loss) * loss

        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def sample(
        self,
        model,
        text_encoder,
        null_embedder,
        z_size,
        prompts,
        device,
        mask=None,
        guidance_scale=None,
        negative_prompts=None,
        additional_args=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        z = torch.randn(*z_size, device=device)

        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(z)
        
        # For performance alignment
        # from source.opensora.utils.inference_utils import apply_mask_strategy
        # mask = apply_mask_strategy(z, [[]], [""], 0, align=5)
        
        assert negative_prompts is None or len(negative_prompts) in [n, 1], \
            "Invalid negative prompts."

        if negative_prompts:
            if len(negative_prompts) == 1: negative_prompts *= n
            prompts = prompts + negative_prompts

        batch_size = len(prompts)
        if context_parallel_util.get_cp_rank() == 0:
            model_args = text_encoder.encode(prompts)
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(model_args['y'])
                context_parallel_util.cp_broadcast(model_args['y_mask'])
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = text_encoder.output_dim
            model_max_length = text_encoder.model_max_length
            y_tensor = torch.zeros([batch_size, 1, model_max_length, caption_channels], dtype=torch.float32, device=device)
            y_mask_tensor = torch.zeros([batch_size, model_max_length], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(y_tensor)
            context_parallel_util.cp_broadcast(y_mask_tensor)
            model_args = {
                "y" : y_tensor,
                "y_mask": y_mask_tensor,
            }

        assert negative_prompts, "Not support uncond training now, pls use negative prompt for uncond."
        if not negative_prompts:
            uncond = null_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
            model_args["y"] = torch.concat([model_args["y"], uncond])

        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = list(np.linspace(self.num_timesteps, 1, self.num_sampling_steps, dtype=np.float32))
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            latent_size = z_size[-3:]
            timesteps = [timestep_transform(t, shift=self.shift, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        model_args["image_cond"] = model_args["image_cond"].repeat(2, 1, 1, 1, 1)
        progress_wrap = partial(tqdm, total=len(timesteps)) if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()

                x0_noise = torch.randn_like(x0)
                if context_parallel_util.get_cp_size() > 1:
                    context_parallel_util.cp_broadcast(x0_noise)

                x_noise = self.scheduler.add_noise(x0, x0_noise, t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)

            t = torch.cat([t, t], 0)
            start = time.time()
            pred = model(z_in, t, **model_args)
            pred = pred[:, :, -z_in.shape[2]:]
            end = time.time()

            print(f"Step {i} Forward time: {end - start:.4f} seconds")
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # When model predict noise-z0, the actual velocity is (v_pred * -1)
            if self.use_reversed_velocity:
                v_pred = -v_pred

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
        return z
