"""Flow-matching samplers used by FramePack inference."""

import math

import torch
from tqdm.auto import trange

from worldfoundry.synthesis.visual_generation.framepack.framepack_runtime.diffusers_helper.utils import (
    repeat_to_batch_size,
)


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    if guidance_rescale == 0:
        return noise_cfg

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg


def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):
        dtype = extra_args["dtype"]
        cfg_scale = extra_args["cfg_scale"]
        cfg_rescale = extra_args["cfg_rescale"]
        concat_latent = extra_args["concat_latent"]

        original_dtype = x.dtype
        sigma = sigma.float()

        x = x.to(dtype)
        timestep = (sigma * t_scale).to(dtype)

        hidden_states = x if concat_latent is None else torch.cat([x, concat_latent.to(x)], dim=1)

        pred_positive = transformer(
            hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args["positive"]
        )[0].float()

        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            pred_negative = transformer(
                hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args["negative"]
            )[0].float()

        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale)

        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)

        return x0.to(dtype=original_dtype)

    return k_model


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class FlowMatchUniPC:
    def __init__(self, model, extra_args, variant="bh1"):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args

    def model_fn(self, x, t):
        return self.model(x, t, **self.extra_args)

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        assert order <= len(model_prev_list)
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = -torch.log(t_prev_0)
        lambda_t = -torch.log(t)
        model_prev_0 = model_prev_list[-1]

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = -torch.log(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError("Bad variant!")

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        use_predictor = len(D1s) > 0

        if use_predictor:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = torch.tensor([0.5], device=b.device) if order == 2 else torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            rhos_p = None

        rhos_c = torch.tensor([0.5], device=b.device) if order == 1 else torch.linalg.solve(R, b)

        x_t_ = expand_dims(t / t_prev_0, dims) * x - expand_dims(h_phi_1, dims) * model_prev_0

        pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0])) if use_predictor else 0

        x_t = x_t_ - expand_dims(B_h, dims) * pred_res
        model_t = self.model_fn(x_t, t)

        corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0])) if D1s is not None else 0

        D1_t = model_t - model_prev_0
        x_t = x_t_ - expand_dims(B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None, disable_pbar=False):
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        for i in trange(len(sigmas) - 1, disable=disable_pbar):
            vec_t = sigmas[i].expand(x.shape[0])

            if i == 0:
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
            elif i < order:
                init_order = i
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)

            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]

            if callback is not None:
                callback({"x": x, "i": i, "denoised": model_prev_list[-1]})

        return model_prev_list[-1]


def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant="bh1"):
    assert variant in ["bh1", "bh2"]
    return FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(
        noise, sigmas=sigmas, callback=callback, disable_pbar=disable
    )


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


@torch.inference_mode()
def sample_hunyuan(
    transformer,
    sampler="unipc",
    initial_latent=None,
    concat_latent=None,
    strength=1.0,
    width=512,
    height=512,
    frames=16,
    real_guidance_scale=1.0,
    distilled_guidance_scale=6.0,
    guidance_rescale=0.0,
    shift=None,
    num_inference_steps=25,
    batch_size=None,
    generator=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    prompt_poolers=None,
    negative_prompt_embeds=None,
    negative_prompt_embeds_mask=None,
    negative_prompt_poolers=None,
    dtype=torch.bfloat16,
    device=None,
    negative_kwargs=None,
    callback=None,
    **kwargs,
):
    device = device or transformer.device

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    latents = torch.randn(
        (batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device
    ).to(device=device, dtype=torch.float32)

    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4

    mu = calculate_flux_mu(seq_length, exp_max=7.0) if shift is None else math.log(shift)

    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)

    k_model = fm_wrapper(transformer)

    if initial_latent is not None:
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(device=device, dtype=torch.float32)
        initial_latent = initial_latent.to(device=device, dtype=torch.float32)
        latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)

    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    sampler_kwargs = {
        "dtype": dtype,
        "cfg_scale": real_guidance_scale,
        "cfg_rescale": guidance_rescale,
        "concat_latent": concat_latent,
        "positive": dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        "negative": dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        ),
    }

    if sampler == "unipc":
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f"Sampler {sampler} is not supported.")

    return results
