"""DDIM sampler used by LVDM / VideoCrafter recipes (including T2V-Turbo).

Legacy stateful solver that predates :class:`~...contracts.DiffusionScheduler`.
Recipes that still need DDIM construct :class:`DDIMSampler` instead of going
through Diffusers.  Numerical helpers live in :mod:`.schedule_math`.
"""

import numpy as np
import torch
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.schedulers.lvdm.schedule_math import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    rescale_noise_cfg,
)
from worldfoundry.core.nn.diffusion.utils import extract_into_tensor, noise_like


class DDIMSampler:
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) is torch.Tensor:
            target_device = torch.device(self.model.device)
            if attr.device != target_device:
                attr = attr.to(target_device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod.to(torch.float32)
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"

        def to_torch(x):
            return x.clone().detach().to(self.model.device).to(torch.float32)

        self.use_scale = getattr(self.model, "use_scale", False)

        if getattr(self.model, "use_dynamic_rescale", False):
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])
        if self.use_scale:
            self.register_buffer("scale_arr", to_torch(self.model.scale_arr))
            self.register_buffer("scale_arr_prev", to_torch(self.model.scale_arr_prev))
            self.register_buffer("ddim_scale_arr", self.scale_arr[self.ddim_timesteps])
            self.register_buffer(
                "ddim_scale_arr_prev",
                torch.cat([self.scale_arr[0:1], self.scale_arr[self.ddim_timesteps[:-1]]]),
            )

        # print(alphas_cumprod)
        # print('calc again')
        # print(torch.log(1. - alphas_cumprod))
        # print('retr')
        # print(self.model.sqrt_alphas_cumprod)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(self.model.alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(self.model.sqrt_alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(self.model.sqrt_one_minus_alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(self.model.log_one_minus_alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(self.model.sqrt_recip_alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(self.model.sqrt_recipm1_alphas_cumprod))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        schedule_verbose=False,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        precision=None,
        fs=None,
        timestep_spacing="uniform",  # uniform_trailing for starting from last timestep
        guidance_rescale=0.0,
        **kwargs,
    ):
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                first_conditioning = conditioning[list(conditioning.keys())[0]]  # noqa: RUF015 - Keeps eager list construction and IndexError.
                if torch.is_tensor(first_conditioning):
                    cbs = first_conditioning.shape[0]
                else:
                    cbs = first_conditioning[0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=schedule_verbose)

        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            verbose=verbose,
            precision=precision,
            fs=fs,
            guidance_rescale=guidance_rescale,
            **kwargs,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        verbose=True,
        precision=None,
        fs=None,
        guidance_rescale=0.0,
        **kwargs,
    ):
        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device) if x_T is None else x_T
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)
                # print('11111')
            if isinstance(precision, torch.dtype):
                img = img.to(dtype=precision)
                # print('2222')
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        ### 2024.5.15 Guangyi adds two lines for gradio progress
        if "gr_progress_bar" in kwargs:
            iterator = kwargs["gr_progress_bar"].tqdm(
                time_range,
                desc=f"DDIM Sampler (Round {kwargs['round_info'][0]}/{kwargs['round_info'][1]})",
                total=total_steps,
            )
        elif verbose:
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        else:
            iterator = time_range

        clean_cond = kwargs.pop("clean_cond", False)

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                if clean_cond:  # noqa: SIM108 - Branches carry per-case comments.
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1.0 - mask) * img  # keep original & modify use img

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                x0=x0,
                fs=fs,
                guidance_rescale=guidance_rescale,
                **kwargs,
            )

            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        uc_type=None,
        conditional_guidance_scale_temporal=None,
        mask=None,
        x0=None,
        guidance_rescale=0.0,
        **kwargs,
    ):
        b, *_, device = *x.shape, x.device
        is_video = x.dim() == 5

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c, **kwargs)  # unet denoiser
        else:
            ### do_classifier_free_guidance
            def drop_duplicate_kwargs(conditioning, call_kwargs):
                if isinstance(conditioning, dict):
                    return {key: value for key, value in call_kwargs.items() if key not in conditioning}
                return call_kwargs

            def disable_motion_conditioning(conditioning, call_kwargs):
                if isinstance(conditioning, dict):
                    conditioning = conditioning.copy()
                    conditioning.pop("features_adapter", None)
                call_kwargs = call_kwargs.copy()
                call_kwargs.pop("features_adapter", None)
                return conditioning, call_kwargs

            disable_motion = "cond_T" in kwargs and bool(torch.all(t < kwargs["cond_T"]).item())
            if isinstance(c, torch.Tensor):
                cond_kwargs = kwargs.copy()
                uncond_kwargs = kwargs.copy()
                uncond = unconditional_conditioning
                if isinstance(uncond, dict):
                    for uk, uv in uncond.items():
                        if uk in uncond_kwargs:
                            uncond_kwargs[uk] = uv
                    uncond = uncond["uc"]
                if disable_motion:
                    cond_kwargs.pop("features_adapter", None)
                    uncond_kwargs.pop("features_adapter", None)
                e_t_cond = self.model.apply_model(x, t, c, **cond_kwargs)
                e_t_uncond = self.model.apply_model(x, t, uncond, **uncond_kwargs)
                temporal_kwargs = cond_kwargs
            elif isinstance(c, dict):
                cond_payload = c
                uncond_payload = unconditional_conditioning
                cond_kwargs = kwargs.copy()
                uncond_kwargs = kwargs.copy()
                if disable_motion:
                    cond_payload, cond_kwargs = disable_motion_conditioning(cond_payload, cond_kwargs)
                    uncond_payload, uncond_kwargs = disable_motion_conditioning(uncond_payload, uncond_kwargs)
                cond_kwargs = drop_duplicate_kwargs(cond_payload, cond_kwargs)
                uncond_kwargs = drop_duplicate_kwargs(uncond_payload, uncond_kwargs)
                e_t_cond = self.model.apply_model(x, t, cond_payload, **cond_kwargs)
                e_t_uncond = self.model.apply_model(x, t, uncond_payload, **uncond_kwargs)
                temporal_kwargs = cond_kwargs
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **temporal_kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **temporal_kwargs)
                model_output = model_output + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", "not implemented"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        )
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        size = (b, 1, 1, 1, 1) if is_video else (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device, dtype=x.dtype)
        a_prev = torch.full(size, alphas_prev[index], device=device, dtype=x.dtype)
        sigma_t = torch.full(size, sigmas[index], device=device, dtype=x.dtype)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device, dtype=x.dtype)
        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if getattr(self.model, "use_dynamic_rescale", False):
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device, dtype=x.dtype)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device, dtype=x.dtype)
            rescale = prev_scale_t / scale_t
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise).to(dtype=x.dtype) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device, dtype=x.dtype)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device, dtype=x.dtype)
            pred_x0 = pred_x0 / scale_t
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
        callback=None,
    ):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            if callback:
                callback(i)
        return x_dec

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
