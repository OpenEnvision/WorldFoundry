"""Vid2World DDIM sampler with KV-cache injection for autoregressive video.

Extends the legacy LVDM DDIM loop with :class:`~.kv_cache.KVCacheManager`
wired into every Vid2World CrossAttention.  Used when decoding long
videos chunk-by-chunk without re-encoding the full context.

Not a DiffusionScheduler Protocol implementation.  Multi-cond
Vid2World sampling lives in :mod:`.ddim_multiplecond_vid2world`.
"""

import numpy as np
from tqdm import tqdm
import torch
from worldfoundry.base_models.diffusion_model.schedulers.lvdm.schedule_math import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from worldfoundry.core.nn.diffusion.utils import noise_like
from worldfoundry.core.nn.diffusion.utils import extract_into_tensor
import copy
from worldfoundry.base_models.diffusion_model.schedulers.lvdm.kv_cache import KVCacheManager


class DDIMSampler(object):
    """Ddim sampler implementation."""
    def __init__(self, model, schedule="linear", **kwargs):
        """Init.

        Args:
            model: The model.
            schedule: The schedule.
        """
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0
    
    def _inject_kv_cache_manager(self, cache_manager):
        """Inject KV cache manager into all CrossAttention layers."""
        from worldfoundry.base_models.diffusion_model.models.networks.lvdm.attention_vid2world import CrossAttention
        def inject_recursive(module):
            """Inject recursive.

            Args:
                module: The module.
            """
            for child in module.children():
                if isinstance(child, CrossAttention):
                    child.kv_cache_manager = cache_manager
                else:
                    inject_recursive(child)
        inject_recursive(self.model.model.diffusion_model)

    def register_buffer(self, name, attr):
        """Register buffer.

        Args:
            name: The name.
            attr: The attr.
        """
        if type(attr) == torch.Tensor:
            target_device = torch.device(self.model.device)
            if attr.device != target_device:
                attr = attr.to(target_device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        """Make schedule.

        Args:
            ddim_num_steps: The ddim num steps.
            ddim_discretize: The ddim discretize.
            ddim_eta: The ddim eta.
            verbose: The verbose.
        """
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform', #uniform_trailing for starting from last timestep
               guidance_rescale=0.0,
               ar=False,
               ar_alter_history_frames=False,
               ar_noise_schedule=None,
               cond_frame=None,
               z=None,
               **kwargs
               ):
        """Sample.

        Args:
            S: The s.
            batch_size: The batch size.
            shape: The shape.
            conditioning: The conditioning.
            callback: The callback.
            normals_sequence: The normals sequence.
            img_callback: The img callback.
            quantize_x0: The quantize x0.
            eta: The eta.
            mask: The mask.
            x0: The x0.
            temperature: The temperature.
            noise_dropout: The noise dropout.
            score_corrector: The score corrector.
            corrector_kwargs: The corrector kwargs.
            verbose: The verbose.
            schedule_verbose: The schedule verbose.
            x_T: The x t.
            log_every_t: The log every t.
            unconditional_guidance_scale: The unconditional guidance scale.
            unconditional_conditioning: The unconditional conditioning.
            precision: The precision.
            fs: The fs.
            timestep_spacing: The timestep spacing.
            guidance_rescale: The guidance rescale.
            ar: The ar.
            ar_alter_history_frames: The ar alter history frames.
            ar_noise_schedule: The ar noise schedule.
            cond_frame: The cond frame.
            z: The z.
        """
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

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
        if ar:
            assert cond_frame is not None and cond_frame > 0, "cond_frame must be provided and greater than 0 in ar setup"
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
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
                                                    ar=ar,
                                                    ar_alter_history_frames=ar_alter_history_frames,
                                                    ar_noise_schedule=ar_noise_schedule,
                                                    cond_frame=cond_frame,
                                                    z=z,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False, # always False
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,precision=None,fs=None,guidance_rescale=0.0,
                      ar=False,ar_alter_history_frames=False,ar_noise_schedule=None,cond_frame=None, z=None,
                      **kwargs):
        """Ddim sampling.

        Args:
            cond: The cond.
            shape: The shape.
            x_T: The x t.
            ddim_use_original_steps: The ddim use original steps.
            callback: The callback.
            timesteps: The timesteps.
            quantize_denoised: The quantize denoised.
            mask: The mask.
            x0: The x0.
            img_callback: The img callback.
            log_every_t: The log every t.
            temperature: The temperature.
            noise_dropout: The noise dropout.
            score_corrector: The score corrector.
            corrector_kwargs: The corrector kwargs.
            unconditional_guidance_scale: The unconditional guidance scale.
            unconditional_conditioning: The unconditional conditioning.
            verbose: The verbose.
            precision: The precision.
            fs: The fs.
            guidance_rescale: The guidance rescale.
            ar: The ar.
            ar_alter_history_frames: The ar alter history frames.
            ar_noise_schedule: The ar noise schedule.
            cond_frame: The cond frame.
            z: The z.
        """
        device = self.model.betas.device        
        b = shape[0] # shape: (B,C,T,H_latent,W_latent)
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        clean_cond = kwargs.pop("clean_cond", False)
        
        # KV cache setup for AR generation
        use_kv_cache = kwargs.pop("use_kv_cache", False) and ar
        kv_cache_manager = None
        if use_kv_cache:
            kv_cache_manager = KVCacheManager(verbose=verbose)
            self._inject_kv_cache_manager(kv_cache_manager)
            if verbose:
                print(f"[KV Cache] Enabled for AR sampling")

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        if ar:
            original_img = img.clone()  # use clone() instead of deepcopy for tensors
            original_cond = cond  # no need to copy, we only read from it
            if unconditional_conditioning is not None:
                original_ucond = unconditional_conditioning  # no need to copy
            else:
                original_ucond = cond
            num_frames = shape[2]
            # initialize the cache for the noisy history frames
            x_noisy_cache = torch.zeros_like(img, device=device)
            small_t = iterator[-2]
            small_ts_single_frame=torch.full((b,1), small_t, device=device, dtype=torch.long)
            small_ts_full=torch.full((b,num_frames), small_t, device=device, dtype=torch.long)
            # pre-allocate zeros tensor for timesteps (optimization)
            zeros_ts = torch.zeros((b, num_frames), device=device, dtype=torch.long)
            # Determine which step to cache: iterator[-2] is the step used for historical frames
            cache_step_index = len(iterator) - 2  # Second to last step
            for t in tqdm(range(num_frames), desc="Sampling frames", total=num_frames):
                if t < cond_frame:
                    if t==cond_frame-1:
                        img=z[:,:,:t+1,:,:] 
                        small_ts=torch.full((b,t+1), small_t, device=device, dtype=torch.long)
                        x_noisy_cache[:,:,:t+1,:,:]=self.model.q_sample(img[:,:,:t+1,:,:], small_ts)
                    continue
                
                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((b,1), step, device=device, dtype=torch.long)
                    if t > 0:
                        # use pre-allocated zeros instead of creating new tensor
                        ts = torch.cat((zeros_ts[:, :t], ts), dim=1)
                        if i ==0:
                            img = torch.cat((img, original_img[:,:,t:t+1,:,:], ), dim=2)
                    else:
                        if i ==0:
                            img = original_img[:,:,:1,:,:]
                    if i==0:
                        clean_img = img.clone()
                    if t > 0:
                        if ar_noise_schedule is not None:
                            if ar_noise_schedule == 1:
                                """ Add deterministic uniform noise to the history frames """
                                if i==0:
                                    # since the second to last frame is last generated and the noisy version of it is not cached
                                    # (except for t==cond, which just resamples the last frame),
                                    # we need to sample the noisy version of the second to last frame
                                    x_noisy=self.model.q_sample(img[:,:,-2:-1,:,:], small_ts_single_frame)
                                    x_noisy_cache[:,:,t-1:t,:,:]=x_noisy
                                img[:,:,:-1,:,:]=x_noisy_cache[:,:,0:t,:,:]
                                ts=torch.cat((small_ts_full[:,:t], ts[:,-1:]), dim=1)
                            else:
                                raise NotImplementedError

                    cond_t = {}
                    ucond_t = {}
                    cond_t['c_crossattn'] = original_cond['c_crossattn']
                    cond_t['c_concat'] = [original_cond['c_concat'][0][:,:,:t+1,:,:]]
                    if 'c_action' in original_cond.keys():
                        cond_t['c_action'] = [original_cond['c_action'][0][:,:t+1,:]]
                        ucond_t['c_action'] = [original_cond['c_action'][0][:,:t+1,:]]
                    ucond_t['c_crossattn'] = original_ucond['c_crossattn']
                    ucond_t['c_concat'] = [original_ucond['c_concat'][0][:,:,:t+1,:,:]]
                    if 'c_action_mask' in original_cond.keys():
                        cond_t['c_action_mask'] = [original_cond['c_action_mask'][0][:,:t+1]]
                        ucond_t['c_action_mask'] = [torch.cat((original_cond['c_action_mask'][0][:,:t], torch.zeros((original_cond['c_action_mask'][0].shape[0], 1), device=device)), dim=1)] # only dropout on the last frame's action
                    # Prepare KV cache info
                    kv_cache_info = None
                    kv_cache_info_no_store = None
                    if kv_cache_manager is not None:
                        # Cache at iterator[-2] step, which matches the noise level used for historical frames
                        should_cache = (i == cache_step_index)
                        # For CFG: only store on first apply_model call (conditional), not on second (unconditional)
                        kv_cache_info = {
                            'current_frame_idx': t,
                            'should_cache': should_cache,
                        }
                        # For unconditional: don't store to avoid overwriting
                        kv_cache_info_no_store = {
                            'current_frame_idx': t,
                            'should_cache': False,  # Never store on unconditional pass
                        }
                    
                    # assert unconditional_guidance_scale == 1.0
                    outs = self.p_sample_ddim(img, cond_t, ts, index=index, use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=ucond_t,
                                            mask=mask,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                            kv_cache_info=kv_cache_info,
                                            kv_cache_info_uncond=kv_cache_info_no_store,
                                            **kwargs)

                    new_img, pred_x0 = outs
                    if t>0:
                        if ar_alter_history_frames:
                            img = new_img
                        else:
                            if i == len(iterator)-1:
                                img = torch.cat((clean_img[:,:,:-1,:,:], new_img[:,:,-1:,:,:]), dim=2)
                            else:
                                # img = torch.cat((img[:,:,:-1,:,:], new_img[:,:,-1:,:,:]), dim=2)
                                img[:,:,-1:,:,:] = new_img[:,:,-1:,:,:]
                    else:
                        img = new_img
                    if index % log_every_t == 0 or index == total_steps - 1:
                        intermediates['x_inter'].append(img)
                        intermediates['pred_x0'].append(pred_x0)
            # Print KV cache stats after AR generation
            if kv_cache_manager is not None:
                kv_cache_manager.print_stats()
        else:
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                ## use mask to blend noised original latent (img_orig) & new sampled latent (img), this is never used
                if mask is not None:
                    assert x0 is not None
                    if clean_cond:
                        img_orig = x0
                    else:
                        img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                    img = img_orig * mask + (1. - mask) * img # keep original & modify use img
                if unconditional_conditioning == None:
                    unconditional_conditioning = copy.deepcopy(cond)
                    unconditional_conditioning['c_action_mask'] = [torch.zeros_like(cond['c_action_mask'][0]).to(device)]
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        mask=mask,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                        **kwargs)


                img, pred_x0 = outs
                if callback: callback(i) # this is none
                if img_callback: img_callback(pred_x0, i) # this is none

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None,mask=None,x0=None,guidance_rescale=0.0,
                      kv_cache_info=None, kv_cache_info_uncond=None, **kwargs):
        """P sample ddim.

        Args:
            x: The x.
            c: The c.
            t: The t.
            index: The index.
            repeat_noise: The repeat noise.
            use_original_steps: The use original steps.
            quantize_denoised: The quantize denoised.
            temperature: The temperature.
            noise_dropout: The noise dropout.
            score_corrector: The score corrector.
            corrector_kwargs: The corrector kwargs.
            unconditional_guidance_scale: The unconditional guidance scale.
            unconditional_conditioning: The unconditional conditioning.
            uc_type: The uc type.
            conditional_guidance_scale_temporal: The conditional guidance scale temporal.
            mask: The mask.
            x0: The x0.
            guidance_rescale: The guidance rescale.
            kv_cache_info: The kv cache info.
            kv_cache_info_uncond: The kv cache info uncond.
        """
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        # import pdb; pdb.set_trace()
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c, kv_cache_info=kv_cache_info, **kwargs) # unet denoiser
        else:
            ### do_classifier_free_guidance
            if isinstance(c, torch.Tensor) or isinstance(c, dict):
                # The cool thing here is since we only add guidance on the most recent action, the same cache can be reused for cond and ucond
                e_t_cond = self.model.apply_model(x, t, c, kv_cache_info=kv_cache_info, **kwargs) # DynamiCrafter/lvdm/models/ddpm3d.py line: 723
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, kv_cache_info=kv_cache_info_uncond, **kwargs) # the only difference here is that we don't store the cache for unconditional
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None: # default is None
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
        
        if self.model.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
            rescale = (prev_scale_t / scale_t)
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):
        """Decode.

        Args:
            x_latent: The x latent.
            cond: The cond.
            t_start: The t start.
            unconditional_guidance_scale: The unconditional guidance scale.
            unconditional_conditioning: The unconditional conditioning.
            use_original_steps: The use original steps.
            callback: The callback.
        """

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        """Stochastic encode.

        Args:
            x0: The x0.
            t: The t.
            use_original_steps: The use original steps.
            noise: The noise.
        """
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
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
