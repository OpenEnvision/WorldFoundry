# https://github.com/LTH14/mar/tree/main/

import math

import mup
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..config import DiffusionGenieConfig
from .diffloss import DiffLoss
from .st_mask_git import STMaskGIT


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking, dim=-1, index=order[:, : mask_len.long()], src=torch.ones(bsz, seq_len).cuda()
    ).bool()
    return masking


class FixedMuReadout(mup.MuReadout):
    def forward(self, x):
        """
        Using `return super(mup.MuReadout, self).forward(self.output_mult * x / self.width_mult())` with `torch.compile`
        results in two divisions by `self.width_mult()` for some reason
        """
        # return F.linear(self.output_mult * x / self.width_mult(), self.weight, self.bias)  # equivalent
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())


class STMAR(STMaskGIT):
    """Spatial-Time MAR with VisionTransformer backbone"""

    def __init__(self, config: DiffusionGenieConfig):
        self.diffloss_w = config.diffloss_w
        self.diffloss_d = config.diffloss_d
        self.num_sampling_steps = config.num_sampling_steps
        self.grad_checkpointing = config.grad_checkpointing

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.patch_size = config.patch_size
        self.vae_stride = config.vae_stride
        self.buffer_size = config.buffer_size
        self.vae_embed_dim = config.vae_embed_dim
        self.maskgit_steps = config.maskgit_steps
        super().__init__(config)

        # --------------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.vae_embed_dim))
        self.token_embed = nn.Linear(
            config.vae_embed_dim * self.config.patch_size**2, config.d_model, bias=False
        )  # hard coded
        cls = FixedMuReadout if config.use_mup else nn.Linear  # (Fixed)MuReadout might slow dow down compiled training?
        self.out_x_proj = cls(config.d_model, config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.z_proj_ln = nn.LayerNorm(config.d_model, eps=1e-6)
        self.seq_len = config.S // (self.config.patch_size**2)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len * config.T, config.d_model))

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=config.vae_embed_dim * self.config.patch_size**2,
            z_channels=config.d_model,
            width=config.diffloss_w,
            depth=config.diffloss_d,
            num_sampling_steps=config.num_sampling_steps,
            grad_checkpointing=config.grad_checkpointing,
        )

        self.diffusion_batch_mul = config.diffusion_batch_mul
        self.initialize_weights()

    def init_action_projectors(
        self,
        domains: list[str],
        d_actions: list[int],
        action_stats: list[list[list[float]]],
        action_network: str = "mlp",
    ):
        super().init_action_projectors(domains, d_actions, action_stats, action_network, use_diffusion=True)
        self.action_diff_losses = nn.ModuleDict()

        # action heads are heterogeneous
        for domain, d_action in zip(self.config.action_domains, self.config.d_actions, strict=False):
            self.action_diff_losses[domain] = DiffLoss(
                target_channels=d_action,
                z_channels=self.config.d_model,
                width=self.diffloss_w,
                depth=self.diffloss_d,
                num_sampling_steps=self.num_sampling_steps,
                grad_checkpointing=self.grad_checkpointing,
            )

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm parameters
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        self.init_weights()

    def set_mup_shapes(self, rescale_params=False):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256  # currently hardcoding to this shape
        base_model = STMAR(base_config)
        if hasattr(self, "action_preprocessor"):
            for base_layer, layer in zip(base_model.decoder.layers, self.decoder.layers, strict=False):
                base_layer.action_projectors = layer.action_projectors
            base_model.action_preprocessor = self.action_preprocessor

        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    def compute_latents(self, x_THW, action_ids: torch.Tensor = None, domain=None, action_mask=None, **kwargs):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        pos_embed_TSC = self.pos_embed_TSC
        diffusion_pos_embed_learned = self.diffusion_pos_embed_learned
        b, t, h, w, c = x_THW.shape
        x_TSC = rearrange(x_THW, "B T H W C -> B T (H W) C").float()
        x_TSC = self.token_embed(x_TSC)
        T = x_TSC.shape[1]

        if action_ids is not None:
            # currently, action_preprocessor just normalizes the actions
            skip_normalization = kwargs.get("skip_normalization", False)
            if not skip_normalization:
                action_ids = self.action_preprocessor[domain[0]](action_ids)
            action_ids = self.action_mlp[domain[0]](action_ids)  # [B, T, D]

            if "concat" in self.config.action_network:
                # randomly dropped the conditioning
                if self.config.action_network == "resampler_concat":
                    action_condition = self.action_projectors[domain[0]](action_ids[:, :T])
                else:
                    action_condition = action_ids[:, :T, None].repeat(
                        1, 1, self.config.action_token_size, 1
                    )  # [B, T, S, C]

                # we add masked tokens between 0 (fully unmasked as in video pred) and 1 (fully masked as in policies) for training losses
                # if we have actions and are trying to predict actions
                x_TSC = torch.concat((x_TSC, action_condition), dim=2)  # [B, T, S, C]

        elif self.config.jointly_predict_actions:
            # all masked when predicting actions and there is no input actions
            action_condition = self.action_mask_tokens[:, :T].repeat(1, 1, self.config.action_token_size, 1)
            x_TSC = torch.concat((x_TSC, action_condition), dim=2)  # [B, T, S, C]

        x_TSC = self.z_proj_ln(x_TSC + pos_embed_TSC[:, : x_TSC.shape[1], : x_TSC.shape[2]])

        # additive position embeddings, using the same vocab space
        domain = domain[0] if domain is not None else None
        x_TSC = self.decoder(x_TSC, action_ids=action_ids, domain=domain)

        # dummy if are not used
        decoded_states = rearrange(diffusion_pos_embed_learned, "B (T H W) C -> B C T H W", T=self.config.T, H=h, W=w)
        decoded_actions = None
        if self.config.jointly_predict_actions:
            decoded_actions = x_TSC[:, :, -self.config.action_token_size :].mean(dim=2)  # pool all tokens

        x_TSC = x_TSC[:, :, : h * w]  # remove action tokens for states
        x_next_TSC = self.decoder_norm(self.out_x_proj(x_TSC))
        x_next_TSC = x_next_TSC + diffusion_pos_embed_learned.view(1, self.config.T, h * w, self.config.d_model)[:, :T]
        decoded_states = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=h, W=w)

        return decoded_states, decoded_actions

    def patchify(self, x):
        bsz, t, h, w, c = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, t, h_, p, w_, p, c)
        x = torch.einsum("nthpwqc->nthwpqc", x)
        x = x.reshape(bsz, t, h_, w_, c * p**2)
        return x

    def unpatchify(self, x):
        # input: B T H W C
        p = self.patch_size
        bsz, t, h, w, _ = x.shape
        c = self.vae_embed_dim
        x = x.reshape(bsz, t, h, w, p, p, c)
        x = torch.einsum("nthwpqc->nthpwqc", x)
        x = x.reshape(bsz, t, h * p, w * p, c)
        return x

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW,
        out_t: int,
        unmask_mode: str = "random",
        action_ids=None,
        domain="default",
        maskgit_steps=8,
        cfg=1.0,
        temperature=1.0,
        cfg_schedule="linear",
        action_only: bool = False,
        state_only: bool = False,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        # init and sample generation orders
        assert out_t, "maskgit_generate requires out_t > 0"
        prompt_THW = self.patchify(prompt_THW)
        bs, _t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        orders = self.sample_orders(bs)  # random order
        sampled_action_token_latent = None

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)

        # patchify the prompt
        latents_CTHW, action_outputs = self.compute_latents(prompt_THW, action_ids=action_ids, domain=domain, **kwargs)
        latents_CHW = latents_CTHW[:, :, out_t]
        orig_latents_CHW = latents_CHW.clone()
        # Return these original logits, not logits after partially sampling.

        for step in range(maskgit_steps):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                latents_CHW, action_outputs = self.compute_latents(
                    prompt_THW, action_ids=action_ids, domain=domain, **kwargs
                )
                latents_CHW = latents_CHW[:, :, out_t]

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / maskgit_steps)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(), torch.minimum(torch.sum(~unmasked, dim=-1, keepdims=True) - 1, mask_len)
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bs, self.seq_len)
            mask = ~unmasked

            if step >= maskgit_steps - 1:
                mask_to_pred = mask[:bs].bool()  # last step
            else:
                mask_to_pred = torch.logical_xor(mask[:bs].bool(), mask_next.bool())
            mask = mask_next

            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            latents_CHW = rearrange(latents_CHW, "b c h w -> b (h w) c")
            latents_CHW = latents_CHW[mask_to_pred.nonzero(as_tuple=True)]

            # copy previously unmasked values from prompt input into sample
            total_mask_len = unmasked.shape[1]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (total_mask_len - unmasked.sum()) / total_mask_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            # need to reshape back
            sampled_token_latent = self.diffloss.sample(
                latents_CHW.contiguous(), temperature, cfg_iter, clip_denoised=True
            )
            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            if action_outputs is not None and self.config.jointly_predict_actions:
                sampled_action_token_latent = self.action_diff_losses[domain[0]].sample(
                    action_outputs.view(-1, action_outputs.shape[-1]), temperature, cfg_iter, clip_denoised=True
                )
                if cfg != 1.0:
                    sampled_action_token_latent, _ = sampled_action_token_latent.chunk(2, dim=0)

            prompt_THW_reshape = rearrange(prompt_THW, "B T H W C -> B T (H W) C")
            prompt_THW_reshape[:, out_t][mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            prompt_THW = rearrange(prompt_THW_reshape, "B T (H W) C -> B T H W C", H=h, W=w)

        # Return the final sample and logits
        prompt_THW = self.unpatchify(prompt_THW)
        return prompt_THW[:, out_t], orig_latents_CHW, sampled_action_token_latent
