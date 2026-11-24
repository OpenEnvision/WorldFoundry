import math

import mup
import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from ..config import GenieConfig
from .attention import BasicCrossAttention
from .factorization_utils import FactorizedEmbedding
from .st_transformer import STTransformerDecoder


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ModulateLayer(nn.Module):
    """
    Modified from the final layer adopted from DiT with token-wise modulation.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)

        self.linear_out = nn.Linear(out_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(model_channels, model_channels), nn.SiLU(), nn.Linear(model_channels, 2 * out_channels, bias=True)
        )
        self.apply(self._init_weights)

    def forward(self, x, c):
        """
        a simple modulation
        """
        x_shape = x.shape
        x = rearrange(x, "(b s) t d -> b s t d", b=len(c))
        c = c[:, None, : x_shape[2]]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_out(x)
        return x.view(x_shape)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


class BasicMLP(nn.Module):
    def __init__(self, d_action, d_model):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_action, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.model(x)

    def _init_weights(self, m):  # TODO: muP?
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


def cosine_schedule(u):
    """u in [0, 1]"""
    if isinstance(u, torch.Tensor):
        cls = torch
    elif isinstance(u, float):
        cls = math
    else:
        raise NotImplementedError(f"Unexpected {type(u)=} {u=}")

    return cls.cos(u * cls.pi / 2)


class ActionStat(nn.Module):
    def __init__(self, input_info):
        super().__init__()
        self.register_buffer("mean", torch.FloatTensor(input_info[0]))
        self.register_buffer("std", torch.FloatTensor(input_info[1]))

    def forward(self, x):
        # x: (B, T, S * D). T window length, S is the stride in the datasets, D action dimensions
        x = rearrange(x, "b t (s d) -> b t s d", d=len(self.mean))
        x = (x - self.mean) / (self.std + 1e-10)
        return rearrange(x, "b t s d -> b t (s d)", d=len(self.mean))

    def extra_repr(self):
        return f"mean={self.mean}, std={self.std}"


class STMaskGIT(nn.Module, PyTorchModelHubMixin):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, config: GenieConfig):
        super().__init__()
        self.h = self.w = math.isqrt(config.S)
        assert self.h**2 == config.S, "Expected S to be square"

        # STTransformerDecoder
        self.decoder = STTransformerDecoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            qk_norm=config.qk_norm,
            use_mup=config.use_mup,
            attn_drop=config.attn_drop,
            mlp_ratio=config.mlp_ratio,
            mlp_bias=config.mlp_bias,
            mlp_drop=config.mlp_drop,
            action_processing=config.action_network,
            random_dummy_action=config.random_dummy_action,
            jointly_predict_actions=config.jointly_predict_actions,
            mask_token_id=config.image_vocab_size,
        )

        # learnable embedding for the maximum image sizes
        self.pos_embed_TSC = torch.nn.Parameter(
            torch.zeros(1, config.T, config.S + config.action_token_size, config.d_model)
        )
        print(f"{self.h=} {self.w=} {config.S=} {config.T=} {config.d_model=}")
        self.mask_token_id = config.image_vocab_size
        self.seq_len = config.S
        self.relevant_action_mask = None
        self.token_embed = FactorizedEmbedding(  # also works for num_factored_vocabs = 1
            factored_vocab_size=config.factored_vocab_size,
            num_factored_vocabs=config.num_factored_vocabs,
            d_model=config.d_model,
            mask_token_id=self.mask_token_id,
        )

        cls = FixedMuReadout if config.use_mup else nn.Linear  # (Fixed)MuReadout might slow dow down compiled training?
        self.out_x_proj = cls(config.d_model, config.factored_vocab_size * config.num_factored_vocabs)
        self.config = config
        self.action_mask_tokens = torch.nn.Parameter(torch.zeros(1, config.T, 1, config.d_model))

        if (self.config.init_actions or self.config.use_actions) and self.config.action_domains is not None:
            self.init_action_projectors(
                self.config.action_domains, self.config.d_actions, self.config.action_stats, self.config.action_network
            )

    def init_action_projectors(
        self,
        domains: list[str],
        d_actions: list[int],
        action_stats: list[list[list[float]]],
        action_network: str = "mlp",
        use_diffusion: bool = False,
    ):
        # initialize the action stems. It's called externally for training.
        # assert len(domains) == len(d_actions)
        self.config.init_actions = True
        self.config.action_domains = domains
        self.config.d_actions = d_actions
        self.config.action_stats = action_stats
        self.action_preprocessor = nn.ModuleDict()
        self.action_mlp = nn.ModuleDict()
        self.action_out_projectors = nn.ModuleDict()

        # initialize for every layer
        print("use diffusion: ", use_diffusion)
        print("init action network:", action_network)
        cls = (
            FixedMuReadout if self.config.use_mup else nn.Linear
        )  # (Fixed)MuReadout might slow dow down compiled training?

        # We currently skip datasets if they fail but `domains` is all specified datasets, so we get misalignment in this case
        assert len(domains) == len(d_actions) == len(action_stats), (
            f"{len(domains)=} {len(d_actions)=} {len(action_stats)=}"
        )
        for domain, d_action, action_stat in zip(domains, d_actions, action_stats, strict=False):
            # by default, we share these modules across layers
            self.action_preprocessor[domain] = ActionStat(action_stat)
            self.action_mlp[domain] = BasicMLP(d_action, self.config.d_model)
            if not use_diffusion:
                self.action_out_projectors[domain] = cls(self.config.d_model, d_action)

        # by default, the conditioning are separate for each layer
        for layer in self.decoder.layers:
            layer.action_projectors = nn.ModuleDict()

            for domain, d_action, _action_stat in zip(domains, d_actions, action_stats, strict=False):
                if "mlp" in action_network:
                    layer.action_projectors[domain] = nn.Identity()

                elif "cross_attention" in action_network:
                    layer.action_projectors[domain] = BasicCrossAttention(
                        num_heads=8, d_model=self.config.d_model, k_model=d_action
                    )

                elif "modulate" in action_network:
                    layer.action_projectors[domain] = ModulateLayer(self.config.d_model, self.config.d_model)

    def init_mask(self, prompt_THW, t=1):
        # since we generate 1 image at a time, the mask should be for a single frame, not across all frames.
        _T, _H, _W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        unmasked = torch.zeros(prompt_THW.size(0), t * self.seq_len, dtype=torch.bool, device=prompt_THW.device)
        return unmasked

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW: torch.LongTensor,
        out_t: int,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
        unmask_mode: str = "random",
        action_ids=None,
        domain="default",
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Performs MaskGIT-style inference to predict frame `out_t`.

        Args:
            prompt_THW: Unfactorized token ids, size (B, T, H, W)
            out_t: Will return predicted unfactorized token ids for this frame.
                Should be >= 1 as the 0th frame is assumed to be given.
                Expects all future frames to be fully masked.
            maskgit_steps: The number of MaskGIT-style inference steps to take.
            temperature: Sampling temperature.
                In the factorized case, sampling is performed for each factorized vocabulary independently.
                If temperature is <= 1e-8, will be greedy (i.e. argmax) instead of actual sampling.
            unmask_mode: The method to determine tokens to unmask during each step of MaskGIT inference.
                Options:
                    - "greedy" for unmasking the most confident tokens, which is matches the original MaskGIT
                    - "random" for randomly choosing tokens to unmask
                "greedy" tends to copy the previous frame, so we default to "random" instead.

        Returns: (sample_HW, factored_logits)
            sample_HW: size (B, H, W) corresponding to predicted unfactorized token ids for frame `out_t`.
            factored_logits: size (B, factored_vocab_size, num_factored_vocabs, H, W).
        """
        # assume we have pre-masked z{out_t}...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:, out_t:] == self.mask_token_id), (
            f"when generating z{out_t}, frames {out_t} and later must be masked"
        )

        bs, _t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        S = h * w

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)
        logits_CTHW, action_outputs = self.compute_logits(prompt_THW, action_ids=action_ids, domain=domain, **kwargs)
        logits_CHW = logits_CTHW[:, :, out_t]
        orig_logits_CHW = logits_CHW.clone()
        # Return these original logits, not logits after partially sampling.

        for step in range(maskgit_steps):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:
                # recompute logits with updated prompt
                # action is one step out so this line is doing it again.
                logits_CHW, action_outputs = self.compute_logits(
                    prompt_THW, action_ids=action_ids, domain=domain, **kwargs
                )
                logits_CHW = logits_CHW[:, :, out_t]

            factored_logits = rearrange(
                logits_CHW,
                "b (num_vocabs vocab_size) h w -> b vocab_size num_vocabs h w",
                vocab_size=self.config.factored_vocab_size,
                num_vocabs=self.config.num_factored_vocabs,
            )

            factored_probs = torch.nn.functional.softmax(factored_logits, dim=1)
            samples_HW = torch.zeros((bs, h, w), dtype=torch.long, device=prompt_THW.device)
            confidences_HW = torch.ones((bs, h, w), dtype=torch.float, device=prompt_THW.device)

            for probs in factored_probs.flip(2).unbind(2):
                if temperature <= 1e-8:  # greedy sampling
                    sample = probs.argmax(dim=1)
                else:
                    # Categorical expects last dim to be channel dim
                    dist = torch.distributions.categorical.Categorical(
                        probs=rearrange(probs, "b vocab_size ... -> b ... vocab_size") / temperature
                    )
                    sample = dist.sample()

                samples_HW *= self.config.factored_vocab_size
                samples_HW += sample
                confidences_HW *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")
            samples_flat = samples_HW.reshape(bs, S)

            if step != maskgit_steps - 1:  # skip masking for last maskgit step
                # use cosine mask scheduling function, n is how many of frame out_t to mask
                n = math.ceil(cosine_schedule((step + 1) / maskgit_steps) * S)

                if unmask_mode == "greedy":
                    # set the n patches with the least confidence to mask_token
                    confidences_flat = confidences_HW.reshape(bs, S)
                elif unmask_mode == "random":
                    # randomize confidences, so that patches are randomly masked
                    confidences_flat = torch.rand_like(confidences_HW).reshape(bs, S)
                    # not probability distribution anymore, but only relative order matters
                else:
                    raise NotImplementedError(
                        f"Expected `unmask_mode` to be one of ['greedy', 'random'], got {unmask_mode}"
                    )

                confidences_flat[unmasked] = torch.inf
                least_confident_tokens = torch.argsort(confidences_flat, dim=1)
                # unmask the (self.config.S - n) most confident tokens
                unmasked.scatter_(1, least_confident_tokens[:, n:], True)
                samples_flat.scatter_(1, least_confident_tokens[:, :n], self.mask_token_id)

            # copy previously unmasked values from prompt input into sample
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples_HW = samples_flat.reshape(-1, h, w)

            # feed back to iteratively decode
            prompt_THW[:, out_t] = samples_HW

        # Return the final sample and logits
        return (
            samples_HW,
            rearrange(
                orig_logits_CHW,
                "B (num_vocabs vocab_size) H W -> B vocab_size num_vocabs H W",
                vocab_size=self.config.factored_vocab_size,
                num_vocabs=self.config.num_factored_vocabs,
                H=h,
                W=w,
            ),
            action_outputs,
        )

    def compute_logits(self, x_THW: torch.Tensor, action_ids: torch.Tensor = None, domain=None, **kwargs):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        h, w = self.h, self.w
        if "h" in kwargs:
            assert "w" in kwargs
            h = kwargs["h"][0]
            w = kwargs["w"][0]

        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")
        x_TSC = self.token_embed(x_TS)
        T = x_TSC.shape[1]

        if action_ids is not None:
            # currently, action_preprocessor just normalizes the actions
            skip_normalization = kwargs.get("skip_normalization", False)
            if not skip_normalization:
                action_ids = self.action_preprocessor[domain[0]](action_ids)
            action_ids = self.action_mlp[domain[0]](action_ids)  # [B, T, D]

            if "concat" in self.config.action_network:
                # randomly dropped the conditioning
                action_condition = action_ids[:, :T, None].repeat(
                    1, 1, self.config.action_token_size, 1
                )  # [B, T, S, D]
                if self.relevant_action_mask is not None and self.config.jointly_predict_actions:
                    action_condition = (
                        self.relevant_action_mask[:, :T] * self.action_mask_tokens[:, :T]
                        + (1 - self.relevant_action_mask[:, :T]) * action_condition[:, :T]
                    )
                x_TSC = torch.concat((x_TSC, action_condition), dim=2)  # [B, T, S, D]

        elif self.config.jointly_predict_actions:
            # all masked there is no input actions and try to predict actions as in policy
            action_condition = self.action_mask_tokens[:, :T].repeat(1, 1, self.config.action_token_size, 1)
            x_TSC = torch.concat((x_TSC, action_condition), dim=2)  # [B, T, S, D]

        # additive position embeddings, using the same vocab space
        domain = domain[0] if domain is not None else None
        x_TSC = self.decoder(
            x_TSC + self.pos_embed_TSC[:, : x_TSC.shape[1], : x_TSC.shape[2]], action_ids=action_ids, domain=domain
        )
        decoded_actions = None
        decoded_states = None

        if self.config.jointly_predict_actions:
            decoded_actions = x_TSC[:, :, -self.config.action_token_size :].mean(dim=2)  # pool all tokens
            decoded_actions = self.action_out_projectors[domain](decoded_actions)

        if self.config.jointly_predict_states:
            x_TSC = x_TSC[:, :, : h * w]  # remove action tokens
            x_next_TSC = self.out_x_proj(x_TSC)
            decoded_states = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=h, W=w)

        # break into actions here
        return decoded_states, decoded_actions

    def init_weights(self):
        """Works with and without muP."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, "infshape"):  # muP
                    mup.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def set_mup_shapes(self, rescale_params=False):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256  # currently hardcoding to this shape
        base_model = STMaskGIT(base_config)
        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Extra logic for muP."""
        model = super().from_pretrained(*args, **kwargs)
        if model.config.use_mup:
            model.set_mup_shapes(rescale_params=False)

        return model


class FixedMuReadout(mup.MuReadout):
    # add init_weights for FixedMuReadout
    def __init__(self, d_input, d_output):
        super().__init__(d_input, d_output)
        self.apply(self._init_weights)

    def _init_weights(self, m):  # TODO: muP?
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Using `return super(mup.MuReadout, self).forward(self.output_mult * x / self.width_mult())` with `torch.compile`
        results in two divisions by `self.width_mult()` for some reason
        """
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())
