"""JING-Flash differences atop the canonical MiniMax-H3 DiT.

Only slice-local text refinement/visibility and camera FiLM are variant-specific.
Projections, attention, RoPE, norms, AdaLN, MLPs and output heads are shared.
"""
from functools import partial

import torch
from torch.nn import functional as F

from ...config import MiniMaxH3DiTArchConfig
from ...dit import MiniMaxH3DiTModel
from .control import Control


def architecture(config):
    names = {
        "num_refiner_layers": "token_refiner_num_layers", "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim", "audio_in_channels": "audio_latents_dim",
        "freq_dim": "timestep_input_dim", "time_embed_hidden_dim": "time_embed_hidden_size",
        "rope_freq_dim": "rope_inv_freq_len",
    }
    fields = MiniMaxH3DiTArchConfig.__dataclass_fields__
    values = {names.get(k, k): v for k, v in config.items() if names.get(k, k) in fields}
    hidden = values.get("hidden_size", 5376)
    values.update(adaln_out_features=18 * hidden, final_adaln_out_features=2 * hidden)
    return MiniMaxH3DiTArchConfig(**values)


def attention_groups(owners, is_media):
    return [((owners == owner).nonzero().flatten(),
             (is_media | (owners == owner)).nonzero().flatten())
            for owner in owners.unique().tolist()]


class JINGTransformer(MiniMaxH3DiTModel):
    def __init__(self, config):
        super().__init__(architecture(config))
        self.jing_config = dict(config)
        control = config.get("control_config") or {"enable": False}
        self.control = Control(control, {
            "hidden_size": self.arch.hidden_size, "patch_size": self.arch.patch_size,
            "num_layers": self.arch.num_layers,
        }).to(dtype=torch.bfloat16) if control["enable"] else None
        dim = self.arch.rope_inv_freq_len
        self.rope.inv_freq = 1 / (float(config.get("rope_theta", 10000)) ** (
            torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim)))

    def forward(self, packed, timestep, timestep_indices, mask=None):
        device = packed.video.device
        video = self.video_patch_proj(packed.video[0].float())
        audio = self.audio_patch_proj(packed.audio[0].float())
        texts = self.condition_proj(packed.text[0].to(self.condition_proj.weight.dtype))
        lengths = torch.tensor((0, *packed.text_lengths), device=device, dtype=torch.int32).cumsum(0)
        text = self.token_refiner(texts, cu_seqlens=lengths)
        hidden = text.new_zeros(len(packed.token_tags), self.hidden_size)
        for indices, values in ((packed.text_indices, text), (packed.video_indices, video),
                                (packed.audio_indices, audio)):
            hidden.index_copy_(0, indices, values.to(hidden.dtype))
        rope = self.build_rope_cache(packed.position_ids[None], device=device)
        temb = F.silu(self.time_embedder(timestep)).to(torch.bfloat16)
        indices = timestep_indices * 3 + packed.token_tags
        encoded, controls = (None, None) if self.control is None else self.control(packed.controls)
        groups = attention_groups(packed.owners, packed.is_media) if mask is None else mask
        cu = torch.tensor([0, len(hidden)], device=device, dtype=torch.int32)
        for index, block in enumerate(self.blocks):
            conditioner = None
            if encoded is not None:
                conditioner = partial(self._condition, injector=self.control.block_injectors[index],
                                      encoded=encoded, indices=controls)
            hidden = block(hidden, adaln_input=temb, combined_indices=indices,
                           rope_cache=rope, cu_seqlens=cu, attention_groups=groups,
                           ffn_conditioner=conditioner)
        video, audio = self.final_layer(hidden, adaln_input=temb, inverse_indices=timestep_indices)
        return video[packed.video_indices][None], audio[packed.audio_indices][None]

    @staticmethod
    def _condition(hidden, *, injector, encoded, indices):
        return injector(hidden[None], encoded, indices)[0]
