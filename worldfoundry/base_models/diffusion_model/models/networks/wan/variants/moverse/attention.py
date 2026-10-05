"""Cached text conditioning for MoVerse's streaming Wan transformer."""

from ...reference_21 import WAN_CROSSATTENTION_CLASSES as _CROSS_ATTENTION_CLASSES
from ...reference_21 import WanT2VCrossAttention as _WanT2VCrossAttention
from worldfoundry.core.attention.sequence.varlen import flash_attention


class WanT2VCrossAttention(_WanT2VCrossAttention):
    """Reuse the shared projections while retaining context keys and values."""

    def forward(self, x, context, context_lens, crossattn_cache=None):
        if crossattn_cache is None:
            return super().forward(x, context, context_lens)

        batch = x.size(0)
        q = self.norm_q(self.q(x)).view(batch, -1, self.num_heads, self.head_dim)
        if not crossattn_cache["is_init"]:
            crossattn_cache["k"] = self.norm_k(self.k(context)).view(
                batch, -1, self.num_heads, self.head_dim
            )
            crossattn_cache["v"] = self.v(context).view(
                batch, -1, self.num_heads, self.head_dim
            )
            crossattn_cache["is_init"] = True

        attended = flash_attention(
            q, crossattn_cache["k"], crossattn_cache["v"], k_lens=context_lens
        )
        return self.o(attended.flatten(2))


WAN_CROSSATTENTION_CLASSES = {
    **_CROSS_ATTENTION_CLASSES,
    "t2v_cross_attn": WanT2VCrossAttention,
}
