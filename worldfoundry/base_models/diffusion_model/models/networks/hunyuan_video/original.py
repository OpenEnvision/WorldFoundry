"""Native checkpoint-compatible HunyuanVideo MM-DiT.

Architecture
------------
``PatchEmbed`` is a stride-``(1,2,2)`` Conv3d: ``B C T H W`` → ``B (T'H'W') C``.
Text tokens are refined by :class:`SingleTokenRefiner` (AdaLN self-attn
stack).  Early :class:`MMDoubleStreamBlock` layers keep video (stream A)
and text (stream B) in separate AdaLN/QKV paths, then concatenate QKV
for joint attention.  Later :class:`MMSingleStreamBlock` layers operate
on the concatenated ``B S C`` sequence.  Optional token-replace
modulation applies a different AdaLN vector to a prefix (i2v first
frame).  :class:`FinalLayer` unpatchifies back to ``B C T H W``.
"""

from functools import partial

import torch
from einops import rearrange, repeat

from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb
from worldfoundry.core.attention import (
    flattened_attention,
    get_cu_seqlens,
    get_nd_rotary_pos_embed,
    scaled_dot_product_attention,
)
from worldfoundry.core.nn import (
    DiTModulation,
    RMSNorm,
)
from worldfoundry.core.nn import (
    ProjectedTimestepEmbedding as TimestepEmbeddings,
)
from worldfoundry.core.nn import (
    apply_gate_with_prefix as apply_gate,
)
from worldfoundry.core.nn import (
    modulate_sequence_with_prefix as modulate,
)

attention = partial(flattened_attention, backend="torch")


class PatchEmbed(torch.nn.Module):
    """3D patch embed: ``B C T H W`` → ``B (T' H' W') C`` via strided Conv3d."""

    def __init__(self, patch_size=(1, 2, 2), in_channels=16, embed_dim=3072):
        super().__init__()
        self.proj = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Flatten the patch grid onto the sequence axis after the conv."""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class IndividualTokenRefinerBlock(torch.nn.Module):
    """One AdaLN-gated self-attn + MLP layer inside the text token refiner."""

    def __init__(self, hidden_size=3072, num_heads=24):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.self_attn_proj = torch.nn.Linear(hidden_size, hidden_size)

        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size * 2, device="cuda", dtype=torch.bfloat16),
        )

    def forward(self, x, c, attn_mask=None):
        """Refine ``B L C`` text tokens; ``c`` is pooled timestep+context AdaLN."""
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        attn = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = rearrange(attn, "B H L D -> B L (H D)")

        x = x + self.self_attn_proj(attn) * gate_msa.unsqueeze(1)
        x = x + self.mlp(self.norm2(x)) * gate_mlp.unsqueeze(1)

        return x


class SingleTokenRefiner(torch.nn.Module):
    """Project LLM text embeddings and refine them before the MM-DiT stack."""

    def __init__(self, in_channels=4096, hidden_size=3072, depth=2):
        super().__init__()
        self.input_embedder = torch.nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbeddings(256, hidden_size, computation_device="cpu")
        self.c_embedder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = torch.nn.ModuleList([IndividualTokenRefinerBlock(hidden_size=hidden_size) for _ in range(depth)])

    def forward(self, x, t, mask=None):
        """``x`` is ``B L C_text``; ``mask`` builds a causal-safe pairwise attn mask."""
        timestep_aware_representations = self.t_embedder(t, dtype=torch.float32)

        mask_float = mask.float().unsqueeze(-1)
        context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations

        x = self.input_embedder(x)

        mask = mask.to(device=x.device, dtype=torch.bool)
        mask = repeat(mask, "B L -> B 1 D L", D=mask.shape[-1])
        mask = mask & mask.transpose(2, 3)
        mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, mask)

        return x


class MMDoubleStreamBlockComponent(torch.nn.Module):
    """One MM-DiT stream (video or text): AdaLN, QKV, then gated residual FFN."""

    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.heads_num = heads_num

        self.mod = DiTModulation(hidden_size, factor=6, zero_init=False)
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.to_qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.norm_q = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.norm_k = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.to_out = torch.nn.Linear(hidden_size, hidden_size)

        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * mlp_width_ratio),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(hidden_size * mlp_width_ratio, hidden_size)
        )

    def forward(self, hidden_states, conditioning, freqs_cis=None, token_replace_vec=None, tr_token=None):
        """Return ``((q,k,v), attn/ff gates, token-replace gates)`` for joint attention."""
        mod1_shift, mod1_scale, mod1_gate, mod2_shift, mod2_scale, mod2_gate = self.mod(conditioning).chunk(6, dim=-1)
        if token_replace_vec is not None:
            assert tr_token is not None
            tr_mod1_shift, tr_mod1_scale, tr_mod1_gate, tr_mod2_shift, tr_mod2_scale, tr_mod2_gate = self.mod(token_replace_vec).chunk(6, dim=-1)
        else:
            tr_mod1_shift, tr_mod1_scale, tr_mod1_gate, tr_mod2_shift, tr_mod2_scale, tr_mod2_gate = None, None, None, None, None, None

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = modulate(
            norm_hidden_states,
            shift=mod1_shift,
            scale=mod1_scale,
            prefix_shift=tr_mod1_shift,
            prefix_scale=tr_mod1_scale,
            prefix_length=tr_token,
        )
        qkv = self.to_qkv(norm_hidden_states)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis, head_first=False)
        return (q, k, v), (mod1_gate, mod2_shift, mod2_scale, mod2_gate), (tr_mod1_gate, tr_mod2_shift, tr_mod2_scale, tr_mod2_gate)

    def process_ff(self, hidden_states, attn_output, mod, mod_tr=None, tr_token=None):
        """Apply gated attn residual then AdaLN-modulated FFN on ``B S C``."""
        mod1_gate, mod2_shift, mod2_scale, mod2_gate = mod
        if mod_tr is not None:
            tr_mod1_gate, tr_mod2_shift, tr_mod2_scale, tr_mod2_gate = mod_tr
        else:
            tr_mod1_gate, tr_mod2_shift, tr_mod2_scale, tr_mod2_gate = None, None, None, None
        hidden_states = hidden_states + apply_gate(self.to_out(attn_output), mod1_gate, tr_mod1_gate, tr_token)
        x = self.ff(
            modulate(
                self.norm2(hidden_states),
                shift=mod2_shift,
                scale=mod2_scale,
                prefix_shift=tr_mod2_shift,
                prefix_scale=tr_mod2_scale,
                prefix_length=tr_token,
            )
        )
        hidden_states = hidden_states + apply_gate(x, mod2_gate, tr_mod2_gate, tr_token)
        return hidden_states


class MMDoubleStreamBlock(torch.nn.Module):
    """Dual-stream MM-DiT block: separate AdaLN/QKV, joint attention, separate FFN."""

    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.component_a = MMDoubleStreamBlockComponent(hidden_size, heads_num, mlp_width_ratio)
        self.component_b = MMDoubleStreamBlockComponent(hidden_size, heads_num, mlp_width_ratio)

    def forward(
        self,
        hidden_states_a,
        hidden_states_b,
        conditioning,
        freqs_cis,
        token_replace_vec=None,
        tr_token=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
    ):
        """Joint-attend video stream A and text stream B; return updated ``B S C`` pairs."""
        (q_a, k_a, v_a), mod_a, mod_tr = self.component_a(hidden_states_a, conditioning, freqs_cis, token_replace_vec, tr_token)
        (q_b, k_b, v_b), mod_b, _ = self.component_b(hidden_states_b, conditioning, freqs_cis=None)

        image_length = q_a.shape[1]
        q = torch.cat((q_a, q_b), dim=1)
        k = torch.cat((k_a, k_b), dim=1)
        v = torch.cat((v_a, v_b), dim=1)
        attn_output = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
        )
        attn_output_a = attn_output[:, :image_length]
        attn_output_b = attn_output[:, image_length:]

        hidden_states_a = self.component_a.process_ff(hidden_states_a, attn_output_a, mod_a, mod_tr, tr_token)
        hidden_states_b = self.component_b.process_ff(hidden_states_b, attn_output_b, mod_b)
        return hidden_states_a, hidden_states_b


class MMSingleStreamBlockOriginal(torch.nn.Module):
    """Legacy fused QKV+MLP single-stream block (kept for checkpoint compatibility)."""

    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.mlp_hidden_dim = hidden_size * mlp_width_ratio

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.q_norm = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.k_norm = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)

        self.pre_norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = torch.nn.GELU(approximate="tanh")
        self.modulation = DiTModulation(hidden_size, factor=3, zero_init=False)

    def forward(self, x, vec, freqs_cis=None, txt_len=256):
        """Attend concatenated ``B (S_img+S_txt) C``; RoPE is applied only to the video prefix."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q_a, q_b = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        k_a, k_b = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        q_a, k_a = apply_rotary_emb(q_a, k_a, freqs_cis, head_first=False)
        q = torch.cat((q_a, q_b), dim=1)
        k = torch.cat((k_a, k_b), dim=1)

        attn_output_a = attention(q[:, :-185].contiguous(), k[:, :-185].contiguous(), v[:, :-185].contiguous())
        attn_output_b = attention(q[:, -185:].contiguous(), k[:, -185:].contiguous(), v[:, -185:].contiguous())
        attn_output = torch.concat([attn_output_a, attn_output_b], dim=1)

        output = self.linear2(torch.cat((attn_output, self.mlp_act(mlp)), 2))
        return x + output * mod_gate.unsqueeze(1)


class MMSingleStreamBlock(torch.nn.Module):
    """Concatenated video+text DiT block; RoPE covers the video prefix only."""

    def __init__(self, hidden_size=3072, heads_num=24, mlp_width_ratio=4):
        super().__init__()
        self.heads_num = heads_num

        self.mod = DiTModulation(hidden_size, factor=3, zero_init=False)
        self.norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.to_qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.norm_q = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.norm_k = RMSNorm(dim=hidden_size // heads_num, eps=1e-6)
        self.to_out = torch.nn.Linear(hidden_size, hidden_size)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * mlp_width_ratio),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(hidden_size * mlp_width_ratio, hidden_size, bias=False)
        )

    def forward(
        self,
        hidden_states,
        conditioning,
        freqs_cis=None,
        txt_len=256,
        token_replace_vec=None,
        tr_token=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
    ):
        """Run gated self-attn + FFN on the joint ``B S C`` sequence."""
        mod_shift, mod_scale, mod_gate = self.mod(conditioning).chunk(3, dim=-1)
        if token_replace_vec is not None:
            assert tr_token is not None
            tr_mod_shift, tr_mod_scale, tr_mod_gate = self.mod(token_replace_vec).chunk(3, dim=-1)
        else:
            tr_mod_shift, tr_mod_scale, tr_mod_gate = None, None, None

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = modulate(
            norm_hidden_states,
            shift=mod_shift,
            scale=mod_scale,
            prefix_shift=tr_mod_shift,
            prefix_scale=tr_mod_scale,
            prefix_length=tr_token,
        )
        qkv = self.to_qkv(norm_hidden_states)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if freqs_cis is not None:
            q_a, q_b = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            k_a, k_b = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            q_a, k_a = apply_rotary_emb(q_a, k_a, freqs_cis, head_first=False)
            q = torch.cat((q_a, q_b), dim=1)
            k = torch.cat((k_a, k_b), dim=1)
        attn_output = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
        )

        hidden_states = hidden_states + apply_gate(self.to_out(attn_output), mod_gate, tr_mod_gate, tr_token)
        hidden_states = hidden_states + apply_gate(self.ff(norm_hidden_states), mod_gate, tr_mod_gate, tr_token)
        return hidden_states


class FinalLayer(torch.nn.Module):
    """AdaLN + linear unpatch projection to ``pt*ph*pw*C`` per video token."""

    def __init__(self, hidden_size=3072, patch_size=(1, 2, 2), out_channels=16):
        super().__init__()

        self.norm_final = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = torch.nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)

        self.adaLN_modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        """Map video tokens ``B S C`` to packed patch channels using timestep AdaLN ``c``."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class HunyuanVideoDiT(torch.nn.Module):
    """Checkpoint-compatible HunyuanVideo MM-DiT (double then single stream)."""

    def __init__(self, in_channels=16, hidden_size=3072, text_dim=4096, num_double_blocks=20, num_single_blocks=40, guidance_embed=True):
        super().__init__()
        self.img_in = PatchEmbed(in_channels=in_channels, embed_dim=hidden_size)
        self.txt_in = SingleTokenRefiner(in_channels=text_dim, hidden_size=hidden_size)
        self.time_in = TimestepEmbeddings(256, hidden_size, computation_device="cpu")
        self.vector_in = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.guidance_in = TimestepEmbeddings(256, hidden_size, computation_device="cpu") if guidance_embed else None
        self.double_blocks = torch.nn.ModuleList([MMDoubleStreamBlock(hidden_size) for _ in range(num_double_blocks)])
        self.single_blocks = torch.nn.ModuleList([MMSingleStreamBlock(hidden_size) for _ in range(num_single_blocks)])
        self.final_layer = FinalLayer(hidden_size)

        # TODO: remove these parameters
        self.dtype = torch.bfloat16
        self.patch_size = [1, 2, 2]
        self.hidden_size = 3072
        self.heads_num = 24
        self.rope_dim_list = [16, 56, 56]

    def unpatchify(self, x, T, H, W):
        """Fold ``B (T'H'W') (C·1·2·2)`` tokens back to ``B C T H W`` (patch ``(1,2,2)``)."""
        x = rearrange(x, "B (T H W) (C pT pH pW) -> B C (T pT) (H pH) (W pW)", H=H, W=W, pT=1, pH=2, pW=2)
        return x


    def prepare_freqs(self, latents):
        """Build 3-axis RoPE for the post-patch ``(T, H/2, W/2)`` lattice from ``B C T H W``."""
        return get_nd_rotary_pos_embed(
            [16, 56, 56],
            [latents.shape[2], latents.shape[3] // 2, latents.shape[4] // 2],
            theta=256,
            use_real=True,
            theta_rescale_factor=1,
            device="cpu",
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prompt_emb: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        pooled_prompt_emb: torch.Tensor = None,
        freqs_cos: torch.Tensor = None,
        freqs_sin: torch.Tensor = None,
        guidance: torch.Tensor = None,
        **kwargs
    ):
        """Denoise ``B C T H W`` latents; return residual in the same layout."""
        B, C, T, H, W = x.shape

        vec = self.time_in(t, dtype=torch.float32) + self.vector_in(pooled_prompt_emb)
        if self.guidance_in is not None:
            vec += self.guidance_in(guidance * 1000, dtype=torch.float32)
        img = self.img_in(x)
        txt = self.txt_in(prompt_emb, t, text_mask)
        img_seq_len = img.shape[1]
        txt_seq_len = txt.shape[1]
        cu_seqlens = get_cu_seqlens(text_mask, img_seq_len)
        max_seqlen = img_seq_len + txt_seq_len

        for block in self.double_blocks:
            img, txt = block(
                img,
                txt,
                vec,
                (freqs_cos, freqs_sin),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
            )

        x = torch.concat([img, txt], dim=1)
        for block in self.single_blocks:
            x = block(
                x,
                vec,
                (freqs_cos, freqs_sin),
                txt_len=txt_seq_len,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
            )

        img = x[:, :img_seq_len]
        img = self.final_layer(img, vec)
        img = self.unpatchify(img, T=T//1, H=H//2, W=W//2)
        return img
