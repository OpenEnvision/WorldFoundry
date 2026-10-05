"""Echo-Infinity bidirectional Wan stem (official packing).

Building-block Wan 2.x graph used by Infinity causal / memory variants.
Memory adapters attach outside this file.  No VACE or linear attn.
"""
import math
import torch
import torch.nn as nn
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from einops import repeat
from .attention import flash_attention
__all__ = ['WanModel']

def sinusoidal_embedding_1d(dim, position):
    """Classic sinusoidal features for 1-D timestep / position ids."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def rope_params(max_seq_len, dim, theta=10000):
    """Build complex cis frequencies for one RoPE axis."""
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_apply(x, grid_sizes, freqs):
    """Apply per-sample 3D RoPE to packed tokens; leave pad tokens unrotated."""
    n, c = (x.size(2), x.size(3) // 2)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)

class WanRMSNorm(nn.Module):
    """RMSNorm with a learned scale; compute in fp32 when the impl upcasts."""

    def __init__(self, dim, eps=1e-05):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        """RMS-normalize the last dim of ``x``."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class WanLayerNorm(nn.LayerNorm):
    """LayerNorm that typically upcasts to fp32, then restores the activation dtype."""

    def __init__(self, dim, eps=1e-06, elementwise_affine=False):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
            elementwise_affine: The elementwise affine.
        """
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return super().forward(x).type_as(x)

class WanSelfAttention(nn.Module):
    """Wan self-attention: RMSNorm Q/K, 3D RoPE, flash or variant kernel."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-06):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """Forward.

        Args:
            x: The x.
            seq_lens: The seq lens.
            grid_sizes: The grid sizes.
            freqs: The freqs.
        """
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)

        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return (q, k, v)
        q, k, v = qkv_fn(x)
        x = flash_attention(q=rope_apply(q, grid_sizes, freqs), k=rope_apply(k, grid_sizes, freqs), v=v, k_lens=seq_lens, window_size=self.window_size)
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanT2VCrossAttention(WanSelfAttention):
    """Wan text or image-text cross-attention."""

    def forward(self, x, context, context_lens, crossattn_cache=None):
        """Forward.

        Args:
            x: The x.
            context: The context.
            context_lens: The context lens.
            crossattn_cache: The crossattn cache.
        """
        b, n, d = (x.size(0), self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if crossattn_cache is not None:
            if not crossattn_cache['is_init']:
                crossattn_cache['is_init'] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache['k'] = k
                crossattn_cache['v'] = v
            else:
                k = crossattn_cache['k']
                v = crossattn_cache['v']
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanGanCrossAttention(WanSelfAttention):
    """GAN-style cross-attention used by some Wan discriminator / aux heads."""

    def forward(self, x, context, crossattn_cache=None):
        """Forward.

        Args:
            x: The x.
            context: The context.
            crossattn_cache: The crossattn cache.
        """
        b, n, d = (x.size(0), self.num_heads, self.head_dim)
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)
        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)
        x = flash_attention(qq, kk, vv)
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanI2VCrossAttention(WanSelfAttention):
    """Wan text or image-text cross-attention."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-06):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        """Forward.

        Args:
            x: The x.
            context: The context.
            context_lens: The context lens.
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = (x.size(0), self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x
WAN_CROSSATTENTION_CLASSES = {'t2v_cross_attn': WanT2VCrossAttention, 'i2v_cross_attn': WanI2VCrossAttention}

class WanAttentionBlock(nn.Module):
    """Wan transformer block: AdaLN self-attn, cross-attn, then AdaLN FFN."""

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-06):
        """Init.

        Args:
            cross_attn_type: The cross attn type.
            dim: The dim.
            ffn_dim: The ffn dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        """Forward.

        Args:
            x: The x.
            e: The e.
            seq_lens: The seq lens.
            grid_sizes: The grid sizes.
            freqs: The freqs.
            context: The context.
            context_lens: The context lens.
        """
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        x = x + y * e[2]

        def cross_attn_ffn(x, context, context_lens, e):
            """Second half of the block: cross-attn then gated FFN."""
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x
        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class GanAttentionBlock(nn.Module):
    """Auxiliary GAN attention block (not on the main denoising residual)."""

    def __init__(self, dim=1536, ffn_dim=8192, num_heads=12, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-06):
        """Init.

        Args:
            dim: The dim.
            ffn_dim: The ffn dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.cross_attn = WanGanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

    def forward(self, x, context):
        """Forward.

        Args:
            x: The x.
            context: The context.
        """

        def cross_attn_ffn(x, context):
            """Cross attn ffn.

            Args:
                x: The x.
                context: The context.
            """
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token
            return y
        x = cross_attn_ffn(x, context)
        return x

class Head(nn.Module):
    """AdaLN-modulated output projection from tokens to packed patch voxels."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-06):
        """Init.

        Args:
            dim: The dim.
            out_dim: The out dim.
            patch_size: The patch size.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        """Forward.

        Args:
            x: The x.
            e: The e.
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x

class MLPProj(torch.nn.Module):
    """CLIP image-token projector (I2V / FLF2V)."""

    def __init__(self, in_dim, out_dim):
        """Init.

        Args:
            in_dim: The in dim.
            out_dim: The out dim.
        """
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim), torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim), torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        """Forward.

        Args:
            image_embeds: The image embeds.
        """
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class RegisterTokens(nn.Module):
    """Learned register / prefix tokens prepended to the packed sequence."""

    def __init__(self, num_registers: int, dim: int):
        """Init.

        Args:
            num_registers: The num registers.
            dim: The dim.
        """
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-06)

    def forward(self):
        """Forward."""
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.normal_(self.register_tokens, std=0.02)

class WanModel(nn.Module, NativeConfigMixin):
    """Official-layout Wan diffusion transformer stem."""
    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-06):
        """Init.

        Args:
            model_type: The model type.
            patch_size: The patch size.
            text_len: The text len.
            in_dim: The in dim.
            dim: The dim.
            ffn_dim: The ffn dim.
            freq_dim: The freq dim.
            text_dim: The text dim.
            out_dim: The out dim.
            num_heads: The num heads.
            num_layers: The num layers.
            window_size: The window size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
        """
        super().__init__()
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        self.init_weights()
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        """Helper function to set gradient checkpointing.

        Args:
            module: The module.
            value: The value.
        """
        self.gradient_checkpointing = value

    def forward(self, *args, **kwargs):
        """Forward."""
        return self._forward(*args, **kwargs)

    def _forward(self, x, t, context, seq_len, classify_mode=False, concat_time_embeddings=False, register_tokens=None, cls_pred_branch=None, gan_ca_blocks=None, clip_fea=None, y=None, sink_recache_after_switch=False):
        """Helper function to forward.

        Args:
            x: The x.
            t: The t.
            context: The context.
            seq_len: The seq len.
            classify_mode: The classify mode.
            concat_time_embeddings: The concat time embeddings.
            register_tokens: The register tokens.
            cls_pred_branch: The cls pred branch.
            gan_ca_blocks: The gan ca blocks.
            clip_fea: The clip fea.
            y: The y.
            sink_recache_after_switch: The sink recache after switch.
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens)

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """

            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None
            final_x = []
            registers = repeat(register_tokens(), 'n d -> b n d', b=x.shape[0])
        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)
            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx:gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1
        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        if classify_mode:
            return (torch.stack(x), final_x)
        return torch.stack(x)

    def _forward_classify(self, x, t, context, seq_len, register_tokens, cls_pred_branch, clip_fea=None, y=None):
        """Helper function to forward classify.

        Args:
            x: The x.
            t: The t.
            context: The context.
            seq_len: The seq len.
            register_tokens: The register tokens.
            cls_pred_branch: The cls pred branch.
            clip_fea: The clip fea.
            y: The y.
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens)

        def create_custom_forward(module):
            """Create custom forward.

            Args:
                module: The module.
            """

            def custom_forward(*inputs, **kwargs):
                """Custom forward."""
                return module(*inputs, **kwargs)
            return custom_forward
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)
        x = self.unpatchify(x, grid_sizes, c=self.dim // 4)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes, c=None):
        """Unpatchify.

        Args:
            x: The x.
            grid_sizes: The grid sizes.
            c: The c.
        """
        c = self.out_dim if c is None else c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Init weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
