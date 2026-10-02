import math
from inspect import isfunction
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

SDP_IS_AVAILABLE = callable(getattr(F, "scaled_dot_product_attention", None))
if not SDP_IS_AVAILABLE:
    print(
        "No SDP backend available, likely because you are running in pytorch versions < 2.0. "
        f"In fact, you are using PyTorch {torch.__version__}. You might want to consider upgrading"
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("No module 'xformers', processing without it")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            mult=4,
            glu=False,
            dropout=0.0,
            zero_init=False
    ):
        super(FeedForward, self).__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU()
            )
            if not glu else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):  # Not used, never mind
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            backend=None,
            zero_init=False,
            **kwargs
    ):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.backend = backend

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None
    ):
        num_heads = self.heads

        if additional_tokens is not None:
            # Get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # Add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=num_heads), (q, k, v))

        out = _worldfoundry_scaled_dot_product_attention(q, k, v, attn_mask=mask, backend=self.backend)

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=num_heads)

        if additional_tokens is not None:
            # Remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):  # We are using this implementation
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            zero_init=False,
            **kwargs
    ):
        super(MemoryEfficientCrossAttention, self).__init__()
        print(
            f"Setting up {self.__class__.__name__}. "
            f"Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a dimension of {dim_head}"
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

        if zero_init:
            nn.init.zeros_(self.to_out[0].weight)
            nn.init.zeros_(self.to_out[0].bias)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            additional_tokens=None,
            batchify_xformers=False
    ):
        if additional_tokens is not None:
            # Get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # Add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v)
        )

        if exists(mask):
            raise NotImplementedError
        else:
            # Actually compute the attention, what we cannot get enough of
            if batchify_xformers:
                max_bs = 32768  # >65536 will result in wrong outputs
                n_batches = math.ceil(q.shape[0] / max_bs)
                out = []
                for i_batch in range(n_batches):
                    batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                    out.append(
                        xformers.ops.memory_efficient_attention(q[batch], k[batch], v[batch], op=self.attention_op)
                    )
                out = torch.cat(out, 0)
            else:
                out = xformers.ops.memory_efficient_attention(q, k, v, op=self.attention_op)

        out = (
            out[None]
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # Remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # Vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # Ampere
    }

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=0.0,
            context_dim=None,
            gated_ff=True,
            use_checkpoint=False,
            disable_self_attn=False,
            attn_mode="softmax",
            sdp_backend=None
    ):
        super(BasicTransformerBlock, self).__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. You are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive")
            if not XFORMERS_IS_AVAILABLE:
                raise ValueError("Please install xformers via e.g. 'pip install xformers==0.0.16'")
            else:
                print("Falling back to xformers efficient attention")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        assert sdp_backend in (None, "math", "flash", "efficient", "cudnn")
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            context_dim=context_dim if self.disable_self_attn else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend
        )  # Is a self-attn if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend
        )  # Is self-attn if context is None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        del use_checkpoint

    def forward(self, x, context=None, additional_tokens=None):
        kwargs = {"x": x}
        if context is not None:
            kwargs.update({"context": context})
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        return self._forward(**kwargs)

    def _forward(self, x, context=None, additional_tokens=None):
        # Spatial self-attn
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,
                       additional_tokens=additional_tokens) + x
        # Spatial cross-attn
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x
        # Feedforward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding) and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image.

    'use_linear' for more efficiency instead of the 1x1 convs.
    """

    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            dropout=0.0,
            context_dim=None,
            disable_self_attn=False,
            use_linear=False,
            attn_type="softmax",
            use_checkpoint=False,
            sdp_backend=None
    ):
        super(SpatialTransformer, self).__init__()
        print(f"Constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")

        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (List, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, List):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: "
                    f"Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. "
                    f"Setting context_dim to {depth * [context_dim[0]]} now"
                )
                # Depth does not match context dims
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "Need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if use_linear:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    use_checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend
                )
                for d in range(depth)
            ]
        )
        if use_linear:
            self.proj_out = zero_module(
                nn.Linear(inner_dim, in_channels)
            )
        else:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # NOTE: If no context is given, cross-attn defaults to self-attn
        if not isinstance(context, List):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # Use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
