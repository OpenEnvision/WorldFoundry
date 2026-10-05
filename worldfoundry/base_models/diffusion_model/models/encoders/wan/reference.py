# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Reference Wan 2.1 T5 encoder (official layout).

Full T5 encoder/decoder graph matching the official Wan 2.1
release layout, used by compatibility loaders.
:class:`T5EncoderModel` / :func:`umt5_xxl` are the usual
entry points.

Native runner path is the encoder-only :mod:`.model`.
Do not mix state dicts between the two without a converter.

Wan UMT5 family, not STEP1 or Gemma.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.core.model_loading import load_state_dict as load_core_state_dict
from worldfoundry.core.model_loading import load_torch_checkpoint
from worldfoundry.core.model_loading import load_torch_state_dict

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    """Fp16 clamp.

    Args:
        x: The x.
    """
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    """Init weights.

    Args:
        m: The m.
    """
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):
    """Gelu implementation."""

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    """Layer norm implementation."""

    def __init__(self, dim, eps=1e-6):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
        """
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    """Attention implementation."""

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        """Init.

        Args:
            dim: The dim.
            dim_attn: The dim attn.
            num_heads: The num heads.
            dropout: The dropout.
        """
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    """Feed forward implementation."""

    def __init__(self, dim, dim_ffn, dropout=0.1):
        """Init.

        Args:
            dim: The dim.
            dim_ffn: The dim ffn.
            dropout: The dropout.
        """
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    """Self attention implementation."""

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """Init.

        Args:
            dim: The dim.
            dim_attn: The dim attn.
            dim_ffn: The dim ffn.
            num_heads: The num heads.
            num_buckets: The num buckets.
            shared_pos: The shared pos.
            dropout: The dropout.
        """
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        """Forward.

        Args:
            x: The x.
            mask: The mask.
            pos_bias: The pos bias.
        """
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):
    """Cross attention implementation."""

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """Init.

        Args:
            dim: The dim.
            dim_attn: The dim attn.
            dim_ffn: The dim ffn.
            num_heads: The num heads.
            num_buckets: The num buckets.
            shared_pos: The shared pos.
            dropout: The dropout.
        """
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        """Forward.

        Args:
            x: The x.
            mask: The mask.
            encoder_states: The encoder states.
            encoder_mask: The encoder mask.
            pos_bias: The pos bias.
        """
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    """Relative embedding implementation."""

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        """Init.

        Args:
            num_buckets: The num buckets.
            num_heads: The num heads.
            bidirectional: The bidirectional.
            max_dist: The max dist.
        """
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        """Forward.

        Args:
            lq: The lq.
            lk: The lk.
        """
        device = self.embedding.weight.device
        if device.type == "meta":
            # Accelerate's sequential CPU offload deliberately replaces child
            # module weights with meta tensors between calls.  This parent
            # module constructs the relative-position indices before the
            # embedding child's pre-forward hook materializes its weight, so
            # deriving the input device from that meta weight produces an
            # unusable meta index tensor.  Use the hook's real execution
            # device when available.
            hook = getattr(self.embedding, "_hf_hook", None)
            execution_device = getattr(hook, "execution_device", None)
            if execution_device is None:
                raise RuntimeError(
                    "T5 relative-position embedding is on meta without an "
                    "Accelerate execution device"
                )
            device = torch.device(execution_device)
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        """Helper function to relative position bucket.

        Args:
            rel_pos: The rel pos.
        """
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):
    """Encoder implementation."""

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """Init.

        Args:
            vocab: The vocab.
            dim: The dim.
            dim_attn: The dim attn.
            dim_ffn: The dim ffn.
            num_heads: The num heads.
            num_layers: The num layers.
            num_buckets: The num buckets.
            shared_pos: The shared pos.
            dropout: The dropout.
        """
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        """Forward.

        Args:
            ids: The ids.
            mask: The mask.
        """
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):
    """Decoder implementation."""

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """Init.

        Args:
            vocab: The vocab.
            dim: The dim.
            dim_attn: The dim attn.
            dim_ffn: The dim ffn.
            num_heads: The num heads.
            num_layers: The num layers.
            num_buckets: The num buckets.
            shared_pos: The shared pos.
            dropout: The dropout.
        """
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        """Forward.

        Args:
            ids: The ids.
            mask: The mask.
            encoder_states: The encoder states.
            encoder_mask: The encoder mask.
        """
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):
    """Model implementation."""

    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        """Init.

        Args:
            vocab_size: The vocab size.
            dim: The dim.
            dim_attn: The dim attn.
            dim_ffn: The dim ffn.
            num_heads: The num heads.
            encoder_layers: The encoder layers.
            decoder_layers: The decoder layers.
            num_buckets: The num buckets.
            shared_pos: The shared pos.
            dropout: The dropout.
        """
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        """Forward.

        Args:
            encoder_ids: The encoder ids.
            encoder_mask: The encoder mask.
            decoder_ids: The decoder ids.
            decoder_mask: The decoder mask.
        """
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    """Helper function to t5.

    Args:
        name: The name.
        encoder_only: The encoder only.
        decoder_only: The decoder only.
        return_tokenizer: The return tokenizer.
        tokenizer_kwargs: The tokenizer kwargs.
        dtype: The dtype.
        device: The device.
    """
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)

    # init tokenizer
    if return_tokenizer:
        from .model import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    """Umt5 xxl."""
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:
    """Encoder model implementation."""

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=None,
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
        load_with_core_loader=False,
        load_with_mmap=False,
        return_full_context=False,
    ):
        """Init.

        Args:
            text_len: The text len.
            dtype: The dtype.
            device: The device.
            checkpoint_path: The checkpoint path.
            tokenizer_path: The tokenizer path.
            shard_fn: The shard fn.
            load_with_core_loader: The load with core loader.
            load_with_mmap: The load with mmap.
            return_full_context: The return full context.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.return_full_context = return_full_context

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device).eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        if load_with_core_loader:
            state_dict = load_core_state_dict(checkpoint_path, device="cpu")
        elif load_with_mmap:
            state_dict = load_torch_checkpoint(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        else:
            state_dict = load_torch_state_dict(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        from .model import HuggingfaceTokenizer

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def to(self, device):
        """To.

        Args:
            device: The device.
        """
        self.device = device
        self.model.to(device)

    def __call__(self, texts, device=None):
        """Call.

        Args:
            texts: The texts.
            device: The device.
        """
        if device is None:
            device = self.device
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        if self.return_full_context:
            for item, seq_len in zip(context, seq_lens):
                item[seq_len:] = 0.0
            return [item for item, _seq_len in zip(context, seq_lens)]
        return [u[:v] for u, v in zip(context, seq_lens)]
