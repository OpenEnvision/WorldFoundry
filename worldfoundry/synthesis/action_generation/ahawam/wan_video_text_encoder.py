import html
import math
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    import ftfy
except ModuleNotFoundError:

    class _FTFY:
        @staticmethod
        def fix_text(text):
            return text

    ftfy = _FTFY()
try:
    import regex as re
except ModuleNotFoundError:
    import re


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import T5Attention


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True, dropout=0.1):
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
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import T5RelativeEmbedding


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5)


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        vocab=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        num_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1,
    ):
        super(WanTextEncoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)

        # Skip costly random init when loading pretrained checkpoints.
        # Verified: checkpoint fully covers this module (missing/unexpected keys are both 0).
        # self.apply(init_weights)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        kwargs.setdefault("local_files_only", True)
        kwargs["local_files_only"] = True
        kwargs["trust_remote_code"] = False
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text
