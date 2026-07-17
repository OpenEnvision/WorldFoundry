import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from worldfoundry.base_models.perception_core.action_recognition.latent_action.vjepa2.modules import (
    Block,
    CrossAttentionBlock,
)


class AttentivePooler(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        init_std=0.02,
        qkv_bias=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cross_attention_block = CrossAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=nn.LayerNorm,
        )
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=False,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(depth - 1)
        )
        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=init_std)
        self.apply(self._init_weights)
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))
        self.cross_attention_block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * max(1, len(self.blocks))))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, value):
        for block in self.blocks:
            if self.use_activation_checkpointing:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        query = self.query_tokens.expand(len(value), -1, -1)
        return self.cross_attention_block(query, value)


class TemporalAttentiveClassifier(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        depth=1,
        num_classes=1000,
        max_temporal_len=128,
        mlp_ratio=4.0,
        init_std=0.02,
        qkv_bias=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            init_std=init_std,
            qkv_bias=qkv_bias,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes)
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_temporal_len, 1, embed_dim))
        trunc_normal_(self.temporal_embed, std=init_std)

    def forward(self, value):
        if value.ndim == 4:
            temporal_length = value.shape[1]
            temporal_embed = self.temporal_embed
            if temporal_length > temporal_embed.shape[1]:
                temporal_embed = F.interpolate(
                    temporal_embed.permute(0, 3, 1, 2),
                    size=(temporal_length, 1),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            else:
                temporal_embed = temporal_embed[:, :temporal_length]
            value = (value + temporal_embed).flatten(1, 2)
        return self.linear(self.pooler(value).squeeze(1))


class FeatureEvaluator(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim=768,
        num_classes=1000,
        num_heads=12,
        depth=1,
        max_temporal_len=128,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.classifier = TemporalAttentiveClassifier(
            embed_dim=model_dim,
            num_heads=num_heads,
            depth=depth,
            num_classes=num_classes,
            max_temporal_len=max_temporal_len,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        for module in self.projector:
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, value):
        return self.classifier(self.projector(value))
