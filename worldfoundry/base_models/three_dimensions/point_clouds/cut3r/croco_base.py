# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo backbone construction for CUT3R inference
# --------------------------------------------------------


"""Module for base_models -> three_dimensions -> point_clouds -> cut3r -> croco_base.py functionality."""

from functools import partial

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from .blocks import Block, DecoderBlock, PatchEmbed
from .pos_embed import RoPE2D, get_2d_sincos_pos_embed


class CrocoConfig(PretrainedConfig):
    """Croco config implementation."""

    model_type = "croco"

    def __init__(
        self,
        img_size=224,  # input image size
        patch_size=16,  # patch_size
        mask_ratio=0.9,  # ratios of masked tokens
        enc_embed_dim=768,  # encoder feature dimension
        enc_depth=12,  # encoder depth
        enc_num_heads=12,  # encoder number of heads in the transformer block
        dec_embed_dim=512,  # decoder feature dimension
        dec_depth=8,  # decoder depth
        dec_num_heads=16,  # decoder number of heads in the transformer block
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder
        pos_embed="cosine",  # positional embedding (either cosine or RoPE100)
    ):
        """Init.

        Args:
            img_size: The img size.
            patch_size: The patch size.
            mask_ratio: The mask ratio.
            enc_embed_dim: The enc embed dim.
            enc_depth: The enc depth.
            enc_num_heads: The enc num heads.
            dec_embed_dim: The dec embed dim.
            dec_depth: The dec depth.
            dec_num_heads: The dec num heads.
            mlp_ratio: The mlp ratio.
            norm_layer: The norm layer.
            norm_im2_in_dec: The norm im2 in dec.
            pos_embed: The pos embed.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_im2_in_dec = norm_im2_in_dec
        self.pos_embed = pos_embed


class CroCoNet(PreTrainedModel):
    """Cro co net implementation."""

    config_class = CrocoConfig
    base_model_prefix = "croco"

    def __init__(self, config: CrocoConfig):
        """Init.

        Args:
            config: The config.
        """

        super().__init__(config)

        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(config.img_size, config.patch_size, config.enc_embed_dim)

        self.pos_embed = config.pos_embed
        if config.pos_embed == "cosine":
            # positional embedding of the encoder
            enc_pos_embed = get_2d_sincos_pos_embed(
                config.enc_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer("enc_pos_embed", torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder
            dec_pos_embed = get_2d_sincos_pos_embed(
                config.dec_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer("dec_pos_embed", torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None  # nothing for cosine
        elif config.pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(config.pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + config.pos_embed)

        # transformer for the encoder
        self.enc_depth = config.enc_depth
        self.enc_embed_dim = config.enc_embed_dim
        self.enc_blocks = nn.ModuleList(
            [
                Block(
                    config.enc_embed_dim,
                    config.enc_num_heads,
                    config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=config.norm_layer,
                    rope=self.rope,
                )
                for i in range(config.enc_depth)
            ]
        )
        self.enc_norm = config.norm_layer(config.enc_embed_dim)

        # masked tokens
        # self._set_mask_token(config.dec_embed_dim)
        self.mask_token = None

        # decoder
        self._set_decoder(
            config.enc_embed_dim,
            config.dec_embed_dim,
            config.dec_num_heads,
            config.dec_depth,
            config.mlp_ratio,
            config.norm_layer,
            config.norm_im2_in_dec,
        )

        # prediction head
        self._set_prediction_head(config.dec_embed_dim, config.patch_size)

        # initializer weights
        self.initialize_weights()

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        """Helper function to set patch embed.

        Args:
            img_size: The img size.
            patch_size: The patch size.
            enc_embed_dim: The enc embed dim.
        """
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        """Helper function to set decoder.

        Args:
            enc_embed_dim: The enc embed dim.
            dec_embed_dim: The dec embed dim.
            dec_num_heads: The dec num heads.
            dec_depth: The dec depth.
            mlp_ratio: The mlp ratio.
            norm_layer: The norm layer.
            norm_im2_in_dec: The norm im2 in dec.
        """
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_prediction_head(self, dec_embed_dim, patch_size):
        """Helper function to set prediction head.

        Args:
            dec_embed_dim: The dec embed dim.
            patch_size: The patch size.
        """
        self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)

    def initialize_weights(self):
        """Initialize weights."""
        # patch embed
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=0.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper function to init weights.

        Args:
            m: The m.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
