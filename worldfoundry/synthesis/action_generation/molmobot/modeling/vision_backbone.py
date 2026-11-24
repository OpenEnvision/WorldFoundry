import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from ..config import BaseConfig, D, StrEnum
from ..preprocessing.image_preprocessor import ImagePreprocessor
from .image_vit import VisionTransformer, VitConfig
from .llm import Activation

log = logging.getLogger(__name__)


class ImagePaddingEmbed(StrEnum):
    """How to embed image padding information"""

    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    """How to pool patch features"""

    attention = "attention"
    attention_meanq = "attention_meanq"
    attention_meanq_2x = "attention_meanq_2x"
    attention_meanq_4x = "attention_meanq_4x"
    attention_2wide = "attention_2wide"
    mean = "mean"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    """How to project the pooled features into the LLM embedding space"""

    random_linear = "random_linear"
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


@dataclass
class MolmoVisionBackboneConfig(BaseConfig):
    """Vision ViT and the Image/Language Connector"""

    vit: VitConfig = field(default_factory=VitConfig)
    """The vision ViT"""

    image_pooling_2d: ImagePooling2DType = ImagePooling2DType.attention_meanq
    """Layer to pool image features"""

    pooling_attention_mask: bool = False
    """Use an attention mask when pooling instead setting masked embeddings to 0"""

    image_projector: ImageProjectType = ImageProjectType.mlp
    """Layer to project pooled image features to the LLM embedding space"""

    image_padding_embed: ImagePaddingEmbed | None = None
    """
    Image padding mode to use to tell the model what parts of the image are padding
    """

    vit_layers: tuple = (-1,)
    """What layers to use from the VIT"""

    skip_unused_layers: bool = True
    """Don't load layers we don't need from the ViT"""

    use_deepstack: bool = False
    """Use deepstack"""

    share_connector: bool = False
    """Share the connector across layers"""

    image_feature_dropout: float = 0.0
    """Dropout for image patch features"""

    connector_activation_checkpointing: bool = True
    """Allow activation checkpoint on the connector components"""

    compile_vit: str | None = "blocks"
    """How to compile the ViT"""

    pool_size_embeds: list[int] | None = None

    compile_connector: str | None = "dynamic"

    normalize_on_gpu: bool = False
    """Run image normalization on the GPU

    Does this will allow image loading to keep the images in uint8 which will reduce
    RAM/shared memory usage significantly
    """
    use_image_augmentation: bool = False
    """Enable stochastic image augmentation during preprocessing"""

    use_resize_bottleneck: bool = False
    """Use a resize bottleneck to set image resolution for better sim to real transfer"""

    def __post_init__(self):
        self.vit_layers = tuple(self.vit_layers)  # type: ignore[assignment]

    def build_preprocessor(self):
        return ImagePreprocessor(
            normalize=self.vit.normalize,
            resize=self.vit.resize_mode,
            pad_value=self.vit.pad_value,
            image_patch_size=self.vit.image_patch_size,
            base_image_input_size=self.vit.image_default_input_size,
            normalize_on_gpu=self.normalize_on_gpu,
            use_image_mask=self.image_padding_embed is not None,
            use_image_augmentation=self.use_image_augmentation,
            use_resize_bottleneck=self.use_resize_bottleneck,
        )

    def build(self, llm_config, device):
        return MolmoVisionBackbone(self, llm_config, device)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "fix_image_padding" in config:
            assert config.image_padding_embed is None or config.fix_image_padding
            del config["fix_image_padding"]
        for k in ["residual_patch_features", "patch_residual", "patch_residual_thresh", "debug", "frame_embedding"]:
            if k in config:
                assert not config[k]
                del config[k]
        for k in ["image_pooling_h", "image_pooling_w"]:
            if k in config:
                assert config.pop(k) == 2
        config.vit = VitConfig.update_legacy_settings(config.vit)
        return config


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(self, config, input_dim: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=device,
        )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        # Activation function.
        self.act = Activation.build(config.activation_type, split_inputs=True)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def reset_parameters(self):
        self.submodule.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig, llm_config, device=None):
        super().__init__()
        self.config = config
        self.image_preprocessor = config.build_preprocessor()

        if config.use_deepstack:
            assert len(config.vit_layers) > 1, "Deepstack type requires at least 2 layers"

        if not config.use_deepstack or config.share_connector:
            self.image_pooling_2d, self.image_projector = self.build_connector(llm_config, device)
        else:
            image_pooling_2d_list, image_project_list = map(
                list,
                zip(*[self.build_connector(llm_config, device) for _ in range(len(config.vit_layers))], strict=False),
            )
            self.image_pooling_2d = nn.ModuleList(image_pooling_2d_list)
            self.image_projector = nn.ModuleList(image_project_list)

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)

        self.vit_layers = []
        for layer in config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(config.vit.image_num_layers + layer)
        last_layer_needed = max(self.vit_layers) + 1

        vit_cfg = self.config.vit
        if last_layer_needed < config.vit.image_num_layers:
            if self.config.skip_unused_layers:
                vit_cfg = replace(vit_cfg, image_num_layers=last_layer_needed)
                self.image_vit: VisionTransformer = vit_cfg.build(device)
            else:
                # Keep unused layers when the checkpoint schema requests them.
                self.image_vit: VisionTransformer = vit_cfg.build(device)
        else:
            self.image_vit: VisionTransformer = vit_cfg.build(device)

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        if config.use_deepstack:
            image_dim = vit_cfg.image_emb_dim
        else:
            image_dim = vit_cfg.image_emb_dim * len(self.config.vit_layers)
        self.pad_embed = None
        if config.image_padding_embed:
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(torch.zeros((image_dim,), device=device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(torch.zeros((2, image_dim), device=device))
            else:
                raise ValueError(config.image_padding_embed)
        if config.pool_size_embeds:
            assert self.pad_embed is None
            self.pad_embed = nn.Parameter(torch.zeros((len(config.pool_size_embeds), image_dim), device=device))
            self._pad_embed_idx = {k * k: i for i, k in enumerate(config.pool_size_embeds)}

    def build_connector(self, llm_config, device):
        config = self.config
        input_dim: int = None
        vit_cfg = config.vit
        pool_dim = vit_cfg.image_emb_dim if config.use_deepstack else vit_cfg.image_emb_dim * len(config.vit_layers)

        from .image_vit import ViTMultiHeadDotProductAttention

        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            image_pooling_2d = ViTMultiHeadDotProductAttention(config.vit, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d in [
            ImagePooling2DType.attention_2wide,
            ImagePooling2DType.attention_meanq_2x,
            ImagePooling2DType.attention_meanq_4x,
        ]:
            mha_cfg = deepcopy(config.vit)
            factor = 4 if config.image_pooling_2d == ImagePooling2DType.attention_meanq_4x else 2
            mha_cfg.image_emb_dim *= factor
            mha_cfg.image_head_dim *= factor
            image_pooling_2d = ViTMultiHeadDotProductAttention(mha_cfg, input_dim=pool_dim)
            input_dim = mha_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack, ImagePooling2DType.mean]:
            image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * vit_cfg.image_emb_dim
            if config.image_pooling_2d == ImagePooling2DType.stack:
                input_dim *= 4
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")

        if config.image_projector == ImageProjectType.mlp:
            image_projector = ImageProjectorMLP(llm_config, input_dim, device=device)
        elif config.image_projector in [ImageProjectType.linear, ImageProjectType.random_linear]:
            image_projector = nn.Linear(input_dim, llm_config.d_model, bias=False, device=device)
            if config.image_projector == ImageProjectType.random_linear:
                image_projector.weight.requires_grad = False
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")

        return image_pooling_2d, image_projector

    @classmethod
    def build(cls, config: MolmoVisionBackboneConfig, outut_dim, device=None) -> "MolmoVisionBackbone":
        return MolmoVisionBackbone(config, outut_dim, device)

    def apply_compile(self, **kwargs):
        if self.config.compile_connector:
            if self.config.compile_connector == "dynamic":
                connect_kwargs = dict(kwargs, dynamic=True)
            elif self.config.compile_connector == "default":
                connect_kwargs = kwargs
            else:
                raise NotImplementedError(self.config.compile_connector)
            if self.image_pooling_2d is not None:
                if not self.config.use_deepstack or self.config.share_connector:
                    self.image_pooling_2d.compile(**connect_kwargs)
                else:
                    for module in self.image_pooling_2d:
                        module.compile(**connect_kwargs)
            if not self.config.use_deepstack or self.config.share_connector:
                self.image_projector.compile(**connect_kwargs)
            else:
                for module in self.image_projector:
                    module.compile(**connect_kwargs)
        if self.config.compile_vit == "blocks":
            for block in self.image_vit.transformer.resblocks:
                block.compile(**kwargs)
        elif self.config.compile_vit is not None:
            raise NotImplementedError(self.config.compile_vit)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        if self.config.normalize_on_gpu:
            images = self.image_preprocessor.normalize_image_tensor(images)
        image_features = self.image_vit(images)

        if cfg.use_deepstack:
            image_features = [
                image_features[layer][:, self.num_prefix_tokens :].view(B, T, N, -1) for layer in self.vit_layers
            ]
        else:
            features = []
            for layer in self.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)

            if self.num_prefix_tokens > 0:
                image_features = image_features[:, 1:]
            image_features = image_features.view(B, T, N, -1)
        return image_features

    def apply_connector(
        self,
        pooling_fn: nn.Module,
        projector_fn: nn.Module,
        image_features: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        batch_size = image_features.shape[0]
        dim = image_features.shape[-1]
        if self.config.pool_size_embeds:
            embed = self.pad_embed[self._pad_embed_idx[pooled_patches_idx.shape[-1]]]
            image_features = image_features + embed.view(1, 1, 1, -1)

        valid = pooled_patches_idx >= 0

        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
        batch_idx = torch.tile(
            batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]]
        )

        # Now [batch, num_high_res_features, pool_dim, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
        # Keep padding masks in the feature dtype.  Casting them to float32 here
        # silently promoted bf16/fp16 activations, which then failed against the
        # connector's reduced-precision Linear weights during inference.
        to_pool = to_pool * valid[:, :, :, None].to(dtype=to_pool.dtype)
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
        attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]]) if self.config.pooling_attention_mask else None

        if cfg.image_pooling_2d in [
            ImagePooling2DType.attention_meanq,
            ImagePooling2DType.attention_meanq_2x,
            ImagePooling2DType.attention_meanq_4x,
        ]:
            if self.config.pooling_attention_mask:
                denom = valid.view(-1, to_pool.shape[-2]).sum(-1)
                denom = torch.clamp(denom, min=1).to(dtype=to_pool.dtype)
                query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
            else:
                query = to_pool.mean(-2, keepdim=True)
            pooled_features = pooling_fn(query, to_pool, attn_mask=attn_mask)
        elif cfg.image_pooling_2d == ImagePooling2DType.mean:
            denom = valid.reshape(-1, to_pool.shape[1]).sum(-1, keepdim=True)
            denom = torch.clamp(denom, min=1).to(dtype=to_pool.dtype)
            pooled_features = to_pool.sum(-2) / denom
        elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
            pooled_features = pooling_fn(to_pool[:, :1, :], to_pool, attn_mask=attn_mask)
        else:
            pooled_features = to_pool

        pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

        # MLP layer to map the feature.
        if cfg.image_projector == ImageProjectType.mlpx2:
            for module in projector_fn:
                pooled_features = module(pooled_features)
        else:
            pooled_features = projector_fn(pooled_features)

        return pooled_features

    def add_image_padding_embed(self, image_features: torch.Tensor, image_masks: torch.Tensor):
        cfg = self.config

        if cfg.image_padding_embed == "pad_embed":
            all_pad = (image_masks == 0).to(dtype=image_features.dtype)
            pad_embed = self.pad_embed[None, None, None, :].to(dtype=image_features.dtype)
            image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
        elif cfg.image_padding_embed == "regress":
            pad_embed = self.pad_embed[None, None, None, :].to(dtype=image_features.dtype)
            padding = torch.maximum(image_masks, torch.zeros_like(image_masks)).to(dtype=image_features.dtype)
            image_features = image_features + pad_embed * torch.unsqueeze(padding, -1)
        elif cfg.image_padding_embed == "pad_and_partial_pad":
            pad_embed = self.pad_embed[:, None, None, None, :].to(dtype=image_features.dtype)
            all_pad = image_masks == 0
            partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=image_features.dtype)
            all_pad = all_pad.to(dtype=image_features.dtype)
            image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
            image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
        else:
            raise ValueError(cfg.image_padding_embed)

        return image_features

    def forward(
        self,
        images: torch.Tensor,
        image_masks: torch.Tensor | None,
        pooled_patches_idx,
    ):
        """Encode and pool image patches for ordinary action inference."""
        cfg = self.config
        image_features = self.encode_image(images)

        if cfg.image_padding_embed:
            if image_masks is None:
                raise ValueError("image_padding_embed requires image_masks.")
            if isinstance(image_features, (list, tuple)):
                image_features = [self.add_image_padding_embed(features, image_masks) for features in image_features]
            else:
                image_features = self.add_image_padding_embed(image_features, image_masks)

        if isinstance(image_features, (list, tuple)):
            image_features = [self.image_feature_dropout(features) for features in image_features]
            if cfg.share_connector:
                pooling_fns = [self.image_pooling_2d] * len(self.vit_layers)
                projector_fns = [self.image_projector] * len(self.vit_layers)
            else:
                pooling_fns = self.image_pooling_2d
                projector_fns = self.image_projector
        else:
            image_features = [self.image_feature_dropout(image_features)]
            pooling_fns = [self.image_pooling_2d]
            projector_fns = [self.image_projector]

        multiple_pooling = isinstance(pooled_patches_idx, (tuple, list))
        pooling_specs = pooled_patches_idx if multiple_pooling else [pooled_patches_idx]
        all_pooled_features = []
        for pooling_indices in pooling_specs:
            valid_token = torch.any(pooling_indices >= 0, dim=-1)
            pooled_features = [
                self.apply_connector(pooling_fn, projector_fn, features, pooling_indices)
                for features, pooling_fn, projector_fn in zip(image_features, pooling_fns, projector_fns, strict=False)
            ]
            value = pooled_features if len(pooled_features) > 1 else pooled_features[0]
            all_pooled_features.append((value, valid_token))

        if multiple_pooling:
            return all_pooled_features
        features, valid_token = all_pooled_features[0]
        if isinstance(features, list):
            return [
                layer_features.reshape(-1, layer_features.shape[-1])[valid_token.reshape(-1)]
                for layer_features in features
            ]
        return features.reshape(-1, features.shape[-1])[valid_token.reshape(-1)]
