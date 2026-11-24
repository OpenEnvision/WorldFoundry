# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_vit_model_arch(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
):
    if "plus" in compact_arch_name:
        model_arch = compact_arch_name.replace("plus", f"{patch_size}plus")
    else:
        model_arch = f"{compact_arch_name}{patch_size}"
    return model_arch


def _make_dinov3_vit_model_url(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
    version: str | None = None,
    weights: Weights | str = Weights.LVD1689M,
    hash: str | None = None,
):
    model_name = "dinov3"
    model_arch = _make_dinov3_vit_model_arch(patch_size=patch_size, compact_arch_name=compact_arch_name)
    version_suffix = f"_{version}" if version else ""
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"{model_name}_{model_arch}"
    model_filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return os.path.join(_DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: float | None = None,
    pos_embed_rope_max_period: float | None = None,
    pos_embed_rope_normalize_coords: str = "separate",
    pos_embed_rope_shift_coords: float | None = None,
    pos_embed_rope_jitter_coords: float | None = None,
    pos_embed_rope_rescale_coords: float | None = None,
    pos_embed_rope_dtype: str = "fp32",
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: float | None = None,
    norm_layer: str = "layernorm",
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    version: str | None = None,
    weights: Weights | str = Weights.LVD1689M,
    hash: str | None = None,
    check_hash: bool = False,
    **kwargs,
):
    from ..models.vision_transformer import DinoVisionTransformer

    vit_kwargs = {
        "img_size": img_size,
        "patch_size": patch_size,
        "in_chans": in_chans,
        "pos_embed_rope_base": pos_embed_rope_base,
        "pos_embed_rope_min_period": pos_embed_rope_min_period,
        "pos_embed_rope_max_period": pos_embed_rope_max_period,
        "pos_embed_rope_normalize_coords": pos_embed_rope_normalize_coords,
        "pos_embed_rope_shift_coords": pos_embed_rope_shift_coords,
        "pos_embed_rope_jitter_coords": pos_embed_rope_jitter_coords,
        "pos_embed_rope_rescale_coords": pos_embed_rope_rescale_coords,
        "pos_embed_rope_dtype": pos_embed_rope_dtype,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "ffn_ratio": ffn_ratio,
        "qkv_bias": qkv_bias,
        "drop_path_rate": drop_path_rate,
        "layerscale_init": layerscale_init,
        "norm_layer": norm_layer,
        "ffn_layer": ffn_layer,
        "ffn_bias": ffn_bias,
        "proj_bias": proj_bias,
        "n_storage_tokens": n_storage_tokens,
        "mask_k_bias": mask_k_bias,
    }
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_vit_model_url(
                patch_size=patch_size,
                compact_arch_name=compact_arch_name,
                version=version,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        model.load_state_dict(state_dict, strict=True)
    else:
        model.init_weights()
    return model


def _make_dinov3_convnext_model_url(
    *,
    compact_arch_name: str = "convnext_base",
    weights: Weights | str = Weights.LVD1689M,
    hash: str | None = None,
):
    model_name = "dinov3"
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"{model_name}_{compact_arch_name}"
    model_filename = f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"
    return os.path.join(_DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_convnext(
    in_chans: int = 3,
    depths: list[int] = [3, 3, 27, 3],  # noqa: B006 - Keeps the upstream default object and signature.
    dims: list[int] = [128, 256, 512, 1024],  # noqa: B006 - Keeps the upstream default object and signature.
    compact_arch_name: str = "convnext_base",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    hash: str | None = None,
    **kwargs,
):
    from ..models.convnext import ConvNeXt

    model_kwargs = {
        "in_chans": in_chans,
        "depths": depths,
        "dims": dims,
        "drop_path_rate": drop_path_rate,
        "layer_scale_init_value": layer_scale_init_value,
    }
    model_kwargs.update(**kwargs)
    model = ConvNeXt(**model_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_convnext_model_url(
                compact_arch_name=compact_arch_name,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


# ─── ViT model factory functions ───


def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-S/16 model (optionally) pretrained on the LVD-1689M dataset."""
    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-S+/16 model (optionally) pretrained on the LVD-1689M dataset."""
    if "hash" not in kwargs:
        kwargs["hash"] = "4057cbaa"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitsplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-B/16 model (optionally) pretrained on the LVD-1689M dataset."""
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitb",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-L/16 model (optionally) pretrained on the LVD-1689M dataset."""
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) != 1:
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        hash_val = matches[0]
        if hash_val == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16plus(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-L+/16 model (optionally) pretrained on the LVD-1689M dataset."""
    if "hash" not in kwargs:
        kwargs["hash"] = "46503df0"
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitlplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-H+/16 model (optionally) pretrained on the LVD-1689M dataset."""
    if "hash" not in kwargs:
        kwargs["hash"] = "7c1da9a5"
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vithplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """DINOv3 ViT-7B/16 model (optionally) pretrained on the LVD-1689M dataset."""
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a955f4ea"
    elif weights == Weights.SAT493M and "hash" not in kwargs:
        kwargs["hash"] = "a6675841"
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b",
        check_hash=check_hash,
        **kwargs,
    )


# ─── ConvNeXt model factory functions ───


def dinov3_convnext_tiny(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    **kwargs,
):
    """DINOv3 ConvNeXt-Tiny model (optionally) pretrained on the LVD-1689M dataset."""
    _hash_convnext = "21b726bb"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext
    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["tiny"]
    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_tiny",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_small(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    **kwargs,
):
    """DINOv3 ConvNeXt-Small model (optionally) pretrained on the LVD-1689M dataset."""
    _hash_convnext = "296db49d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext
    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["small"]
    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_small",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_base(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    **kwargs,
):
    """DINOv3 ConvNeXt-Base model (optionally) pretrained on the LVD-1689M dataset."""
    _hash_convnext = "801f2ba9"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext
    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["base"]
    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_base",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_large(
    *,
    pretrained: bool = True,
    weights: Weights | str = Weights.LVD1689M,
    **kwargs,
):
    """DINOv3 ConvNeXt-Large model (optionally) pretrained on the LVD-1689M dataset."""
    _hash_convnext = "61fa432d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext
    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["large"]
    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_large",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


# ─── Convenience DINOv3 wrapper (pretrained=False) for downstream models ───


def DINOv3(model_name: str):
    """Create a DINOv3 backbone without pretrained weights (for downstream model loading)."""
    model_zoo = {
        "vits": ("vit_small", "mlp"),
        "vitsplus": ("vit_so400m", "swiglu"),
        "vitb": ("vit_base", "mlp"),
        "vitl": ("vit_large", "mlp"),
        "vitlplus": ("vit_large", "swiglu"),
        "vithplus": ("vit_huge2", "swiglu"),
        "vit7b": ("vit_7b", "swiglu64"),
    }
    arch_name, ffn_layer = model_zoo[model_name]

    # Map arch_name to the embed_dim/depth/num_heads needed by _make_dinov3_vit
    vit_configs = {
        "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "ffn_ratio": 4},
        "vit_so400m": {"embed_dim": 1152, "depth": 27, "num_heads": 18, "ffn_ratio": 3.777777778},
        "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "ffn_ratio": 4},
        "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "ffn_ratio": 4},
        "vit_huge2": {"embed_dim": 1280, "depth": 32, "num_heads": 20, "ffn_ratio": 4},
        "vit_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "ffn_ratio": 3},
    }
    vit_cfg = vit_configs[arch_name]
    qkv_bias = arch_name != "vit_7b"

    return _make_dinov3_vit(
        ffn_layer=ffn_layer,
        pretrained=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        qkv_bias=qkv_bias,
        embed_dim=vit_cfg["embed_dim"],
        depth=vit_cfg["depth"],
        num_heads=vit_cfg["num_heads"],
        ffn_ratio=vit_cfg["ffn_ratio"],
        compact_arch_name=model_name,
    )
