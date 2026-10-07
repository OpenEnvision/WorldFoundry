"""Hierarchical typed pipeline configuration (Pydantic v2).

Section models that belong to a runtime policy live next to that policy and are
re-exported here so ``PipelineConfig`` stays the single YAML entry point.

  paths      → runtime weight loading (+ optional paths.dit_export regional block AOTI)
  dit        → DiffusionTransformer3D(...)
  generation → Kandinsky6Pipeline defaults
  cache      → runtime.cache
  text_embedder → Kandinsky6TextEmbedder (optional NF4 Qwen)
  beautifier → prompt rewrite before encode (qwen25 reuses that Qwen)
  audio_vae  → mel autoencoder (tod_vae weights)
  vocoder    → BigVGAN (separate checkpoint)
  offload    → runtime.offload
  compile    → runtime.compile / runtime.aoti
  attention  → runtime.kernels
  piflow     → runtime.sampler
  profile    → runtime.profile
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel as _Cfg
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.cache import CacheConfig, CacheModeName, MagCacheConfig, NaviCacheConfig
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.compile import AotiCompileConfig, CompileConfig
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.kernels import AttentionConfig
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.offload import OffloadConfig
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.profile import ProfileConfig, ProfileMode
from worldfoundry.base_models.diffusion_model.schedulers.kandinsky6.sampler_config import PiFlowConfig

__all__ = (
    "AotiCompileConfig",
    "AttentionConfig",
    "AudioVAEConfig",
    "BeautifierConfig",
    "CacheConfig",
    "CacheModeName",
    "CompileConfig",
    "DiTConfig",
    "GenerationConfig",
    "MagCacheConfig",
    "NaviCacheConfig",
    "OffloadConfig",
    "PathsConfig",
    "ProfileConfig",
    "ProfileMode",
    "PiFlowConfig",
    "PipelineConfig",
    "TextEmbedderConfig",
    "VocoderConfig",
    "load_config",
)


BeautifierName = Literal["none", "qwen25", "qwen35_9b", "qwen38_27b", "gigachat"]


class BeautifierConfig(_Cfg):
    """Prompt beautifier selected by name.

    ``qwen25`` reuses the text encoder. ``qwen35_9b``, ``qwen38_27b``, and
    ``gigachat`` are registered by the internal package. ``model_path`` is a
    local snapshot for the Qwen variants.
    """

    name: BeautifierName = "qwen25"
    model_path: str | None = None


class PathsConfig(_Cfg):
    dit: str
    vae: str
    qwen: str
    clip: str
    dit_export: str | None = None  # regional AOTI .pt2 for visual_transformer_blocks[*]


class DiTConfig(_Cfg):
    in_visual_dim: int = 16
    out_visual_dim: int = 16
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    time_dim: int = 512
    patch_size: tuple[int, int, int] = (1, 2, 2)
    model_dim: int = 1792
    ff_dim: int = 7168
    num_text_blocks: int = 2
    num_visual_blocks: int = 32
    axes_dims: tuple[int, int, int] = (16, 24, 24)
    visual_cond: bool = True
    is_multimodal: bool = False
    # I2VA: number of visual token-type ids (0=off, 2=gen/ref for tail_cond)
    visual_token_type_num_embeddings: int = 0
    # Audio / fused
    in_audio_dim: int = 20
    out_audio_dim: int = 20
    model_dim_a: int | None = None
    time_dim_a: int | None = None
    ff_dim_a: int | None = None
    axes_dims_a: tuple[int, int, int] | None = None
    audio_freqs_scaling: float = 1.0
    text_token_padding: bool = False
    ca_rope: bool = False
    cross_gates: bool = False
    fix_modulation: bool = False

    @field_validator("patch_size", "axes_dims", "axes_dims_a", mode="before")
    @classmethod
    def _as_tuple(cls, v):
        if v is None:
            return v
        return tuple(v)


class GenerationConfig(_Cfg):
    height: int = 512
    width: int = 768
    sample_frames: int | None = None
    latent_frames: int | None = None
    num_steps: int = 50
    guidance_weight: float = 5.0
    scheduler_scale: float = 10.0
    scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0)
    # Visual conditioning scheme for I2V / I2VA (K5 inference.visual_cond_scheme)
    visual_cond_scheme: Literal["pretrain", "i2v", "tail_cond_first_frame"] = "pretrain"
    # I2V/I2VA: when height/width are omitted, resize the input image to fit this
    # pixel area with sides divisible by ``image_divisibility`` (K5 i2v resize_image).
    # None → height * width from this config section.
    max_area: int | None = None
    image_divisibility: int = 16

    @field_validator("scale_factor", mode="before")
    @classmethod
    def _as_tuple(cls, v):
        return tuple(v)


class TextEmbedderConfig(_Cfg):
    max_length: int = 1024
    # NF4 Qwen via bitsandbytes (``kandinsky.runtime.quantize.nf4``). Saves ~10–12 GiB VRAM.
    quantized_qwen: bool = False


class AudioVAEConfig(_Cfg):
    """Mel autoencoder. The vocoder is ``VocoderConfig``, not a field of this section."""

    mode: Literal["44k"] = "44k"
    need_vae_encoder: bool = True
    need_vae_decoder: bool = True
    tod_vae_ckpt: str
    scaling_factor: float = 1.0


class VocoderConfig(_Cfg):
    """BigVGAN directory: Diffusers ``diffusion_pytorch_model.safetensors`` or ``bigvgan_generator.pt``."""

    ckpt: str


class PipelineConfig(_Cfg):
    """Settings that construct a pipeline.

    ``checkpoint`` names a repo in ``kandinsky.runtime.weights.CHECKPOINTS``.
    Relative ``paths`` are resolved against that snapshot when the factory builds
    the pipeline. Policy sections left at their defaults do not change the pipe.
    """

    checkpoint: str = "pro-distill"
    paths: PathsConfig
    dit: DiTConfig = Field(default_factory=DiTConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    piflow: PiFlowConfig = Field(default_factory=PiFlowConfig)
    text_embedder: TextEmbedderConfig = Field(default_factory=TextEmbedderConfig)
    beautifier: BeautifierConfig = Field(default_factory=BeautifierConfig)
    audio_vae: AudioVAEConfig | None = None
    vocoder: VocoderConfig | None = None
    offload: OffloadConfig = Field(default_factory=OffloadConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)

    @model_validator(mode="after")
    def _audio_components_together(self) -> PipelineConfig:
        if (self.audio_vae is None) != (self.vocoder is None):
            raise ValueError("audio_vae and vocoder must be configured together")
        return self


_CONFIGS = Path(__file__).resolve().parents[3] / "data/models/runtime/configs/kandinsky6"


def _read_mapping(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return raw


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _checkpoint_file(name: str) -> Path:
    path = _CONFIGS / "checkpoints" / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown checkpoint {name!r}; no file {path}")
    return path


def _module_file(name: str) -> Path:
    path = _CONFIGS / "modules" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"unknown DiT module {name!r}; no file {path}")
    return path


def _per_grid_overrides(module: dict, overrides: dict, raw: dict) -> dict:
    """Drop head sizes that are already ``base * n_grid``.

    Distilled checkpoints store the expanded PiFlow head. ``PiFlowDiffusionTransformer3D``
    multiplies the per-grid width itself, so the config keeps the module width.
    """
    piflow = raw.get("piflow") or {}
    n_grid = piflow.get("dx_num_grid_points")
    if not piflow.get("enabled") or not isinstance(n_grid, int):
        return overrides
    kept = dict(overrides)
    for key in ("out_visual_dim", "out_audio_dim"):
        expanded = kept.get(key)
        base = module.get(key)
        if isinstance(expanded, int) and isinstance(base, int) and expanded == base * n_grid:
            del kept[key]
    return kept


def _expand_dit(raw: dict) -> dict:
    """Turn ``dit: transformer_pro`` plus ``dit_overrides`` into a DiT mapping."""
    dit = raw.get("dit")
    if not isinstance(dit, str):
        raw.pop("dit_overrides", None)
        raw.pop("scheduler", None)
        return raw
    module = json.loads(_module_file(dit).read_text(encoding="utf-8"))
    if not isinstance(module, dict):
        raise ValueError(f"DiT module {dit!r} must be a mapping")
    module.pop("scale_factor", None)
    overrides = raw.pop("dit_overrides", None) or {}
    if not isinstance(overrides, dict):
        raise ValueError("dit_overrides must be a mapping")
    raw["dit"] = {**module, **_per_grid_overrides(module, overrides, raw)}
    raw.pop("scheduler", None)
    return raw


def _compose(raw: dict) -> dict:
    """Fill a device preset from the checkpoint file it names.

    A preset without ``paths`` is a device file. Its ``checkpoint`` selects
    ``configs/checkpoints/<name>.yaml``. Sections in the preset override that file.
    """
    if "paths" in raw:
        return _expand_dit(raw)
    name = raw.get("checkpoint")
    if not isinstance(name, str) or not name:
        raise ValueError("a device config must set checkpoint")
    base = _expand_dit(_read_mapping(_checkpoint_file(name)))
    return _deep_merge(base, raw)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a device preset or a full pipeline YAML.

    A device preset names ``checkpoint``. When the caller does not pass another
    checkpoint, that name is the snapshot the factory downloads and runs.
    """
    return PipelineConfig.model_validate(_compose(_read_mapping(Path(path))))
