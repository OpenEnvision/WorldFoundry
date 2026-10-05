"""Canonical checkpoint-bundle loading for native Wan inference variants.

Loads an official Wan directory into explicit roles (DiT, T5, VAE, tokenizer,
optional CLIP) without a model-manager backend.  Recipe factories call these
helpers, then wrap the DiT in a :class:`~..models.denoisers.wan.WanDenoiser`
that satisfies the runner ``Denoiser`` Protocol.

Each helper fails if a role glob does not match exactly one file, or if the
tokenizer directory ``google/umt5-xxl`` is missing.  Architecture is inferred
from the state-dict layout; pass ``transformer_config`` when inference fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.core.model_loading.checkpoints import load_tensor_state_dict
from worldfoundry.core.model_loading import load_model, load_state_dict

from ..models.autoencoders.wan import (
    WanVideoVAE,
    WanVideoVAE38,
    WanVideoVAEStateDictConverter,
)
from ..models.encoders.wan import (
    WanImageEncoder,
    WanImageEncoderStateDictConverter,
    WanTextEncoder,
)
from ..models.networks.wan import WanModel, WanModelStateDictConverter


@dataclass(slots=True)
class WanInferenceComponents:
    """Explicit Wan roles loaded from one official checkpoint directory.

    Attributes:
        dit: Transformer used as the denoiser inner network.
        text_encoder: UMT5 text encoder.
        vae: Wan 2.1 or 2.2 video VAE (class chosen from the filename).
        tokenizer_path: Directory ``<root>/google/umt5-xxl``.
        image_encoder: Optional CLIP tower for I2V bundles.
    """

    dit: torch.nn.Module
    text_encoder: WanTextEncoder
    vae: WanVideoVAE
    tokenizer_path: Path
    image_encoder: WanImageEncoder | None = None


@dataclass(slots=True)
class WanConditioningComponents:
    """Wan text/VAE roles that can be shared by one or more denoisers.

    Same fields as :class:`WanInferenceComponents` except ``dit``.  Dual-expert
    recipes load this once and attach two DiTs.
    """

    text_encoder: WanTextEncoder
    vae: WanVideoVAE
    tokenizer_path: Path
    image_encoder: WanImageEncoder | None = None


def _one(root: Path, patterns: tuple[str, ...], role: str) -> Path:
    matches = [path for pattern in patterns for path in sorted(root.glob(pattern))]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Wan {role} expected one file under {root}; found {[path.name for path in matches]}"
        )
    return matches[0]


def _transformer_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if not paths:
        paths = sorted(root.glob("diffusion_pytorch_model.safetensors"))
    if not paths:
        raise FileNotFoundError(f"Wan transformer checkpoint not found under {root}")
    return paths


def load_wan_transformer_checkpoint(
    checkpoint_root: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    transformer_class: type[torch.nn.Module] = WanModel,
    transformer_config: dict | None = None,
    state_dict_prefixes: tuple[str, ...] = (),
    strict: bool = True,
) -> torch.nn.Module:
    """Load a Wan transformer directory or tensor checkpoint into one role.

    ``state_dict_prefixes`` handles training containers such as
    ``pipe.dit.*`` without teaching the common loader about a specific model.
    Prefixes are optional: an already-flat canonical state dict passes through.
    """

    root = Path(checkpoint_root).expanduser().resolve()
    if root.is_dir():
        transformer_files = _transformer_paths(root)
        transformer_state = load_state_dict(
            [str(path) for path in transformer_files],
            torch_dtype=torch_dtype,
            device="cpu",
        )
    elif root.is_file():
        if root.suffix == ".safetensors":
            transformer_state = load_state_dict(
                str(root),
                torch_dtype=torch_dtype,
                device="cpu",
            )
        else:
            transformer_state = load_tensor_state_dict(
                root,
                map_location="cpu",
                mmap=True,
            )
            transformer_state = {
                name: value.to(dtype=torch_dtype)
                for name, value in transformer_state.items()
            }
        for prefix in state_dict_prefixes:
            prefixed = {
                name[len(prefix) :]: value
                for name, value in transformer_state.items()
                if name.startswith(prefix)
            }
            if prefixed:
                transformer_state = prefixed
                break
    else:
        raise FileNotFoundError(f"Wan transformer checkpoint not found: {root}")

    converter = WanModelStateDictConverter()
    if any(".attn1." in key or key.startswith("condition_embedder.") for key in transformer_state):
        transformer_state, detected_config = converter.from_diffusers(transformer_state)
    else:
        transformer_state, detected_config = converter.from_civitai(transformer_state)
    config = dict(detected_config or {})
    config.update(transformer_config or {})
    if not config:
        raise ValueError(
            "Wan transformer architecture could not be inferred; pass transformer_config explicitly"
        )
    return load_model(
        transformer_class,
        path=None,
        config=config,
        torch_dtype=torch_dtype,
        device=device,
        state_dict=transformer_state,
        strict=strict,
    )


def load_wan_conditioning_components(
    checkpoint_root: str | Path,
    *,
    image_conditioned: bool,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> WanConditioningComponents:
    """Load Wan text, tokenizer, VAE, and optional image-conditioning roles."""

    root = Path(checkpoint_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Wan checkpoint directory not found: {root}")

    text_path = _one(root, ("models_t5_umt5-xxl-enc-bf16.pth", "models_t5_*.safetensors"), "text encoder")
    text_encoder = load_model(
        WanTextEncoder,
        str(text_path),
        torch_dtype=torch_dtype,
        device=device,
    )

    vae = load_wan_vae_checkpoint(
        root,
        torch_dtype=torch_dtype,
        device=device,
    )

    tokenizer_path = root / "google" / "umt5-xxl"
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Wan tokenizer directory not found: {tokenizer_path}")

    image_encoder = None
    if image_conditioned:
        image_path = _one(
            root,
            (
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "models_clip_*.safetensors",
            ),
            "image encoder",
        )
        image_encoder = load_model(
            WanImageEncoder,
            str(image_path),
            torch_dtype=torch.float32,
            device=device,
            state_dict_converter=WanImageEncoderStateDictConverter().from_civitai,
        )

    return WanConditioningComponents(
        text_encoder=text_encoder,
        vae=vae,
        tokenizer_path=tokenizer_path,
        image_encoder=image_encoder,
    )


def load_wan_vae_checkpoint(
    checkpoint_root: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> WanVideoVAE:
    """Load only the Wan VAE role from an official checkpoint directory."""

    root = Path(checkpoint_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Wan checkpoint directory not found: {root}")
    vae_path = _one(
        root,
        ("Wan2.2_VAE.pth", "Wan2.1_VAE.pth", "Wan*_VAE.safetensors"),
        "VAE",
    )
    vae_class = WanVideoVAE38 if "2.2" in vae_path.name else WanVideoVAE
    return load_model(
        vae_class,
        str(vae_path),
        torch_dtype=torch_dtype,
        device=device,
        state_dict_converter=WanVideoVAEStateDictConverter().from_civitai,
    )


def load_wan_inference_components(
    checkpoint_root: str | Path,
    *,
    image_conditioned: bool,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    transformer_class: type[torch.nn.Module] = WanModel,
    transformer_config: dict | None = None,
    transformer_strict: bool = True,
) -> WanInferenceComponents:
    """Load a complete official Wan bundle without a model-manager backend."""

    dit = load_wan_transformer_checkpoint(
        checkpoint_root,
        torch_dtype=torch_dtype,
        device=device,
        transformer_class=transformer_class,
        transformer_config=transformer_config,
        strict=transformer_strict,
    )
    conditioning = load_wan_conditioning_components(
        checkpoint_root,
        image_conditioned=image_conditioned,
        torch_dtype=torch_dtype,
        device=device,
    )
    return WanInferenceComponents(
        dit=dit,
        text_encoder=conditioning.text_encoder,
        vae=conditioning.vae,
        tokenizer_path=conditioning.tokenizer_path,
        image_encoder=conditioning.image_encoder,
    )


__all__ = [
    "WanConditioningComponents",
    "WanInferenceComponents",
    "load_wan_conditioning_components",
    "load_wan_inference_components",
    "load_wan_transformer_checkpoint",
    "load_wan_vae_checkpoint",
]
