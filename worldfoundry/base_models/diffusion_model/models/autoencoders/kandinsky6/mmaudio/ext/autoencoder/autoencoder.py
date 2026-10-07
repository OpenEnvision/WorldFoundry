import json
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.bigvgan_v2.bigvgan import load_hparams_from_json
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.bigvgan_v2.env import AttrDict
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder.distributions import DiagonalGaussianDistribution
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder.vae import VAE

_DIFFUSERS_WEIGHTS = "diffusion_pytorch_model.safetensors"
_BUNDLED_VOCODER_CONFIG = Path(__file__).resolve().parents[1] / "bigvgan_v2" / "config.json"
_VAE_PREFIX = "vae."


def build_bigvgan_v2(vocoder_config: dict | AttrDict) -> BigVGANv2:
    """Build an inference-ready BigVGAN-v2 from its serialized configuration."""
    model = BigVGANv2(vocoder_config if isinstance(vocoder_config, AttrDict) else AttrDict(vocoder_config))
    model.remove_weight_norm()
    return model


def vocoder_hparams(config_file: str | Path) -> AttrDict:
    """Hyperparameters for BigVGAN.

    A training ``config.json`` is used as stored. A Diffusers ``MMAudioVocoder``
    config only lists the architecture fields, so the bundled BigVGAN defaults
    fill ``resblock`` and ``activation``.
    """
    raw = json.loads(Path(config_file).read_text(encoding="utf-8"))
    if "resblock" in raw and "activation" in raw:
        return load_hparams_from_json(str(config_file))
    base = json.loads(_BUNDLED_VOCODER_CONFIG.read_text(encoding="utf-8"))
    base.update({key: value for key, value in raw.items() if not key.startswith("_")})
    return AttrDict(base)


def vae_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mel-VAE weights from a flat file or a Diffusers audio-VAE snapshot.

    The Diffusers file also stores the vocoder and mel converter. Only the ``vae.``
    tensors belong to this module.
    """
    if any(key.startswith(_VAE_PREFIX) for key in state):
        return {key[len(_VAE_PREFIX) :]: value for key, value in state.items() if key.startswith(_VAE_PREFIX)}
    return state


def _load_weight_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path))
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "generator" in loaded:
        return loaded["generator"]
    return loaded


def _resolve_weight_file(root: Path, legacy_name: str) -> Path | None:
    safetensors = root / _DIFFUSERS_WEIGHTS
    if safetensors.is_file():
        return safetensors
    legacy = root / legacy_name
    if legacy.is_file():
        return legacy
    return None


def load_bigvgan_v2(
    vocoder_ckpt_path: str,
    *,
    vocoder_config: dict | None = None,
) -> BigVGANv2:
    """Load BigVGAN-v2 from a local directory (avoids HubMixin/huggingface_hub API drift)."""
    root = Path(vocoder_ckpt_path)
    weight_file = _resolve_weight_file(root, "bigvgan_generator.pt")
    config_file = root / "config.json"
    if weight_file is None or not config_file.is_file():
        raise FileNotFoundError(
            f"Expected config.json and {_DIFFUSERS_WEIGHTS} or bigvgan_generator.pt under {vocoder_ckpt_path}"
        )
    hparams = AttrDict(vocoder_config) if vocoder_config is not None else vocoder_hparams(config_file)
    model = BigVGANv2(hparams)
    state = _load_weight_file(weight_file)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    return model


class AutoEncoderModule(nn.Module):
    """Mel autoencoder. The BigVGAN vocoder is a separate module."""

    def __init__(
        self,
        *,
        vae_ckpt_path: str | None = None,
        mode: Literal["44k"],
        need_vae_encoder: bool = True,
        need_vae_decoder: bool = True,
    ):
        super().__init__()
        if mode != "44k":
            raise ValueError(f"Unknown model: {mode}")
        self.vae: VAE = VAE(data_dim=128, embed_dim=40, hidden_dim=512).eval()
        if vae_ckpt_path is not None:
            weight_file = _resolve_weight_file(Path(vae_ckpt_path), "")
            source = weight_file if weight_file is not None else Path(vae_ckpt_path)
            if not source.is_file():
                raise FileNotFoundError(f"No audio VAE weights under {vae_ckpt_path}")
            self.vae.load_state_dict(vae_state_dict(_load_weight_file(source)))
        self.vae.remove_weight_norm()

        if not need_vae_decoder:
            del self.vae.decoder
        if not need_vae_encoder:
            del self.vae.encoder

        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)
