"""Checkpoint loading for MAGI-2-preview real weights.

The published ``sand-ai/MAGI-2-preview`` checkpoints use several formats that
must be reconciled against the ported single-GPU modules (verified key-diffs
against the real weights, all 0-missing on the model side):

* **preview / refiner DiT** — sharded safetensors whose keys match the ported
  modules verbatim (1371 / 281 keys, 0 missing / 0 unexpected). Load directly.
* **video VAE** — ``Wan2.2_VAE.pth`` whose 196 keys match ``Magi2VideoVAE.vae``
  directly. Load into the inner ``.vae`` module.
* **turbo decoder** — a training ``.ckpt``; the generator lives at
  ``state_dict['gen_model']`` (90 keys). 80 map to the decode module; the extra
  ``aligned_feature_projection_heads`` + ``latents_mean/std`` are training-only
  and dropped (strict=False).
* **audio VAE** — a full Stable-Audio pipeline safetensors; the autoencoder is
  under the ``pretransform.model.`` prefix (365 keys, encoder+decoder), with
  legacy ``weight_g``/``weight_v`` weight-norm names remapped to the
  ``parametrizations.weight.original0/1`` form when the module uses the modern
  parametrization.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import torch


def _load_sharded_safetensors(component_dir: Path) -> dict[str, torch.Tensor]:
    """Merge an index-mapped or globbed ``*.safetensors`` directory into one dict."""

    from safetensors.torch import load_file

    index = glob.glob(str(component_dir / "*.safetensors.index.json"))
    merged: dict[str, torch.Tensor] = {}
    if index:
        weight_map = json.loads(Path(index[0]).read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            merged.update(load_file(str(component_dir / shard)))
        return merged
    for shard in sorted(component_dir.glob("*.safetensors")):
        merged.update(load_file(str(shard)))
    return merged


def _remap_weight_norm(state_dict: dict[str, torch.Tensor], model_keys: set[str]) -> dict[str, torch.Tensor]:
    """Rename legacy ``weight_g``/``weight_v`` to parametrization form if needed."""

    needs_param = any(".parametrizations.weight.original0" in k for k in model_keys)
    if not needs_param:
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            out[key[: -len(".weight_g")] + ".parametrizations.weight.original0"] = value
        elif key.endswith(".weight_v"):
            out[key[: -len(".weight_v")] + ".parametrizations.weight.original1"] = value
        else:
            out[key] = value
    return out


def load_magi2_dit_weights(model: torch.nn.Module, component_dir: str | Path, *, strict: bool = True):
    """Load a preview/refiner DiT checkpoint dir (sharded safetensors) directly."""

    sd = _load_sharded_safetensors(Path(component_dir))
    result = model.load_state_dict(sd, strict=strict, assign=True)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_magi2_video_vae_weights(video_vae: torch.nn.Module, pth_path: str | Path, *, strict: bool = True):
    """Load ``Wan2.2_VAE.pth`` into ``Magi2VideoVAE`` (keys map to ``.vae``)."""

    sd = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    inner = getattr(video_vae, "vae", video_vae)
    # The ported inner VAE constructs on the meta device (no eager weights), so
    # assign=True replaces the meta params with the loaded tensors.
    result = inner.load_state_dict(sd, strict=strict, assign=True)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_magi2_turbo_weights(turbo: torch.nn.Module, ckpt_path: str | Path):
    """Load the turbo decoder from a training ``.ckpt`` (``state_dict.gen_model``).

    Training-only tensors (``aligned_feature_projection_heads``,
    ``latents_mean``/``latents_std``) are absent from the decode module and
    dropped via ``strict=False``.
    """

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    gen = ckpt["state_dict"]["gen_model"]
    result = turbo.load_state_dict(gen, strict=False)
    # Only tolerate the known training-only unexpected keys; no missing decode keys.
    return list(result.missing_keys), list(result.unexpected_keys)


def load_magi2_audio_vae_weights(audio_vae: torch.nn.Module, safetensors_path: str | Path, *, strict: bool = False):
    """Load the Stable-Audio autoencoder from the ``pretransform.model.`` prefix."""

    from safetensors.torch import load_file

    full = load_file(str(safetensors_path))
    prefix = "pretransform.model."
    sd = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
    model_keys = set(dict(audio_vae.named_parameters()) | dict(audio_vae.named_buffers()))
    sd = _remap_weight_norm(sd, model_keys)
    result = audio_vae.load_state_dict(sd, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


__all__ = [
    "load_magi2_audio_vae_weights",
    "load_magi2_dit_weights",
    "load_magi2_turbo_weights",
    "load_magi2_video_vae_weights",
]
