"""Shared native Wan2.2 conditioning for interactive model runtimes.

Accepts the released Diffusers directory layout while using WorldFoundry's
UMT5, VAE38, checkpoint converters and native loader. No external Wan model
implementation is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ..models.autoencoders.wan import WanVideoVAE38, convert_diffusers_wan22_vae_state_dict
from ..models.encoders.wan import WanTextEncoder, convert_diffusers_umt5_encoder_state_dict
from ..optimizations import RuntimePolicy
from . import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader


def local_weights(directory: str | Path) -> CheckpointSpec:
    root = Path(directory).expanduser()
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"Ambiguous checkpoint indexes under {root}")
    if indexes:
        manifest = json.loads(indexes[0].read_text())
        names = sorted(set(manifest["weight_map"].values()))
        if any(Path(name).name != name for name in names):
            raise ValueError("Checkpoint shard names must be local filenames")
        files = [root / name for name in names]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
    else:
        files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights under {root}")
    return CheckpointSpec(source=tuple(str(path) for path in files))


class WanInteractiveTextEncoder(nn.Module):
    """Preserve released prompt padding while loading the native UMT5 tower."""

    def __init__(self, pretrained_dir: str | Path, max_length: int = 512):
        super().__init__()
        from transformers import AutoTokenizer

        root = Path(pretrained_dir).expanduser()
        self.text_encoder = (
            NativeModuleLoader()
            .load(
                ModuleLoadSpec(
                    module_class=WanTextEncoder,
                    state_dict_converter=convert_diffusers_umt5_encoder_state_dict,
                ),
                local_weights(root / "text_encoder"),
                RuntimePolicy(device="cpu", dtype=torch.bfloat16),
            )
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        self.max_length = int(max_length)

    @torch.inference_mode()
    def padded(self, prompts: list[str], *, fixed: bool = False):
        batch = self.tokenizer(
            prompts,
            padding="max_length" if fixed else "longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids, mask = batch.input_ids, batch.attention_mask
        if ids.shape[1] < 512:
            padding = 512 - ids.shape[1]
            ids = torch.nn.functional.pad(ids, (0, padding), value=self.tokenizer.pad_token_id)
            mask = torch.nn.functional.pad(mask, (0, padding))
        device = next(self.text_encoder.parameters()).device
        ids, mask = ids.to(device), mask.to(device)
        context = self.text_encoder(ids, mask)
        context = context.masked_fill(~mask.bool().unsqueeze(-1), 0)
        return context, mask.sum(1).clamp_min(512).to(torch.int32)

    def encode(self, prompts: list[str]):
        context, lengths = self.padded(prompts)
        return torch.cat([row[: int(n)] for row, n in zip(context, lengths)]), lengths

    def forward(self, text_prompts: list[str]):
        return {"prompt_embeds": self.padded(text_prompts, fixed=True)[0]}


class WanInteractiveVAE(nn.Module):
    """Native normalized VAE38 with BFCHW latent/video boundaries."""

    def __init__(self, pretrained_dir: str | Path):
        super().__init__()
        self.model = (
            NativeModuleLoader()
            .load(
                ModuleLoadSpec(
                    module_class=WanVideoVAE38,
                    state_dict_converter=convert_diffusers_wan22_vae_state_dict,
                ),
                local_weights(Path(pretrained_dir) / "vae"),
                RuntimePolicy(device="cpu", dtype=torch.bfloat16),
            )
            .eval()
            .requires_grad_(False)
        )

    @torch.inference_mode()
    def encode(self, frames: torch.Tensor):
        if frames.dtype != torch.uint8:
            raise ValueError("reference frames must use uint8 BCTHW pixels")
        parameter = next(self.model.parameters())
        pixels = frames.to(parameter).div(127.5).sub(1)
        return self.model.encode(pixels, parameter.device).float().permute(0, 2, 1, 3, 4)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor):
        parameter = next(self.model.parameters())
        value = latents.permute(0, 2, 1, 3, 4).to(parameter)
        return self.model.decode(value, parameter.device).float().permute(0, 2, 1, 3, 4)
