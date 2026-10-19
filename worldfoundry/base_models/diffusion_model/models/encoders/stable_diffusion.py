# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Native CLIP text conditioning shared by SD2 inpainting and Marigold."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...components import ComponentBuildContext
from ...contracts import Conditioning, DiffusionRequest
from ...loaders import (
    MaterializedCheckpoint,
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
    checkpoint_json_config,
)


class SDCLIPModule(torch.nn.Module):
    """Stable wrapper around the checkpoint's CLIP text architecture."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        from transformers import CLIPTextConfig, CLIPTextModel

        self.model = CLIPTextModel(CLIPTextConfig.from_dict(dict(checkpoint_config)))


def clip_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    """Resolve CLIP architecture from the recipe-bound config resource."""
    return {"checkpoint_config": checkpoint_json_config(checkpoint, "text_encoder/config.json")}


def convert_clip_state_dict(state: Mapping[str, object]) -> Mapping[str, object]:
    """Prefix the shared wrapper and omit the historical nonpersistent buffer."""
    return {"model." + key: value for key, value in state.items() if key != "text_model.embeddings.position_ids"}


class SDCLIPConditioner:
    """SD uses 77 tokens; Marigold uses the two unpadded empty-text tokens."""

    def __init__(self, model: SDCLIPModule, tokenizer: Any, *, empty_text: bool) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.empty_text = empty_text

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        """Encode only the branches used by this native request."""

        def encode_text(texts: tuple[str, ...]) -> dict[str, object]:
            tokens = self.tokenizer(
                list(texts),
                padding="do_not_pad" if self.empty_text else "max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            # Released SD2/Marigold CLIP does not use the tokenizer padding mask.
            hidden = self.model.model(tokens.input_ids.to(device))[0].to(dtype=dtype)
            return {"context": hidden}

        positive = encode_text(("",) * request.batch_size if self.empty_text else request.prompts)
        negative = {}
        if not self.empty_text and request.sampling.guidance_scale != 1:
            negative = encode_text(request.negative_prompts or ("",) * request.batch_size)
        return Conditioning(positive=positive, negative=negative)


def build_sd_clip_conditioner(context: ComponentBuildContext) -> SDCLIPConditioner:
    """Restore CLIP with canonical loading and local tokenizer resources."""
    from transformers import CLIPTokenizer

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=SDCLIPModule, config_resolver=clip_config, state_dict_converter=convert_clip_state_dict
        ),
        context.require_checkpoint(),
        context.policy,
    )
    if not isinstance(model, SDCLIPModule):
        raise TypeError("SD CLIP loader returned the wrong component")
    model.requires_grad_(False)
    tokenizer_assets = NativeCheckpointResolver().materialize(context.require_checkpoint("tokenizer"))
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_assets.directory("tokenizer"), local_files_only=True)
    return SDCLIPConditioner(model, tokenizer, empty_text=bool(context.component_options["empty_text"]))
