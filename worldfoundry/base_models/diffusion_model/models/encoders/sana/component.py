"""Sana Gemma-2 :class:`~...contracts.ConditionEncoder`.

Tokenizes prompts (optionally prefixed with the official English
enhancement template) and encodes them with Gemma-2.  Returns
``Conditioning.positive[\"context\"]`` / ``negative[\"context\"]`` for
the Sana DiT cross-attention.  Does not rewrite non-English user
prompts; the prefix is English-only instruction text from the release.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import torch
from transformers import AutoTokenizer, Gemma2Config, Gemma2Model
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import (
    MaterializedCheckpoint,
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
)

SANA_PROMPT_PREFIX = "\n".join(
    (
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        'Here are examples of how to transform or refine prompts:',
        '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
        '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
        'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
        'User Prompt: ',
    )
)


class SanaGemmaModule(torch.nn.Module):
    """Gemma-2 encoder with the same key surface as the published checkpoint."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        config = Gemma2Config.from_dict(dict(checkpoint_config))
        config.use_cache = False
        self.model = Gemma2Model(config)


def _gemma_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    config_path = checkpoint.root / "config.json"
    if not config_path.is_file():
        config_path = checkpoint.root / "text_encoder" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Sana Gemma config does not exist: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Sana Gemma config.json must contain an object")
    return {"checkpoint_config": value}


def _gemma_state_dict(state: Mapping[str, object]) -> Mapping[str, object]:
    """Accept both encoder-only and CausalLM-prefixed Gemma checkpoints."""

    converted: dict[str, object] = {}
    for key, value in state.items():
        source = str(key)
        if source.startswith("model.model."):
            source = source.removeprefix("model.")
        elif not source.startswith("model."):
            source = f"model.{source}"
        if source.startswith("model.layers.") or source in {
            "model.embed_tokens.weight",
            "model.norm.weight",
        }:
            converted[source] = value
    return converted


class SanaPromptConditioner:
    """Encode positive and negative prompts into the canonical Sana context."""

    def __init__(
        self,
        encoder: SanaGemmaModule,
        tokenizer,
        *,
        max_length: int = 300,
        enhance_prompt: bool = True,
    ) -> None:
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enhance_prompt = bool(enhance_prompt)
        if self.max_length < 2:
            raise ValueError("Sana max_length must be at least two")
        self.tokenizer.padding_side = "right"

    @torch.no_grad()
    def _branch(
        self,
        prompts: Sequence[str],
        *,
        enhance: bool,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        texts = [f"{SANA_PROMPT_PREFIX}{prompt}" if enhance else prompt for prompt in prompts]
        prefix_tokens = len(self.tokenizer.encode(SANA_PROMPT_PREFIX)) if enhance else 2
        encoded_length = prefix_tokens + self.max_length - 2 if enhance else self.max_length
        tokens = self.tokenizer(
            texts,
            max_length=encoded_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device=device)
        attention_mask = tokens.attention_mask.to(device=device)
        hidden = self.encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        if enhance:
            select = torch.tensor(
                [0, *range(encoded_length - self.max_length + 1, encoded_length)],
                device=device,
            )
            hidden = hidden.index_select(1, select)
            attention_mask = attention_mask.index_select(1, select)
        return {
            "context": hidden[:, None],
            "context_mask": attention_mask,
        }

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Encode prompts (plus optional motion-score suffix) to Gemma-2 context."""
        motion_score = int(request.inputs.get("motion_score", 0))
        if motion_score < 0:
            raise ValueError("Sana motion_score must be non-negative")
        positive_prompts = (
            tuple(f"{prompt.strip()} motion score: {motion_score}." for prompt in request.prompts)
            if motion_score > 0
            else request.prompts
        )
        positive = self._branch(
            positive_prompts,
            enhance=self.enhance_prompt,
            device=device,
        )
        negative_prompts = request.negative_prompts or ("",) * request.batch_size
        negative = self._branch(negative_prompts, enhance=False, device=device)
        positive = {
            "context": positive["context"].to(device=device, dtype=dtype),
            "context_mask": positive["context_mask"].to(device=device),
        }
        negative = {
            "context": negative["context"].to(device=device, dtype=dtype),
            "context_mask": negative["context_mask"].to(device=device),
        }
        shared = {
            "cfg_scale": torch.full(
                (request.batch_size,),
                request.sampling.guidance_scale,
                device=device,
                dtype=dtype,
            ),
            "img_hw": torch.tensor(
                [[request.height, request.width]],
                device=device,
                dtype=dtype,
            ).expand(request.batch_size, -1),
            "aspect_ratio": torch.full(
                (request.batch_size, 1),
                float(request.height) / float(request.width),
                device=device,
                dtype=dtype,
            ),
        }
        return Conditioning(positive=positive, negative=negative, shared=shared)


def _sana_vram_module_map() -> Mapping[type[torch.nn.Module], type[torch.nn.Module]]:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    return {
        torch.nn.Linear: AutoWrappedLinear,
        torch.nn.Embedding: AutoWrappedModule,
        torch.nn.LayerNorm: AutoWrappedModule,
        Gemma2RMSNorm: AutoWrappedModule,
    }


def build_sana_prompt_conditioner(context: ComponentBuildContext) -> SanaPromptConditioner:
    encoder = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=SanaGemmaModule,
            config_resolver=_gemma_config,
            state_dict_converter=_gemma_state_dict,
            vram_module_map=_sana_vram_module_map(),
            layer_container="model.layers",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(encoder, SanaGemmaModule):
        raise TypeError(f"expected SanaGemmaModule, got {type(encoder).__name__}")
    resources = NativeCheckpointResolver().materialize(context.require_checkpoint("tokenizer"))
    tokenizer_root = resources.root / "tokenizer"
    if not tokenizer_root.is_dir():
        tokenizer_root = resources.directory()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
    return SanaPromptConditioner(
        encoder,
        tokenizer,
        max_length=int(context.component_options.get("max_length", 300)),
        enhance_prompt=bool(context.component_options.get("enhance_prompt", True)),
    )


__all__ = [
    "SANA_PROMPT_PREFIX",
    "SanaGemmaModule",
    "SanaPromptConditioner",
    "build_sana_prompt_conditioner",
]
