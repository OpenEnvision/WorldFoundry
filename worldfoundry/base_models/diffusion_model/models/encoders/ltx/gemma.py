"""Gemma 3 / Gemma 4 text encoder for LTX split checkpoints.

Derived from Lightricks/LTX-2@a95ab856 (LTX-2.x Community License; see
THIRD-PARTY-NOTICES). :class:`GemmaAssets` reads a Hugging Face Gemma root or a
single text-encoder ``.safetensors`` file that embeds the config, tokenizer and
processor sidecars. :class:`LTXGemmaTextEncoder` runs the language model for
prompt hidden states and, for instruct checkpoints, prompt enhancement. The
``populate_*`` functions fill the non-persistent rotary and embedding-scale
buffers that meta-device construction leaves unset.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch
import transformers
from tokenizers import Tokenizer
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.video_processing_utils import BaseVideoProcessor

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.tokenizer import LTXVGemmaTokenizer, PaddingSide


def _find_matching_file(root_path: str, pattern: str) -> Path:
    """
    Recursively search for files matching a glob pattern and return the first match.
    """
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return matches[0]


GEMMA_CONFIG_METADATA_KEY = "gemma_config"


TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"


HF_ASSET_TENSOR_PREFIX = "hf_asset__"


_REQUIRED_ASSET_FILENAMES = (
    "tokenizer_config.json",
    "processor_config.json",
)


# Older / Comfy packs may store these as metadata strings instead of ``hf_asset__`` tensors.
_METADATA_FALLBACK_FILENAMES = (
    *_REQUIRED_ASSET_FILENAMES,
    "chat_template.jinja",
    "generation_config.json",
)


# Collect by pattern (not allowlist) so future processor sidecars are not silently dropped.
_ASSET_GLOBS = ("*.json", "*.jinja")


_ASSET_GLOB_EXCLUDES = frozenset({"config.json", "tokenizer.json"})


def is_safetensors_file(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix == ".safetensors"


@dataclass(frozen=True, slots=True)
class GemmaAssets:
    """In-memory Gemma HF assets loaded from a directory root or a single TE file."""

    source: str
    config_dict: Mapping[str, Any]
    tokenizer_json: bytes
    sidecars: Mapping[str, bytes]
    weight_paths: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> GemmaAssets:
        p = Path(path)
        if p.is_dir():
            return cls.from_root(p)
        if is_safetensors_file(p):
            return cls.from_single_file(p)
        raise FileNotFoundError(f"Gemma path {path!r} is neither a directory nor a .safetensors file.")

    @classmethod
    def from_root(cls, root: Path) -> GemmaAssets:
        """Load an HF Gemma directory (flat or nested ``_readout_proj``)."""
        root = Path(root)
        config_path = _find_matching_file(str(root), "config.json")
        tokenizer_path = _find_matching_file(str(root), "tokenizer.json")
        config_dict = json.loads(config_path.read_bytes())
        tokenizer_json = tokenizer_path.read_bytes()

        # Nested roots can duplicate basenames; keep the first hit (matches find_matching_file).
        sidecars: dict[str, bytes] = {}
        for pattern in _ASSET_GLOBS:
            for path in sorted(root.rglob(pattern)):
                if not path.is_file() or path.name in _ASSET_GLOB_EXCLUDES:
                    continue
                sidecars.setdefault(path.name, path.read_bytes())

        assets = cls(
            source=str(root),
            config_dict=config_dict,
            tokenizer_json=tokenizer_json,
            sidecars=sidecars,
            weight_paths=resolve_gemma_weight_paths(str(root)),
        )
        assets._require_sidecars(*_REQUIRED_ASSET_FILENAMES)
        return assets

    @classmethod
    def from_single_file(cls, path: Path) -> GemmaAssets:
        path = Path(path)
        with safetensors.safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            raw_config = meta.get(GEMMA_CONFIG_METADATA_KEY)
            if raw_config is None:
                raise ValueError(
                    f"Safetensors text-encoder {path} is missing metadata key "
                    f"{GEMMA_CONFIG_METADATA_KEY!r} (JSON-encoded HuggingFace config)."
                )
            config_dict = json.loads(raw_config)

            keys = set(f.keys())
            if TOKENIZER_JSON_TENSOR_KEY not in keys:
                raise ValueError(f"Safetensors text-encoder {path} is missing tensor {TOKENIZER_JSON_TENSOR_KEY!r}.")
            tokenizer_json = _tensor_to_bytes(f.get_tensor(TOKENIZER_JSON_TENSOR_KEY))

            sidecars: dict[str, bytes] = {}
            for key in keys:
                if not key.startswith(HF_ASSET_TENSOR_PREFIX):
                    continue
                name = key.removeprefix(HF_ASSET_TENSOR_PREFIX)
                sidecars[name] = _tensor_to_bytes(f.get_tensor(key))

            for name in _METADATA_FALLBACK_FILENAMES:
                if name in sidecars or name not in meta:
                    continue
                sidecars[name] = meta[name].encode()

        assets = cls(
            source=str(path),
            config_dict=config_dict,
            tokenizer_json=tokenizer_json,
            sidecars=sidecars,
            weight_paths=(str(path.resolve()),),
        )
        assets._require_sidecars(*_REQUIRED_ASSET_FILENAMES)
        return assets

    def sidecar_bytes(self, name: str) -> bytes:
        try:
            return self.sidecars[name]
        except KeyError as exc:
            raise KeyError(f"Gemma assets from {self.source!r} are missing sidecar {name!r}.") from exc

    def sidecar_json(self, name: str) -> dict[str, Any]:
        return json.loads(self.sidecar_bytes(name))

    def _require_sidecars(self, *names: str) -> None:
        missing = [name for name in names if name not in self.sidecars]
        if missing:
            raise ValueError(
                f"Gemma assets from {self.source!r} are missing required sidecar(s): {', '.join(missing)}. "
                f"Embed as {HF_ASSET_TENSOR_PREFIX}<name> (or metadata for small JSON)."
            )


TOKENIZER_MAX_LENGTH = 1024


# ``added_tokens_decoder`` is for from_pretrained reconstruction; as a kwarg it TypeErrors.
_TOKENIZER_CONFIG_SKIP = frozenset(
    {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
)


_SUBPROCESSOR_BASES: dict[str, type] = {
    "image_processor": BaseImageProcessor,
    "feature_extractor": FeatureExtractionMixin,
    "video_processor": BaseVideoProcessor,
}


# gemma3 stores image-processor config in preprocessor_config.json, not nested in processor_config.
_SUBPROCESSOR_SIDECAR_FALLBACKS: dict[str, tuple[str, ...]] = {
    "image_processor": ("preprocessor_config.json",),
    "feature_extractor": ("preprocessor_config.json",),
    "video_processor": ("video_preprocessor_config.json", "preprocessor_config.json"),
}


def _resolve_transformers_class(name: str, base: type) -> type:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Invalid transformers class name {name!r}.")
    cls = getattr(transformers, name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, base):
        raise ValueError(f"transformers.{name} is not a subclass of {base.__name__}.")
    return cls


def _resolve_subprocessor_class(attribute: str, type_name: str) -> type:
    """Resolve a Processor sub-component class from its declared ``*_type`` name.
    For ``image_processor``, prefer the PIL backend (``*Pil``) when available. That matches
    the legacy disk loader's ``AutoImageProcessor.from_pretrained(..., use_fast=False)``
    (now ``backend="pil"``). The torchvision ``Gemma4ImageProcessor`` path produces
    slightly different pixel values and shifts I2V enhance goldens.
    """
    base = _SUBPROCESSOR_BASES[attribute]
    if attribute == "image_processor" and not type_name.endswith("Pil"):
        pil_cls = getattr(transformers, f"{type_name}Pil", None)
        if isinstance(pil_cls, type) and issubclass(pil_cls, base):
            return pil_cls
    return _resolve_transformers_class(type_name, base)


def _chat_template_from_assets(assets: GemmaAssets) -> str | None:
    if (tpl := assets.sidecars.get("chat_template.jinja")) is not None:
        return tpl.decode()
    raw = assets.sidecars.get("chat_template.json")
    if raw is None:
        return None
    loaded = json.loads(raw)
    if isinstance(loaded, str):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("chat_template"), str):
        return loaded["chat_template"]
    return None


def build_gemma_hf_config(assets: GemmaAssets) -> PretrainedConfig:
    model_type = assets.config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Gemma config from {assets.source!r} is missing model_type.")
    try:
        config_cls = CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Gemma model_type={model_type!r} in assets from {assets.source!r}.") from exc
    return config_cls.from_dict(dict(assets.config_dict))


def build_gemma_hf_tokenizer(assets: GemmaAssets, max_length: int = TOKENIZER_MAX_LENGTH) -> PreTrainedTokenizerFast:
    tokenizer_cfg = assets.sidecar_json("tokenizer_config.json")
    kwargs = {k: v for k, v in tokenizer_cfg.items() if k not in _TOKENIZER_CONFIG_SKIP}
    if (tpl := _chat_template_from_assets(assets)) is not None:
        kwargs.setdefault("chat_template", tpl)
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(assets.tokenizer_json),
        model_max_length=max_length,
        **kwargs,
    )


def build_gemma_processor(assets: GemmaAssets, hf_tokenizer: PreTrainedTokenizerBase) -> ProcessorMixin:
    processor_cfg = assets.sidecar_json("processor_config.json")
    processor_class_name = processor_cfg.get("processor_class")
    if not processor_class_name:
        raise ValueError(f"processor_config.json from {assets.source!r} is missing processor_class.")
    processor_class = _resolve_transformers_class(processor_class_name, ProcessorMixin)

    components: dict[str, Any] = {"tokenizer": hf_tokenizer}
    for name in processor_class.get_attributes():
        if name in components:
            continue
        if name not in _SUBPROCESSOR_BASES:
            raise ValueError(
                f"Unsupported processor attribute {name!r} in {processor_class_name} (assets from {assets.source!r})."
            )
        if name in processor_cfg and isinstance(processor_cfg[name], dict):
            sub_cfg = dict(processor_cfg[name])
        else:
            sub_cfg = None
            for fname in _SUBPROCESSOR_SIDECAR_FALLBACKS.get(name, ()):
                if fname in assets.sidecars:
                    sub_cfg = assets.sidecar_json(fname)
                    break
            if sub_cfg is None:
                raise ValueError(
                    f"No nested {name!r} config in processor_config.json and no fallback sidecar "
                    f"for assets from {assets.source!r}."
                )
        type_key = f"{name}_type"
        type_name = sub_cfg.get(type_key)
        if not type_name:
            raise ValueError(f"Missing {type_key!r} for processor attribute {name!r} (assets from {assets.source!r}).")
        concrete = _resolve_subprocessor_class(name, type_name)
        components[name] = concrete.from_dict(sub_cfg)

    extra = {
        k: v for k, v in processor_cfg.items() if k not in processor_class.get_attributes() and k != "processor_class"
    }
    return processor_class(**components, **extra)


def resolve_gemma_weight_paths(gemma_model_path: str) -> tuple[str, ...]:
    path = Path(gemma_model_path)
    if is_safetensors_file(path):
        return (str(path.resolve()),)
    model_folder = _find_matching_file(gemma_model_path, "model*.safetensors").parent
    return tuple(str(p) for p in sorted(model_folder.rglob("*.safetensors")))


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().numpy()
    if arr.dtype == np.uint8:
        return arr.tobytes()
    # Comfy may store tokenizer_json as int8 (or other integer dtypes).
    return arr.astype(np.uint8).tobytes()


# Sampling decoding for the Gemma 3 self-enhance path.
GEMMA3_ENHANCE_GENERATION_KWARGS: dict[str, Any] = {"do_sample": True, "temperature": 0.7, "max_new_tokens": 512}


# Greedy decoding for Gemma 4 instruct enhancers.
GEMMA4_ENHANCE_GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "no_repeat_ngram_size": 5,
    "max_new_tokens": 600,
}


class LTXGemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full Gemma
    multimodal-generation model (including lm_head) is loaded.
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: ProcessorMixin | None = None,
        dtype: torch.dtype = torch.bfloat16,
        execution_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype
        self.execution_device = execution_device

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Run a single fused Gemma forward over a batch of prompts.
        Calls the inner model (self.model.model) to skip lm_head logits computation
        (~500 MiB saving). The tokenizer pads every prompt to ``max_length`` (1024),
        so the inputs stack into a single ``[N, 1024]`` batch with no further padding
        logic; per-prompt outputs are sliced back to ``[1, 1024, D]`` / ``[1, 1024]``
        in the original order.
        """
        if not prompts:
            return []
        tokenized = [self.tokenizer.tokenize_with_weights(t)["gemma"] for t in prompts]
        input_ids = torch.tensor(
            [[tok for tok, _ in pairs] for pairs in tokenized],
            device=self.execution_device or self.model.device,
        )
        attention_mask = torch.tensor(
            [[w for _, w in pairs] for pairs in tokenized],
            device=self.execution_device or self.model.device,
        )
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return [(tuple(h[i : i + 1] for h in hidden_states), attention_mask[i : i + 1]) for i in range(len(prompts))]

    def _model_type(self) -> str:
        return getattr(getattr(self.model, "config", None), "model_type", "") or ""

    def _default_generation_kwargs(self) -> dict[str, Any]:
        model_type = self._model_type()
        if model_type == "gemma3":
            return dict(GEMMA3_ENHANCE_GENERATION_KWARGS)
        if model_type == "gemma4":
            return dict(GEMMA4_ENHANCE_GENERATION_KWARGS)
        raise ValueError(f"Unsupported Gemma enhancer model_type={model_type!r}; expected 'gemma3' or 'gemma4'.")

    def _default_system_prompt(self, *, t2v: bool) -> str:
        model_type = self._model_type()
        if model_type == "gemma3":
            return default_gemma3_t2v_system_prompt() if t2v else default_gemma3_i2v_system_prompt()
        if model_type == "gemma4":
            return default_gemma4_t2v_system_prompt() if t2v else default_gemma4_i2v_system_prompt()
        raise ValueError(f"Unsupported Gemma enhancer model_type={model_type!r}; expected 'gemma3' or 'gemma4'.")

    def _enhance(
        self,
        messages: list[dict],
        image: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.execution_device or self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        gen_kwargs = self._default_generation_kwargs()
        if static_cache:
            gen_kwargs["cache_implementation"] = "static"
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        # fork_rng device pinning is only supported for CUDA; MPS/CPU fork CPU RNG only.
        device = self.execution_device or self.model.device
        fork_devices = [device] if device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self._default_system_prompt(t2v=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed, static_cache=static_cache)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self._default_system_prompt(t2v=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed, static_cache=static_cache)


@functools.lru_cache(maxsize=4)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}") as f:
        return f.read()


def default_gemma3_i2v_system_prompt() -> str:
    return _load_system_prompt("gemma3_i2v_system_prompt.txt")


def default_gemma3_t2v_system_prompt() -> str:
    return _load_system_prompt("gemma3_t2v_system_prompt.txt")


def default_gemma4_i2v_system_prompt() -> str:
    return _load_system_prompt("gemma4_i2v_system_prompt.txt")


def default_gemma4_t2v_system_prompt() -> str:
    return _load_system_prompt("gemma4_t2v_system_prompt.txt")


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Left-pad a tensor (prepend the padding) with the given value.
    Decoder-only LLMs must be LEFT-padded for generation: with right-padding the trailing
    pad positions become the sequence end, so generate() continues from a pad position
    instead of the real last token. Left-padding keeps the real last token at the end
    while still aligning the sequence length to a multiple of 8.
    See https://huggingface.co/docs/transformers/en/llm_tutorial#padding-side
    """
    pad = torch.full(
        (1, padding_length),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([pad, tensor], dim=1)


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def build_gemma_tokenizer(assets: GemmaAssets) -> LTXVGemmaTokenizer:
    return LTXVGemmaTokenizer(
        build_gemma_hf_tokenizer(assets),
        TOKENIZER_MAX_LENGTH,
        padding_side=PaddingSide.LEFT,
    )


def _populate_rotary_v5(l_model: torch.nn.Module, config: object) -> None:
    """transformers >=5 layout: single rotary_emb with per-layer-type buffers.
    Gemma 4's global ("full_attention") layers use a wider head_dim than the local layers, so when
    such a layer uses "proportional" RoPE the init must read ``global_head_dim`` instead of the
    default ``head_dim``. Gemma 3 never uses "proportional" RoPE, so the branch is inert there.
    """
    rope_emb = l_model.rotary_emb
    for layer_type in dict.fromkeys(config.layer_types):
        rope_params = config.rope_parameters[layer_type]
        if rope_params is None:
            continue
        rope_type = rope_params["rope_type"]
        if rope_type == "default":
            inv_freq, attn_scaling = rope_emb.compute_default_rope_parameters(config, layer_type=layer_type)
        else:
            init_kwargs = {"layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                init_kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, attn_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, **init_kwargs)
        rope_emb.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        rope_emb.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(rope_emb, f"{layer_type}_attention_scaling", attn_scaling)


def _populate_language_model_buffers(module: LTXGemmaTextEncoder) -> LTXGemmaTextEncoder:
    """Populate the non-persistent rotary and embed-scale buffers shared by all Gemma families."""
    model = module.model
    l_model = model.model.language_model
    config = model.config.text_config

    _populate_rotary_v5(l_model, config)

    embed_scale = torch.tensor(config.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale, persistent=False)

    # Dense gemma4 also has per-layer input embeddings; unified/gemma3 do not.
    if hasattr(l_model, "embed_tokens_per_layer"):
        per_layer_scale = torch.tensor(config.hidden_size_per_layer_input**0.5, device="cpu")
        l_model.embed_tokens_per_layer.register_buffer("embed_scale", per_layer_scale, persistent=False)

    return module


def populate_gemma3_buffers(module: LTXGemmaTextEncoder) -> LTXGemmaTextEncoder:
    module = _populate_language_model_buffers(module)

    v_tower = module.model.model.vision_tower
    positions_length = len(v_tower.embeddings.position_ids[0])
    position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
    v_tower.embeddings.register_buffer("position_ids", position_ids, persistent=False)

    return module


def populate_gemma4_unified_buffers(module: LTXGemmaTextEncoder) -> LTXGemmaTextEncoder:
    """Populate buffers for the Gemma 4 unified text encoder.
    The unified model is encoder-free: vision/audio inputs go through lightweight embedders
    (model.model.embed_vision / .embed_audio) that carry no RoPE, so unlike Gemma 3 there is no
    vision_tower to seed.
    """
    return _populate_language_model_buffers(module)


def populate_gemma4_dense_buffers(module: LTXGemmaTextEncoder) -> LTXGemmaTextEncoder:
    """Populate buffers for dense Gemma 4 instruct (enhance-capable).
    Same language-model buffers as unified, plus vision-tower RoPE (needed for I2V enhance).
    """
    module = _populate_language_model_buffers(module)

    vision_tower = module.model.model.vision_tower
    rotary = vision_tower.encoder.rotary_emb
    rope_fn = (
        ROPE_INIT_FUNCTIONS[rotary.rope_type]
        if rotary.rope_type != "default"
        else rotary.compute_default_rope_parameters
    )
    inv_freq, _ = rope_fn(rotary.config)
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    rotary.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    return module


def convert_gemma4_state_dict(state_dict: Mapping[str, torch.Tensor], *, model_type: str) -> Mapping[str, torch.Tensor]:
    """Map the released unified / instruct Gemma-4 layout, including the tied LM head."""
    from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

    flat = "model.layers.0.post_feedforward_layernorm.weight" in state_dict
    prefixes = {"model.language_model.": "model.model.language_model."}
    if model_type == "gemma4_unified":
        if flat:
            prefixes.update(
                {
                    "model.layers.": "model.model.language_model.layers.",
                    "model.embed_tokens.": "model.model.language_model.embed_tokens.",
                    "model.norm.": "model.model.language_model.norm.",
                    "vision_model.": "model.model.embed_vision.",
                    "multi_modal_projector.embedding_projection.": "model.model.embed_vision.multimodal_embedder.embedding_projection.",
                    "audio_projector.": "model.model.embed_audio.",
                }
            )
        prefixes.update(
            {
                "model.embed_audio.": "model.model.embed_audio.",
                "model.embed_vision.embedding_projection.": "model.model.embed_vision.multimodal_embedder.embedding_projection.",
                "model.vision_embedder.": "model.model.embed_vision.",
            }
        )
    elif model_type == "gemma4":
        prefixes.update(
            {
                f"model.{name}.": f"model.model.{name}."
                for name in (
                    "vision_tower",
                    "audio_tower",
                    "embed_vision",
                    "embed_audio",
                    "multi_modal_projector",
                )
            }
        )
    else:
        raise ValueError(f"Unsupported Gemma-4 model_type: {model_type!r}")
    keys = {}
    for key in state_dict:
        for source, destination in prefixes.items():
            if key.startswith(source):
                target = destination + key.removeprefix(source)
                keys[target] = key
                if target == "model.model.language_model.embed_tokens.weight":
                    keys["model.lm_head.weight"] = key
                break
    return TransformedStateDict(state_dict, keys)
