"""Cosmos Predict 2.5 Reason1 (Qwen2.5-VL) :class:`~...contracts.ConditionEncoder`.

Tokenizes prompts with an optional English system prefix and encodes
them with the Qwen2.5-VL text tower (layer-concat strategy).  Returns
``Conditioning.positive[\"context\"]`` for the Predict 2.5 DiT.  Does
not rewrite user prompt strings.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn

from worldfoundry.core.model_loading.text_embeddings import (
    DEFAULT_EMBEDDING_CACHE_MAX_BYTES,
    EmbeddingConcatStrategy,
    validate_prompt_encoder_options,
)

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import (
    CheckpointSpec,
    MaterializedCheckpoint,
    ModuleLoadSpec,
    NativeCheckpointResolver,
    NativeModuleLoader,
)
from ....optimizations import OffloadPolicy, RuntimePolicy

_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."
_DEFAULT_EMBEDDING_CACHE_MAX_BYTES = DEFAULT_EMBEDDING_CACHE_MAX_BYTES


def _qwen_class(name: str):
    """Resolve optional Qwen implementations only when an encoder is built."""

    module = import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    if name == "Qwen2_5_VLRMSNorm" and not hasattr(module, name):
        # Transformers 4.57 exports the shared Qwen2 implementation.
        name = "Qwen2RMSNorm"
    return getattr(module, name)


def __getattr__(name: str):
    """Keep the previous Qwen exports available without eager imports."""

    if name in {"Qwen2_5_VLRMSNorm", "Qwen2_5_VLRotaryEmbedding", "Qwen2_5_VLTextModel"}:
        return _qwen_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Cosmos25TextBackbone(nn.Module):
    """Language-only slice of the Reason1 Qwen2.5-VL checkpoint."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = _qwen_class("Qwen2_5_VLTextModel")(config)

    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)


def _reason1_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint.root, local_files_only=True, trust_remote_code=False)
    return {"config": config.text_config}


def convert_reason1_text_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Discard the unused vision tower and LM head from Reason1."""

    return {key: value for key, value in state_dict.items() if key.startswith("model.")}


class Cosmos25PromptConditioner:
    """Reason1 conditioning with optional CPU execution and bounded batch reuse.

    Caching is off by default. Enabled caches use insertion-order eviction and
    independent host copies, invalidate on normal weight/device/dtype/tokenizer
    changes, and bypass training mode or unversioned inference weights. Call
    ``clear_embedding_cache`` after changes through unsupported raw ``.data``
    writes or custom tokenizer internals.
    """

    def __init__(
        self,
        model: Cosmos25TextBackbone,
        tokenizer,
        *,
        sequence_length: int = 512,
        embedding_concat_strategy: str = "full_concat",
        n_layers_per_group: int = 5,
        run_on_cpu: bool = False,
        embedding_cache_size: int = 0,
        embedding_cache_max_bytes: int = _DEFAULT_EMBEDDING_CACHE_MAX_BYTES,
        output_device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.embedding_concat_strategy = str(embedding_concat_strategy)
        self.n_layers_per_group = int(n_layers_per_group)
        valid = {str(item) for item in EmbeddingConcatStrategy}
        if self.embedding_concat_strategy not in valid:
            raise ValueError(f"invalid embedding_concat_strategy: {self.embedding_concat_strategy}")
        if self.n_layers_per_group <= 0:
            raise ValueError("n_layers_per_group must be positive")
        validate_prompt_encoder_options(
            run_on_cpu=run_on_cpu,
            embedding_cache_size=embedding_cache_size,
            embedding_cache_max_bytes=embedding_cache_max_bytes,
        )
        self.run_on_cpu = run_on_cpu
        self.embedding_cache_size = embedding_cache_size
        self.embedding_cache_max_bytes = embedding_cache_max_bytes
        self.output_device = None if output_device is None else torch.device(output_device)
        # Entries are complete batches: splitting misses into individual prompts
        # changes GEMM shapes and can change BF16 results. Host storage is bounded
        # by bytes as well as entries (full_concat is about 100 MiB per prompt).
        self._embedding_cache: OrderedDict[tuple[str, ...], torch.Tensor] = OrderedDict()
        self._embedding_cache_bytes = 0
        self._embedding_cache_signature: tuple | None = None
        self._embedding_cache_lock = threading.RLock()

    def clear_embedding_cache(self) -> None:
        """Release cached host tensors after explicit model/tokenizer updates."""

        with self._embedding_cache_lock:
            self._embedding_cache.clear()
            self._embedding_cache_bytes = 0
            self._embedding_cache_signature = None

    def _cache_signature(self, device: torch.device, dtype: torch.dtype) -> tuple | None:
        tensors = (*self.model.parameters(), *self.model.buffers())
        try:
            weights = tuple((id(value), value._version, value.device, value.dtype) for value in tensors)
        except RuntimeError:
            # Inference tensors lack version counters; do not reuse embeddings
            # when ordinary in-place weight updates cannot be detected.
            return None
        compute_type = "cpu" if self.run_on_cpu else device.type
        model_config = getattr(getattr(self.model, "model", self.model), "config", None)
        return (
            weights, id(self.model), id(self.tokenizer),
            repr(getattr(self.tokenizer, "chat_template", None)),
            repr(getattr(self.tokenizer, "special_tokens_map", None)),
            getattr(self.tokenizer, "vocab_size", None),
            len(self.tokenizer), self.sequence_length, self.embedding_concat_strategy,
            self.n_layers_per_group, self.run_on_cpu, _SYSTEM_PROMPT, device, dtype,
            torch.is_autocast_enabled(compute_type), torch.get_autocast_dtype(compute_type),
            self.embedding_cache_size, self.embedding_cache_max_bytes,
            getattr(model_config, "_attn_implementation", None),
            torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)(),
        )

    @torch.inference_mode()
    def _encode(self, prompts: Sequence[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if not self.embedding_cache_size or not self.embedding_cache_max_bytes:
            if self._embedding_cache:
                self.clear_embedding_cache()
            return self._encode_uncached(prompts, device=device, dtype=dtype)
        with self._embedding_cache_lock:
            training = any(module.training for module in self.model.modules())
            signature = None if training else self._cache_signature(device, dtype)
            if signature != self._embedding_cache_signature or signature is None:
                self.clear_embedding_cache()
                self._embedding_cache_signature = signature
            if signature is None:
                return self._encode_uncached(prompts, device=device, dtype=dtype)
            key = tuple(prompts)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                # Consumers may modify their conditioning tensor in place.
                return cached.to(device=device, dtype=dtype).clone()
            output = self._encode_uncached(prompts, device=device, dtype=dtype)
            size_bytes = output.numel() * output.element_size()
            if size_bytes <= self.embedding_cache_max_bytes:
                # Blocking D2H also makes a hit from another host thread safe.
                cached = output.detach().to(device="cpu", copy=True)
                while self._embedding_cache and (
                    len(self._embedding_cache) >= self.embedding_cache_size
                    or self._embedding_cache_bytes + size_bytes > self.embedding_cache_max_bytes
                ):
                    _, evicted = self._embedding_cache.popitem(last=False)
                    self._embedding_cache_bytes -= evicted.numel() * evicted.element_size()
                self._embedding_cache[key] = cached
                self._embedding_cache_bytes += size_bytes
            return output

    def _encode_uncached(self, prompts: Sequence[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            ids = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=self.sequence_length,
                truncation=True,
                padding="max_length",
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            rows.append(torch.tensor(ids, dtype=torch.long))
        input_ids = torch.stack(rows).to(device="cpu" if self.run_on_cpu else device)
        hidden_states = self.model(input_ids).hidden_states
        normalized = [
            (value - value.mean(dim=-1, keepdim=True)) / (value.std(dim=-1, keepdim=True) + 1e-8)
            for value in hidden_states[1:]
        ]
        if self.embedding_concat_strategy == str(EmbeddingConcatStrategy.FULL_CONCAT):
            output = torch.cat(normalized, dim=-1)
        elif self.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING):
            output = torch.stack(normalized).mean(dim=0)
        else:
            groups = [
                torch.stack(normalized[index : index + self.n_layers_per_group]).mean(dim=0)
                for index in range(0, len(normalized), self.n_layers_per_group)
            ]
            output = torch.cat(groups, dim=-1)
        return output.to(device=device, dtype=dtype)

    def compute_text_embeddings_online(
        self,
        data_batch: Mapping[str, object],
        input_caption_key: str,
    ) -> torch.Tensor:
        """Compatibility surface for model-internal multi-view conditioners."""

        raw_prompts = data_batch[input_caption_key]
        if isinstance(raw_prompts, str) or not isinstance(raw_prompts, Sequence):
            raw_prompts = [raw_prompts]
        prompts: list[str] = []
        for value in raw_prompts:
            if isinstance(value, str):
                prompts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                prompts.append(" ".join(str(item) for item in value))
            else:
                prompts.append(str(value))
        parameter = next(self.model.parameters())
        return self._encode(prompts, device=self.output_device or parameter.device, dtype=parameter.dtype)

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        positive = {"context": self._encode(request.prompts, device=device, dtype=dtype)}
        negative: dict[str, torch.Tensor] = {}
        if request.sampling.guidance_scale != 1.0:
            prompts = request.negative_prompts or (("",) * request.batch_size)
            negative["context"] = self._encode(prompts, device=device, dtype=dtype)
        return Conditioning(
            positive=positive,
            negative=negative,
            shared={"fps": float(request.inputs.get("fps", request.inputs.get("frame_rate", 16.0)))},
        )


def _load_reason1_backbone(
    checkpoint_spec: CheckpointSpec,
    policy: RuntimePolicy,
) -> Cosmos25TextBackbone:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos25TextBackbone,
            config_resolver=_reason1_config,
            state_dict_converter=convert_reason1_text_state_dict,
            vram_module_map={
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.RMSNorm: AutoWrappedModule,
                _qwen_class("Qwen2_5_VLRMSNorm"): AutoWrappedModule,
                _qwen_class("Qwen2_5_VLRotaryEmbedding"): AutoWrappedModule,
            },
            layer_container="model.layers",
        ),
        checkpoint_spec,
        policy,
    )
    if not isinstance(model, Cosmos25TextBackbone):
        raise TypeError(f"expected Cosmos25TextBackbone, got {type(model).__name__}")
    return model


def _load_reason1_tokenizer(checkpoint_spec: CheckpointSpec):
    from transformers import AutoTokenizer

    checkpoint = NativeCheckpointResolver().materialize(checkpoint_spec)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    return tokenizer


def build_cosmos25_prompt_conditioner(context: ComponentBuildContext) -> Cosmos25PromptConditioner:
    encoder_options = validate_prompt_encoder_options(
        run_on_cpu=context.component_options.get("run_on_cpu", False),
        embedding_cache_size=context.component_options.get("embedding_cache_size", 0),
        embedding_cache_max_bytes=context.component_options.get(
            "embedding_cache_max_bytes", _DEFAULT_EMBEDDING_CACHE_MAX_BYTES
        ),
    )
    run_on_cpu = encoder_options["run_on_cpu"]
    encoder_policy = _reason1_encoder_policy(context.policy, run_on_cpu=run_on_cpu)
    model = _load_reason1_backbone(context.require_checkpoint("weights"), encoder_policy)
    tokenizer = _load_reason1_tokenizer(context.require_checkpoint("tokenizer"))
    return Cosmos25PromptConditioner(
        model,
        tokenizer,
        sequence_length=int(context.component_options.get("sequence_length", 512)),
        embedding_concat_strategy=str(
            context.component_options.get("embedding_concat_strategy", "full_concat")
        ),
        n_layers_per_group=int(context.component_options.get("n_layers_per_group", 5)),
        **encoder_options,
        output_device=context.policy.device if run_on_cpu else None,
    )


def _reason1_encoder_policy(policy: RuntimePolicy, *, run_on_cpu: bool) -> RuntimePolicy:
    if not isinstance(run_on_cpu, bool):
        raise TypeError("run_on_cpu must be a bool")
    if not run_on_cpu:
        return policy
    if policy.quantization.mode.value != "none":
        raise ValueError("CPU Reason1 encoding requires unquantized encoder weights")
    # CPU execution owns its placement; the DiT's GPU offload policy is separate.
    return replace(policy, device="cpu", offload=OffloadPolicy())


def load_cosmos_reason1_prompt_encoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sequence_length: int = 512,
    embedding_concat_strategy: str = "full_concat",
    n_layers_per_group: int = 5,
    run_on_cpu: bool = False,
    embedding_cache_size: int = 0,
    embedding_cache_max_bytes: int = _DEFAULT_EMBEDDING_CACHE_MAX_BYTES,
) -> Cosmos25PromptConditioner:
    """Load one shared language-only Cosmos Reason1 prompt encoder."""

    validate_prompt_encoder_options(
        run_on_cpu=run_on_cpu,
        embedding_cache_size=embedding_cache_size,
        embedding_cache_max_bytes=embedding_cache_max_bytes,
    )
    source = str(checkpoint_path)
    if source.startswith("hf://"):
        from worldfoundry.core.io.assets.easy_io import resolve_checkpoint_path

        source = resolve_checkpoint_path(source)
    checkpoint = CheckpointSpec(source=source)
    policy = _reason1_encoder_policy(RuntimePolicy(device=device, dtype=dtype), run_on_cpu=run_on_cpu)
    return Cosmos25PromptConditioner(
        _load_reason1_backbone(checkpoint, policy),
        _load_reason1_tokenizer(checkpoint),
        sequence_length=sequence_length,
        embedding_concat_strategy=embedding_concat_strategy,
        n_layers_per_group=n_layers_per_group,
        run_on_cpu=run_on_cpu,
        embedding_cache_size=embedding_cache_size,
        embedding_cache_max_bytes=embedding_cache_max_bytes,
        output_device=device if run_on_cpu else None,
    )


@dataclass(slots=True)
class CosmosReason1TextEncoderConfig:
    compute_online: bool = False
    embedding_concat_strategy: str = "full_concat"
    n_layers_per_group: int = 5
    ckpt_path: str = "hf://nvidia/Cosmos-Reason1-7B"
    sequence_length: int = 512
    run_on_cpu: bool = False
    embedding_cache_size: int = 0
    embedding_cache_max_bytes: int = _DEFAULT_EMBEDDING_CACHE_MAX_BYTES


class CosmosReason1TextEncoder:
    """Thin legacy-facing view of the shared native prompt encoder."""

    def __init__(
        self,
        config: CosmosReason1TextEncoderConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = config
        self.encoder = load_cosmos_reason1_prompt_encoder(
            config.ckpt_path,
            device=device,
            embedding_concat_strategy=str(config.embedding_concat_strategy),
            n_layers_per_group=int(config.n_layers_per_group),
            sequence_length=int(getattr(config, "sequence_length", 512)),
            run_on_cpu=getattr(config, "run_on_cpu", False),
            embedding_cache_size=getattr(config, "embedding_cache_size", 0),
            embedding_cache_max_bytes=getattr(
                config, "embedding_cache_max_bytes", _DEFAULT_EMBEDDING_CACHE_MAX_BYTES
            ),
        )
        self.model = self.encoder.model

    def compute_text_embeddings_online(
        self,
        data_batch: Mapping[str, object],
        input_caption_key: str,
    ) -> torch.Tensor:
        return self.encoder.compute_text_embeddings_online(data_batch, input_caption_key)


__all__ = [
    "CosmosReason1TextEncoder",
    "CosmosReason1TextEncoderConfig",
    "Cosmos25PromptConditioner",
    "Cosmos25TextBackbone",
    "build_cosmos25_prompt_conditioner",
    "convert_reason1_text_state_dict",
    "load_cosmos_reason1_prompt_encoder",
]
