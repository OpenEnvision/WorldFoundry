"""Wan denoiser adapter for the native diffusion execution contract.

Wraps a checkpoint-compatible Wan DiT as a :class:`~...contracts.Denoiser`.
The runner calls ``__call__(DenoiserInput) -> DenoiserOutput``; this module
maps ``conditioning`` tensors onto ``WanModel`` kwargs and applies optional
CUDA Graph / TeaCache mixins from :mod:`.graph_wrapped`.  Weights are loaded
through :class:`~...loaders.module.NativeModuleLoader`, never inside the
sampling loop.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock

import torch

from worldfoundry.core.model_loading import hash_state_dict_keys

from ...components import BuildPurpose, ComponentBuildContext, ComponentKey, ComponentKind
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ...loaders.wan_vsa import (
    canonical_wan_vsa_gate_key,
    convert_wan_vsa_gate_state_dict,
    resolve_wan_vsa_gate_config,
    validate_wan_vsa_gate_shapes,
)
from ...optimizations import parse_torch_dtype
from ...optimizations.wan.feature_cache import (
    WanFeatureCacheConfig,
    resolve_wan_feature_cache,
)
from ..initializers.wan.component import WAN_DENOISE_MASK_IS_ALL_ONES
from ..networks.wan.model import RMSNorm, WanModel
from .graph_wrapped import (
    FeatureCacheDenoiserMixin,
    GraphWrappedDenoiserMixin,
    acceleration_runtime_call,
    resolve_cuda_graph_option,
    validate_cuda_graph_options,
)

WAN21_T2V_1P3B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "eps": 1e-6,
}

WAN21_T2V_14B_CONFIG = {
    **WAN21_T2V_1P3B_CONFIG,
    "dim": 5120,
    "ffn_dim": 13824,
    "num_heads": 40,
    "num_layers": 40,
}

WAN22_T2V_A14B_CONFIG = dict(WAN21_T2V_14B_CONFIG)

WAN21_I2V_14B_CONFIG = {
    **WAN21_T2V_14B_CONFIG,
    "has_image_input": True,
    "in_dim": 36,
}

# Some Wan2.1 research checkpoints (for example WoW 1.3B) train the
# first-frame VAE condition without adding the CLIP cross-attention branch.
# This is still the canonical Wan graph: only its explicit conditioning roles
# differ from the released 14B I2V checkpoint.
WAN21_VAE_I2V_1P3B_CONFIG = {
    **WAN21_T2V_1P3B_CONFIG,
    "in_dim": 36,
    "require_vae_embedding": True,
    "require_clip_embedding": False,
}

# Wan2.2 A14B I2V conditions the latent stream with the VAE mask/latent
# tensor but does not own the Wan2.1 CLIP image branch.  Keeping that split in
# one canonical graph is required by Fun-Control and FantasyWorld releases.
WAN22_I2V_A14B_CONFIG = {
    **WAN21_T2V_14B_CONFIG,
    "has_image_input": False,
    "in_dim": 36,
    "require_vae_embedding": True,
    "require_clip_embedding": False,
}

WAN22_TI2V_5B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "per_token_timestep": True,
}

_WAN_DENOISER_OPTION_KEYS = frozenset(
    {
        "accelerations",
        "adacache",
        "blocktaylorseer",
        "cuda_graph",
        "custom",
        "dualblock",
        "dynamicblock",
        "feature_cache",
        "firstblock",
        "fused_rope",
        "inplace_residual",
        "magcache",
        "peft_adapter_path",
        "rope_precision",
        "rms_norm_precision",
        "taylorseer",
        "teacache",
        "teacache_thresh",
        "weight_dtype",
    }
)


def _read_wan_peft_config(adapter_path: Path) -> dict[str, object]:
    config_path = adapter_path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Wan PEFT adapter config: {config_path}") from error
    if not isinstance(config, dict):
        raise TypeError("Wan PEFT adapter_config.json must contain one JSON object")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"unsupported Wan PEFT adapter type: {config.get('peft_type')!r}")
    return config


def _validated_wan_peft_adapter(
    context: ComponentBuildContext,
    value: object,
) -> Path:
    if context.purpose is BuildPurpose.TRAINING:
        raise ValueError("Wan inference PEFT adapters cannot be supplied to a training component build")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError("Wan peft_adapter_path must be a non-empty local filesystem path")

    adapter_path = Path(value).expanduser()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Wan PEFT adapter directory does not exist: {adapter_path}")
    config = _read_wan_peft_config(adapter_path)
    base_model_id = config.get("base_model_name_or_path")
    if base_model_id not in (None, "") and str(base_model_id) != context.model_id:
        raise ValueError(
            "Wan PEFT adapter base model differs from the selected model: "
            f"adapter={base_model_id!r}, selected={context.model_id!r}"
        )
    return adapter_path


def _merge_wan_peft_adapter(model: torch.nn.Module, adapter_path: Path) -> None:
    from worldfoundry.core.model_loading.peft import audit_wan_lora_targets, merge_peft_adapter

    config = _read_wan_peft_config(adapter_path)
    target_modules = config.get("target_modules")
    if not target_modules or not isinstance(target_modules, (str, list)):
        raise ValueError("Wan PEFT adapter must declare target_modules")
    target_names = audit_wan_lora_targets(model)
    auto_mapping = config.get("auto_mapping")
    if isinstance(auto_mapping, Mapping):
        base_class = auto_mapping.get("base_model_class")
        if base_class not in (None, "", type(model).__name__):
            raise ValueError(
                "Wan PEFT adapter base architecture differs from the loaded model: "
                f"adapter={base_class!r}, loaded={type(model).__name__!r}"
            )
    try:
        from peft import PeftModel
    except ModuleNotFoundError as error:
        raise RuntimeError("loading a Wan LoRA adapter requires the PEFT dependency") from error
    loaded = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    targeted = set(getattr(loaded, "targeted_module_names", ()))
    if targeted != set(target_names):
        raise ValueError("Wan PEFT adapter targets are incompatible with the loaded Wan graph")
    merged = merge_peft_adapter(loaded)
    if merged is not model:
        raise RuntimeError("PEFT merge did not return the loaded Wan base model")

SKYREELS_V2_DF_1P3B_CONFIG = {
    **WAN21_T2V_1P3B_CONFIG,
    "inject_sample_info": True,
}

SKYREELS_V3_R2V_14B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
}


WAN_CIVITAI_CONFIGS_BY_HASH = {
    "9269f8db9040a9d860eaca435be61814": WAN21_T2V_1P3B_CONFIG,
    "aafcfd9672c3a2456dc46e1cb6e52c70": WAN21_T2V_14B_CONFIG,
    "6bfcfb3b342cb286ce886889d519a77e": WAN21_I2V_14B_CONFIG,
    "6d6ccde6845b95ad9114ab993d917893": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 36,
    },
    "349723183fc063b2bfc10bb2835cf677": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
    },
    "efa44cddf936c70abd0ea28b6cbe946c": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
    },
    "3ef3b1f8e1dab83d5b71fd7b617f859f": {
        **WAN21_I2V_14B_CONFIG,
        "has_image_pos_emb": True,
    },
    "70ddad9d3a133785da5ea371aae09504": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
        "has_ref_conv": True,
    },
    "26bde73488a92e64cc20b0a7485b9e5b": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
        "has_ref_conv": True,
    },
    "ac6a5aa74f4a0aab6f64eb9a72f19901": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
    },
    "b61c605c2adbd23124d152ed28e049ae": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
    },
}


def infer_native_wan_transformer_config(
    state_dict: Mapping[str, object],
) -> dict[str, object]:
    """Infer the canonical Wan graph from checkpoint tensor structure.

    Released research checkpoints often keep native Wan parameter names but
    have a new key hash after changing input conditioning.  The architecture
    is nevertheless fully described by the patch/head/block tensor shapes and
    optional module keys, so those checkpoints should not need a model-specific
    backend or an ever-growing hash table.
    """

    def tensor(name: str) -> torch.Tensor:
        value = state_dict.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Wan checkpoint is missing tensor {name!r}")
        return value

    patch = tensor("patch_embedding.weight")
    head = tensor("head.head.weight")
    text = tensor("text_embedding.0.weight")
    time = tensor("time_embedding.0.weight")
    ffn = tensor("blocks.0.ffn.0.weight")
    if patch.ndim != 5:
        raise ValueError(f"Wan patch embedding must be Conv3d-shaped, got {tuple(patch.shape)}")

    block_ids = {
        int(name.split(".", 2)[1])
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor)
        and name.startswith("blocks.")
        and len(name.split(".", 2)) == 3
        and name.split(".", 2)[1].isdigit()
    }
    if not block_ids or block_ids != set(range(max(block_ids) + 1)):
        raise ValueError("Wan checkpoint blocks must form one contiguous zero-based sequence")

    patch_size = tuple(int(size) for size in patch.shape[2:])
    patch_volume = math.prod(patch_size)
    if head.shape[0] % patch_volume:
        raise ValueError(
            f"Wan output projection {tuple(head.shape)} is incompatible with patch size {patch_size}"
        )
    dim = int(patch.shape[0])
    if dim % 128:
        raise ValueError(f"Wan hidden dimension {dim} is not divisible by the canonical head width 128")

    has_image_input = any(
        name.startswith("img_emb.") or ".cross_attn.k_img." in name
        for name in state_dict
    )
    has_control_adapter = any(name.startswith("control_adapter.") for name in state_dict)
    config: dict[str, object] = {
        "has_image_input": has_image_input,
        "patch_size": patch_size,
        "in_dim": int(patch.shape[1]),
        "dim": dim,
        "ffn_dim": int(ffn.shape[0]),
        "freq_dim": int(time.shape[1]),
        "text_dim": int(text.shape[1]),
        "out_dim": int(head.shape[0] // patch_volume),
        "num_heads": dim // 128,
        "num_layers": max(block_ids) + 1,
        "eps": 1e-6,
        "has_image_pos_emb": "img_emb.emb_pos" in state_dict,
        "has_ref_conv": any(name.startswith("ref_conv.") for name in state_dict),
        "add_control_adapter": has_control_adapter,
        "require_vae_embedding": int(patch.shape[1]) != int(head.shape[0] // patch_volume),
        "require_clip_embedding": has_image_input,
        "inject_sample_info": "fps_embedding.weight" in state_dict,
    }
    if has_control_adapter:
        adapter_weight = tensor("control_adapter.conv.weight")
        config["in_dim_control_adapter"] = int(adapter_weight.shape[1] // 64)
    if validate_wan_vsa_gate_shapes(
        state_dict,
        dim=dim,
        num_layers=max(block_ids) + 1,
    ):
        config["vsa_gate_compress"] = True
    return config


def convert_diffusers_wan_transformer_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Map Diffusers or already-native Wan names onto the native graph.

    Official Wan releases use ``diffusion_pytorch_model`` filenames for both
    layouts.  Stable-Video-Infinity stages the native Wan2.1 shards under that
    name, so filename-based loading may legitimately invoke this converter on
    native keys.
    """

    replacements = (
        (".attn1.to_out.0.", ".self_attn.o."),
        (".attn1.to_q.", ".self_attn.q."),
        (".attn1.to_k.", ".self_attn.k."),
        (".attn1.to_v.", ".self_attn.v."),
        (".attn1.norm_q.", ".self_attn.norm_q."),
        (".attn1.norm_k.", ".self_attn.norm_k."),
        (".attn2.to_out.0.", ".cross_attn.o."),
        (".attn2.to_q.", ".cross_attn.q."),
        (".attn2.to_k.", ".cross_attn.k."),
        (".attn2.to_v.", ".cross_attn.v."),
        (".attn2.norm_q.", ".cross_attn.norm_q."),
        (".attn2.norm_k.", ".cross_attn.norm_k."),
        (".ffn.net.0.proj.", ".ffn.0."),
        (".ffn.net.2.", ".ffn.2."),
        (".ffn.fc_in.", ".ffn.0."),
        (".ffn.fc_out.", ".ffn.2."),
        (".norm2.", ".norm3."),
        (".scale_shift_table", ".modulation"),
    )
    roots = {
        "condition_embedder.text_embedder.linear_1.": "text_embedding.0.",
        "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
        "condition_embedder.time_embedder.linear_1.": "time_embedding.0.",
        "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
        "condition_embedder.time_proj.": "time_projection.1.",
        "proj_out.": "head.head.",
        "scale_shift_table": "head.modulation",
    }
    native_roots = (
        "patch_embedding.",
        "text_embedding.",
        "time_embedding.",
        "time_projection.",
        "img_emb.",
        "head.",
    )
    converted: dict[str, object] = {}
    for source, value in state_dict.items():
        gate_target = canonical_wan_vsa_gate_key(source)
        target = gate_target or (
            source if source.startswith(native_roots) else roots.get(source)
        )
        if target is None:
            for prefix, replacement in roots.items():
                if source.startswith(prefix):
                    target = replacement + source[len(prefix) :]
                    break
        if target is None and source.startswith("blocks."):
            target = source
            for old, new in replacements:
                target = target.replace(old, new)
        if target is None:
            raise KeyError(f"unsupported Diffusers Wan transformer parameter: {source}")
        if target in converted:
            raise KeyError(f"Wan transformer conversion produced duplicate parameter: {target}")
        converted[target] = value
    return converted


class WanModelStateDictConverter:
    """Detect released Wan checkpoint layouts for native model construction."""

    def from_diffusers(self, state_dict):
        converted = convert_diffusers_wan_transformer_state_dict(state_dict)
        config = (
            WAN21_T2V_14B_CONFIG
            if hash_state_dict_keys(state_dict)
            == "cb104773c6c2cb6df4f9529ad5c60d0b"
            else infer_native_wan_transformer_config(converted)
        )
        config = dict(config)
        if validate_wan_vsa_gate_shapes(
            converted,
            dim=int(config["dim"]),
            num_layers=int(config["num_layers"]),
        ):
            config["vsa_gate_compress"] = True
        return converted, config

    def from_civitai(self, state_dict):
        raw_hash = hash_state_dict_keys(state_dict)
        filtered = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("vace")
            and not (name.startswith("control") and not name.startswith("control_adapter"))
        }
        checkpoint_hash = raw_hash if raw_hash in {
            "ac6a5aa74f4a0aab6f64eb9a72f19901",
            "b61c605c2adbd23124d152ed28e049ae",
        } else hash_state_dict_keys(filtered)
        converted = convert_wan_vsa_gate_state_dict(filtered)
        config = WAN_CIVITAI_CONFIGS_BY_HASH.get(checkpoint_hash)
        if config is None:
            config = infer_native_wan_transformer_config(converted)
        config = dict(config)
        if validate_wan_vsa_gate_shapes(
            converted,
            dim=int(config["dim"]),
            num_layers=int(config["num_layers"]),
        ):
            config["vsa_gate_compress"] = True
        return converted, config


class WanDenoiser(FeatureCacheDenoiserMixin, GraphWrappedDenoiserMixin):
    """Expose a checkpoint-compatible Wan DiT through ``Denoiser``.

    Expects tensor ``context`` (and optional image / channel conditions).
    Missing required tensors raise :exc:`TypeError`.  Dual-expert Wan 2.2
    uses :class:`Wan22DualExpertDenoiser` instead.
    """

    def __init__(
        self,
        model: WanModel,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
        reference_condition_key: str | None = None,
        channel_condition_key: str | None = None,
        manage_autocast: bool = True,
        inplace_residual: bool = False,
        enable_cuda_graph: bool = False,
        teacache_threshold: float | None = None,
        feature_cache_config: WanFeatureCacheConfig | None = None,
    ) -> None:
        if not isinstance(manage_autocast, bool):
            raise TypeError("Wan manage_autocast must be a bool")
        if not isinstance(inplace_residual, bool):
            raise TypeError("Wan inplace_residual must be a bool")
        if teacache_threshold is not None and teacache_threshold < 0:
            raise ValueError("Wan TeaCache threshold must be non-negative")
        if teacache_threshold is not None and feature_cache_config is not None:
            raise ValueError("Wan legacy TeaCache threshold and feature_cache_config are exclusive")
        if getattr(model, "_worldfoundry_easycache_config", None) is not None and (
            teacache_threshold is not None or feature_cache_config is not None
        ):
            raise ValueError("EasyCache and existing feature-cache configurations are exclusive")
        self.model = model
        self.compute_dtype = compute_dtype
        self.reference_condition_key = reference_condition_key
        self.channel_condition_key = channel_condition_key
        self.manage_autocast = manage_autocast
        self.inplace_residual = inplace_residual
        parameter_source = getattr(model, "parameters", None)
        parameter = (
            next(parameter_source(), None)
            if callable(parameter_source)
            else None
        )
        # A managed autocast context is useful when FP32-resident weights are
        # intentionally executed in BF16/FP16.  It is pure dispatcher/cache
        # overhead once both the resident DiT and its inputs already use the
        # compute dtype.  Cache this load-time invariant instead of scanning a
        # multi-billion-parameter module on every CFG branch.
        self._managed_autocast_is_redundant = bool(
            manage_autocast
            and compute_dtype in {torch.float16, torch.bfloat16}
            and parameter is not None
            and parameter.dtype == compute_dtype
        )
        self._teacache_threshold = teacache_threshold
        self._feature_cache_config = feature_cache_config
        self._feature_cache_requests: dict[str, dict[str, object]] = {}
        self._feature_cache_receipt_snapshots: dict[str, str] = {}
        self._static_cross_kv_receipt_snapshots: dict[str, str] = {}
        self._inplace_residual_receipt_snapshots: dict[str, str] = {}
        self._timestep_receipt_snapshots: dict[str, str] = {}
        self._inplace_residual_runtime: dict[str, object] = {
            "requested": inplace_residual,
            "request_id": None,
            "calls": 0,
            "inplace_calls": 0,
            "functional_calls": 0,
            "autograd_fallback_calls": 0,
            "feature_cache_fallback_calls": 0,
            "finalized": False,
        }
        self._timestep_runtime: dict[str, object] = {
            "request_id": None,
            "calls": 0,
            "global_calls": 0,
            "per_token_calls": 0,
            "explicit_all_ones_calls": 0,
            "finalized": False,
        }
        self._feature_cache_epoch_counter = 0
        self._optimization_request_windows: dict[str, dict[str, object]] = {}
        self._init_graph_runner(
            model,
            enabled=enable_cuda_graph,
            extra_key=f"wan:{type(model).__module__}.{type(model).__qualname__}",
        )

    def _inplace_residual_state(self) -> dict[str, object]:
        """Return lazy-compatible in-place dispatch telemetry.

        A few lightweight tests and third-party adapters construct a denoiser
        with ``__new__`` to exercise cache/report logic without loading a Wan
        checkpoint.  Lazily creating the telemetry keeps those callers on the
        disabled functional path instead of turning an optional speedup into a
        new construction requirement.
        """

        state = self.__dict__.get("_inplace_residual_runtime")
        if isinstance(state, dict):
            return state
        state = {
            "requested": bool(getattr(self, "inplace_residual", False)),
            "request_id": None,
            "calls": 0,
            "inplace_calls": 0,
            "functional_calls": 0,
            "autograd_fallback_calls": 0,
            "feature_cache_fallback_calls": 0,
            "finalized": False,
        }
        self.__dict__["_inplace_residual_runtime"] = state
        return state

    def _timestep_state(self) -> dict[str, object]:
        """Return lazy-compatible scalar/per-token timestep telemetry."""

        state = self.__dict__.get("_timestep_runtime")
        if isinstance(state, dict):
            return state
        state = {
            "request_id": None,
            "calls": 0,
            "global_calls": 0,
            "per_token_calls": 0,
            "explicit_all_ones_calls": 0,
            "finalized": False,
        }
        self.__dict__["_timestep_runtime"] = state
        return state

    def _reset_request_optimization_state(self) -> None:
        """Invalidate state and telemetry that must not cross requests."""

        self._begin_graph_request_window()
        self._inplace_residual_state().update(
            request_id=None,
            calls=0,
            inplace_calls=0,
            functional_calls=0,
            autograd_fallback_calls=0,
            feature_cache_fallback_calls=0,
            finalized=False,
        )
        self._timestep_state().update(
            request_id=None,
            calls=0,
            global_calls=0,
            per_token_calls=0,
            explicit_all_ones_calls=0,
            finalized=False,
        )
        cross_kv_cache = getattr(self.model, "_worldfoundry_static_cross_kv", None)
        if cross_kv_cache is not None:
            cross_kv_cache.invalidate()
        fused_rope_state = getattr(
            self.model,
            "_worldfoundry_fused_rope_runtime",
            None,
        )
        if fused_rope_state is not None:
            fused_rope_state.reset_request_window()
        sequence_parallel_state = getattr(
            self.model,
            "_worldfoundry_sequence_parallel",
            None,
        )
        if sequence_parallel_state is not None:
            sequence_parallel_state.reset_request_window()
        qkv_fusion_state = getattr(
            self.model,
            "_worldfoundry_qkv_fusion",
            None,
        )
        if qkv_fusion_state is not None:
            reset_qkv_window = getattr(qkv_fusion_state, "reset_request_window", None)
            if callable(reset_qkv_window):
                reset_qkv_window()
        compile_runtime = getattr(
            self.model,
            "_worldfoundry_compile_runtime",
            None,
        )
        if isinstance(compile_runtime, dict):
            compile_runtime["request_calls"] = 0
            compile_runtime["request_failures"] = 0
            compile_runtime["request_last_error"] = None
        offload_handle = getattr(
            self.model,
            "_worldfoundry_layerwise_cpu_offload_handle",
            None,
        )
        reset_offload_window = getattr(
            offload_handle,
            "reset_request_window",
            None,
        )
        if callable(reset_offload_window):
            reset_offload_window()
        from worldfoundry.core.acceleration.quantization.linear import (
            reset_quantization_runtime_window,
        )

        reset_quantization_runtime_window(self.model)
    def _begin_request_optimization_state(self, model_input: DenoiserInput) -> None:
        """Open one explicit request window exactly once, regardless of CFG order."""

        request_id = model_input.request_id
        cache_enabled = (
            getattr(self, "_feature_cache_config", None) is not None
            or getattr(self, "_teacache_threshold", None) is not None
        )
        if not isinstance(request_id, str) or not request_id.strip():
            if cache_enabled:
                self._explicit_request_id(model_input)
            # Compatibility for dense direct callers. Stateful cache paths
            # never enter this heuristic.
            if model_input.step_index == 0 and model_input.branch == "positive":
                self._reset_request_optimization_state()
                approximate_state = getattr(
                    self.model,
                    "_worldfoundry_approximate_attention",
                    None,
                )
                if approximate_state is not None:
                    from ...optimizations.approximate_attention import (
                        reset_approximate_attention,
                    )

                    reset_approximate_attention(approximate_state)
            return

        self._feature_cache_context_var().set(request_id)
        windows = self.__dict__.setdefault("_optimization_request_windows", {})
        should_reset = False
        with self._feature_cache_lock():
            window = windows.get(request_id)
            if window is None:
                window = {
                    "request_id": request_id,
                    "complete": False,
                }
                windows[request_id] = window
                should_reset = True
            if model_input.step_index >= model_input.total_steps - 1:
                window["complete"] = True
            if len(windows) > 64:
                completed_ids = [
                    key
                    for key, value in windows.items()
                    if bool(value.get("complete")) and key != request_id
                ]
                for key in completed_ids[: max(len(windows) - 32, 0)]:
                    windows.pop(key, None)
        if should_reset:
            self._reset_request_optimization_state()

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Finalize request telemetry and release request-owned cache tensors."""

        from worldfoundry.core.acceleration.plugins import end_acceleration_request

        try:
            self._finalize_request(request_id, error=error)
        finally:
            end_acceleration_request(self.model, self, request_id)

    def _finalize_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Complete all receipts before releasing acceleration ownership."""
        self._feature_cache_context_var().set(request_id)
        cleanup_errors: list[BaseException] = []
        try:
            self.end_feature_cache_request(request_id, error=error)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        approximate_state = getattr(
            self.model,
            "_worldfoundry_approximate_attention",
            None,
        )
        if approximate_state is not None:
            from ...optimizations.approximate_attention import (
                finalize_approximate_attention_request,
            )

            try:
                finalize_approximate_attention_request(
                    approximate_state,
                    request_id,
                    error=error,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        with self._feature_cache_lock():
            self.__dict__.setdefault("_optimization_request_windows", {}).pop(
                request_id,
                None,
            )
        cross_kv_cache = getattr(self.model, "_worldfoundry_static_cross_kv", None)
        if cross_kv_cache is not None:
            cross_kv_report = dict(cross_kv_cache.report())
            cross_kv_report.update(
                request_id=request_id,
                finalized=True,
                release_reason="error" if error is not None else "completed",
            )
            if error is not None:
                cross_kv_report["error_type"] = type(error).__name__
            with self._feature_cache_lock():
                snapshots = self.__dict__.setdefault(
                    "_static_cross_kv_receipt_snapshots",
                    {},
                )
                snapshots[request_id] = json.dumps(
                    cross_kv_report,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                while len(snapshots) > 32:
                    oldest_request_id = next(iter(snapshots))
                    snapshots.pop(oldest_request_id, None)
            cross_kv_cache.invalidate()
        inplace_receipt = dict(self._inplace_residual_state())
        inplace_receipt.update(
            request_id=request_id,
            finalized=True,
            release_reason="error" if error is not None else "completed",
        )
        if error is not None:
            inplace_receipt["error_type"] = type(error).__name__
        with self._feature_cache_lock():
            snapshots = self.__dict__.setdefault(
                "_inplace_residual_receipt_snapshots",
                {},
            )
            snapshots[request_id] = json.dumps(
                inplace_receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            while len(snapshots) > 32:
                oldest_request_id = next(iter(snapshots))
                snapshots.pop(oldest_request_id, None)
        timestep_receipt = dict(self._timestep_state())
        timestep_receipt.update(
            request_id=request_id,
            finalized=True,
            release_reason="error" if error is not None else "completed",
        )
        if error is not None:
            timestep_receipt["error_type"] = type(error).__name__
        with self._feature_cache_lock():
            snapshots = self.__dict__.setdefault(
                "_timestep_receipt_snapshots",
                {},
            )
            snapshots[request_id] = json.dumps(
                timestep_receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            while len(snapshots) > 32:
                oldest_request_id = next(iter(snapshots))
                snapshots.pop(oldest_request_id, None)
        if cleanup_errors:
            primary = cleanup_errors[0]
            add_note = getattr(primary, "add_note", None)
            if callable(add_note):
                for extra in cleanup_errors[1:]:
                    add_note(
                        "additional Wan request cleanup failed: "
                        f"{type(extra).__name__}: {extra}"
                    )
            raise primary

    @acceleration_runtime_call
    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        self._begin_request_optimization_state(model_input)
        try:
            context = model_input.conditioning["context"]
        except KeyError as error:
            raise KeyError("Wan denoising requires a 'context' conditioning tensor") from error
        if not isinstance(context, torch.Tensor):
            raise TypeError("Wan 'context' conditioning must be a tensor")

        latents = model_input.latents
        resident_compute = bool(
            self.manage_autocast
            and getattr(self, "_managed_autocast_is_redundant", False)
        )
        network_dtype = self.compute_dtype if resident_compute else latents.dtype
        model_latents = latents.to(dtype=network_dtype)
        if resident_compute:
            context = context.to(device=latents.device, dtype=network_dtype)
        timestep = model_input.timestep.to(
            device=latents.device,
            dtype=torch.float32,
        ).reshape(-1)
        if timestep.numel() == 1 and model_input.latents.shape[0] != 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        if timestep.numel() != model_input.latents.shape[0]:
            raise ValueError("Wan timestep must be scalar or have one value per latent sample")
        if self.model.per_token_timestep:
            all_ones_semantics = model_input.conditioning.get(
                WAN_DENOISE_MASK_IS_ALL_ONES,
                False,
            )
            if not isinstance(all_ones_semantics, bool):
                raise TypeError(
                    f"{WAN_DENOISE_MASK_IS_ALL_ONES} must be a bool when provided"
                )
            denoise_mask = model_input.conditioning.get("denoise_mask")
            if not isinstance(denoise_mask, torch.Tensor):
                raise TypeError("per-token Wan denoising requires a tensor denoise_mask")
            if denoise_mask.ndim != 5 or denoise_mask.shape[0] != latents.shape[0]:
                raise ValueError("Wan denoise_mask must have shape [B,C,T,H,W]")
            if not all_ones_semantics:
                denoise_mask = denoise_mask.to(
                    device=latents.device,
                    dtype=latents.dtype,
                )
                patch_height, patch_width = self.model.patch_size[1:]
                token_mask = denoise_mask[
                    :, 0, :, ::patch_height, ::patch_width
                ].flatten(1)
                timestep = timestep[:, None] * token_mask
        else:
            all_ones_semantics = False
        timestep_state = self._timestep_state()
        timestep_state["request_id"] = model_input.request_id
        timestep_state["calls"] = int(timestep_state["calls"]) + 1
        if timestep.ndim == 1:
            timestep_state["global_calls"] = int(
                timestep_state["global_calls"]
            ) + 1
            if all_ones_semantics:
                timestep_state["explicit_all_ones_calls"] = int(
                    timestep_state["explicit_all_ones_calls"]
                ) + 1
        else:
            timestep_state["per_token_calls"] = int(
                timestep_state["per_token_calls"]
            ) + 1
        model_kwargs = {}
        use_gradient_checkpointing = model_input.conditioning.get(
            "use_gradient_checkpointing",
            False,
        )
        use_gradient_checkpointing_offload = model_input.conditioning.get(
            "use_gradient_checkpointing_offload",
            False,
        )
        if not isinstance(use_gradient_checkpointing, bool):
            raise TypeError("Wan use_gradient_checkpointing must be a bool")
        if not isinstance(use_gradient_checkpointing_offload, bool):
            raise TypeError("Wan use_gradient_checkpointing_offload must be a bool")
        model_kwargs.update(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        if self.model.inject_sample_info:
            fps = int(model_input.conditioning.get("fps", 24))
            fps_id = 0 if fps == 16 else 1
            model_kwargs["fps"] = torch.full(
                (model_input.latents.shape[0],),
                fps_id,
                device=model_input.latents.device,
                dtype=torch.long,
            )
        channel_condition_key = getattr(self, "channel_condition_key", None)
        if channel_condition_key is not None:
            channel_condition = model_input.conditioning.get(channel_condition_key)
            if not isinstance(channel_condition, torch.Tensor):
                raise TypeError(
                    f"Wan channel conditioning {channel_condition_key!r} must be a tensor"
                )
            channel_condition = channel_condition.to(
                device=latents.device,
                dtype=network_dtype,
            )
            if (
                channel_condition.shape[0] != latents.shape[0]
                or channel_condition.shape[2:] != latents.shape[2:]
            ):
                raise ValueError(
                    "Wan channel conditioning must match latent batch and spatiotemporal geometry: "
                    f"{tuple(channel_condition.shape)} vs {tuple(latents.shape)}"
                )
            model_latents = torch.cat((model_latents, channel_condition), dim=1)
        reference_condition_key = getattr(self, "reference_condition_key", None)
        if reference_condition_key is not None:
            reference = model_input.conditioning.get(reference_condition_key)
            if not isinstance(reference, torch.Tensor):
                raise TypeError(
                    f"Wan reference conditioning {reference_condition_key!r} must be a tensor"
                )
            reference = reference.to(device=latents.device, dtype=network_dtype)
            if bool(model_input.conditioning.get("drop_secondary_condition", False)):
                reference = torch.zeros_like(reference)
            if reference.shape[:2] != latents.shape[:2] or reference.shape[-2:] != latents.shape[-2:]:
                raise ValueError(
                    "Wan reference latents must match generated batch/channel/spatial geometry: "
                    f"{tuple(reference.shape)} vs {tuple(latents.shape)}"
                )
            model_latents = torch.cat((model_latents, reference), dim=2)

        if self.model.has_image_input:
            condition_latents = model_input.conditioning.get("condition_latents")
            clip_feature = model_input.conditioning.get("clip_feature")
            if not isinstance(condition_latents, torch.Tensor):
                raise TypeError("Wan image-to-video denoising requires tensor condition_latents")
            if not isinstance(clip_feature, torch.Tensor):
                raise TypeError("Wan image-to-video denoising requires tensor clip_feature")
            model_kwargs.update(
                y=condition_latents.to(device=latents.device, dtype=network_dtype),
                clip_feature=clip_feature.to(device=latents.device, dtype=network_dtype),
            )

        feature_cache_kwargs = self.feature_cache_kwargs(model_input)
        model_kwargs.update(feature_cache_kwargs)
        inplace_runtime = self._inplace_residual_state()
        inplace_runtime["request_id"] = model_input.request_id
        inplace_runtime["calls"] = int(inplace_runtime["calls"]) + 1
        inplace_residual_requested = bool(
            getattr(self, "inplace_residual", False)
        )
        if inplace_residual_requested:
            autograd_active = torch.is_grad_enabled()
            feature_cache_active = feature_cache_kwargs.get("feature_cache") is not None
            if not autograd_active and not feature_cache_active:
                # Keep this internal flag absent on every unsafe path so
                # variant hooks and cache controllers see the canonical
                # functional block contract.
                model_kwargs["_worldfoundry_inplace_residual"] = True
                inplace_runtime["inplace_calls"] = (
                    int(inplace_runtime["inplace_calls"]) + 1
                )
            else:
                inplace_runtime["functional_calls"] = (
                    int(inplace_runtime["functional_calls"]) + 1
                )
                if autograd_active:
                    inplace_runtime["autograd_fallback_calls"] = (
                        int(inplace_runtime["autograd_fallback_calls"]) + 1
                    )
                if feature_cache_active:
                    inplace_runtime["feature_cache_fallback_calls"] = (
                        int(inplace_runtime["feature_cache_fallback_calls"]) + 1
                    )
        else:
            inplace_runtime["functional_calls"] = (
                int(inplace_runtime["functional_calls"]) + 1
            )
        approximate_state = getattr(
            self.model,
            "_worldfoundry_approximate_attention",
            None,
        )
        if approximate_state is not None:
            from ...optimizations.approximate_attention import (
                advance_approximate_step,
            )

            # CFG invokes positive and negative branches at the same denoise
            # step. Set the explicit index instead of incrementing per call.
            request_id = model_input.request_id
            if isinstance(request_id, str) and request_id.strip():
                advance_approximate_step(
                    approximate_state,
                    step=model_input.step_index,
                    total_steps=model_input.total_steps,
                    request_id=request_id,
                    branch=model_input.branch,
                    routed_steps=bool(
                        model_input.conditioning.get(
                            "_worldfoundry_approximate_routed_steps",
                            False,
                        )
                    ),
                )
            else:
                advance_approximate_step(
                    approximate_state,
                    step=model_input.step_index,
                    total_steps=model_input.total_steps,
                )

        autocast_enabled = bool(
            self.manage_autocast
            and self.compute_dtype in {torch.float16, torch.bfloat16}
            and not resident_compute
        )
        autocast_context = (
            torch.autocast(
                device_type=latents.device.type,
                dtype=self.compute_dtype,
            )
            if autocast_enabled
            else nullcontext()
        )
        with autocast_context:
            sample = self._run_network(
                x=model_latents,
                timestep=timestep,
                context=context,
                **model_kwargs,
            )
        sample = sample[:, :, : latents.shape[2]]
        return DenoiserOutput(sample=sample)

    def runtime_optimization_report(
        self,
        request_id: str | None = None,
    ) -> dict[str, object]:
        """Merge policy audit with dispatch state for one selected request."""

        applied = getattr(self.model, "_worldfoundry_applied_optimizations", None)
        if applied is None:
            requested: dict[str, object] = {}
            effective: dict[str, object] = {}
            fallbacks: list[object] = []
            quality_tier = "exact"
        else:
            snapshot = applied.to_optimization_snapshot()
            requested = dict(snapshot.requested)
            effective = dict(snapshot.effective)
            fallbacks = list(snapshot.fallbacks)
            quality_tier = snapshot.quality_tier

        acceleration_session = getattr(self.model, "_worldfoundry_accelerations", None)
        if acceleration_session is not None or "accelerations" in effective:
            effective["accelerations"] = (
                acceleration_session.report()
                if acceleration_session is not None
                else {"installed": [], "execution_verified": False}
            )
            if acceleration_session is not None and any(
                handle.details.get("approximate") for handle in acceleration_session.handles
            ):
                quality_tier = "approximate"

        autocast_requested = bool(
            self.manage_autocast
            and self.compute_dtype in {torch.float16, torch.bfloat16}
        )
        autocast_elided = bool(
            autocast_requested
            and getattr(self, "_managed_autocast_is_redundant", False)
        )
        requested["denoiser_autocast"] = (
            str(self.compute_dtype) if autocast_requested else False
        )
        effective["denoiser_compute_dtype"] = str(self.compute_dtype)
        effective["denoiser_autocast_context"] = (
            "elided-resident-dtype"
            if autocast_elided
            else ("enabled" if autocast_requested else "disabled")
        )

        selected_inplace_request_id = request_id
        if selected_inplace_request_id is None:
            selected_inplace_request_id = self._feature_cache_context_var().get()
        encoded_inplace = (
            self.__dict__.get("_inplace_residual_receipt_snapshots", {}).get(
                selected_inplace_request_id
            )
            if isinstance(selected_inplace_request_id, str)
            else None
        )
        if isinstance(encoded_inplace, str):
            inplace_runtime = json.loads(encoded_inplace)
            if not isinstance(inplace_runtime, dict):
                raise TypeError(
                    "Wan in-place residual receipt must decode to an object"
                )
        else:
            inplace_runtime = dict(self._inplace_residual_state())
        inplace_residual_requested = bool(
            getattr(self, "inplace_residual", False)
        )
        requested["inplace_residual"] = inplace_residual_requested
        inplace_calls = int(inplace_runtime["inplace_calls"])
        functional_calls = int(inplace_runtime["functional_calls"])
        if not inplace_residual_requested:
            inplace_effective = "disabled-functional"
        elif inplace_calls and not functional_calls:
            inplace_effective = "in-place-executed"
        elif inplace_calls:
            inplace_effective = "partially-in-place"
        elif int(inplace_runtime["calls"]):
            inplace_effective = "functional-safety-fallback"
        else:
            inplace_effective = "pending"
        effective["inplace_residual"] = inplace_effective
        if inplace_residual_requested and functional_calls:
            reason = (
                "inplace_residual: functional safety path executed "
                f"(autograd={int(inplace_runtime['autograd_fallback_calls'])}, "
                "feature_cache="
                f"{int(inplace_runtime['feature_cache_fallback_calls'])})"
            )
            if reason not in fallbacks:
                fallbacks.append(reason)

        selected_timestep_request_id = request_id
        if selected_timestep_request_id is None:
            selected_timestep_request_id = self._feature_cache_context_var().get()
        encoded_timestep = (
            self.__dict__.get("_timestep_receipt_snapshots", {}).get(
                selected_timestep_request_id
            )
            if isinstance(selected_timestep_request_id, str)
            else None
        )
        if isinstance(encoded_timestep, str):
            timestep_runtime = json.loads(encoded_timestep)
            if not isinstance(timestep_runtime, dict):
                raise TypeError("Wan timestep receipt must decode to an object")
        else:
            timestep_runtime = dict(self._timestep_state())
        if int(timestep_runtime["per_token_calls"]) > 0:
            timestep_effective = "per-token"
        elif int(timestep_runtime["explicit_all_ones_calls"]) > 0:
            timestep_effective = "global-explicit-all-ones"
        elif int(timestep_runtime["global_calls"]) > 0:
            timestep_effective = "global"
        else:
            timestep_effective = "pending"
        effective["wan_timestep_mode"] = timestep_effective

        graph = self.graph_report()
        requested["cuda_graph"] = graph is not None
        if graph is not None:
            graph_captures = int(graph["capture"])
            graph_replays = int(graph["replay"])
            graph_eager = int(graph["eager"])
            graph_failures = int(graph["capture_failed"])
            if graph_eager > 0 or graph_failures > 0:
                graph_effective = (
                    "partial-eager"
                    if graph_replays > 0
                    else "eager"
                )
                fallbacks.append(
                    "cuda_graph: current request used eager execution or had a "
                    "capture failure"
                )
            elif graph_captures > 0 and graph_replays > 0:
                graph_effective = "captured-and-replayed"
            elif graph_replays > 0:
                graph_effective = "replayed-existing-graph"
            else:
                graph_effective = "pending"
            effective["cuda_graph"] = graph_effective
            effective["cuda_graph_report"] = dict(graph)

        feature_cache = self.feature_cache_report(request_id)
        requested["feature_cache"] = (
            feature_cache["algorithm"] if feature_cache["enabled"] else False
        )
        if feature_cache["enabled"]:
            effective["feature_cache"] = feature_cache["effective"]
            effective["feature_cache_report"] = feature_cache
            quality_tier = "approximate"
        if feature_cache["algorithm"] in {"teacache", "adaptive-residual-legacy"}:
            requested["teacache"] = feature_cache["threshold"]
            effective["teacache"] = feature_cache["effective"]
            effective["teacache_report"] = feature_cache

        compile_runtime = getattr(self.model, "_worldfoundry_compile_runtime", None)
        if isinstance(compile_runtime, Mapping):
            runtime_compile = dict(compile_runtime)
            runtime_compile["lifetime_calls"] = int(
                compile_runtime.get("calls", 0)
            )
            runtime_compile["lifetime_failures"] = int(
                compile_runtime.get("failures", 0)
            )
            runtime_compile["lifetime_last_error"] = compile_runtime.get(
                "last_error"
            )
            runtime_compile["calls"] = int(
                compile_runtime.get(
                    "request_calls",
                    compile_runtime.get("calls", 0),
                )
            )
            runtime_compile["failures"] = int(
                compile_runtime.get(
                    "request_failures",
                    compile_runtime.get("failures", 0),
                )
            )
            runtime_compile["last_error"] = compile_runtime.get(
                "request_last_error",
                compile_runtime.get("last_error"),
            )
            if not bool(runtime_compile.get("wrapper_installed")):
                compile_effective = "eager (compile setup fallback)"
            elif int(runtime_compile.get("failures", 0)):
                compile_effective = "compile-wrapper-failed"
            elif int(runtime_compile.get("calls", 0)):
                compile_effective = "compile-wrapper-executed"
            else:
                compile_effective = "compile-wrapper-installed (lazy)"
            effective["compile"] = compile_effective
            effective["compile_runtime"] = runtime_compile

        approximate_state = getattr(
            self.model,
            "_worldfoundry_approximate_attention",
            None,
        )
        approximate = None
        if approximate_state is not None:
            from ...optimizations.approximate_attention import (
                approximate_attention_report,
            )

            approximate = approximate_attention_report(
                approximate_state,
                request_id=request_id,
            )
            effective["approximate_attention_kernel"] = approximate["effective_kernel"]
            effective["approximate_attention_runtime"] = approximate
            if str(approximate["effective_kernel"]).startswith("exact"):
                reason = (
                    "approximate_attention: requested sparse execution but runtime used "
                    f"{approximate['effective_kernel']}"
                )
                if reason not in fallbacks:
                    fallbacks.append(reason)

        cross_kv_cache = getattr(self.model, "_worldfoundry_static_cross_kv", None)
        cross_kv = None
        selected_request_id = request_id
        if selected_request_id is None:
            selected_request_id = self._feature_cache_context_var().get()
        if cross_kv_cache is not None:
            encoded_cross_kv = None
            if isinstance(selected_request_id, str):
                encoded_cross_kv = self.__dict__.get(
                    "_static_cross_kv_receipt_snapshots",
                    {},
                ).get(selected_request_id)
            if isinstance(encoded_cross_kv, str):
                decoded_cross_kv = json.loads(encoded_cross_kv)
                if not isinstance(decoded_cross_kv, dict):
                    raise TypeError(
                        "static cross-KV receipt snapshot must decode to an object"
                    )
                cross_kv = decoded_cross_kv
            else:
                cross_kv = cross_kv_cache.report()
        if cross_kv is not None:
            cross_kv_effective = str(cross_kv["effective"])
            effective["static_cross_kv_runtime"] = cross_kv_effective
            if int(cross_kv["bypasses"]) > 0:
                fallback = (
                    "static_cross_kv: request used dense processor bypasses "
                    f"for kwargs={cross_kv['bypass_kwargs']}"
                )
                if fallback not in fallbacks:
                    fallbacks.append(fallback)
            elif int(cross_kv["misses"]) > 0 and int(cross_kv["hits"]) <= 0:
                fallback = (
                    "static_cross_kv: request projected K/V but recorded no cache hit"
                )
                if fallback not in fallbacks:
                    fallbacks.append(fallback)

        from ...optimizations.qkv_fusion import qkv_fusion_report

        qkv_fusion = qkv_fusion_report(self.model)
        if qkv_fusion is not None:
            effective["fuse_qkv_runtime"] = qkv_fusion["execution"]
            compile_request_calls = (
                int(runtime_compile.get("calls", 0))
                if isinstance(compile_runtime, Mapping)
                else 0
            )
            compiled_qkv_executed = (
                int(qkv_fusion.get("lifetime_compiled_graph_traces", 0)) > 0
                and compile_request_calls > 0
            )
            if (
                int(qkv_fusion["fused_blocks"]) > 0
                and int(qkv_fusion["eager_projection_calls"]) <= 0
                and not compiled_qkv_executed
            ):
                fallback = (
                    "fuse_qkv: merged projections were installed but no forward "
                    "execution receipt was recorded"
                )
                if fallback not in fallbacks:
                    fallbacks.append(fallback)

        fused_rope_state = getattr(
            self.model,
            "_worldfoundry_fused_rope_runtime",
            None,
        )
        fused_rope = None
        if fused_rope_state is not None:
            from ...optimizations.fused_rope import fused_rope_runtime_report

            fused_rope = fused_rope_runtime_report(fused_rope_state)
            effective["fused_rope_runtime"] = fused_rope["effective"]
            eager_failures = sum(
                int(fused_rope[name])
                for name in (
                    "torch_fallback_calls",
                    "provider_failures",
                    "quarantined_skips",
                    "malformed_receipts",
                )
            )
            if int(fused_rope["eager_calls"]) > 0 and eager_failures > 0:
                fallback = (
                    "fused_rope: eager execution used a torch fallback, failed/"
                    "quarantined provider, or malformed dispatch receipt"
                )
                if fallback not in fallbacks:
                    fallbacks.append(fallback)
        sequence_parallel_state = getattr(
            self.model,
            "_worldfoundry_sequence_parallel",
            None,
        )
        sequence_parallel = None
        if sequence_parallel_state is not None:
            from ...optimizations.sequence_parallel import sequence_parallel_report

            sequence_parallel = sequence_parallel_report(sequence_parallel_state)

        from worldfoundry.core.acceleration.quantization.linear import (
            quantization_runtime_report,
        )

        quantization = quantization_runtime_report(self.model)
        if quantization is not None:
            quantization_effective = str(quantization["effective"])
            effective["quantization"] = quantization_effective
            effective["quantization_runtime"] = quantization
            fallback_compute_calls = int(
                quantization.get("packed_weight_calls") or 0
            ) + int(quantization.get("dense_fallback_calls") or 0)
            if fallback_compute_calls > 0:
                reasons = "; ".join(
                    str(reason) for reason in quantization["fallback_reasons"]
                )
                fallback = (
                    "quantization: runtime used dequantize/dense fallback"
                    + (f" ({reasons})" if reasons else "")
                )
                if fallback not in fallbacks:
                    fallbacks.append(fallback)

        offload_handle = getattr(
            self.model,
            "_worldfoundry_layerwise_cpu_offload_handle",
            None,
        )
        offload_reporter = getattr(offload_handle, "report", None)
        offload = offload_reporter() if callable(offload_reporter) else None
        if isinstance(offload, Mapping):
            offload = dict(offload)
            offload_used = int(
                dict(offload.get("request") or {}).get("forward_calls", 0)
            ) > 0
            if bool(offload.get("effective")):
                effective["offload"] = "async-double-buffer-executed"
            elif offload_used:
                effective["offload"] = "block-offload-not-overlapped"
                issues = "; ".join(
                    str(issue) for issue in offload.get("issues", ())
                )
                fallback = (
                    "offload block: asynchronous double-buffer execution was "
                    "not proven"
                    + (f" ({issues})" if issues else "")
                )
                if fallback not in fallbacks:
                    fallbacks.append(fallback)
            effective["offload_runtime"] = offload

        from worldfoundry.core.attention import attention_dispatch_report
        from worldfoundry.core.kernels import kernel_dispatch_report

        return {
            "requested": requested,
            "effective": effective,
            "fallbacks": fallbacks,
            "quality_tier": quality_tier,
            "runtime": {
                "cuda_graph": graph,
                "teacache": feature_cache,
                "feature_cache": feature_cache,
                "compile": runtime_compile if isinstance(compile_runtime, Mapping) else None,
                "approximate_attention": approximate,
                "fused_rope": fused_rope,
                "qkv_fusion": qkv_fusion,
                "static_cross_kv": cross_kv,
                "sequence_parallel": sequence_parallel,
                "quantization": quantization,
                "offload": offload,
                "denoiser_autocast_context": effective[
                    "denoiser_autocast_context"
                ],
                "inplace_residual": inplace_runtime,
                "wan_timestep": timestep_runtime,
                "attention_dispatch": attention_dispatch_report(),
                "kernel_dispatch": kernel_dispatch_report(),
            },
        }


def build_wan21_t2v_1p3b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the Wan2.1 T2V 1.3B DiT with shared core placement policy."""

    return _build_wan_denoiser(context, config=WAN21_T2V_1P3B_CONFIG)


def build_skyreels_v2_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the SkyReels-V2 1.3B graph on the shared Wan architecture."""

    return _build_wan_denoiser(context, config=SKYREELS_V2_DF_1P3B_CONFIG)


def build_wan21_t2v_14b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the Wan2.1 T2V 14B checkpoint on the shared Wan graph."""

    return _build_wan_denoiser(context, config=WAN21_T2V_14B_CONFIG)


def build_wan22_t2v_a14b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load either released Wan2.2 A14B expert on the native Wan graph."""

    return _build_wan_denoiser(context, config=WAN22_T2V_A14B_CONFIG)


class Wan22DualExpertDenoiser:
    """Route released Wan2.2 A14B steps with request-local receipts."""

    def __init__(
        self,
        high_noise: WanDenoiser,
        low_noise: WanDenoiser,
        *,
        boundary_ratio: float,
        num_train_timesteps: int = 1000,
    ) -> None:
        boundary = float(boundary_ratio)
        if not 0.0 < boundary < 1.0:
            raise ValueError("Wan2.2 A14B boundary_ratio must be in (0, 1)")
        if int(num_train_timesteps) <= 0:
            raise ValueError("Wan2.2 A14B num_train_timesteps must be positive")
        self.high_noise = high_noise
        self.low_noise = low_noise
        self.boundary_ratio = boundary
        self.num_train_timesteps = int(num_train_timesteps)
        self._route_lock = RLock()
        self._route_current_request: ContextVar[str | None] = ContextVar(
            f"worldfoundry_wan22_route_request_{id(self)}",
            default=None,
        )
        self._route_requests: dict[str, dict[str, object]] = {}
        self._route_receipt_snapshots: dict[str, str] = {}
        self._route_epoch_counter = 0
        # Compatibility telemetry for direct callers that do not have a
        # runner-created request id. Explicit requests never use it as proof.
        self._route_calls = {"high-noise": 0, "low-noise": 0}
        self._last_expert: str | None = None
        self._validate_easycache_experts()

    def _easycache_enabled(self) -> bool:
        return any(
            getattr(getattr(expert, "_feature_cache_config", None), "algorithm", None) == "easycache"
            for expert in (self.high_noise, self.low_noise)
        )

    def _validate_easycache_experts(self) -> None:
        if not self._easycache_enabled():
            return
        if any(
            type(expert) is not WanDenoiser or getattr(expert._feature_cache_config, "algorithm", None) != "easycache"
            for expert in (self.high_noise, self.low_noise)
        ):
            raise ValueError("dual-expert EasyCache requires EasyCache on both native Wan denoisers")
        if self.high_noise is self.low_noise or self.high_noise.model is self.low_noise.model:
            raise ValueError("dual-expert EasyCache requires independent expert models and denoisers")
        for expert in (self.high_noise, self.low_noise):
            if (
                getattr(expert, "_graph_runner", None) is not None
                or getattr(expert, "inplace_residual", False)
                or getattr(expert, "_teacache_threshold", None) is not None
                or getattr(expert.model, "training", False)
                or any(
                    getattr(expert.model, name, None) is not None
                    for name in (
                        "_worldfoundry_compile_runtime",
                        "_worldfoundry_sequence_parallel",
                        "_worldfoundry_layerwise_cpu_offload_handle",
                        "_worldfoundry_approximate_attention",
                        "_worldfoundry_static_cross_kv",
                    )
                )
            ):
                raise ValueError("dual-expert EasyCache requires resident eager experts without Graph/SP or other caches")

    def validate_request_execution(self, *, cfg_parallel_degree: int, cfg_gate_step: float) -> None:
        if self._easycache_enabled() and (cfg_parallel_degree != 1 or cfg_gate_step != 1.0):
            raise ValueError("dual-expert EasyCache requires dense single-rank CFG execution")

    def prepare_request_schedule(self, request_id: str, schedule) -> None:
        """Freeze both expert phases before the first CFG evaluation."""
        if not self._easycache_enabled():
            return
        from worldfoundry.core.acceleration.plugins import acceleration_runtime_scope

        from ...optimizations.wan.expert_schedule import WanExpertSchedule

        self._validate_easycache_experts()
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("dual-expert EasyCache requires an explicit request_id")
        plan = WanExpertSchedule.build(
            schedule, boundary_ratio=self.boundary_ratio, num_train_timesteps=self.num_train_timesteps
        )
        self._route_current_request.set(request_id)
        with self._route_lock:
            state = self._route_request_state(request_id, create=True)
            assert state is not None
            previous = state.get("expert_schedule")
            if previous is not None:
                if previous != plan:
                    raise ValueError("expert schedule changed during an active request")
                if state["events"]:
                    raise ValueError("cannot prepare an expert schedule after request execution started")
                return
            state["expert_schedule"] = plan
            state["branch_global_steps"] = {}
            state["expert_branch_steps"] = {}
            # Guard both models even before an expert's first actual call.
            for expert in (self.high_noise, self.low_noise):
                with acceleration_runtime_scope(expert.model, expert, request_id):
                    pass

    def _route_request_state(
        self,
        request_id: str,
        *,
        create: bool,
    ) -> dict[str, object] | None:
        with self._route_lock:
            state = self._route_requests.get(request_id)
            if state is None and create:
                self._route_epoch_counter += 1
                state = {
                    "request_id": request_id,
                    "request_epoch": self._route_epoch_counter,
                    "route_calls": {"high-noise": 0, "low-noise": 0},
                    "branch_calls": {},
                    "events": [],
                    "last_expert": None,
                }
                self._route_requests[request_id] = state
            return state

    def route_receipt(self, request_id: str | None = None) -> dict[str, object]:
        """Return the active or finalized route proof for one request."""

        if request_id is None:
            request_id = self._route_current_request.get()
        state = (
            self._route_request_state(request_id, create=False)
            if isinstance(request_id, str) and request_id.strip()
            else None
        )
        if state is not None:
            with self._route_lock:
                return {
                    "request_id": request_id,
                    "request_epoch": int(state["request_epoch"]),
                    "request_local": True,
                    "finalized": False,
                    "release_reason": None,
                    "last_expert": state["last_expert"],
                    "route_calls": dict(state["route_calls"]),
                    "branch_calls": dict(state["branch_calls"]),
                    "events": [dict(event) for event in state["events"]],
                    **(
                        {"expert_schedule": state["expert_schedule"].receipt()}
                        if state.get("expert_schedule") is not None
                        else {}
                    ),
                }
        if isinstance(request_id, str):
            with self._route_lock:
                encoded = self._route_receipt_snapshots.get(request_id)
            if isinstance(encoded, str):
                decoded = json.loads(encoded)
                if not isinstance(decoded, dict):
                    raise TypeError("Wan2.2 route receipt must decode to an object")
                return decoded
        return {
            "request_id": None,
            "request_epoch": 0,
            "request_local": False,
            "finalized": False,
            "release_reason": None,
            "last_expert": self._last_expert,
            "route_calls": dict(self._route_calls),
            "branch_calls": {},
            "events": [],
        }

    def route_lifecycle_report(self) -> dict[str, int]:
        """Expose bounded route ownership without retaining model tensors."""

        with self._route_lock:
            return {
                "live_requests": len(self._route_requests),
                "receipt_snapshots": len(self._route_receipt_snapshots),
                "max_receipt_snapshots": 32,
            }

    def expert_for_timestep(self, timestep: torch.Tensor) -> str:
        sigmas = timestep.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(sigmas).all()):
            raise ValueError("Wan2.2 A14B timesteps must be finite")
        sigmas = sigmas / float(self.num_train_timesteps)
        high = sigmas >= self.boundary_ratio
        if bool(high.all()):
            return "high-noise"
        if bool((~high).all()):
            return "low-noise"
        raise ValueError("one Wan2.2 inference call cannot mix high- and low-noise experts")

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        local_step = None
        if self._easycache_enabled():
            self._validate_easycache_experts()
            if model_input.conditioning.get("_worldfoundry_cfg_parallel_request", False):
                raise ValueError("dual-expert EasyCache requires dense single-rank CFG execution")
            state = self._route_request_state(model_input.request_id, create=False)
            if state is None or state.get("expert_schedule") is None:
                raise ValueError("dual-expert EasyCache requires a prepared full request schedule")
            with self._route_lock:
                local_step = state["expert_schedule"].select(model_input)
                branch = str(model_input.branch)
                previous = state["branch_global_steps"].get(branch)
                if (
                    (previous is None and local_step.local_index != 0)
                    or (previous is not None and model_input.step_index != previous + 1)
                    or (previous is None and branch in {"positive", "unconditional"} and model_input.step_index != 0)
                ):
                    raise ValueError("expert CFG branch steps must be unique and contiguous from a phase boundary")
                state["branch_global_steps"][branch] = model_input.step_index
                state["expert_branch_steps"][(local_step.expert, branch)] = local_step.local_index
        expert = self.expert_for_timestep(model_input.timestep)
        if local_step is not None and local_step.expert != expert:
            raise ValueError("denoiser routing differs from the prepared expert schedule")
        denoiser = self.high_noise if expert == "high-noise" else self.low_noise
        request_id = model_input.request_id
        if isinstance(request_id, str) and request_id.strip():
            self._route_current_request.set(request_id)
            state = self._route_request_state(request_id, create=True)
            assert state is not None
            branch = str(model_input.branch)
            with self._route_lock:
                route_calls = state["route_calls"]
                branch_calls = state["branch_calls"]
                events = state["events"]
                assert isinstance(route_calls, dict)
                assert isinstance(branch_calls, dict)
                assert isinstance(events, list)
                route_calls[expert] = int(route_calls[expert]) + 1
                branch_calls[branch] = int(branch_calls.get(branch, 0)) + 1
                events.append(
                    {
                        "expert": expert,
                        "branch": branch,
                        "step_index": int(model_input.step_index),
                        **(
                            {"local_step_index": local_step.local_index, "local_total_steps": local_step.local_count}
                            if local_step is not None
                            else {}
                        ),
                    }
                )
                state["last_expert"] = expert
        else:
            cfg_parallel_request = bool(
                model_input.conditioning.get(
                    "_worldfoundry_cfg_parallel_request",
                    False,
                )
            )
            if model_input.step_index == 0 and (
                model_input.branch == "positive" or cfg_parallel_request
            ):
                self._route_calls = {"high-noise": 0, "low-noise": 0}
                self._last_expert = None
            self._route_calls[expert] += 1
            self._last_expert = expert
        routed_conditioning = dict(model_input.conditioning)
        routed_conditioning["_worldfoundry_approximate_routed_steps"] = True
        routed_input = model_input.with_updates(conditioning=routed_conditioning)
        if local_step is not None:
            routed_input = routed_input.with_updates(step_index=local_step.local_index, total_steps=local_step.local_count)
        output = denoiser(routed_input)
        return output.with_updates(extras={**dict(output.extras), "expert": expert})

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Finalize both experts and freeze a bounded immutable route receipt."""

        cleanup_errors: list[BaseException] = []
        state = self._route_request_state(request_id, create=False)
        if error is None and state is not None and state.get("expert_schedule") is not None:
            plan = state["expert_schedule"]
            with self._route_lock:
                phase_counts = {step.expert: step.local_count for step in plan.steps}
                unfinished = not state["expert_branch_steps"] or any(
                    last_step != phase_counts[expert] - 1
                    for (expert, branch), last_step in state["expert_branch_steps"].items()
                )
                unfinished = unfinished or any(
                    last_step != len(plan.steps) - 1
                    for branch, last_step in state["branch_global_steps"].items()
                    if branch in {"positive", "unconditional"}
                )
            if unfinished:
                error = ValueError("expert EasyCache request ended before its active CFG phases completed")
                cleanup_errors.append(error)
        for denoiser in (self.high_noise, self.low_noise):
            finalize = getattr(denoiser, "end_request", None)
            if callable(finalize):
                try:
                    finalize(request_id, error=error)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)

        if state is not None:
            report = self.route_receipt(request_id)
            report["finalized"] = True
            report["release_reason"] = "error" if error is not None else "completed"
            if error is not None:
                report["error_type"] = type(error).__name__
            with self._route_lock:
                removed = self._route_requests.pop(request_id, None)
                if removed is not None:
                    self._route_receipt_snapshots[request_id] = json.dumps(
                        report,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    while len(self._route_receipt_snapshots) > 32:
                        oldest_request_id = next(iter(self._route_receipt_snapshots))
                        self._route_receipt_snapshots.pop(oldest_request_id, None)
        self._route_current_request.set(request_id)

        if cleanup_errors:
            primary = cleanup_errors[0]
            add_note = getattr(primary, "add_note", None)
            if callable(add_note):
                for extra in cleanup_errors[1:]:
                    add_note(
                        "additional expert cleanup failed: "
                        f"{type(extra).__name__}: {extra}"
                    )
            raise primary

    def runtime_optimization_report(self) -> dict[str, object]:
        """Aggregate both experts without hiding asymmetric fallbacks."""

        route_receipt = self.route_receipt()
        request_id = route_receipt.get("request_id")
        reports = {}
        for name, denoiser in (
            ("high-noise", self.high_noise),
            ("low-noise", self.low_noise),
        ):
            reporter = getattr(denoiser, "runtime_optimization_report", None)
            if callable(reporter):
                if isinstance(denoiser, WanDenoiser):
                    reports[name] = dict(
                        reporter(
                            request_id=request_id
                            if isinstance(request_id, str)
                            else None
                        )
                    )
                else:
                    reports[name] = dict(reporter())
            else:
                reports[name] = {
                    "requested": {},
                    "effective": {},
                    "fallbacks": [
                        f"{type(denoiser).__name__} exposes no runtime optimization report"
                    ],
                    "quality_tier": "unknown",
                    "runtime": {},
                }

        def merged_mapping(field: str) -> dict[str, object]:
            high = dict(reports["high-noise"].get(field, {}))
            low = dict(reports["low-noise"].get(field, {}))
            merged = {}
            for key in sorted(set(high) | set(low)):
                high_value = high.get(key)
                low_value = low.get(key)
                merged[key] = (
                    high_value
                    if high_value == low_value
                    else {"high-noise": high_value, "low-noise": low_value}
                )
            return merged

        quality_order = {
            "exact": 0,
            "numerically-approximate": 1,
            "approximate": 2,
            "unknown": 3,
        }
        quality_tier = max(
            (str(report.get("quality_tier", "unknown")) for report in reports.values()),
            key=lambda value: quality_order.get(value, 3),
        )
        fallbacks = [
            f"{expert}: {fallback}"
            for expert, report in reports.items()
            for fallback in report.get("fallbacks", [])
        ]
        effective = merged_mapping("effective")
        last_expert = route_receipt.get("last_expert")
        route_calls = dict(route_receipt["route_calls"])
        effective["dual_expert_last_route"] = last_expert or "pending"
        effective["dual_expert_route_calls"] = route_calls
        effective["dual_expert_route_receipt"] = route_receipt
        return {
            "requested": merged_mapping("requested"),
            "effective": effective,
            "fallbacks": fallbacks,
            "quality_tier": quality_tier,
            "runtime": {
                "dual_expert": {
                    "boundary_ratio": self.boundary_ratio,
                    "num_train_timesteps": self.num_train_timesteps,
                    "request_id": route_receipt.get("request_id"),
                    "last_expert": last_expert,
                    "route_calls": route_calls,
                    "route_receipt": route_receipt,
                    "lifecycle": self.route_lifecycle_report(),
                    "experts": reports,
                }
            },
        }


@dataclass(slots=True)
class CausalWanExpertCache:
    """Request-local self- and cross-attention state for one CausalWan expert."""

    kv_cache: list[dict[str, torch.Tensor]]
    cross_attention_cache: list[dict[str, object]]


@dataclass(slots=True)
class CausalWanCacheBundle:
    """Independent high/low expert caches carried by one guidance branch."""

    high_noise: CausalWanExpertCache
    low_noise: CausalWanExpertCache


class FastVideoCausalWanDenoiser:
    """FastVideo CausalWan2.2 dual expert with framework-owned cache state.

    Sampling calls route by shifted model timestep. A cache-commit call runs
    both experts so either branch can serve the next temporal block without a
    stale prefix, matching the released MoE rollout contract.
    """

    variant = "fastvideo-causal-wan2.2"

    def __init__(
        self,
        high_noise: torch.nn.Module,
        low_noise: torch.nn.Module,
        *,
        boundary_ratio: float = 0.875,
        block_size: int = 3,
        cache_window: int = 21,
        sink_size: int = 0,
        text_length: int = 512,
        num_train_timesteps: int = 1000,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not 0.0 < float(boundary_ratio) < 1.0:
            raise ValueError("FastVideo CausalWan boundary_ratio must be in (0, 1)")
        if min(int(block_size), int(cache_window), int(text_length), int(num_train_timesteps)) <= 0:
            raise ValueError("FastVideo CausalWan sizes must be positive")
        if int(sink_size) < 0 or int(sink_size) >= int(cache_window):
            raise ValueError("FastVideo CausalWan sink_size must be in [0, cache_window)")
        self.high_noise = high_noise
        self.low_noise = low_noise
        self.boundary_ratio = float(boundary_ratio)
        self.block_size = int(block_size)
        self.cache_window = int(cache_window)
        self.sink_size = int(sink_size)
        self.text_length = int(text_length)
        self.num_train_timesteps = int(num_train_timesteps)
        self.compute_dtype = compute_dtype

    @staticmethod
    def _model_geometry(model: torch.nn.Module) -> tuple[int, int, int]:
        layers = int(getattr(model, "num_layers"))
        heads = int(getattr(model, "num_heads"))
        dim = int(getattr(model, "dim"))
        if min(layers, heads, dim) <= 0 or dim % heads:
            raise ValueError("CausalWan expert exposes invalid layer/head geometry")
        return layers, heads, dim // heads

    def _create_expert_cache(
        self,
        model: torch.nn.Module,
        *,
        batch_size: int,
        frame_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CausalWanExpertCache:
        layers, heads, head_dim = self._model_geometry(model)
        cache_tokens = self.cache_window * frame_sequence_length
        kv_cache = [
            {
                "k": torch.zeros(
                    batch_size,
                    cache_tokens,
                    heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "v": torch.zeros(
                    batch_size,
                    cache_tokens,
                    heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
            }
            for _ in range(layers)
        ]
        cross_attention_cache: list[dict[str, object]] = [
            {
                "k": torch.zeros(
                    batch_size,
                    self.text_length,
                    heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "v": torch.zeros(
                    batch_size,
                    self.text_length,
                    heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "is_init": False,
            }
            for _ in range(layers)
        ]
        return CausalWanExpertCache(kv_cache, cross_attention_cache)

    def create_kv_cache(
        self,
        *,
        batch_size: int,
        n_views: int,
        latent_frames_per_view: int,
        frame_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CausalWanCacheBundle:
        del latent_frames_per_view
        if int(n_views) != 1:
            raise ValueError("FastVideo CausalWan supports one temporal view per rollout")
        if int(frame_sequence_length) <= 0:
            raise ValueError("frame_sequence_length must be positive")
        kwargs = {
            "batch_size": int(batch_size),
            "frame_sequence_length": int(frame_sequence_length),
            "device": device,
            "dtype": dtype,
        }
        return CausalWanCacheBundle(
            high_noise=self._create_expert_cache(self.high_noise, **kwargs),
            low_noise=self._create_expert_cache(self.low_noise, **kwargs),
        )

    def expert_for_timestep(self, timestep: torch.Tensor) -> str:
        values = timestep.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("FastVideo CausalWan timesteps must be finite")
        high = values >= self.boundary_ratio * float(self.num_train_timesteps)
        if bool(high.all()):
            return "high-noise"
        if bool((~high).all()):
            return "low-noise"
        raise ValueError("one FastVideo CausalWan call cannot mix experts")

    @staticmethod
    def _frame_timesteps(model_input: DenoiserInput) -> torch.Tensor:
        batch, frames = model_input.latents.shape[0], model_input.latents.shape[2]
        values = model_input.timestep.to(
            device=model_input.latents.device,
            dtype=torch.float32,
        ).reshape(-1)
        if values.numel() == 1:
            return values.expand(batch, frames)
        if values.numel() == batch:
            return values[:, None].expand(batch, frames)
        if values.numel() == batch * frames:
            return values.reshape(batch, frames)
        raise ValueError("FastVideo CausalWan timestep must be scalar, per-sample, or per-frame")

    def _run_expert(
        self,
        model: torch.nn.Module,
        cache: CausalWanExpertCache,
        model_input: DenoiserInput,
    ) -> torch.Tensor:
        context = model_input.conditioning.get("context")
        if not isinstance(context, torch.Tensor):
            raise TypeError("FastVideo CausalWan requires tensor 'context' conditioning")
        current_start = int(model_input.conditioning.get("current_start", 0))
        cache_start = int(model_input.conditioning.get("cache_start", current_start))
        latents = model_input.latents
        sequence_length = int(
            latents.shape[2]
            * latents.shape[-2]
            * latents.shape[-1]
            // 4
        )
        autocast_enabled = self.compute_dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(
            device_type=latents.device.type,
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
            sample = model(
                x=latents,
                t=self._frame_timesteps(model_input),
                context=context,
                seq_len=sequence_length,
                kv_cache=cache.kv_cache,
                crossattn_cache=cache.cross_attention_cache,
                current_start=current_start,
                cache_start=cache_start,
            )
        if not isinstance(sample, torch.Tensor):
            raise TypeError("FastVideo CausalWan expert must return a tensor")
        return sample.to(dtype=latents.dtype)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        bundle = model_input.conditioning.get("kv_cache")
        if not isinstance(bundle, CausalWanCacheBundle):
            raise TypeError("FastVideo CausalWan requires a CausalWanCacheBundle")
        if model_input.branch.endswith("cache-commit"):
            high = self._run_expert(self.high_noise, bundle.high_noise, model_input)
            low = self._run_expert(self.low_noise, bundle.low_noise, model_input)
            return DenoiserOutput(
                sample=low,
                extras={"committed_experts": ("high-noise", "low-noise"), "high": high},
            )
        expert = self.expert_for_timestep(model_input.timestep)
        if expert == "high-noise":
            sample = self._run_expert(self.high_noise, bundle.high_noise, model_input)
        else:
            sample = self._run_expert(self.low_noise, bundle.low_noise, model_input)
        return DenoiserOutput(sample=sample, extras={"expert": expert})


def build_fastvideo_causal_wan22_denoiser(
    context: ComponentBuildContext,
) -> FastVideoCausalWanDenoiser:
    """Load both Diffusers-format CausalWan2.2 experts on the causal Wan graph."""

    from ..networks.wan.variants.forcing.causal_forcing import CausalWanModel

    options = context.component_options
    supported = {
        "block_size",
        "boundary_ratio",
        "cache_window",
        "num_train_timesteps",
        "sink_size",
        "text_length",
        "weight_dtype",
    }
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(f"unsupported FastVideo CausalWan denoiser options: {unknown}")
    block_size = int(options.get("block_size", 3))
    cache_window = int(options.get("cache_window", 21))
    sink_size = int(options.get("sink_size", 0))
    # The released pair contains roughly 28B parameters in total.  Defaulting
    # both experts to fp32 ignores the runtime policy and cannot fit on an
    # 80-GiB accelerator; inherit the requested inference dtype unless a
    # caller deliberately asks for a different storage dtype.
    weight_dtype = options.get("weight_dtype", context.policy.dtype)
    if not isinstance(weight_dtype, torch.dtype):
        raise TypeError("FastVideo CausalWan weight_dtype must be a torch.dtype")
    config = {
        "model_type": "t2v",
        "patch_size": (1, 2, 2),
        "text_len": int(options.get("text_length", 512)),
        "in_dim": 16,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "local_attn_size": cache_window,
        "sink_size": sink_size,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    def load(checkpoint_name: str) -> torch.nn.Module:
        model = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=CausalWanModel,
                config=config,
                state_dict_converter=convert_diffusers_wan_transformer_state_dict,
                vram_module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                layer_container="blocks",
            ),
            context.require_checkpoint(checkpoint_name),
            replace(context.policy, dtype=weight_dtype),
        )
        model.num_frame_per_block = block_size
        return model

    return FastVideoCausalWanDenoiser(
        load("high_weights"),
        load("low_weights"),
        boundary_ratio=float(options.get("boundary_ratio", 0.875)),
        block_size=block_size,
        cache_window=cache_window,
        sink_size=sink_size,
        text_length=int(options.get("text_length", 512)),
        num_train_timesteps=int(options.get("num_train_timesteps", 1000)),
        compute_dtype=context.policy.dtype,
    )


def _build_wan22_dual_expert_denoiser(
    context: ComponentBuildContext,
    *,
    config: dict[str, object],
    channel_condition_key: str | None = None,
) -> Wan22DualExpertDenoiser:
    from ...optimizations.plugins import acceleration_policy

    policy = acceleration_policy(context)
    accelerations = policy.options.get("accelerations", {})
    if (
        isinstance(accelerations, Mapping)
        and accelerations.get("easycache") is not None
        and accelerations.get("easycache") is not False
    ):
        incompatible = [
            name
            for name in (
                "cuda_graph",
                "sequence_parallel",
                "sp_degree",
                "device_map",
                "feature_cache",
                "teacache",
                "magcache",
                "adacache",
                "taylorseer",
                "blocktaylorseer",
                "custom",
                "approximate_attention",
                "inplace_residual",
                "static_cross_kv",
            )
            if policy.options.get(name)
        ]
        if policy.compile or policy.offload.mode.value != "none":
            incompatible.append("compile/offload")
        if any(policy.options.get(key, 1) != 1 for key in ("cfg_parallel", "cfg_parallel_degree")):
            incompatible.append("cfg_parallel")
        if any(
            policy.options.get(key) is not None and float(policy.options[key]) != 1.0
            for key in ("cfg_gate_step", "cfg_gate_fraction")
        ):
            incompatible.append("cfg_gate")
        if incompatible:
            raise ValueError(f"dual-expert EasyCache conflicts with unvalidated options: {incompatible}")
    expert_options = {
        key: value
        for key, value in context.component_options.items()
        if key in _WAN_DENOISER_OPTION_KEYS
    }

    def build(name: str, checkpoint_name: str) -> WanDenoiser:
        expert_context = ComponentBuildContext(
            model_id=context.model_id,
            key=ComponentKey(ComponentKind.DENOISER, name),
            policy=context.policy,
            purpose=context.purpose,
            checkpoints={"weights": context.require_checkpoint(checkpoint_name)},
            recipe_options=context.recipe_options,
            component_options=expert_options,
        )
        return _build_wan_denoiser(
            expert_context,
            config=config,
            channel_condition_key=channel_condition_key,
        )

    return Wan22DualExpertDenoiser(
        build("high-noise", "high_weights"),
        build("low-noise", "low_weights"),
        boundary_ratio=float(context.component_options.get("boundary_ratio", 0.875)),
        num_train_timesteps=int(context.component_options.get("num_train_timesteps", 1000)),
    )


def build_wan22_t2v_a14b_dual_denoiser(
    context: ComponentBuildContext,
) -> Wan22DualExpertDenoiser:
    return _build_wan22_dual_expert_denoiser(context, config=WAN22_T2V_A14B_CONFIG)


def build_wan22_i2v_a14b_dual_denoiser(
    context: ComponentBuildContext,
) -> Wan22DualExpertDenoiser:
    return _build_wan22_dual_expert_denoiser(
        context,
        config=WAN22_I2V_A14B_CONFIG,
        channel_condition_key="condition_latents",
    )


def build_wan21_i2v_14b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load the Wan2.1 I2V 14B checkpoint on the shared Wan graph."""

    return _build_wan_denoiser(context, config=WAN21_I2V_14B_CONFIG)


def build_wan22_ti2v_5b_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load Wan2.2 TI2V 5B on the shared Wan graph with token timesteps."""

    return _build_wan_denoiser(context, config=WAN22_TI2V_5B_CONFIG)


def build_skyreels_v3_denoiser(context: ComponentBuildContext) -> WanDenoiser:
    """Load SkyReels-V3 R2V weights into the shared native Wan graph."""

    return _build_wan_denoiser(
        context,
        config=SKYREELS_V3_R2V_14B_CONFIG,
        state_dict_converter=convert_diffusers_wan_transformer_state_dict,
        reference_condition_key="reference_latents",
    )


def _build_wan_denoiser(
    context: ComponentBuildContext,
    *,
    config: dict[str, object],
    state_dict_converter=None,
    reference_condition_key: str | None = None,
    channel_condition_key: str | None = None,
) -> WanDenoiser:
    from ...optimizations.plugins import acceleration_policy

    validate_cuda_graph_options(context)
    feature_cache_config = resolve_wan_feature_cache(context)
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    unknown_options = sorted(set(context.component_options) - _WAN_DENOISER_OPTION_KEYS)
    if unknown_options:
        raise ValueError(f"unsupported Wan denoiser options: {unknown_options}")
    weight_dtype_value = context.component_options.get(
        "weight_dtype",
        context.policy.options.get("dit_weight_dtype", torch.float32),
    )
    weight_dtype = (
        weight_dtype_value
        if isinstance(weight_dtype_value, torch.dtype)
        else parse_torch_dtype(weight_dtype_value, owner="Wan denoiser weights")
    )
    if weight_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Wan denoiser weights support only fp16, bf16, or fp32")
    fused_rope = bool(
        context.component_options.get(
            "fused_rope",
            context.policy.options.get("fused_rope", False),
        )
    )
    inplace_residual_value = context.component_options.get(
        "inplace_residual",
        context.policy.options.get("inplace_residual", False),
    )
    if not isinstance(inplace_residual_value, bool):
        raise TypeError("Wan inplace_residual must be a bool")
    inplace_residual = inplace_residual_value
    rope_precision = str(
        context.component_options.get(
            "rope_precision",
            context.policy.options.get("rope_precision", "fp32"),
        )
    ).strip().lower()
    if rope_precision not in {"fp32", "fp64"}:
        raise ValueError("Wan rope_precision must be 'fp32' or 'fp64'")
    rms_norm_precision = str(
        context.component_options.get(
            "rms_norm_precision",
            context.policy.options.get("rms_norm_precision", "fp32"),
        )
    ).strip().casefold()
    if rms_norm_precision not in {"fp32", "input"}:
        raise ValueError("Wan rms_norm_precision must be 'fp32' or 'input'")
    if fused_rope and rope_precision != "fp32":
        raise ValueError(
            "Wan fused_rope currently supports rope_precision='fp32' only; "
            "use fused_rope=False for the exact fp64 RoPE path"
        )
    if fused_rope and rms_norm_precision == "input":
        raise ValueError(
            "Wan fused_rope currently requires rms_norm_precision='fp32'; "
            "use the non-fused complex RoPE path for LightX2V input-dtype RMS parity"
        )
    adapter_value = context.component_options.get("peft_adapter_path")
    adapter_path = (
        None
        if adapter_value is None
        else _validated_wan_peft_adapter(context, adapter_value)
    )

    def merge_training_adapter(model: torch.nn.Module) -> None:
        if adapter_path is None:
            return
        _merge_wan_peft_adapter(model, adapter_path)

    fastvideo_sla_projection_state: dict[str, object] = {}

    def resolve_approximate_attention_state() -> Mapping[str, object]:
        # The converter below captures only the tiny learned proj_l tensors.
        # Transfer ownership exactly once: the adapter clones FP32 masters and
        # computes immutable fingerprints during construction, so retaining a
        # second checkpoint copy in this closure only extends tensor lifetime.
        resolved = dict(fastvideo_sla_projection_state)
        fastvideo_sla_projection_state.clear()
        return resolved

    def resolve_checkpoint_architecture(checkpoint) -> Mapping[str, object]:
        return resolve_wan_vsa_gate_config(
            checkpoint,
            dim=int(config["dim"]),
            num_layers=int(config["num_layers"]),
        )

    def convert_checkpoint_state_dict(
        checkpoint_state: Mapping[str, object],
    ) -> Mapping[str, object]:
        from ...optimizations.sparse_linear_attention import (
            split_fastvideo_sla_projection_weights,
        )

        model_checkpoint_state, projection_state = (
            split_fastvideo_sla_projection_weights(checkpoint_state)
        )
        fastvideo_sla_projection_state.clear()
        fastvideo_sla_projection_state.update(projection_state)
        if state_dict_converter is not None:
            converted = state_dict_converter(model_checkpoint_state)
        elif any(
            ".attn1." in name or name.startswith("condition_embedder.")
            for name in model_checkpoint_state
        ):
            # FastVideo publishes its VSA-QAT transformer in Diffusers layout,
            # while the ordinary Wan recipes use native keys.  Detect from the
            # tensor namespace so a local QAT checkpoint override remains
            # strict without changing dense official checkpoints.
            converted = convert_diffusers_wan_transformer_state_dict(
                model_checkpoint_state
            )
        else:
            converted = model_checkpoint_state
        return convert_wan_vsa_gate_state_dict(converted)

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanModel,
            config=config,
            config_resolver=resolve_checkpoint_architecture,
            state_dict_converter=convert_checkpoint_state_dict,
            approximate_attention_state_resolver=(
                resolve_approximate_attention_state
            ),
            supports_approximate_attention=True,
            supports_acceleration_plugins=True,
            vram_module_map={
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            layer_container="blocks",
            post_load_hook=merge_training_adapter if adapter_path is not None else None,
        ),
        context.require_checkpoint("weights"),
        replace(acceleration_policy(context), dtype=weight_dtype),
    )
    if not isinstance(model, WanModel):
        raise TypeError(f"expected WanModel, got {type(model).__name__}")
    model._worldfoundry_dit_weight_dtype = weight_dtype
    model._worldfoundry_fused_rope = fused_rope
    model._worldfoundry_rope_precision = rope_precision
    model._worldfoundry_inplace_residual = inplace_residual
    configured_rms_norms = model.set_rms_norm_precision(rms_norm_precision)
    if fused_rope:
        from ...optimizations.fused_rope import install_fused_rope_runtime

        install_fused_rope_runtime(model)
    applied = getattr(model, "_worldfoundry_applied_optimizations", None)
    if applied is not None:
        applied.requested["rope_precision"] = rope_precision
        applied.effective["rope_precision"] = (
            f"complex-{rope_precision}"
            if not fused_rope
            else f"fused-{rope_precision}"
        )
        applied.requested["rms_norm_precision"] = rms_norm_precision
        applied.effective["rms_norm_precision"] = {
            "mode": rms_norm_precision,
            "configured_modules": configured_rms_norms,
        }
        applied.record_model_kernel(
            "fused_rope",
            requested=fused_rope,
            effective=(
                f"hidden_qk_rmsnorm_rope_3d:{rope_precision}"
                if fused_rope
                else "complex_rope"
            ),
            approximate=fused_rope and rope_precision == "fp32",
        )
        applied.record_model_kernel(
            "inplace_residual",
            requested=inplace_residual,
            effective="no-grad-cache-free-dispatch-installed",
        )
    return WanDenoiser(
        model,
        compute_dtype=context.policy.dtype,
        reference_condition_key=reference_condition_key,
        channel_condition_key=channel_condition_key,
        manage_autocast=context.purpose is not BuildPurpose.TRAINING,
        inplace_residual=inplace_residual,
        enable_cuda_graph=resolve_cuda_graph_option(context),
        feature_cache_config=feature_cache_config,
    )


__all__ = [
    "CausalWanCacheBundle",
    "CausalWanExpertCache",
    "FastVideoCausalWanDenoiser",
    "SKYREELS_V2_DF_1P3B_CONFIG",
    "SKYREELS_V3_R2V_14B_CONFIG",
    "WAN21_I2V_14B_CONFIG",
    "WAN21_VAE_I2V_1P3B_CONFIG",
    "WAN22_I2V_A14B_CONFIG",
    "WAN21_T2V_1P3B_CONFIG",
    "WAN21_T2V_14B_CONFIG",
    "WAN22_T2V_A14B_CONFIG",
    "WAN22_TI2V_5B_CONFIG",
    "WanDenoiser",
    "Wan22DualExpertDenoiser",
    "WanModelStateDictConverter",
    "WAN_CIVITAI_CONFIGS_BY_HASH",
    "infer_native_wan_transformer_config",
    "build_fastvideo_causal_wan22_denoiser",
    "build_skyreels_v2_denoiser",
    "build_skyreels_v3_denoiser",
    "build_wan21_i2v_14b_denoiser",
    "convert_diffusers_wan_transformer_state_dict",
    "build_wan21_t2v_14b_denoiser",
    "build_wan22_t2v_a14b_denoiser",
    "build_wan22_t2v_a14b_dual_denoiser",
    "build_wan22_i2v_a14b_dual_denoiser",
    "build_wan21_t2v_1p3b_denoiser",
    "build_wan22_ti2v_5b_denoiser",
]
