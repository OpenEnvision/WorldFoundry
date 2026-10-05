"""Framework-owned execution and memory policies for diffusion components.

Public surface consumed by loaders and recipes.  :class:`RuntimePolicy` is
defined in ``worldfoundry.core.model_loading.policy`` and re-exported here
so diffusion code shares one placement vocabulary with other model families.

Parsers in :mod:`.policy` are the only place public option strings
(``bf16``, ``block``, ``balanced``) become typed policy objects.  Kernel
transforms (QKV fusion, TeaCache, STA/VSA) live in sibling modules and are
applied by :class:`~..loaders.module.NativeModuleLoader`, not by the runner.
"""

from .sparse_linear_attention import (
    FastVideoSLAAdapter,
    FastVideoSLAConfig,
    FastVideoSLAContractError,
    FastVideoSLALayerSpec,
    FastVideoSLAUnavailableError,
    resolve_fastvideo_sla_projection_weights,
    split_fastvideo_sla_projection_weights,
)
from .sparse_mask_attention import (
    LightX2VSparseAdapter,
    LightX2VSparseConfig,
    LightX2VSparseResult,
    LightX2VSparseUnavailableError,
    parse_lightx2v_sparse,
)
from .policy import (
    AttentionBackend,
    OffloadMode,
    OffloadPolicy,
    QuantizationMode,
    QuantizationPolicy,
    RuntimePolicy,
    parse_attention_backend,
    parse_device_map,
    parse_offload_policy,
    parse_quantization_policy,
    parse_torch_dtype,
)

__all__ = [
    "AttentionBackend",
    "FastVideoSLAAdapter",
    "FastVideoSLAConfig",
    "FastVideoSLAContractError",
    "FastVideoSLALayerSpec",
    "FastVideoSLAUnavailableError",
    "LightX2VSparseAdapter",
    "LightX2VSparseConfig",
    "LightX2VSparseResult",
    "LightX2VSparseUnavailableError",
    "OffloadMode",
    "OffloadPolicy",
    "QuantizationMode",
    "QuantizationPolicy",
    "RuntimePolicy",
    "parse_attention_backend",
    "parse_device_map",
    "parse_offload_policy",
    "parse_quantization_policy",
    "parse_lightx2v_sparse",
    "parse_torch_dtype",
    "resolve_fastvideo_sla_projection_weights",
    "split_fastvideo_sla_projection_weights",
]
