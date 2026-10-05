"""Inference contracts and process-wide runtime bootstrap.

**Spec dataclasses** — declarative input/output/checkpoint contracts:

- :class:`InferenceFieldSpec` / :class:`InferenceArtifactSpec` — task I/O fields.
- :class:`InferenceTaskProfile` / :class:`InferenceVariantSpec` / :class:`ModelInferenceSpec` — family catalog.

**Runtime bootstrap** — process-wide torch/SDPA configuration:

- :func:`install_worldfoundry_inference_infra` / :func:`worldfoundry_inference_context`
- :func:`autocast_context` / :func:`compile_module_if_enabled`
- :func:`wrap_runner_for_worldfoundry_core`

The per-model spec catalog deliberately lives outside this package, in
``worldfoundry.runtime.inference_catalog``: core owns model-neutral primitives
and must not encode model identity.
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}
_ORIGINAL_SDPA: Callable[..., Any] | None = None
_PRE_INSTALL_TORCH_STATE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Inference spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WorldFoundryInferenceInfraState:
    """Observable process-wide inference acceleration state.

    Args:
        installed: Whether :func:`install_worldfoundry_inference_infra` ran.
        sdpa_patched: Whether the compatibility SDPA monkey-patch is active.
        attention_backend: Normalized attention backend policy (``auto``,
            ``flash``, ``math``, …). External names such as
            ``flash_attention_2`` are accepted by the installer and mapped to
            the SDPA policy ``auto``.
        matmul_precision: Current float32 matmul precision (``highest`` /
            ``high`` / ``medium``).
        tf32_enabled: Whether CUDA TF32 matmul/cudnn execution is enabled.
    """

    installed: bool = False
    sdpa_patched: bool = False
    attention_backend: str = "auto"
    matmul_precision: str = "high"
    tf32_enabled: bool = True


@dataclass(frozen=True)
class InferenceFieldSpec:
    """User-facing input field contract for one inference task profile."""

    field_id: str
    label: str
    kind: str = "string"
    target: str = "call_kwargs"
    required: bool = False
    default: Any = None
    choices: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this field contract to a JSON-ready mapping."""
        return {
            "field_id": self.field_id,
            "label": self.label,
            "kind": self.kind,
            "target": self.target,
            "required": self.required,
            "default": self.default,
            "choices": list(self.choices),
            "description": self.description,
        }


@dataclass(frozen=True)
class InferenceArtifactSpec:
    """Output artifact contract emitted by an inference task profile."""

    artifact_id: str
    kind: str
    required: bool = False
    preview: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this artifact contract to a JSON-ready mapping."""
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "required": self.required,
            "preview": self.preview,
            "description": self.description,
        }


@dataclass(frozen=True)
class InferenceTaskProfile:
    """Runnable inference task profile for a model family or variant."""

    task_id: str
    label: str
    inputs: tuple[InferenceFieldSpec, ...]
    outputs: tuple[InferenceArtifactSpec, ...]
    description: str = ""
    default_call_kwargs: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this task profile, including nested field/artifact specs."""
        return {
            "task_id": self.task_id,
            "label": self.label,
            "description": self.description,
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "default_call_kwargs": dict(self.default_call_kwargs),
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class InferenceCheckpointRef:
    """Checkpoint reference used by a concrete inference variant."""

    role: str
    uri: str
    required: bool = True
    status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this checkpoint reference (role, URI, required, status)."""
        return {
            "role": self.role,
            "uri": self.uri,
            "required": self.required,
            "status": self.status,
        }


@dataclass(frozen=True)
class InferenceVariantSpec:
    """Concrete checkpoint/runtime variant under a model family."""

    variant_id: str
    label: str
    checkpoints: tuple[InferenceCheckpointRef, ...] = ()
    status: str = "unknown"
    load_kwargs: Mapping[str, Any] = field(default_factory=dict)
    call_kwargs: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def primary_checkpoint_uri(self) -> str:
        """URI of the ``primary`` / ``base`` / ``checkpoint`` role, else the first."""
        if not self.checkpoints:
            return ""
        for checkpoint in self.checkpoints:
            if checkpoint.role in {"primary", "base", "checkpoint"}:
                return checkpoint.uri
        return self.checkpoints[0].uri

    def checkpoint_map(self) -> dict[str, str]:
        """Map each checkpoint role onto its URI."""
        return {checkpoint.role: checkpoint.uri for checkpoint in self.checkpoints}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this variant, including ``model_ref`` and checkpoint map."""
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "status": self.status,
            "model_ref": self.primary_checkpoint_uri,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "checkpoint_map": self.checkpoint_map(),
            "load_kwargs": dict(self.load_kwargs),
            "call_kwargs": dict(self.call_kwargs),
            "aliases": list(self.aliases),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ModelInferenceSpec:
    """Inference contract shared by Studio, CLI, manifests, and eval."""

    model_family_id: str
    display_name: str
    variants: tuple[InferenceVariantSpec, ...]
    tasks: tuple[InferenceTaskProfile, ...]
    default_variant_id: str = "default"
    default_task_id: str = "default"
    aliases: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def variant(self, variant_id: str | None = None) -> InferenceVariantSpec:
        """Look up a variant by id or alias; default is ``default_variant_id``.

        Raises:
            ValueError: No variant matches the (normalized) identifier.
        """
        requested = normalise_infer_id(variant_id or self.default_variant_id)
        for variant in self.variants:
            keys = (variant.variant_id, *variant.aliases)
            if requested in {normalise_infer_id(item) for item in keys}:
                return variant
        supported = ", ".join(variant.variant_id for variant in self.variants)
        raise ValueError(
            f"Unknown inference variant {variant_id!r} for {self.model_family_id}. Choose one of: {supported}"
        )

    def task(self, task_id: str | None = None) -> InferenceTaskProfile:
        """Look up a task profile by id or alias; default is ``default_task_id``.

        Raises:
            ValueError: No task matches the (normalized) identifier.
        """
        requested = normalise_infer_id(task_id or self.default_task_id)
        for task in self.tasks:
            keys = (task.task_id, *task.aliases)
            if requested in {normalise_infer_id(item) for item in keys}:
                return task
        supported = ", ".join(task.task_id for task in self.tasks)
        raise ValueError(f"Unknown inference task {task_id!r} for {self.model_family_id}. Choose one of: {supported}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the family contract, including nested variants and tasks."""
        return {
            "model_family_id": self.model_family_id,
            "display_name": self.display_name,
            "default_variant_id": self.default_variant_id,
            "default_task_id": self.default_task_id,
            "variants": [item.to_dict() for item in self.variants],
            "tasks": [item.to_dict() for item in self.tasks],
            "aliases": list(self.aliases),
            "notes": list(self.notes),
        }


_STATE = WorldFoundryInferenceInfraState()


def normalise_infer_id(value: str | None) -> str:
    """Canonicalize a model-family/variant/task identifier for lookups."""

    return str(value or "").strip().lower().replace("_", "-")


# ---------------------------------------------------------------------------
# Runtime infra bootstrap
# ---------------------------------------------------------------------------


def inference_infra_state() -> WorldFoundryInferenceInfraState:
    """Return the process-global inference infra state (do not mutate casually).

    The returned dataclass is the live singleton updated by
    :func:`install_worldfoundry_inference_infra` and
    :func:`uninstall_worldfoundry_inference_infra`.
    """

    return _STATE


def install_worldfoundry_inference_infra(
    *,
    attention_backend: str | None = None,
    matmul_precision: str | None = None,
    enable_tf32: bool | None = None,
    patch_sdpa: bool | None = None,
) -> WorldFoundryInferenceInfraState:
    """Install WorldFoundry core inference optimizations for this process.

    Environment controls:
    - ``WORLDFOUNDRY_USE_CORE_INFRA=0`` disables installation.
    - ``WORLDFOUNDRY_ATTENTION_BACKEND=auto|flash|cudnn|efficient|math`` selects
      the SDPA backend policy. Attention dispatch backend names (for example
      ``flash_attention_2``, ``sage_attention``, ``xformers``) are also
      accepted; they resolve the SDPA policy to ``auto`` and are honoured by
      ``worldfoundry.core.attention`` instead.
    - ``WORLDFOUNDRY_MATMUL_PRECISION=highest|high|medium`` selects PyTorch
      float32 matmul precision.
    - ``WORLDFOUNDRY_ENABLE_TF32=0`` disables TF32 backend flags.
    - ``WORLDFOUNDRY_PATCH_SDPA=0`` avoids monkey-patching PyTorch SDPA calls.
    - ``WORLDFOUNDRY_DETERMINISTIC=1`` enables ``torch.use_deterministic_algorithms``
      and CuDNN deterministic mode (XC-23).

    Use :func:`uninstall_worldfoundry_inference_infra` (or the
    :func:`worldfoundry_inference_infra_disabled` context manager) to restore
    the original SDPA function and pre-install torch backend flags.
    """

    if _env_flag("WORLDFOUNDRY_USE_CORE_INFRA", default=True) is False:
        return _STATE

    backend = _normalize_attention_backend(
        attention_backend or os.getenv("WORLDFOUNDRY_ATTENTION_BACKEND") or _STATE.attention_backend
    )
    precision = str(matmul_precision or os.getenv("WORLDFOUNDRY_MATMUL_PRECISION") or _STATE.matmul_precision).strip()
    use_tf32 = _env_flag("WORLDFOUNDRY_ENABLE_TF32", default=True) if enable_tf32 is None else bool(enable_tf32)
    should_patch_sdpa = _env_flag("WORLDFOUNDRY_PATCH_SDPA", default=True) if patch_sdpa is None else bool(patch_sdpa)

    _capture_pre_install_torch_state()
    _configure_torch_backends(matmul_precision=precision, enable_tf32=use_tf32)
    if _env_flag("WORLDFOUNDRY_DETERMINISTIC", default=False):
        try:
            from worldfoundry.core.utils.tensors.torch import set_deterministic

            set_deterministic(True)
        except Exception as exc:
            logger.warning("WORLDFOUNDRY_DETERMINISTIC requested but could not be applied: %s", exc)
    if should_patch_sdpa:
        _patch_torch_sdpa()

    _STATE.installed = True
    _STATE.attention_backend = backend
    _STATE.matmul_precision = precision
    _STATE.tf32_enabled = use_tf32
    return _STATE


@contextmanager
def worldfoundry_inference_context() -> Iterator[None]:
    """Run model inference under the shared WorldFoundry core runtime policy.

    Installs process-wide infra if needed, then yields inside
    ``torch.no_grad()`` when torch is importable. A missing torch install
    still enters the context so CPU-only control planes do not crash.
    """

    install_worldfoundry_inference_infra()
    try:
        import torch
    except Exception:
        with nullcontext():
            yield
        return

    with torch.no_grad():
        yield


def autocast_context(
    device: Any,
    *,
    dtype: Any | None = None,
    enabled: bool = True,
) -> Any:
    """Return a CUDA autocast context and a no-op context for non-CUDA devices.

    *device* may be a tensor, ``torch.device``, or device string. CPU /
    NPU / missing-torch paths return :func:`contextlib.nullcontext`.
    """

    try:
        import torch
    except Exception:
        return nullcontext()

    if _device_type(device) != "cuda":
        return nullcontext()

    kwargs: dict[str, Any] = {"device_type": "cuda", "enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.amp.autocast(**kwargs)


def compile_module_if_enabled(
    module: Any,
    *,
    enabled: bool | None = None,
    label: str | None = None,
    backend: str | None = None,
    mode: str | None = None,
    fullgraph: bool | None = None,
    dynamic: bool | None = None,
    options: dict[str, Any] | None = None,
) -> Any:
    """Compile one module with persistent cache and wrapper reuse when enabled.

    Honours ``WORLDFOUNDRY_TORCH_COMPILE`` and the matching
    ``_BACKEND`` / ``_MODE`` / ``_FULLGRAPH`` / ``_DYNAMIC`` /
    ``_STRICT`` env vars when the explicit kwargs are omitted. Already
    compiled wrappers (``_worldfoundry_core_compiled``) are returned
    unchanged. Delegates to
    :func:`worldfoundry.core.execution.compile_cache.compile_module_cached`.
    """

    should_compile = _env_flag("WORLDFOUNDRY_TORCH_COMPILE", default=False) if enabled is None else bool(enabled)
    if not should_compile or getattr(module, "_worldfoundry_core_compiled", False):
        return module

    selected_backend = backend or os.getenv("WORLDFOUNDRY_TORCH_COMPILE_BACKEND")
    selected_mode = mode or os.getenv("WORLDFOUNDRY_TORCH_COMPILE_MODE")
    if fullgraph is not None:
        selected_fullgraph = bool(fullgraph)
    elif os.getenv("WORLDFOUNDRY_TORCH_COMPILE_FULLGRAPH") is not None:
        selected_fullgraph = _env_flag("WORLDFOUNDRY_TORCH_COMPILE_FULLGRAPH", default=False)
    else:
        selected_fullgraph = False
    if dynamic is not None:
        selected_dynamic = bool(dynamic)
    elif os.getenv("WORLDFOUNDRY_TORCH_COMPILE_DYNAMIC") is not None:
        selected_dynamic = _env_flag("WORLDFOUNDRY_TORCH_COMPILE_DYNAMIC", default=False)
    else:
        selected_dynamic = None

    from worldfoundry.core.execution.compile_cache import CompilePolicy, compile_module_cached

    compiled = compile_module_cached(
        module,
        policy=CompilePolicy(
            backend=selected_backend or "inductor",
            mode=selected_mode or "default",
            fullgraph=selected_fullgraph,
            dynamic=selected_dynamic,
        ),
        namespace=label or "core-inference",
        options=options,
        strict=_env_flag("WORLDFOUNDRY_TORCH_COMPILE_STRICT", default=False),
    )
    if compiled is module:
        return module

    try:
        setattr(compiled, "_worldfoundry_core_compiled", True)
        if label is not None:
            setattr(compiled, "_worldfoundry_core_compile_label", label)
    except Exception:
        pass
    return compiled


def wrap_runner_for_worldfoundry_core(runner: Any) -> Any:
    """Wrap a runner instance so ``generate`` always uses core inference infra.

    Installs infra, then replaces ``runner.generate`` with a wrapper that
    enters :func:`worldfoundry_inference_context`. Idempotent: a second
    call returns the same instance. Runners without ``generate`` are
    returned unchanged.
    """

    install_worldfoundry_inference_infra()
    if getattr(runner, "_worldfoundry_core_infra_wrapped", False):
        return runner
    generate = getattr(runner, "generate", None)
    if not callable(generate):
        return runner

    @wraps(generate)
    def generate_with_worldfoundry_core(*args: Any, **kwargs: Any) -> Any:
        """Run the original ``generate`` inside the process-wide inference context."""
        with worldfoundry_inference_context():
            return generate(*args, **kwargs)

    try:
        setattr(runner, "generate", generate_with_worldfoundry_core)
        setattr(runner, "_worldfoundry_core_infra_wrapped", True)
    except Exception:
        return runner
    return runner


def _device_type(device: Any) -> str:
    """Best-effort device kind (``cuda``/``cpu``/…) without requiring torch at import time."""
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        if isinstance(device, torch.Tensor):
            return device.device.type
        if isinstance(device, torch.device):
            return device.type
    return str(device).split(":", maxsplit=1)[0]


def _capture_pre_install_torch_state() -> None:
    """Snapshot restorable torch backend state before the first install."""

    global _PRE_INSTALL_TORCH_STATE

    if _PRE_INSTALL_TORCH_STATE is not None:
        return
    try:
        import torch
    except Exception:
        return

    snapshot: dict[str, Any] = {}
    try:
        snapshot["matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        snapshot["matmul_precision"] = None
    matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    snapshot["matmul_allow_tf32"] = (
        bool(matmul_backend.allow_tf32) if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32") else None
    )
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    snapshot["cudnn_allow_tf32"] = (
        bool(cudnn_backend.allow_tf32) if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32") else None
    )
    _PRE_INSTALL_TORCH_STATE = snapshot


def _configure_torch_backends(*, matmul_precision: str, enable_tf32: bool) -> None:
    """Apply matmul precision and the real TF32 switches; missing torch is a silent no-op."""
    try:
        import torch
    except Exception:
        return

    if hasattr(torch, "set_float32_matmul_precision") and matmul_precision:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception as exc:
            logger.warning("Failed to set float32 matmul precision to %r: %s", matmul_precision, exc)

    # CC-33: the matmul TF32 switch lives at ``torch.backends.cuda.matmul.allow_tf32``
    # (verified on torch 2.7). ``setattr(torch.backends.cuda, "allow_tf32", ...)``
    # only created a dead attribute, so matmul TF32 was previously controlled
    # solely by ``set_float32_matmul_precision`` and could not be disabled.
    # Numerical-behavior impact: with ``enable_tf32=False`` matmul TF32 is now
    # actually disabled (previously it silently stayed on under the default
    # ``matmul_precision="high"``).
    matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        if enable_tf32 and matmul_precision == "highest":
            # An explicit full-precision request wins over the TF32 default;
            # forcing allow_tf32=True would silently demote "highest" to "high".
            logger.info(
                "matmul_precision='highest' requested; leaving matmul TF32 disabled "
                "despite enable_tf32=True."
            )
        else:
            previous = bool(matmul_backend.allow_tf32)
            matmul_backend.allow_tf32 = bool(enable_tf32)
            if previous != bool(enable_tf32):
                logger.info(
                    "Numerical-behavior impact: torch.backends.cuda.matmul.allow_tf32 changed %s -> %s.",
                    previous,
                    bool(enable_tf32),
                )
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        try:
            cudnn_backend.allow_tf32 = bool(enable_tf32)
        except Exception as exc:
            logger.warning("Failed to set torch.backends.cudnn.allow_tf32=%s: %s", bool(enable_tf32), exc)


def _patch_torch_sdpa() -> None:
    """Install the process-wide SDPA wrapper once; skip if already patched or torch is absent."""
    global _ORIGINAL_SDPA

    try:
        import torch.nn.functional as F
    except Exception:
        return

    current = getattr(F, "scaled_dot_product_attention", None)
    if not callable(current):
        return
    if getattr(current, "_worldfoundry_core_sdpa", False):
        _STATE.sdpa_patched = True
        return

    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = current

    def worldfoundry_core_sdpa(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Replacement SDPA: honour the installed backend policy and normalize fully-masked rows."""
        return _call_sdpa_with_backend(query, key, value, *args, **kwargs)

    setattr(worldfoundry_core_sdpa, "_worldfoundry_core_sdpa", True)
    F.scaled_dot_product_attention = worldfoundry_core_sdpa
    _STATE.sdpa_patched = True
    logger.info(
        "Patched torch.nn.functional.scaled_dot_product_attention process-wide with the "
        "WorldFoundry SDPA policy wrapper (adds fully-masked-row normalization). Set "
        "WORLDFOUNDRY_PATCH_SDPA=0 to disable, or call "
        "worldfoundry.core.execution.inference.uninstall_worldfoundry_inference_infra() to restore the original."
    )


def _unpatch_torch_sdpa() -> bool:
    """Restore the original ``F.scaled_dot_product_attention`` if we patched it."""

    try:
        import torch.nn.functional as F
    except Exception:
        return False

    current = getattr(F, "scaled_dot_product_attention", None)
    if _ORIGINAL_SDPA is None or not getattr(current, "_worldfoundry_core_sdpa", False):
        _STATE.sdpa_patched = False
        return False
    F.scaled_dot_product_attention = _ORIGINAL_SDPA
    _STATE.sdpa_patched = False
    logger.info("Restored the original torch.nn.functional.scaled_dot_product_attention.")
    return True


def uninstall_worldfoundry_inference_infra() -> WorldFoundryInferenceInfraState:
    """Undo :func:`install_worldfoundry_inference_infra` process-wide changes.

    Restores the original ``F.scaled_dot_product_attention`` and, when a
    pre-install snapshot exists, the float32 matmul precision and TF32 flags
    captured before the first install.
    """

    global _PRE_INSTALL_TORCH_STATE

    _unpatch_torch_sdpa()
    snapshot = _PRE_INSTALL_TORCH_STATE
    if snapshot is not None:
        try:
            import torch

            if snapshot.get("matmul_precision") is not None:
                torch.set_float32_matmul_precision(snapshot["matmul_precision"])
            matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
            if matmul_backend is not None and snapshot.get("matmul_allow_tf32") is not None:
                matmul_backend.allow_tf32 = snapshot["matmul_allow_tf32"]
            cudnn_backend = getattr(torch.backends, "cudnn", None)
            if cudnn_backend is not None and snapshot.get("cudnn_allow_tf32") is not None:
                cudnn_backend.allow_tf32 = snapshot["cudnn_allow_tf32"]
        except Exception as exc:
            logger.warning("Failed to fully restore pre-install torch backend state: %s", exc)
        _PRE_INSTALL_TORCH_STATE = None
    _STATE.installed = False
    return _STATE


@contextmanager
def worldfoundry_inference_infra_disabled() -> Iterator[None]:
    """Temporarily restore vanilla torch behavior inside the scope.

    Useful for A/B numerical comparisons against the unpatched SDPA. On exit
    the previous configuration is reinstalled if it was installed before.
    """

    was_installed = _STATE.installed
    previous_backend = _STATE.attention_backend
    previous_precision = _STATE.matmul_precision
    previous_tf32 = _STATE.tf32_enabled
    previous_sdpa_patched = _STATE.sdpa_patched
    uninstall_worldfoundry_inference_infra()
    try:
        yield
    finally:
        if was_installed:
            install_worldfoundry_inference_infra(
                attention_backend=previous_backend,
                matmul_precision=previous_precision,
                enable_tf32=previous_tf32,
                patch_sdpa=previous_sdpa_patched,
            )


def _call_sdpa_with_backend(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
    """Call the original SDPA inside the requested kernel context; fall back if the API drifted."""
    if _ORIGINAL_SDPA is None:
        raise RuntimeError("WorldFoundry SDPA patch was installed without an original SDPA function.")

    backends = _resolve_sdpa_backends(_STATE.attention_backend, query)
    if not backends:
        return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

    attn_mask = kwargs.get("attn_mask", args[0] if args else None)

    try:
        import torch
    except Exception:
        return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

    attention = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention, "sdpa_kernel", None) if attention is not None else None
    if not callable(sdpa_kernel):
        output = _ORIGINAL_SDPA(query, key, value, *args, **kwargs)
        return _normalize_fully_masked_sdpa_rows(output, attn_mask, query, key)

    try:
        context = sdpa_kernel(backends=backends, set_priority=True)
    except TypeError:
        try:
            context = sdpa_kernel(backends=backends, set_priority_order=True)
        except TypeError:
            try:
                context = sdpa_kernel(backends=backends)
            except TypeError:
                context = sdpa_kernel(backends)

    with context:
        output = _ORIGINAL_SDPA(query, key, value, *args, **kwargs)
    return _normalize_fully_masked_sdpa_rows(output, attn_mask, query, key)


def _normalize_fully_masked_sdpa_rows(output: Any, attn_mask: Any, query: Any, key: Any) -> Any:
    """Zero fully-masked attention rows so NaNs from empty softmax do not leak into generate."""
    from worldfoundry.core.attention import normalize_fully_masked_rows

    return normalize_fully_masked_rows(output, attn_mask, query, key)


def _resolve_sdpa_backends(policy: str, query: Any) -> list[Any]:
    """Map a policy name to ``SDPBackend`` values; ``auto`` and CPU non-math stay empty (torch default)."""
    if policy == "auto":
        return []

    try:
        import torch
    except Exception:
        return []

    attention = getattr(torch.nn, "attention", None)
    backend_type = getattr(attention, "SDPBackend", None) if attention is not None else None
    if backend_type is None:
        return []
    is_cuda = bool(getattr(getattr(query, "device", None), "type", None) == "cuda")
    if not is_cuda and policy != "math":
        return []

    backend_map = {
        "math": getattr(backend_type, "MATH", None),
        "efficient": getattr(backend_type, "EFFICIENT_ATTENTION", None),
        "cudnn": getattr(backend_type, "CUDNN_ATTENTION", None),
        "flash": getattr(backend_type, "FLASH_ATTENTION", None),
    }
    names = (policy,)
    return [backend for name in names if (backend := backend_map.get(name)) is not None]


_SDPA_POLICY_ALIASES = {
    "": "auto",
    "default": "auto",
    "sdpa": "auto",
    "mem_efficient": "efficient",
    "memory_efficient": "efficient",
    "flash_attention": "flash",
}
_SDPA_POLICIES = frozenset({"auto", "flash", "cudnn", "efficient", "math"})
_DISPATCH_BACKEND_POLICY_WARNED: set[str] = set()


def _normalize_attention_backend(value: str) -> str:
    """Resolve ``WORLDFOUNDRY_ATTENTION_BACKEND`` into an SDPA kernel policy.

    The same environment variable is also read by the attention dispatch layer
    (``worldfoundry.core.attention.backends.probe``) with a wider vocabulary
    (``flash_attention_2``, ``sage_attention``, ``xformers``...). Values from
    that vocabulary are accepted here instead of crashing (CC-31): the SDPA
    kernel policy resolves to ``auto`` and the dispatch layer stays
    responsible for honouring the requested provider.
    """

    normalized = str(value).strip().lower().replace("-", "_")
    normalized = _SDPA_POLICY_ALIASES.get(normalized, normalized)
    if normalized in _SDPA_POLICIES:
        return normalized

    canonical: str | None
    try:
        from worldfoundry.core.attention.backends.probe import normalize_attention_backend

        canonical = normalize_attention_backend(normalized)
    except ValueError:
        raise ValueError(
            "WORLDFOUNDRY_ATTENTION_BACKEND must be an SDPA kernel policy (auto, flash, cudnn, "
            "efficient, math) or an attention dispatch backend name accepted by "
            f"worldfoundry.core.attention.backends.probe.normalize_attention_backend (got {value!r})."
        ) from None
    except ImportError:
        # torch (and therefore the dispatch layer) is unavailable; accept the
        # value so torch-less tooling can still parse configuration.
        canonical = normalized

    if normalized not in _DISPATCH_BACKEND_POLICY_WARNED:
        _DISPATCH_BACKEND_POLICY_WARNED.add(normalized)
        warnings.warn(
            f"WORLDFOUNDRY_ATTENTION_BACKEND={value!r} names an attention dispatch backend "
            f"(canonical: {canonical!r}), not an SDPA kernel policy. The SDPA kernel policy for "
            "the core inference infra resolves to 'auto'; the attention dispatch layer "
            f"(worldfoundry.core.attention) is responsible for honouring {canonical!r}. Use one of "
            "auto/flash/cudnn/efficient/math to pin the SDPA kernel policy explicitly.",
            DeprecationWarning,
            stacklevel=3,
        )
    return "auto"


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean env flag; unrecognized strings keep *default* so typos do not flip policy."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    if normalized in _TRUE_VALUES:
        return True
    return default


__all__ = [
    "InferenceArtifactSpec",
    "InferenceCheckpointRef",
    "InferenceFieldSpec",
    "InferenceTaskProfile",
    "InferenceVariantSpec",
    "ModelInferenceSpec",
    "WorldFoundryInferenceInfraState",
    "autocast_context",
    "compile_module_if_enabled",
    "inference_infra_state",
    "install_worldfoundry_inference_infra",
    "normalise_infer_id",
    "uninstall_worldfoundry_inference_infra",
    "worldfoundry_inference_context",
    "worldfoundry_inference_infra_disabled",
    "wrap_runner_for_worldfoundry_core",
]
