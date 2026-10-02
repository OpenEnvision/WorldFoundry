"""Inference runtime helpers: adaptive batch, OOM detection, token limits.

Responsibility
    Shrink batch on allocator OOM, resolve ``max_new_tokens`` / batch size
    from scoped env vars, and optionally compile a stable evaluator forward.

Boundaries
    Not the CUDA Graph runner — that is ``inference_graph``. Not training
    AMP. Compiler failures are remembered on the callable owner so a flaky
    Inductor kernel is not retried every batch; allocator OOM is never
    swallowed as a compile failure.

Public surface
    :func:`adaptive_batched_inference`, :func:`resolve_inference_batch_size`,
    :func:`resolve_generation_max_new_tokens`, :func:`is_accelerator_out_of_memory`,
    :func:`to`, :func:`arch_invariant_rand`, :func:`get_data_batch_size`,
    :func:`disabled_train`, :func:`timer`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from worldfoundry.core.observability.colors import Color

from worldfoundry.core.utils.tensors.torch import set_random_seed

_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled", "none"}
_TRUE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}


# ──────────────────────────────────────────────────────────────────────────
# OOM / env — scoped WORLDFOUNDRY_EVAL_* overrides beat automatic sizing
# ──────────────────────────────────────────────────────────────────────────


def is_accelerator_out_of_memory(exc: BaseException) -> bool:
    """Return whether an exception reports an accelerator allocation failure."""

    if type(exc).__name__ == "OutOfMemoryError":
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "alloc_failed",
            "cudnn_status_alloc_failed",
        )
    )


def _batch_size_env_names(scope: str | None) -> tuple[str, ...]:
    """Scoped ``WORLDFOUNDRY_EVAL_<SCOPE>_BATCH_SIZE`` then the global fallback."""
    names: list[str] = []
    if scope:
        normalized = "".join(character if character.isalnum() else "_" for character in scope).upper()
        normalized = normalized.strip("_")
        if normalized:
            names.append(f"WORLDFOUNDRY_EVAL_{normalized}_BATCH_SIZE")
    names.append("WORLDFOUNDRY_EVAL_BATCH_SIZE")
    return tuple(names)


def _scoped_eval_env_names(scope: str | None, suffix: str) -> tuple[str, ...]:
    """Build ``WORLDFOUNDRY_EVAL_<SCOPE>_<SUFFIX>`` then the unscope fallback."""
    names: list[str] = []
    if scope:
        normalized = "".join(character if character.isalnum() else "_" for character in scope).upper()
        normalized = normalized.strip("_")
        if normalized:
            names.append(f"WORLDFOUNDRY_EVAL_{normalized}_{suffix}")
    names.append(f"WORLDFOUNDRY_EVAL_{suffix}")
    return tuple(names)


def _resolve_batch_size_buckets(
    batch_size: int,
    *,
    scope: str | None,
    explicit: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve graph/compile-friendly batch buckets bounded by ``batch_size``."""

    values: Sequence[int] | None = explicit
    source: str | None = None
    if values is None:
        for env_name in _scoped_eval_env_names(scope, "BATCH_BUCKETS"):
            configured = os.getenv(env_name, "").strip()
            if not configured:
                continue
            source = env_name
            if configured.casefold() in {"off", "false", "none", "disabled"}:
                return ()
            try:
                values = tuple(int(item.strip()) for item in configured.split(",") if item.strip())
            except ValueError as exc:
                raise ValueError(f"{env_name} must be a comma-separated list of positive integers") from exc
            break
    if values is None:
        return ()

    buckets: set[int] = set()
    for item in values:
        value = int(item)
        if value < 1:
            label = source or "batch_size_buckets"
            raise ValueError(f"{label} must contain only positive integers")
        if value <= batch_size:
            buckets.add(value)
    return tuple(sorted(buckets))


def _scoped_eval_value(scope: str | None, suffix: str) -> str | None:
    """First non-empty scoped env value, or ``None`` if unset."""
    for env_name in _scoped_eval_env_names(scope, suffix):
        value = os.getenv(env_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _scoped_eval_flag(scope: str | None, suffix: str, *, default: bool) -> bool:
    """Parse a scoped boolean env; unknown spellings raise :exc:`ValueError`."""
    value = _scoped_eval_value(scope, suffix)
    if value is None:
        return default
    normalized = value.casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    names = ", ".join(_scoped_eval_env_names(scope, suffix))
    raise ValueError(f"{names} must be a boolean value")


def _compiler_runtime_failure(exc: BaseException) -> bool:
    """Detect Dynamo / Inductor / Triton compile errors (not allocator OOM)."""
    module = type(exc).__module__.casefold()
    if module.startswith(("torch._dynamo", "torch._inductor", "triton")):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "backendcompilerfailed",
            "backend compiler failed",
            "torch.compile",
            "torchdynamo",
            "inductorerror",
        )
    )


def _compile_failure_owner(function: Callable[..., Any]) -> Any:
    """Bound-method owner if present, else the callable itself (failure cache key)."""
    owner = getattr(function, "__self__", None)
    return function if owner is None else owner


def _has_persistent_compile_owner(function: Callable[..., Any]) -> bool:
    """Return whether a compiled wrapper can be reused beyond this batch call."""

    if getattr(function, "__self__", None) is not None:
        return True
    if hasattr(function, "_modules") and hasattr(function, "forward"):
        return True
    return not bool(getattr(function, "__closure__", None))


def _power_of_two_batch_buckets(maximum: int) -> tuple[int, ...]:
    """``1, 2, 4, …, maximum`` so compile/Graph capture sees a small shape set."""
    buckets: list[int] = []
    value = 1
    while value < maximum:
        buckets.append(value)
        value *= 2
    buckets.append(maximum)
    return tuple(buckets)


def _compile_eval_forward(
    forward: Callable[[torch.Tensor], torch.Tensor],
    *,
    scope: str | None,
    mode: str | None,
    strict: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile one stable-shape evaluator and fall back on compiler failures."""

    selected_mode = mode or _scoped_eval_value(scope, "COMPILE_MODE") or "reduce-overhead"
    selected_backend = _scoped_eval_value(scope, "COMPILE_BACKEND") or "inductor"
    failure_key = (scope or "batched", selected_backend, selected_mode)
    owner = _compile_failure_owner(forward)
    failures = getattr(owner, "_worldfoundry_eval_compile_failures", ())
    if failure_key in failures:
        return forward

    from worldfoundry.core.execution.compile_cache import CompilePolicy, compile_callable_cached

    compiled = compile_callable_cached(
        forward,
        policy=CompilePolicy(
            backend=selected_backend,
            mode=selected_mode,
            fullgraph=False,
            dynamic=False,
        ),
        namespace=f"evaluation-{scope or 'batched'}",
        strict=strict,
    )
    if compiled is forward:
        return forward

    compiled_active = True

    def compiled_or_eager(batch: torch.Tensor) -> torch.Tensor:
        """Run compiled; on a non-OOM compiler error, remember and fall back."""
        nonlocal compiled_active
        if not compiled_active:
            return forward(batch)
        try:
            return compiled(batch)
        except BaseException as exc:
            if (
                strict
                or is_accelerator_out_of_memory(exc)
                or isinstance(exc, (KeyboardInterrupt, SystemExit))
                or not _compiler_runtime_failure(exc)
            ):
                raise
            compiled_active = False
            remembered = set(getattr(owner, "_worldfoundry_eval_compile_failures", ()))
            remembered.add(failure_key)
            try:
                setattr(owner, "_worldfoundry_eval_compile_failures", remembered)
            except Exception:
                pass
            return forward(batch)

    return compiled_or_eager


def resolve_generation_max_new_tokens(default: int, *, scope: str | None = None) -> int:
    """Resolve a bounded generation budget with optional per-evaluator overrides.

    ``WORLDFOUNDRY_EVAL_<SCOPE>_MAX_NEW_TOKENS`` takes precedence over
    ``WORLDFOUNDRY_EVAL_MAX_NEW_TOKENS``. This keeps concise judge prompts from
    running for thousands of tokens while allowing unusually verbose models to
    opt back into a larger budget without code changes.
    """

    default = int(default)
    if default < 1:
        raise ValueError("default must be a positive integer")
    for env_name in _scoped_eval_env_names(scope, "MAX_NEW_TOKENS"):
        configured = os.getenv(env_name, "").strip()
        if not configured:
            continue
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a positive integer") from exc
        if value < 1:
            raise ValueError(f"{env_name} must be a positive integer")
        return value
    return default


def _coerce_torch_device(device: torch.device | str | int) -> torch.device:
    """Treat a bare int as ``cuda:<n>`` so env / CLI device ids stay consistent."""
    return torch.device("cuda", device) if isinstance(device, int) else torch.device(device)


def resolve_inference_batch_size(
    default_batch_size: int,
    *,
    device: torch.device | str | int | None = None,
    scope: str | None = None,
    minimum: int = 1,
    maximum: int | None = None,
    reference_free_memory_gib: float = 24.0,
) -> int:
    """Choose a capability-neutral inference batch size from free device memory.

    The default is treated as the batch size for roughly 24 GiB of free memory.
    Smaller devices start conservatively, while 80 GiB and newer high-memory
    accelerators can use larger batches.  Scoped and global environment values
    take precedence over automatic sizing.
    """

    default_batch_size = int(default_batch_size)
    minimum = int(minimum)
    configured_maximum = None if maximum is None else int(maximum)
    automatic_maximum = configured_maximum if configured_maximum is not None else max(default_batch_size * 4, minimum)
    if default_batch_size < 1:
        raise ValueError("default_batch_size must be positive")
    if minimum < 1:
        raise ValueError("minimum must be positive")
    if automatic_maximum < minimum:
        raise ValueError("maximum must be greater than or equal to minimum")
    if reference_free_memory_gib <= 0:
        raise ValueError("reference_free_memory_gib must be positive")

    for env_name in _batch_size_env_names(scope):
        configured = os.getenv(env_name, "").strip()
        if not configured:
            continue
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a positive integer") from exc
        if value < 1:
            raise ValueError(f"{env_name} must be a positive integer")
        value = max(value, minimum)
        return min(value, configured_maximum) if configured_maximum is not None else value

    resolved_device = (
        _coerce_torch_device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        return min(max(default_batch_size, minimum), automatic_maximum)

    try:
        free_bytes, _ = torch.cuda.mem_get_info(resolved_device)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return min(max(default_batch_size, minimum), automatic_maximum)
    reference_bytes = float(reference_free_memory_gib) * (1024.0**3)
    memory_scaled = int(default_batch_size * (float(free_bytes) / reference_bytes))
    return min(max(memory_scaled, minimum), automatic_maximum)


def adaptive_batched_inference(
    inputs: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_size: int,
    device: torch.device | str | int | None = None,
    output_device: torch.device | str | int | None = None,
    minimum_batch_size: int = 1,
    batch_size_buckets: Sequence[int] | None = None,
    pad_to_batch_size: bool = False,
    scope: str | None = None,
    compile_forward: bool | None = None,
    compile_mode: str | None = None,
    persistent_forward: bool = False,
) -> torch.Tensor:
    """Run tensor inference in bounded, optionally shape-stable batches.

    Only the failed slice is retried.  Successful outputs retain input order,
    and an allocation failure at batch size one is propagated unchanged.
    ``pad_to_batch_size`` repeats the final real item and trims its outputs;
    this is intended for independent per-item encoders so Inductor and CUDA
    Graphs can reuse a small set of static leading dimensions.
    ``persistent_forward`` declares that a closure is reused across multiple
    calls, allowing throughput mode to retain one compiled wrapper. It does not
    enable compilation by itself.
    """

    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"inputs must be a torch.Tensor, got {type(inputs)!r}")
    if inputs.ndim == 0 or len(inputs) == 0:
        raise ValueError("inputs must contain at least one batch item")
    if not callable(forward):
        raise TypeError("forward must be callable")
    batch_size = int(batch_size)
    minimum_batch_size = int(minimum_batch_size)
    if batch_size < 1 or minimum_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if minimum_batch_size > batch_size:
        raise ValueError("minimum_batch_size cannot exceed batch_size")

    target_device = inputs.device if device is None else _coerce_torch_device(device)
    target_output_device = None if output_device is None else _coerce_torch_device(output_device)
    should_compile = (
        _scoped_eval_flag(scope, "TORCH_COMPILE", default=False) if compile_forward is None else bool(compile_forward)
    )
    compile_candidate = (
        should_compile
        and target_device.type == "cuda"
        and torch.cuda.is_available()
        and (
            compile_forward is True
            or bool(persistent_forward)
            or _has_persistent_compile_owner(forward)
        )
    )
    if compile_candidate:
        forward = _compile_eval_forward(
            forward,
            scope=scope,
            mode=compile_mode,
            strict=_scoped_eval_flag(scope, "TORCH_COMPILE_STRICT", default=False),
        )
        if batch_size_buckets is None and _scoped_eval_value(scope, "BATCH_BUCKETS") is None:
            # Static powers-of-two mirror serving-engine capture buckets while
            # the exact configured maximum keeps padding below a 2x bound.
            batch_size_buckets = _power_of_two_batch_buckets(batch_size)
    desired_batch_size = min(batch_size, len(inputs))
    buckets = _resolve_batch_size_buckets(
        batch_size,
        scope=scope,
        explicit=batch_size_buckets,
    )
    current_batch_size = desired_batch_size
    if pad_to_batch_size and buckets:
        current_batch_size = next(
            (bucket for bucket in buckets if bucket >= desired_batch_size),
            desired_batch_size,
        )
    offset = 0
    final_output: torch.Tensor | None = None

    while offset < len(inputs):
        count = min(current_batch_size, len(inputs) - offset)
        execution_count = current_batch_size if pad_to_batch_size else count
        batch: torch.Tensor | None = None
        output: torch.Tensor | None = None
        retry_batch_size: int | None = None
        try:
            batch = inputs.narrow(0, offset, count).to(target_device, non_blocking=True)
            if execution_count > count:
                padding = batch[-1:].expand(execution_count - count, *batch.shape[1:])
                batch = torch.cat((batch, padding), dim=0)
            with torch.inference_mode():
                output = forward(batch)
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"forward must return a torch.Tensor, got {type(output)!r}")
            if output.ndim == 0:
                output = output.reshape(1)
            if len(output) != execution_count:
                raise ValueError(
                    f"forward must preserve the leading batch dimension: expected {execution_count}, got {len(output)}"
                )
            output = output.detach().narrow(0, 0, count)
            if target_output_device is not None:
                output = output.to(target_output_device)
            if final_output is None:
                final_output = output.new_empty((len(inputs), *output.shape[1:]))
            elif (
                final_output.shape[1:] != output.shape[1:]
                or final_output.dtype != output.dtype
                or final_output.device != output.device
            ):
                raise ValueError(
                    "forward output shape, dtype, and device must remain stable across batches"
                )
            # reduce-overhead may replay a CUDA Graph whose output storage is
            # overwritten by the next call. Copy directly into the final owning
            # buffer before replay; this replaces both the old per-batch clone
            # and the final torch.cat with one transfer.
            final_output.narrow(0, offset, count).copy_(output)
        except Exception as exc:
            if not is_accelerator_out_of_memory(exc) or execution_count <= minimum_batch_size:
                raise
            batch = None
            output = None
            smaller_buckets = tuple(bucket for bucket in buckets if minimum_batch_size <= bucket < execution_count)
            retry_batch_size = (
                smaller_buckets[-1] if smaller_buckets else max((execution_count + 1) // 2, minimum_batch_size)
            )
        if retry_batch_size is not None:
            # Leave the exception handler before releasing allocator blocks so
            # Python no longer retains failed-forward tensors via traceback.
            current_batch_size = retry_batch_size
            if target_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        offset += count

    if final_output is None:  # pragma: no cover - non-empty inputs are enforced above.
        raise AssertionError("adaptive inference produced no output")
    return final_output


# ──────────────────────────────────────────────────────────────────────────
# Tree move / RNG / batch size — eval helpers that stay out of the denoise loop
# ──────────────────────────────────────────────────────────────────────────


def to(data: Any, *, device=None, dtype=None, memory_format=torch.preserve_format) -> Any:
    """Recursively ``.to`` tensor leaves; skip channels-last on the wrong rank.

    Mappings and non-string sequences keep their type. Non-tensor leaves
    pass through. ``channels_last`` / ``channels_last_3d`` are ignored unless
    rank is 4 / 5 so NHWC is never applied to a 2-D feature.
    """
    if isinstance(data, torch.Tensor):
        if memory_format == torch.channels_last and data.dim() != 4:
            memory_format = torch.preserve_format
        if memory_format == torch.channels_last_3d and data.dim() != 5:
            memory_format = torch.preserve_format
        return data.to(device=device, dtype=dtype, memory_format=memory_format, non_blocking=True)
    if isinstance(data, Mapping):
        return type(data)(
            {key: to(value, device=device, dtype=dtype, memory_format=memory_format) for key, value in data.items()}
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return type(data)(to(value, device=device, dtype=dtype, memory_format=memory_format) for value in data)
    return data


def arch_invariant_rand(shape, dtype: torch.dtype, device, seed: int | None = None) -> torch.Tensor:
    """Host ``RandomState`` Gaussian so CPU vs CUDA generators cannot diverge."""
    array = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(array).to(dtype=dtype, device=device)


def get_data_batch_size(data: Any) -> int:
    """Infer leading batch length from a tensor, mapping, or sequence tree.

    Failure: :exc:`ValueError` when no tensor / sequence leaf is found.
    """
    if isinstance(data, torch.Tensor):
        return len(data)
    if isinstance(data, Mapping):
        for value in data.values():
            try:
                return get_data_batch_size(value)
            except ValueError:
                pass
    if isinstance(data, Sequence) and data:
        return len(data) if isinstance(data[0], torch.Tensor) else get_data_batch_size(data[0])
    raise ValueError("unable to infer batch size")


def disabled_train(self, mode: bool = True):
    """No-op ``train()`` replacement so eval modules cannot flip BN / dropout."""
    del mode
    return self


@contextmanager
def timer(name: str):
    """Log wall time of a block; import logger lazily to stay CPU-importable."""
    started = time.perf_counter()
    yield
    from worldfoundry.core.distributed.runtime.logging import log

    log.info("{} takes {:.4f}s", name, time.perf_counter() - started)


__all__ = [
    "Color",
    "adaptive_batched_inference",
    "arch_invariant_rand",
    "disabled_train",
    "get_data_batch_size",
    "is_accelerator_out_of_memory",
    "resolve_generation_max_new_tokens",
    "resolve_inference_batch_size",
    "set_random_seed",
    "timer",
    "to",
]
