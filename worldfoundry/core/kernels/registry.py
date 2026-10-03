"""Runtime registry for optional in-tree accelerator kernels.

PyTorch remains the semantic source of truth. A Triton / native kernel is
chosen per workload signature; a failed signature is quarantined so one
bad shape does not disable the backend globally. CUDA OOM is never turned
into a silent fallback — retrying usually raises peak memory.

Callers use :mod:`worldfoundry.core.kernels.diffusion` public ops, not
this registry directly. :func:`kernel_dispatch_report` /
``clear_kernel_dispatch_cache`` (on the diffusion facade) reset the cache
after a device change or a hot-reloaded extension.

Not this module:
    Persistent winner JSON lives in :mod:`.autotune_cache`. Capability
    profiles live in :mod:`.capabilities`. Operator-facing thresholds live
    in :mod:`.diffusion`.

Public surface:

- :data:`KERNEL_REGISTRY` — process-wide candidate table.
- :func:`kernel_dispatch_receipt_scope` — request-local execution receipt.
- :func:`kernel_autotune_enabled` / :func:`kernel_dispatch_report` /
  :func:`clear_kernel_dispatch_cache`.
- :class:`KernelNotSupported` — wrapper-raised contract miss (quarantined).
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable

KernelCallable = Callable[..., Any]
KernelPredicate = Callable[..., bool]

_KERNEL_DISPATCH_RECEIPT: ContextVar[MutableMapping[str, Any] | None] = ContextVar(
    "worldfoundry_kernel_dispatch_receipt",
    default=None,
)


# ──────────────────────────────────────────────────────────────────────────
# Request-local receipts — prove a cached selection actually executed
# ──────────────────────────────────────────────────────────────────────────


@contextmanager
def kernel_dispatch_receipt_scope(
    receipt: MutableMapping[str, Any],
) -> Iterator[None]:
    """Capture the concrete implementation selected by logical kernel calls.

    The registry's selection cache is useful diagnostic state, but by itself it
    does not prove that a request executed a cached implementation.  A model
    optimization can use this request-local scope around its public kernel call
    and retain an execution receipt without changing tensor return values.
    Context variables keep concurrent Workspace jobs isolated.
    """

    token = _KERNEL_DISPATCH_RECEIPT.set(receipt)
    try:
        yield
    finally:
        _KERNEL_DISPATCH_RECEIPT.reset(token)


def _publish_dispatch_receipt(
    *,
    op: str,
    implementation: str,
    backend: str,
    accelerated: bool,
    fallback: bool,
    cache_hit: bool,
    failures: list[str],
    quarantined: list[str],
    reason: str | None = None,
) -> None:
    """Append one dispatch event when a request-local receipt scope is active.

    Failure: a non-list ``dispatches`` bag raises :class:`TypeError` so a
    corrupted receipt is not silently dropped.
    """

    receipt = _KERNEL_DISPATCH_RECEIPT.get()
    if receipt is None:
        return
    dispatches = receipt.setdefault("dispatches", [])
    if not isinstance(dispatches, list):
        raise TypeError("kernel dispatch receipt 'dispatches' must be a list")
    dispatches.append(
        {
            "op": str(op),
            "implementation": str(implementation),
            "backend": str(backend),
            "accelerated": bool(accelerated),
            "fallback": bool(fallback),
            "cache_hit": bool(cache_hit),
            "failures": list(failures),
            "quarantined": list(quarantined),
            "reason": reason,
        }
    )


def _outputs_equivalent(reference: Any, candidate: Any) -> bool:
    """Conservatively compare exact-fallback and accelerator outputs."""

    import torch

    if isinstance(reference, torch.Tensor):
        if not isinstance(candidate, torch.Tensor):
            return False
        if (
            reference.shape != candidate.shape
            or reference.dtype != candidate.dtype
            or reference.device != candidate.device
        ):
            return False
        if not (reference.is_floating_point() or reference.is_complex()):
            return bool(torch.equal(reference, candidate))
        if reference.dtype == torch.bfloat16:
            rtol, atol = 2e-2, 2e-2
        elif reference.dtype == torch.float16:
            rtol, atol = 5e-3, 5e-3
        elif reference.dtype in {torch.float32, torch.complex64}:
            rtol, atol = 3e-4, 3e-5
        else:
            rtol, atol = 1e-5, 1e-7
        try:
            return bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol, equal_nan=True))
        except RuntimeError:
            return False
    if isinstance(reference, (tuple, list)):
        return (
            type(reference) is type(candidate)
            and len(reference) == len(candidate)
            and all(
                _outputs_equivalent(expected, actual) for expected, actual in zip(reference, candidate, strict=True)
            )
        )
    if isinstance(reference, dict):
        return (
            isinstance(candidate, dict)
            and reference.keys() == candidate.keys()
            and all(_outputs_equivalent(reference[key], candidate[key]) for key in reference)
        )
    try:
        return bool(reference == candidate)
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Candidate table — priority then name; autotune is opt-in per registration
# ──────────────────────────────────────────────────────────────────────────


class KernelNotSupported(RuntimeError):
    """Raised by a kernel wrapper when a workload is outside its contract."""


@dataclass(frozen=True)
class KernelCandidate:
    """One implementation of a logical operator."""

    op: str
    backend: str
    name: str
    priority: int
    implementation: KernelCallable
    predicate: KernelPredicate
    autotune: bool = False


class KernelRegistry:
    """Select accelerator kernels while retaining an explicit fallback."""

    def __init__(self, *, failure_limit: int = 1024, selection_limit: int = 4096) -> None:
        """Bound quarantine and selection caches so a long process cannot grow without limit."""

        self._candidates: dict[str, list[KernelCandidate]] = defaultdict(list)
        self._failures: set[tuple[object, ...]] = set()
        self._failure_order: deque[tuple[object, ...]] = deque()
        self._failure_messages: dict[tuple[object, ...], str] = {}
        self._failure_limit = int(failure_limit)
        self._selection_cache: OrderedDict[tuple[object, ...], KernelCandidate | None] = OrderedDict()
        self._autotune_records: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
        self._selection_limit = int(selection_limit)
        self._lock = threading.Lock()

    def register(
        self,
        op: str,
        *,
        backend: str,
        name: str,
        implementation: KernelCallable,
        predicate: KernelPredicate,
        priority: int = 0,
        autotune: bool = False,
    ) -> KernelCandidate:
        """Register one implementation; duplicate ``name`` for the same op is refused."""

        candidate = KernelCandidate(
            op=str(op),
            backend=str(backend),
            name=str(name),
            priority=int(priority),
            implementation=implementation,
            predicate=predicate,
            autotune=bool(autotune),
        )
        entries = self._candidates[candidate.op]
        if any(item.name == candidate.name for item in entries):
            raise ValueError(f"kernel {candidate.name!r} is already registered for {candidate.op!r}")
        entries.append(candidate)
        entries.sort(key=lambda item: (-item.priority, item.name))
        return candidate

    @staticmethod
    def _invoke_candidate(
        candidate: KernelCandidate,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a CUDA candidate with the input tensor's device made current.

        Workspace can execute jobs for several GPUs in different threads of
        one process. Triton launches use the calling thread's current CUDA
        context, which is not necessarily the device that owns the input
        pointer. PyTorch operators normally install their own device guard;
        optional custom candidates need the same protection here.
        """

        import torch

        tensor = next(
            (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor) and value.is_cuda),
            None,
        )
        if tensor is None:
            return candidate.implementation(*args, **kwargs)
        with torch.cuda.device(tensor.device):
            return candidate.implementation(*args, **kwargs)

    def dispatch(
        self,
        op: str,
        fallback: KernelCallable,
        *args: Any,
        signature: tuple[object, ...],
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Select a candidate or *fallback* for one workload *signature*.

        Cache key is ``(op, requested backend, autotune flag, *signature)``.
        Optional-kernel failures quarantine that signature; CUDA OOM,
        :class:`KeyboardInterrupt`, and :class:`SystemExit` always re-raise.
        """

        requested = (backend if backend is not None else os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto")).strip().casefold() or "auto"
        failures: list[str] = []
        quarantined: list[str] = []
        if requested in {"torch", "pytorch", "native", "off", "disabled"}:
            result = fallback(*args, **kwargs)
            _publish_dispatch_receipt(
                op=op,
                implementation="torch",
                backend="torch",
                accelerated=False,
                fallback=True,
                cache_hit=False,
                failures=failures,
                quarantined=quarantined,
                reason="torch backend explicitly requested",
            )
            return result

        selection_key = (op, requested, kernel_autotune_enabled(), *signature)
        cached, cache_hit = self._cached_selection(selection_key)
        skipped: set[str] = set()
        if cache_hit:
            if cached is None:
                result = fallback(*args, **kwargs)
                _publish_dispatch_receipt(
                    op=op,
                    implementation="torch",
                    backend="torch",
                    accelerated=False,
                    fallback=True,
                    cache_hit=True,
                    failures=failures,
                    quarantined=quarantined,
                    reason="cached torch selection",
                )
                return result
            failure_key = (cached.name, *signature)
            if failure_key not in self._failures:
                try:
                    result = self._invoke_candidate(cached, args, kwargs)
                    _publish_dispatch_receipt(
                        op=op,
                        implementation=cached.name,
                        backend=cached.backend,
                        accelerated=True,
                        fallback=False,
                        cache_hit=True,
                        failures=failures,
                        quarantined=quarantined,
                    )
                    return result
                except BaseException as exc:
                    if _is_out_of_memory(exc) or isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    if not _is_optional_kernel_failure(exc):
                        raise
                    self._drop_selection(selection_key)
                    if cached.autotune:
                        self._invalidate_persistent_candidate(cached, signature, args, kwargs)
                    self._remember_failure(failure_key, exc)
                    warnings.warn(
                        f"Kernel {cached.name!r} failed for this workload; using the next eligible fallback: {exc}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    failures.append(f"{cached.name}: {type(exc).__name__}: {exc}")
                    skipped.add(cached.name)
            else:
                quarantined.append(cached.name)

        for candidate in self._candidates.get(op, ()):
            if candidate.name in skipped:
                continue
            if requested != "auto" and requested not in {candidate.backend.casefold(), candidate.name.casefold()}:
                continue
            failure_key = (candidate.name, *signature)
            if failure_key in self._failures:
                quarantined.append(candidate.name)
                continue
            try:
                if not candidate.predicate(*args, **kwargs):
                    continue
                if requested == "auto" and candidate.autotune and kernel_autotune_enabled():
                    selected, result, record = self._autotune_candidate(
                        candidate,
                        fallback,
                        args,
                        kwargs,
                        signature,
                    )
                    self._remember_selection(selection_key, selected)
                    self._remember_autotune(selection_key, record)
                    if selected is None:
                        _publish_dispatch_receipt(
                            op=op,
                            implementation="torch",
                            backend="torch",
                            accelerated=False,
                            fallback=True,
                            cache_hit=bool(record.get("cache_hit")),
                            failures=failures,
                            quarantined=quarantined,
                            reason=f"autotune winner: {record.get('winner', 'torch')}",
                        )
                    else:
                        _publish_dispatch_receipt(
                            op=op,
                            implementation=selected.name,
                            backend=selected.backend,
                            accelerated=True,
                            fallback=False,
                            cache_hit=bool(record.get("cache_hit")),
                            failures=failures,
                            quarantined=quarantined,
                        )
                    return result
                result = self._invoke_candidate(candidate, args, kwargs)
                self._remember_selection(selection_key, candidate)
                _publish_dispatch_receipt(
                    op=op,
                    implementation=candidate.name,
                    backend=candidate.backend,
                    accelerated=True,
                    fallback=False,
                    cache_hit=False,
                    failures=failures,
                    quarantined=quarantined,
                )
                return result
            except BaseException as exc:
                if _is_out_of_memory(exc) or isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if not _is_optional_kernel_failure(exc):
                    raise
                if candidate.autotune:
                    self._invalidate_persistent_candidate(candidate, signature, args, kwargs)
                self._remember_failure(failure_key, exc)
                warnings.warn(
                    f"Kernel {candidate.name!r} failed for this workload; using the PyTorch fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                failures.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
        self._remember_selection(selection_key, None)
        result = fallback(*args, **kwargs)
        _publish_dispatch_receipt(
            op=op,
            implementation="torch",
            backend="torch",
            accelerated=False,
            fallback=True,
            cache_hit=False,
            failures=failures,
            quarantined=quarantined,
            reason="no accelerated candidate completed",
        )
        return result

    def _autotune_candidate(
        self,
        candidate: KernelCandidate,
        fallback: KernelCallable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        signature: tuple[object, ...],
    ) -> tuple[KernelCandidate | None, Any, dict[str, object]]:
        """Measure one safe eager candidate against its semantic fallback."""

        import torch

        tensor = next(
            (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor) and value.is_cuda),
            None,
        )
        if tensor is None:
            result = candidate.implementation(*args, **kwargs)
            return (
                candidate,
                result,
                {
                    "op": candidate.op,
                    "candidate": candidate.name,
                    "winner": candidate.name,
                    "reason": "no_cuda_tensor",
                },
            )

        from worldfoundry.core.kernels.autotune_cache import (
            load_persistent_selection,
            store_persistent_selection,
        )

        cached = load_persistent_selection(
            op=candidate.op,
            candidate=candidate.name,
            signature=signature,
            tensor=tensor,
        )
        if cached is not None:
            winner = str(cached["winner"])
            record = dict(cached.get("record", {}))
            record.update(
                {
                    "op": candidate.op,
                    "candidate": candidate.name,
                    "winner": winner,
                    "cache_hit": True,
                    "cache_path": cached.get("path"),
                }
            )
            selected = candidate if winner == candidate.name else None
            result = self._invoke_candidate(candidate, args, kwargs) if selected is not None else fallback(*args, **kwargs)
            return selected, result, record

        try:
            capability: object = tuple(int(value) for value in torch.cuda.get_device_capability(tensor.device))
        except (AssertionError, RuntimeError, TypeError, ValueError):
            capability = "unknown"
        record_context: dict[str, object] = {
            "device": str(tensor.device),
            "compute_capability": capability,
            "dtype": str(tensor.dtype),
            "shape": tuple(int(value) for value in tensor.shape),
        }

        try:
            iterations = int(os.getenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ITERS", "5") or "5")
        except ValueError:
            iterations = 5
        iterations = min(max(iterations, 1), 50)
        try:
            rounds = int(os.getenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ROUNDS", "3") or "3")
        except ValueError:
            rounds = 3
        rounds = min(max(rounds, 1), 9)

        def elapsed_ms(function: KernelCallable) -> float:
            """Time *iterations* launches after one warmup; requires a CUDA tensor."""

            warmup = function(*args, **kwargs)
            del warmup
            torch.cuda.synchronize(tensor.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            timed_result = None
            for _ in range(iterations):
                timed_result = function(*args, **kwargs)
            end.record()
            end.synchronize()
            del timed_result
            return float(start.elapsed_time(end)) / iterations

        with torch.cuda.device(tensor.device):
            # Validate the exact device/dtype/shape before a kernel is allowed
            # into the selection cache. This is especially important for a new
            # GPU architecture whose Triton lowering was not available during
            # development: a successful launch alone is not proof of fidelity.
            reference = fallback(*args, **kwargs)
            candidate_result = candidate.implementation(*args, **kwargs)
            if not _outputs_equivalent(reference, candidate_result):
                del candidate_result
                record = {
                    **record_context,
                    "op": candidate.op,
                    "candidate": candidate.name,
                    "winner": "torch",
                    "reason": "numerical_mismatch",
                    "iterations": iterations,
                    "rounds": rounds,
                    "cache_hit": False,
                }
                store_persistent_selection(
                    op=candidate.op,
                    candidate=candidate.name,
                    signature=signature,
                    tensor=tensor,
                    winner="torch",
                    record=record,
                )
                return (
                    None,
                    reference,
                    record,
                )
            del reference, candidate_result

            # Alternate measurement order and compare medians so clock ramp,
            # cache warmth, and one noisy launch do not encode an A100-specific
            # decision into a Hopper/Blackwell (or legacy GPU) selection.
            fallback_samples: list[float] = []
            candidate_samples: list[float] = []
            for round_index in range(rounds):
                measurements = (
                    (("torch", fallback), (candidate.name, candidate.implementation))
                    if round_index % 2 == 0
                    else ((candidate.name, candidate.implementation), ("torch", fallback))
                )
                for name, function in measurements:
                    measured = elapsed_ms(function)
                    if name == "torch":
                        fallback_samples.append(measured)
                    else:
                        candidate_samples.append(measured)
            fallback_ms = float(median(fallback_samples))
            candidate_ms = float(median(candidate_samples))

        # Require a real margin; selecting a fused kernel on measurement noise
        # makes short workloads slower and is worse than retaining PyTorch.
        selected = candidate if candidate_ms < fallback_ms * 0.98 else None
        winner = candidate.name if selected is not None else "torch"
        record = {
            **record_context,
            "op": candidate.op,
            "candidate": candidate.name,
            "winner": winner,
            "candidate_ms": candidate_ms,
            "torch_ms": fallback_ms,
            "iterations": iterations,
            "rounds": rounds,
            "cache_hit": False,
        }
        store_persistent_selection(
            op=candidate.op,
            candidate=candidate.name,
            signature=signature,
            tensor=tensor,
            winner=winner,
            record=record,
        )
        result = self._invoke_candidate(candidate, args, kwargs) if selected is not None else fallback(*args, **kwargs)
        return (
            selected,
            result,
            record,
        )

    @staticmethod
    def _invalidate_persistent_candidate(
        candidate: KernelCandidate,
        signature: tuple[object, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Drop one stale disk winner without obscuring the original failure."""

        try:
            import torch

            tensor = next(
                (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor) and value.is_cuda),
                None,
            )
            if tensor is None:
                return
            from worldfoundry.core.kernels.autotune_cache import invalidate_persistent_selection

            invalidate_persistent_selection(
                op=candidate.op,
                candidate=candidate.name,
                signature=signature,
                tensor=tensor,
            )
        except Exception:
            return

    def _cached_selection(self, key: tuple[object, ...]) -> tuple[KernelCandidate | None, bool]:
        """LRU-touch one selection; ``(None, False)`` means the key was never stored."""

        with self._lock:
            if key not in self._selection_cache:
                return None, False
            selected = self._selection_cache.pop(key)
            self._selection_cache[key] = selected
            return selected, True

    def _remember_selection(self, key: tuple[object, ...], selected: KernelCandidate | None) -> None:
        """Store *selected* (or explicit PyTorch ``None``) and evict the oldest entry."""

        with self._lock:
            self._selection_cache.pop(key, None)
            self._selection_cache[key] = selected
            while len(self._selection_cache) > self._selection_limit:
                self._selection_cache.popitem(last=False)

    def _drop_selection(self, key: tuple[object, ...]) -> None:
        """Forget one cached winner after a later optional-kernel failure."""

        with self._lock:
            self._selection_cache.pop(key, None)

    def _remember_autotune(self, key: tuple[object, ...], record: dict[str, object]) -> None:
        """Retain one timing/mismatch record for :meth:`report` diagnostics."""

        with self._lock:
            self._autotune_records.pop(key, None)
            self._autotune_records[key] = record
            while len(self._autotune_records) > self._selection_limit:
                self._autotune_records.popitem(last=False)

    def _remember_failure(self, key: tuple[object, ...], exc: BaseException) -> None:
        """Quarantine one ``(candidate, *signature)`` so the next call skips it."""

        with self._lock:
            if key in self._failures:
                return
            if len(self._failure_order) >= self._failure_limit:
                expired = self._failure_order.popleft()
                self._failures.discard(expired)
                self._failure_messages.pop(expired, None)
            self._failures.add(key)
            self._failure_order.append(key)
            self._failure_messages[key] = f"{type(exc).__name__}: {exc}"

    def report(self) -> dict[str, object]:
        """Snapshot candidates, cache occupancy, autotune winners, and quarantine text."""

        from worldfoundry.core.kernels.autotune_cache import persistent_selection_cache_info

        selections: dict[str, dict[str, int]] = {}
        with self._lock:
            for key, candidate in self._selection_cache.items():
                op = str(key[0])
                implementation = "torch" if candidate is None else candidate.name
                counts = selections.setdefault(op, {})
                counts[implementation] = counts.get(implementation, 0) + 1
        return {
            "requested_backend": os.getenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto"),
            "operators": {
                op: [
                    {
                        "name": item.name,
                        "backend": item.backend,
                        "priority": item.priority,
                        "autotune": item.autotune,
                    }
                    for item in candidates
                ]
                for op, candidates in sorted(self._candidates.items())
            },
            "selection_cache_entries": len(self._selection_cache),
            "selected_implementations": selections,
            "autotune_enabled": kernel_autotune_enabled(),
            "persistent_autotune_cache": persistent_selection_cache_info(),
            "autotune_records": tuple(self._autotune_records.values()),
            "failed_signatures": len(self._failures),
            "failures": tuple(self._failure_messages.values()),
        }

    def clear_failures(self) -> None:
        """Drop quarantine, in-memory selections, and autotune records together."""

        with self._lock:
            self._failures.clear()
            self._failure_order.clear()
            self._failure_messages.clear()
            self._selection_cache.clear()
            self._autotune_records.clear()


# ──────────────────────────────────────────────────────────────────────────
# Failure classification — OOM re-raises; Triton/contract misses quarantine
# ──────────────────────────────────────────────────────────────────────────


def _is_out_of_memory(exc: BaseException) -> bool:
    """Detect allocator failures that must not become a silent PyTorch fallback."""

    message = str(exc).casefold()
    return "out of memory" in message or "alloc_failed" in message or type(exc).__name__ == "OutOfMemoryError"


def _is_optional_kernel_failure(exc: BaseException) -> bool:
    """Return whether *exc* is a recoverable kernel-contract or compiler miss.

    Genuine application :class:`RuntimeError` values (wrong shapes from the
    caller) still propagate. Marker strings cover PTX / SM / Triton compile
    failures that should not abort inference.
    """

    if isinstance(exc, (ImportError, OSError, KernelNotSupported)):
        return True
    module = type(exc).__module__.casefold()
    if module.startswith("triton"):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    markers = (
        "not supported",
        "unsupported",
        "no kernel image",
        "invalid device function",
        "out of resource",
        "ptxas",
        "triton",
        "requires sm",
    )
    return any(marker in message for marker in markers)


# ──────────────────────────────────────────────────────────────────────────
# Process-wide table — env gates for first-use timing vs forced PyTorch
# ──────────────────────────────────────────────────────────────────────────


KERNEL_REGISTRY = KernelRegistry()


def kernel_autotune_enabled() -> bool:
    """Return whether first-use eager kernel measurements are enabled.

    Prefer ``WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED`` (XC-9). The older
    ``WORLDFOUNDRY_KERNEL_AUTOTUNE`` name remains a fallback.
    """

    configured = os.getenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED")
    if configured is None or not configured.strip():
        configured = os.getenv("WORLDFOUNDRY_KERNEL_AUTOTUNE", "")
    return configured.strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
        "enable",
        "enabled",
    }


def kernel_dispatch_report() -> dict[str, object]:
    """Return registered implementations and quarantined failure counts."""

    return KERNEL_REGISTRY.report()


def clear_kernel_dispatch_cache() -> None:
    """Clear workload selections and runtime failure quarantine."""

    KERNEL_REGISTRY.clear_failures()


__all__ = [
    "KERNEL_REGISTRY",
    "KernelCandidate",
    "KernelNotSupported",
    "KernelRegistry",
    "clear_kernel_dispatch_cache",
    "kernel_dispatch_receipt_scope",
    "kernel_autotune_enabled",
    "kernel_dispatch_report",
]
