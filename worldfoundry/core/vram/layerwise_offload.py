"""Layerwise CPU offload with asynchronous CUDA prefetch.

Each transformer block keeps a CPU master copy; a pre-forward hook copies
the next layer onto GPU while the current layer runs. That overlaps H2D
with compute, unlike :class:`~worldfoundry.core.vram.memory.DynamicSwapInstaller`
which moves on attribute access (no prefetch).

:class:`LayerwiseOffloadHandle` retains hook handles so :meth:`disable`
can restore parameters (CC-27). The mutation scope blocks LoRA / optimizer
writes from racing a swap. Do not stack this with TP/CP on the same
module — those expect resident shards.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch import nn

from worldfoundry.core.observability.nvtx import nvtx_range

_compiler_disable = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)

# ──────────────────────────────────────────────────────────────────────────
# Public handle — retain RemovableHandle objects so disable() can undo offload
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class LayerwiseOffloadHandle:
    """Handle returned by ``enable_layerwise_cpu_offload``.

    Holds the registered hook handles so the offload can be undone with
    :meth:`disable` (CC-27); previously the ``RemovableHandle`` objects were
    dropped and a model could never leave offload mode without a rebuild.
    """

    enabled: bool
    layer_count: int
    reason: str = ""
    _model: nn.Module | None = field(default=None, repr=False, compare=False)
    _layer_hooks: list = field(default_factory=list, repr=False, compare=False)
    _model_hooks: list = field(default_factory=list, repr=False, compare=False)
    _metrics: _LayerwiseOffloadMetrics | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def reset_request_window(self) -> None:
        """Reset request-local counters without discarding a prefetched layer."""

        if self._metrics is not None:
            self._metrics.reset_request_window()

    def report(self) -> dict[str, Any]:
        """Return auditable lifetime and request-local execution evidence.

        ``enabled`` only proves that hooks were installed.  ``effective`` is
        true after a forward actually used the two-layer CUDA window with
        asynchronous H2D copies and no emergency synchronous materialization.
        Pageable host tensors are surfaced explicitly because
        ``non_blocking=True`` alone does not prove copy/compute overlap.
        """

        metrics = self._metrics
        if metrics is None:
            return {
                "requested": self.enabled,
                "enabled": self.enabled,
                "effective": False,
                "mode": "disabled",
                "layer_count": self.layer_count,
                "reason": self.reason or "no layerwise offload metrics",
            }
        return metrics.report(
            enabled=self.enabled,
            layer_count=self.layer_count,
            reason=self.reason,
        )

    def disable(self) -> bool:
        """Remove offload hooks and restore real parameters onto each layer.

        Parameters are restored from the CPU master copies (on CPU); move the
        model to the desired device afterwards. Returns True when hooks were
        removed. Handles created for an already-enabled model
        (``reason="already enabled"``) do not own hooks and return False.
        """

        if not self.enabled or self._model is None or not self._layer_hooks:
            self.enabled = False
            return False
        if self._metrics is not None:
            # A primed/prefetched tensor may still be owned by the copy stream.
            # Synchronize that stream before dropping its final Python
            # reference so allocator reuse cannot race the in-flight H2D.
            self._metrics.copy_stream.synchronize()
        for hook in self._model_hooks:
            hook.remove()
        self._model_hooks.clear()
        for layer, state, hook_handles in self._layer_hooks:
            for hook_handle in hook_handles:
                hook_handle.remove()
            for name, param in layer.named_parameters(recurse=True):
                cpu_tensor = state.cpu_named_parameters.get(name)
                if cpu_tensor is not None:
                    param.data = cpu_tensor
            state.gpu_named_parameters.clear()
            state.cpu_named_parameters.clear()
            state.next_state = None
            for attribute in ("_worldfoundry_layerwise_cpu_offload", "_worldfoundry_layerwise_cpu_offload_state"):
                layer.__dict__.pop(attribute, None)
        self._model.__dict__.pop("_worldfoundry_layerwise_cpu_offload", None)
        self._layer_hooks.clear()
        self._model = None
        self._metrics = None
        self.enabled = False
        return True


@dataclass
class _LayerwiseOffloadMetrics:
    """Shared counters for one two-layer CUDA prefetch window."""

    copy_stream: torch.cuda.Stream = field(repr=False, compare=False)
    pin_memory_requested: bool = True
    primed: bool = False
    buffer_mode: str = "per-parameter"
    device_buffer_bytes: int = 0
    active_layers: set[int] = field(default_factory=set, repr=False)
    lifetime: dict[str, int] = field(default_factory=dict, repr=False)
    request: dict[str, int] = field(default_factory=dict, repr=False)

    _COUNTERS = (
        "forward_calls",
        "prefetch_calls",
        "async_copy_tensors",
        "async_copy_bytes",
        "synchronous_copy_tensors",
        "synchronous_copy_bytes",
        "wait_calls",
        "release_calls",
        "pinned_cpu_tensors",
        "pageable_cpu_tensors",
        "peak_active_layers",
    )

    def __post_init__(self) -> None:
        self.lifetime = {name: 0 for name in self._COUNTERS}
        self.request = {name: 0 for name in self._COUNTERS}

    def count(self, name: str, amount: int = 1, *, request: bool = True) -> None:
        self.lifetime[name] = int(self.lifetime.get(name, 0)) + int(amount)
        if request:
            self.request[name] = int(self.request.get(name, 0)) + int(amount)

    def set_layer_active(self, state_id: int, active: bool) -> None:
        if active:
            self.active_layers.add(state_id)
            active_count = len(self.active_layers)
            self.lifetime["peak_active_layers"] = max(
                self.lifetime["peak_active_layers"],
                active_count,
            )
            self.request["peak_active_layers"] = max(
                self.request["peak_active_layers"],
                active_count,
            )
        else:
            self.active_layers.discard(state_id)

    def reset_request_window(self) -> None:
        self.request = {name: 0 for name in self._COUNTERS}
        # One layer is deliberately kept warm across denoising calls.  The
        # request peak starts at the physical state rather than pretending
        # the cache is empty.
        self.request["peak_active_layers"] = len(self.active_layers)

    def report(
        self,
        *,
        enabled: bool,
        layer_count: int,
        reason: str,
    ) -> dict[str, Any]:
        request = dict(self.request)
        used = request["forward_calls"] > 0
        async_h2d = request["async_copy_tensors"] > 0
        no_sync_h2d = request["synchronous_copy_tensors"] == 0
        pinned = request["pageable_cpu_tensors"] == 0
        bounded_window = request["peak_active_layers"] <= min(2, layer_count)
        effective = bool(enabled and used and async_h2d and no_sync_h2d and pinned and bounded_window)
        issues: list[str] = []
        if used and not async_h2d:
            issues.append("no asynchronous H2D copy executed in the request window")
        if request["synchronous_copy_tensors"]:
            issues.append("emergency synchronous H2D materialization executed")
        if request["pageable_cpu_tensors"]:
            issues.append("one or more CPU masters were not pinned")
        if not bounded_window:
            issues.append("more than two transformer layers were CUDA-resident")
        return {
            "requested": True,
            "enabled": bool(enabled),
            "effective": effective,
            "mode": "async-double-buffer",
            "layer_count": int(layer_count),
            "pin_memory_requested": bool(self.pin_memory_requested),
            "primed": bool(self.primed),
            "buffer_mode": self.buffer_mode,
            "device_buffer_bytes": self.device_buffer_bytes,
            "active_layers": len(self.active_layers),
            "request": request,
            "lifetime": dict(self.lifetime),
            "issues": issues,
            "reason": reason or None,
        }


# ──────────────────────────────────────────────────────────────────────────
# Enable / mutation scope — opt-in; disabled handle on non-CUDA or no ModuleList
# ──────────────────────────────────────────────────────────────────────────


def enable_layerwise_cpu_offload(
    model: nn.Module,
    *,
    layer_container: str | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = True,
    reuse_buffers: bool = False,
) -> LayerwiseOffloadHandle:
    """Attach hooks to a ModuleList or an explicitly named Sequential stack.

    The helper keeps parameters on CPU and moves one layer at a time to CUDA.
    The next layer is prefetched on a separate CUDA stream while the current
    layer runs. It is intentionally opt-in and returns a disabled handle on
    non-CUDA systems. Keep the returned handle if you need to switch the
    model back to full-device residency later via ``handle.disable()``.
    Opt in with ``reuse_buffers=True`` to pack contiguous parameters into CPU
    storage and two reusable CUDA slots. The per-parameter path is the default;
    non-contiguous weights also retain that path to preserve their layout.
    The reusable path is inference-only and forbids shared parameter storage
    across layers. Its mutation scope still supports in-place weight edits.
    """

    if not torch.cuda.is_available():
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason="CUDA is not available")
    if getattr(model, "_worldfoundry_layerwise_cpu_offload", False):
        return LayerwiseOffloadHandle(enabled=True, layer_count=0, reason="already enabled")

    target_device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
    if target_device.type != "cuda":
        return LayerwiseOffloadHandle(enabled=False, layer_count=0, reason=f"target device is {target_device.type!r}")
    if target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())

    layers = _find_layer_container(model, layer_container)
    if layers is None or len(layers) == 0:
        return LayerwiseOffloadHandle(
            enabled=False,
            layer_count=0,
            reason="no ModuleList or explicitly named Sequential layer container found",
        )
    if any(getattr(layer, "_worldfoundry_layerwise_cpu_offload", False) for layer in layers):
        raise ValueError("one or more layers already belong to another offload handle")

    packed = reuse_buffers and all(p.is_contiguous() for layer in layers for p in layer.parameters())
    if packed:
        storage_owners: dict[tuple[torch.device, int], int] = {}
        for index, layer in enumerate(layers):
            for parameter in layer.parameters():
                if parameter.is_meta or parameter.layout != torch.strided:
                    raise ValueError("load ordinary tensor weights before enabling layerwise offload")
                if not parameter.numel():
                    continue
                key = (parameter.device, parameter.untyped_storage().data_ptr())
                if key in storage_owners:
                    raise ValueError("reusable layerwise buffers do not support shared parameter storage")
                storage_owners[key] = index

    stream = torch.cuda.Stream(device=target_device)
    metrics = _LayerwiseOffloadMetrics(
        copy_stream=stream,
        pin_memory_requested=pin_memory,
        buffer_mode="packed-two-slot" if packed else "per-parameter",
    )
    pool = _PackedBufferPool(target_device, stream) if packed else None
    states = [
        (_LayerwiseOffloadState if pool is None else _PackedLayerwiseOffloadState)(
            layer,
            target_device,
            stream,
            metrics=metrics,
            state_id=index,
            pin_memory=pin_memory,
            **({"pool": pool} if pool is not None else {}),
        )
        for index, layer in enumerate(layers)
    ]
    layer_hooks: list[tuple[nn.Module, _LayerwiseOffloadState, list]] = []
    try:
        for index, state in enumerate(states):
            # Wrap around so the last layer prefetches the first for the next step.
            state.next_state = states[(index + 1) % len(states)]
            state.offload_to_cpu()
            layer = layers[index]
            if getattr(layer, "_worldfoundry_layerwise_cpu_offload", False):
                continue
            pre_handle = layer.register_forward_pre_hook(_make_pre_hook(state), with_kwargs=True)
            post_handle = layer.register_forward_hook(
                _make_post_hook(state),
                with_kwargs=True,
                always_call=True,
            )
            handles = [pre_handle, post_handle]
            if packed:
                handles.extend(
                    (
                        layer.register_state_dict_pre_hook(_reject_packed_serialization),
                        layer.register_load_state_dict_pre_hook(_reject_packed_serialization),
                    )
                )
            layer_hooks.append((layer, state, handles))
            setattr(layer, "_worldfoundry_layerwise_cpu_offload", True)
            setattr(layer, "_worldfoundry_layerwise_cpu_offload_state", state)
        if pool is not None:
            pool.initialize(states)
            metrics.device_buffer_bytes = pool.nbytes
        setattr(model, "_worldfoundry_layerwise_cpu_offload", True)
        # Match the upstream double-buffer lifecycle: the first CUDA buffer is
        # populated before layer 0 is entered, so the first timed forward does not
        # fall back to a synchronous H2D copy.  The pre-hook waits on this stream,
        # then starts layer 1 while layer 0 computes.
        states[0].prefetch(request_counter=False)
        metrics.primed = True
    except BaseException:
        # Restore even a layer snapshotted before its hooks were installed.
        hooked = {id(layer) for layer, _, _ in layer_hooks}
        rollback_hooks = layer_hooks + [(state.module, state, []) for state in states if id(state.module) not in hooked]
        LayerwiseOffloadHandle(True, len(states), _model=model, _layer_hooks=rollback_hooks, _metrics=metrics).disable()
        raise
    model_hooks = []
    if pool is not None:

        def release_failed_request(module, args, output):
            if output is None:
                for state in states:
                    state.release_gpu_params()

        model_hooks.append(model.register_forward_hook(release_failed_request, always_call=True))
    return LayerwiseOffloadHandle(
        enabled=True,
        layer_count=len(states),
        _model=model,
        _layer_hooks=layer_hooks,
        _model_hooks=model_hooks,
        _metrics=metrics,
    )


def layerwise_offload_mutation_scope(module: nn.Module) -> Iterator[None]:
    """Temporarily materialize offloaded parameters for in-place mutations."""

    state = getattr(module, "_worldfoundry_layerwise_cpu_offload_state", None)
    if state is None:
        return nullcontext()
    return state.mutate_params_scope()


# ──────────────────────────────────────────────────────────────────────────
# Per-layer CPU master + GPU cache — prefetch overlaps H2D with compute
# ──────────────────────────────────────────────────────────────────────────


class _LayerwiseOffloadState:
    """CPU master copy, GPU cache, and next-layer prefetch for one block.

    Placeholders replace ``param.data`` so Parameter identity (optimizer /
    hook targets) stays stable while bytes live on CPU. Do not stack with
    TP/CP: those expect resident shards, not ping-pong storage.
    """

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        async_copy_stream: torch.cuda.Stream,
        *,
        metrics: _LayerwiseOffloadMetrics,
        state_id: int,
        pin_memory: bool,
    ) -> None:
        """Bind one module to a compute device and a shared prefetch stream."""
        self.module = module
        self.device = device
        self.async_copy_stream = async_copy_stream
        self.metrics = metrics
        self.state_id = int(state_id)
        self.pin_memory = pin_memory
        self.cpu_named_parameters: dict[str, torch.Tensor] = {}
        self.gpu_named_parameters: dict[str, torch.Tensor] = {}
        self.next_state: _LayerwiseOffloadState | None = None

    @_compiler_disable
    def offload_to_cpu(self) -> None:
        """Snapshot weights to CPU (optionally pinned) and drop GPU storage.

        ``pin_memory`` can raise on some allocators; the copy still lives on
        pageable CPU RAM so offload continues rather than aborting setup.
        """
        for name, param in self.module.named_parameters(recurse=True):
            cpu_tensor = param.detach().to("cpu")
            if self.pin_memory:
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except RuntimeError:
                    pass
            counter = "pinned_cpu_tensors" if cpu_tensor.is_pinned() else "pageable_cpu_tensors"
            # Setup inventory is lifetime evidence.  Request counters are
            # incremented when a copy actually consumes the CPU master.
            self.metrics.count(counter, request=False)
            self.cpu_named_parameters[name] = cpu_tensor
            param.data = _tensor_placeholder(param.data, self.device)

    @_compiler_disable
    def wait_and_materialize(self) -> None:
        """Block the compute stream on prefetch, then bind GPU copies to params."""
        torch.cuda.current_stream(self.device).wait_stream(self.async_copy_stream)
        self.metrics.count("wait_calls")
        named_parameters = dict(self.module.named_parameters(recurse=True))
        for name, param in named_parameters.items():
            if name not in self.cpu_named_parameters:
                continue
            if name not in self.gpu_named_parameters:
                cpu_tensor = self.cpu_named_parameters[name]
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
                self.gpu_named_parameters[name] = gpu_tensor
                self.metrics.count("synchronous_copy_tensors")
                self.metrics.count(
                    "synchronous_copy_bytes",
                    cpu_tensor.numel() * cpu_tensor.element_size(),
                )
                self.metrics.count("pinned_cpu_tensors" if cpu_tensor.is_pinned() else "pageable_cpu_tensors")
            param.data = self.gpu_named_parameters[name]
        if self.gpu_named_parameters:
            self.metrics.set_layer_active(self.state_id, True)

    @_compiler_disable
    def prefetch(self, *, request_counter: bool = True) -> None:
        """Copy missing GPU weights on the async stream; skip already-resident names.

        ``record_stream`` lets the compute stream reuse the copy after
        :meth:`wait_and_materialize` without a full device sync.
        """
        self.metrics.count("prefetch_calls", request=request_counter)
        compute_stream = torch.cuda.current_stream(self.device)
        copied = False
        with torch.cuda.stream(self.async_copy_stream):
            for name, cpu_tensor in self.cpu_named_parameters.items():
                if name in self.gpu_named_parameters:
                    continue
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
                gpu_tensor.record_stream(compute_stream)
                self.gpu_named_parameters[name] = gpu_tensor
                copied = True
                self.metrics.count(
                    "async_copy_tensors",
                    request=request_counter,
                )
                self.metrics.count(
                    "async_copy_bytes",
                    cpu_tensor.numel() * cpu_tensor.element_size(),
                    request=request_counter,
                )
                self.metrics.count(
                    "pinned_cpu_tensors" if cpu_tensor.is_pinned() else "pageable_cpu_tensors",
                    request=request_counter,
                )
        if copied:
            self.metrics.set_layer_active(self.state_id, True)

    @_compiler_disable
    def release_gpu_params(self) -> None:
        """Drop GPU copies after forward so only the CPU master remains."""
        named_parameters = dict(self.module.named_parameters(recurse=True))
        for name, param in named_parameters.items():
            if name not in self.cpu_named_parameters:
                continue
            if name in self.gpu_named_parameters:
                param.data = _tensor_placeholder(param.data, self.device)
                del self.gpu_named_parameters[name]
        if not self.gpu_named_parameters:
            self.metrics.set_layer_active(self.state_id, False)
        self.metrics.count("release_calls")

    @contextmanager
    def mutate_params_scope(self) -> Iterator[None]:
        """Materialize, yield for in-place writes, then refresh the CPU master.

        LoRA / optimizer updates must not race a swap. After the body the GPU
        cache is discarded and :meth:`offload_to_cpu` snapshots the mutated
        weights so later prefetch sees the write.
        """
        self.wait_and_materialize()
        try:
            yield
        finally:
            self.cpu_named_parameters.clear()
            self.gpu_named_parameters.clear()
            self.offload_to_cpu()


# ──────────────────────────────────────────────────────────────────────────
@dataclass
class _PackedSlot:
    buffers: dict[torch.dtype, torch.Tensor]
    owner: int | None = None
    released: torch.cuda.Event | None = None


class _PackedBufferPool:
    """Two long-lived device slots, fenced against outstanding compute reads."""

    def __init__(self, device: torch.device, stream: torch.cuda.Stream) -> None:
        self.device = device
        self.stream = stream
        self.slots: list[_PackedSlot] = []

    def initialize(self, states: list[_PackedLayerwiseOffloadState]) -> None:
        capacities: dict[torch.dtype, int] = {}
        for state in states:
            for dtype, buffer in state.host_buffers.items():
                capacities[dtype] = max(capacities.get(dtype, 0), buffer.numel())
        with torch.cuda.device(self.device):
            self.slots = [
                _PackedSlot(
                    {dtype: torch.empty(size, dtype=dtype, device=self.device) for dtype, size in capacities.items()}
                )
                for _ in range(min(2, len(states)))
            ]
            self.stream.wait_stream(torch.cuda.current_stream(self.device))

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for slot in self.slots for t in slot.buffers.values())

    def acquire(self, owner: int) -> _PackedSlot:
        for slot in self.slots:
            if slot.owner is None:
                if slot.released is not None:
                    self.stream.wait_event(slot.released)
                slot.owner = owner
                return slot
        raise RuntimeError("layerwise offload requires sequential execution with at most two resident layers")


class _PackedLayerwiseOffloadState(_LayerwiseOffloadState):
    """Packed CPU masters and parameter views into a reusable device slot."""

    def __init__(self, *args, pool: _PackedBufferPool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = pool
        self.host_buffers: dict[torch.dtype, torch.Tensor] = {}
        self.slot: _PackedSlot | None = None
        self.ready: torch.cuda.Event | None = None

    @_compiler_disable
    def offload_to_cpu(self) -> None:
        # Build the complete snapshot before changing any Parameter. Views
        # preserve parameter identities, with 256-byte aligned packed offsets.
        grouped: dict[torch.dtype, list[tuple[str, nn.Parameter]]] = {}
        for name, parameter in self.module.named_parameters():
            grouped.setdefault(parameter.dtype, []).append((name, parameter))
        masters: dict[str, torch.Tensor] = {}
        buffers: dict[torch.dtype, torch.Tensor] = {}
        with torch.no_grad():
            for dtype, parameters in grouped.items():
                alignment = max(1, 256 // parameters[0][1].element_size())
                offsets = []
                size = 0
                for _, parameter in parameters:
                    size = (size + alignment - 1) // alignment * alignment
                    offsets.append(size)
                    size += parameter.numel()
                try:
                    host = torch.empty(size, dtype=dtype, device="cpu", pin_memory=self.pin_memory)
                except RuntimeError:
                    host = torch.empty(size, dtype=dtype, device="cpu")
                for (name, parameter), offset in zip(parameters, offsets):
                    view = host[offset : offset + parameter.numel()].view(parameter.shape)
                    view.copy_(parameter.detach())
                    masters[name] = view
                buffers[dtype] = host
        self.host_buffers = buffers
        self.cpu_named_parameters = masters
        for name, parameter in self.module.named_parameters():
            cpu = masters[name]
            self.metrics.count("pinned_cpu_tensors" if cpu.is_pinned() else "pageable_cpu_tensors", request=False)
            parameter.data = _tensor_placeholder(parameter.data, self.device)

    @_compiler_disable
    def prefetch(self, *, request_counter: bool = True) -> None:
        self.metrics.count("prefetch_calls", request=request_counter)
        if self.slot is not None:
            return
        slot = self.pool.acquire(self.state_id)
        self.slot = slot
        with nvtx_range("worldfoundry.offload.h2d"), torch.cuda.stream(self.async_copy_stream):
            for dtype, host in self.host_buffers.items():
                slot.buffers[dtype][: host.numel()].copy_(host, non_blocking=True)
                self.metrics.count("async_copy_tensors", request=request_counter)
                self.metrics.count("async_copy_bytes", host.numel() * host.element_size(), request=request_counter)
                self.metrics.count(
                    "pinned_cpu_tensors" if host.is_pinned() else "pageable_cpu_tensors", request=request_counter
                )
            self.ready = torch.cuda.Event()
            self.ready.record(self.async_copy_stream)
        for name, cpu in self.cpu_named_parameters.items():
            self.gpu_named_parameters[name] = slot.buffers[cpu.dtype][
                cpu.storage_offset() : cpu.storage_offset() + cpu.numel()
            ].view(cpu.shape)
        self.metrics.set_layer_active(self.state_id, True)

    @_compiler_disable
    def wait_and_materialize(self) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("reusable layerwise offload requires torch.no_grad() or inference_mode()")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("reusable layerwise offload does not support CUDA Graph capture")
        for name, parameter in self.module.named_parameters():
            cpu = self.cpu_named_parameters[name]
            if parameter.dtype != cpu.dtype or parameter.device != self.device:
                raise RuntimeError("disable layerwise offload before changing parameter dtype or device")
        if self.slot is None:
            self.prefetch()
        stream = torch.cuda.current_stream(self.device)
        if self.ready is not None:
            stream.wait_event(self.ready)
        self.metrics.count("wait_calls")
        for name, parameter in self.module.named_parameters():
            gpu = self.gpu_named_parameters[name]
            gpu.record_stream(stream)
            parameter.data = gpu

    @_compiler_disable
    def release_gpu_params(self) -> None:
        if self.slot is not None:
            # Reusing memory requires a compute->copy fence, not just the
            # copy->compute fence used when materializing the next layer.
            released = torch.cuda.Event()
            released.record(torch.cuda.current_stream(self.device))
            self.slot.released = released
            self.slot.owner = None
            self.slot = None
            self.ready = None
        super().release_gpu_params()

    @contextmanager
    def mutate_params_scope(self) -> Iterator[None]:
        with torch.no_grad():
            self.wait_and_materialize()
        try:
            yield
        finally:
            # Snapshot edited values before unbinding the slot. The blocking
            # D2H copy completes before the new host buffers can be prefetched.
            self.offload_to_cpu()
            self.release_gpu_params()


def _reject_packed_serialization(*args, **kwargs) -> None:
    raise RuntimeError("disable layerwise offload before saving or loading a state_dict")


# Hooks and container lookup — first ModuleList unless a dotted path is given
# ──────────────────────────────────────────────────────────────────────────


def _find_layer_container(
    model: nn.Module, layer_container: str | None
) -> nn.ModuleList | nn.Sequential | None:
    """Resolve an explicit layer stack, or discover the first ModuleList.

    A dotted ``layer_container`` is required when the first ``ModuleList`` is
    not the transformer stack (e.g. a vision tower). Sequential is accepted
    only for an explicit path, so discovery cannot mistake an MLP for the
    complete transformer stack. Wan's CLIP encoder uses a Sequential stack.
    """
    if layer_container:
        current: object = model
        for part in layer_container.split("."):
            current = getattr(current, part)
        return current if isinstance(current, (nn.ModuleList, nn.Sequential)) else None
    for module in model.modules():
        for child in module.children():
            if isinstance(child, nn.ModuleList):
                return child
    return None


def _make_pre_hook(state: _LayerwiseOffloadState):
    """Build a pre-forward hook that materializes this layer and prefetches the next."""

    @_compiler_disable
    def pre_hook(module: nn.Module, args, kwargs):
        """Wait for this layer's copy, then start H2D for ``next_state``."""
        state.metrics.count("forward_calls")
        state.wait_and_materialize()
        if state.next_state is not None:
            state.next_state.prefetch()
        return args, kwargs

    return pre_hook


def _make_post_hook(state: _LayerwiseOffloadState):
    """Build a post-forward hook that frees this layer's GPU copies."""

    @_compiler_disable
    def post_hook(module: nn.Module, args, kwargs, output):
        """Release GPU params; the output tensor is returned unchanged."""
        state.release_gpu_params()
        return output

    return post_hook


def _tensor_placeholder(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Zero-size storage with the same rank/dtype so Parameter identity survives.

    A scalar uses ``(0,)`` because ``empty(())`` is still one element. The
    placeholder lives on ``device`` so later ``.to(device)`` is a no-op.
    """
    shape = (0,) if tensor.ndim <= 0 else (0,) * tensor.ndim
    return torch.empty(shape, dtype=tensor.dtype, device=device)


__all__ = [
    "LayerwiseOffloadHandle",
    "enable_layerwise_cpu_offload",
    "layerwise_offload_mutation_scope",
]
