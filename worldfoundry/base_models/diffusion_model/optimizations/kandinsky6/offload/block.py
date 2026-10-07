"""Inference-only, double-buffered block offload, external to model/algo code.

CPU snapshots are immutable: no D2H copy is needed after a forward. Two reusable
CUDA slots bound weight residency even when the CPU queues many forwards ahead
of the GPU. A slot's H2D waits for its previous compute event; compute waits only
for its own ready event, not the next block's transfer. Block forward decorators
are installed *outside* regional torch.compile, with no offload branches in DiT.

Weights/buffers must not be mutated after registration or during inference.
This handle is single-request, single-compute-stream, and does not support
training, whole-model compilation, CUDA graphs, or externally bound AOTI weights.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
from torch import nn


@dataclass
class _Binding:
    owner: nn.Module
    name: str
    parameter: bool
    cpu: torch.Tensor

    def bind(self, tensor: torch.Tensor) -> None:
        if self.parameter:
            self.owner._parameters[self.name].data = tensor
        else:
            self.owner._buffers[self.name] = tensor


class _TensorGroup:
    """Registered tensors, including integer/nonpersistent/FP8 buffers."""

    def __init__(self, modules: Sequence[nn.Module]):
        self.bindings: list[_Binding] = []
        self._owners: set[int] = set()
        self.add_modules(modules)

    def add_modules(self, modules: Sequence[nn.Module]) -> None:
        """Include newly materialized children of a lazy, CPU-loaded stage."""
        for module in modules:
            if id(module) in self._owners:
                continue
            for parameter, values in ((True, module._parameters), (False, module._buffers)):
                for name, value in values.items():
                    if value is None:
                        continue
                    if value.device.type == "meta":
                        raise ValueError("Materialize meta tensors before registering block offload")
                    if parameter and type(value) is not nn.Parameter:
                        raise ValueError("Block offload requires ordinary parameters (not quantized subclasses)")
                    # Left exactly as loaded — safetensors hands back file-backed
                    # tensors, and pinning them would turn the whole checkpoint
                    # into memory the kernel may never reclaim.
                    cpu = value.detach().to("cpu")
                    binding = _Binding(module, name, parameter, cpu)
                    binding.bind(cpu)
                    self.bindings.append(binding)
            self._owners.add(id(module))

    def restore(self) -> None:
        for binding in self.bindings:
            binding.bind(binding.cpu)

    def bind(self, tensors: Sequence[torch.Tensor]) -> None:
        for binding, tensor in zip(self.bindings, tensors, strict=True):
            binding.bind(tensor)


class _Slot:
    """One residency buffer: CUDA tensors plus the pinned staging they arrive through.

    Asynchronous H2D needs page-locked source memory, but pinning the weights
    themselves costs the whole checkpoint. Pinning only two blocks' worth of
    staging keeps the transfer asynchronous and leaves the weights reclaimable.
    """

    def __init__(self, *, staged: bool):
        self.tensors: list[torch.Tensor] = []
        self.staging: list[torch.Tensor] = []
        self.staged = staged
        self.signature: tuple = ()
        self.group: _TensorGroup | None = None
        self.ready: torch.cuda.Event | None = None
        self.done: torch.cuda.Event | None = None

    def load(self, group: _TensorGroup, stream: torch.cuda.Stream) -> None:
        signature = tuple((b.cpu.shape, b.cpu.stride(), b.cpu.dtype) for b in group.bindings)
        # The staging buffers may still be the source of the previous transfer.
        if self.staged and self.ready is not None:
            self.ready.synchronize()
        with torch.cuda.stream(stream):
            if self.done is not None:
                stream.wait_event(self.done)
            if self.ready is not None:
                stream.wait_event(self.ready)
            if signature != self.signature:
                # Changing text/fused block layouts is rare. Drain this slot before
                # replacing storage, so allocator use cannot grow with queue depth.
                stream.synchronize()
                self.tensors.clear()
                self.staging.clear()
                self.tensors = [torch.empty_like(b.cpu, device=stream.device) for b in group.bindings]
                if self.staged:
                    self.staging = [torch.empty_like(b.cpu, pin_memory=True) for b in group.bindings]
                self.signature = signature
            if self.staged:
                # Pageable -> pinned on this thread (this is where a file-backed
                # weight is faulted in), then pinned -> CUDA without blocking.
                for binding, staging in zip(group.bindings, self.staging, strict=True):
                    staging.copy_(binding.cpu)
                sources = self.staging
            else:
                sources = [b.cpu for b in group.bindings]
            for source, tensor in zip(sources, self.tensors, strict=True):
                tensor.copy_(source, non_blocking=self.staged)
            self.ready = stream.record_event()
        self.group = group

    def clear(self) -> None:
        # Called only after the stage streams have drained.
        self.tensors.clear()
        self.staging.clear()
        self.signature = ()
        self.group = None
        self.ready = self.done = None


class BlockOffload:
    """Stage offload + one-block lookahead for every DiT text/visual stack.

    Register the bare DiT or CacheDiT as ``dit``. Other stages stay CPU-resident
    until ``use``; stage-level prefetch hints are deliberately ignored to keep
    the text encoder and VAEs out of VRAM during denoising. NF4 Qwen, when used,
    retains its existing device-bound behavior and is *not* block-offloaded.
    """

    strategy = "block"
    _STACKS = (
        "text_blocks",
        "video_text_blocks",
        "audio_text_blocks",
        "video_text_transformer_blocks",
        "audio_text_transformer_blocks",
        "visual_blocks",
        "text_transformer_blocks",
        "adapter.blocks",
        "visual_transformer_blocks",
    )

    def __init__(self, compute_device: torch.device, *, pin_memory: bool = True):
        self.compute_device = torch.device(compute_device)
        if self.compute_device.type != "cuda":
            raise ValueError("BlockOffload requires a CUDA compute device")
        if not pin_memory:
            raise ValueError("BlockOffload requires pin_memory=True for asynchronous H2D")
        self.pin_memory = pin_memory
        if self.compute_device.index is None:
            self.compute_device = torch.device("cuda", torch.cuda.current_device())
        self._xfer = torch.cuda.Stream(device=self.compute_device)
        self._slots = (_Slot(staged=True), _Slot(staged=True))
        self._stages: dict[str, _TensorGroup] = {}
        self._stage_roots: dict[str, nn.Module] = {}
        self._facades: dict[str, Any] = {}
        self._blocks: list[_TensorGroup] = []
        self._active: set[str] = set()
        self._compute: torch.cuda.Stream | None = None
        self._next_slot = 0

    def release(self, *names: str) -> None:
        """Stage residency ends in ``use``. The pipeline calls this after every request."""
        del names

    def register(self, name: str, module: Any) -> None:
        if self._active or name in self._stages:
            raise RuntimeError(f"Cannot register active or duplicate offload stage: {name}")
        if name == "dit":
            self._register_dit(module)
            return
        if isinstance(module, nn.Module):
            modules = list(module.modules())
            self._stage_roots[name] = module
        else:
            # TextEmbedder facade: NF4 Qwen cannot be moved via .data/.to().
            attrs = ("clip",) if getattr(module, "quantized_qwen", False) else ("qwen", "clip")
            modules = [sub for attr in attrs for sub in getattr(module, attr).modules()]
            self._facades[name] = module
        self._stages[name] = _TensorGroup(modules)

    def _register_dit(self, module: nn.Module) -> None:
        bare = getattr(module, "module", module)
        stacks = []
        seen = set()
        for name in self._STACKS:
            try:
                stack = bare.get_submodule(name)
            except AttributeError:
                continue
            if not isinstance(stack, nn.ModuleList):
                raise ValueError(f"Expected a ModuleList for DiT block stack {name!r}")
            if id(stack) not in seen:
                stacks.append(stack)
                seen.add(id(stack))
        if not stacks:
            raise ValueError("Block offload expects a DiT with text/visual block stacks")
        blocks = [block for stack in stacks for block in stack]
        if len({id(block) for block in blocks}) != len(blocks):
            raise ValueError("Block offload does not support shared/reused block objects")
        excluded = {id(sub) for stack in stacks for sub in stack.modules()}
        stage_modules = [sub for sub in bare.modules() if id(sub) not in excluded]
        # A shared Parameter is one mutable .data binding: restoring a block
        # would also move a still-resident output projection (or another block)
        # to CPU. Reject cross-group aliases before pinning or wrapping anything.
        tensor_groups: dict[int, int] = {}
        for index, modules in enumerate([stage_modules, *(list(block.modules()) for block in blocks)]):
            for sub in modules:
                for value in (*sub._parameters.values(), *sub._buffers.values()):
                    if value is not None and tensor_groups.setdefault(id(value), index) != index:
                        raise ValueError("Block offload does not support tensors shared across residency groups")
        self._stages["dit"] = _TensorGroup(stage_modules)
        for stack in stacks:
            groups = [_TensorGroup(list(block.modules())) for block in stack]
            self._blocks.extend(groups)
            for index, block in enumerate(stack):
                following = groups[index + 1] if index + 1 < len(groups) else None
                block.forward = self._decorate(block.forward, groups[index], following)

    def _decorate(self, forward, group: _TensorGroup, following: _TensorGroup | None):
        @wraps(forward)
        def wrapped(*args, **kwargs):
            with self._use_block(group, following):
                return forward(*args, **kwargs)

        # Keep transfer orchestration out of Dynamo; forward itself may already
        # be regionally compiled, and is called with CUDA tensors bound in place.
        return torch.compiler.disable(wrapped)

    def _prefetch(self, group: _TensorGroup, *, exclude: _Slot | None = None) -> _Slot:
        for slot in self._slots:
            if slot.group is group:
                return slot
        index = self._next_slot
        if self._slots[index] is exclude:
            index = 1 - index
        slot = self._slots[index]
        self._next_slot = 1 - index
        slot.load(group, self._xfer)
        return slot

    @contextmanager
    def _use_block(self, group: _TensorGroup, following: _TensorGroup | None) -> Iterator[None]:
        if torch.is_grad_enabled():
            raise RuntimeError("BlockOffload is inference-only; gradients were enabled inside the stage")
        if "dit" not in self._active:
            raise RuntimeError("Use offloaded DiT inside `with offload.use('dit')`")
        compute = torch.cuda.current_stream(self.compute_device)
        if compute != self._compute:
            raise RuntimeError("BlockOffload requires a single compute stream per stage")
        slot = self._prefetch(group)
        compute.wait_event(slot.ready)
        group.bind(slot.tensors)
        try:
            # no_grad alone still lets autocast cache a CUDA cast of every FP32
            # Parameter. That defeats bounded block residency (notably SR DiT).
            # Preserve the caller's autocast dtype/enabled state, and restore its
            # cache policy on exit; never clear another component's cast cache.
            with torch.autocast(
                "cuda",
                enabled=torch.is_autocast_enabled("cuda"),
                cache_enabled=False,
            ):
                yield
            # After the forward is enqueued, not before it: staging the next
            # block is real CPU work now, and it has to overlap this block's
            # compute instead of delaying its launch.
            if following is not None:
                self._prefetch(following, exclude=slot)
        finally:
            slot.done = compute.record_event()
            group.restore()
            # Keep slot storage, not its logical residency: even repeated CFG
            # passes and cache skips must reload the right block on demand.
            slot.group = None

    @contextmanager
    def use(self, *names: str, prefetch: str | Sequence[str] | None = None) -> Iterator[None]:
        del prefetch  # Deliberately no overlap between *whole* pipeline stages.
        if torch.is_grad_enabled():
            raise RuntimeError("BlockOffload is inference-only; use torch.no_grad() or inference_mode()")
        if self._active:
            raise RuntimeError("BlockOffload does not support nested/concurrent stage use")
        # SR latent-upscaler banks may materialize another scale on CPU between
        # requests. Register those children before staging, never during forward.
        for name in names:
            if name in self._stage_roots:
                self._stages[name].add_modules(list(self._stage_roots[name].modules()))
        groups = [self._stages[name] for name in names]
        self._active = set(names)
        self._compute = torch.cuda.current_stream(self.compute_device)
        resident: list[_Slot] = []
        try:
            # Non-block weights live on CUDA only for the enclosing stage.
            for group in groups:
                slot = _Slot(staged=False)
                resident.append(slot)
                slot.load(group, self._xfer)
                self._compute.wait_event(slot.ready)
                group.bind(slot.tensors)
            for name in names:
                if name in self._facades:
                    self._facades[name].device = self.compute_device
            yield
        finally:
            # Stage boundary (not per block): make cleanup, exceptions, and
            # immediate reuse on another compute stream safe.
            self._compute.synchronize()
            self._xfer.synchronize()
            for group in (*groups, *self._blocks):
                group.restore()
            for slot in (*resident, *self._slots):
                slot.clear()
            for name in names:
                if name in self._facades:
                    self._facades[name].device = torch.device("cpu")
            self._active.clear()
            self._compute = None
            self._next_slot = 0
            with torch.cuda.device(self.compute_device):
                torch.cuda.empty_cache()
