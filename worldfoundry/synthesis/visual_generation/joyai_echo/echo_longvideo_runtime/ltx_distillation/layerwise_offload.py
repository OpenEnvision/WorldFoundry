"""Layerwise CPU weight streaming for the Echo 1.5 DiT.

The manager keeps an immutable CPU master copy of every Transformer block and
only materializes the blocks needed by the current forward on CUDA. Copies use
a dedicated stream so the next block can overlap with current-block compute.
Immutable inference weights never travel from CUDA back to CPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerwiseOffloadReport:
    enabled: bool
    block_count: int
    resident_blocks: int
    prefetch_blocks: int
    cpu_weight_bytes: int
    pinned_weight_bytes: int


@dataclass
class _TensorRecord:
    parameter: torch.nn.Parameter
    cpu_tensor: torch.Tensor


class DiTLayerwiseOffload:
    """Stream ``transformer_blocks`` between CPU and one CUDA device.

    The first ``resident_blocks`` stay on CUDA for the denoising stage. All
    remaining blocks are prefetched in execution order and released after
    their forward hook. This is inference-only: weights must stay immutable.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        *,
        execution_device: torch.device | str,
        resident_blocks: int = 0,
        prefetch_blocks: int = 1,
        pin_memory: bool = True,
    ) -> None:
        self.generator = generator
        self.execution_device = torch.device(execution_device)
        if self.execution_device.type != "cuda":
            raise ValueError("DiT layerwise offload requires a CUDA execution device")
        self.blocks = list(generator.model.velocity_model.transformer_blocks)
        if not 0 <= resident_blocks <= len(self.blocks):
            raise ValueError(f"resident_blocks must be between 0 and {len(self.blocks)}, got {resident_blocks}")
        if prefetch_blocks < 1:
            raise ValueError("prefetch_blocks must be at least 1")

        self.resident_blocks = resident_blocks
        self.prefetch_blocks = prefetch_blocks
        self.pin_memory = pin_memory
        self._records: list[list[_TensorRecord]] = []
        self._events: dict[int, torch.cuda.Event] = {}
        self._loaded: set[int] = set()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._copy_stream: torch.cuda.Stream | None = None
        self._gpu_placeholders: dict[torch.dtype, torch.Tensor] = {}
        self._active = False
        self._cpu_weight_bytes = 0
        self._pinned_weight_bytes = 0

        self._capture_cpu_weights()
        self._register_hooks()

    @property
    def report(self) -> LayerwiseOffloadReport:
        return LayerwiseOffloadReport(
            enabled=True,
            block_count=len(self.blocks),
            resident_blocks=self.resident_blocks,
            prefetch_blocks=self.prefetch_blocks,
            cpu_weight_bytes=self._cpu_weight_bytes,
            pinned_weight_bytes=self._pinned_weight_bytes,
        )

    @staticmethod
    def _unique_parameters(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
        seen: set[int] = set()
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter

    def _cpu_copy(self, value: torch.Tensor) -> torch.Tensor:
        source = value.detach()
        if source.device.type != "cpu":
            source = source.to("cpu")
        result = torch.empty_strided(
            source.size(),
            source.stride(),
            dtype=source.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        result.copy_(source)
        return result

    def _capture_cpu_weights(self) -> None:
        """Capture CPU masters, then leave one-element registered placeholders."""

        if any(parameter.device.type != "cpu" for block in self.blocks for parameter in block.parameters()):
            raise ValueError("attach DiT layerwise offload while Transformer blocks are on CPU")

        for block in self.blocks:
            block_records: list[_TensorRecord] = []
            for parameter in self._unique_parameters(block):
                try:
                    cpu_tensor = self._cpu_copy(parameter)
                except RuntimeError as error:
                    if not self.pin_memory:
                        raise
                    # Pinned memory is an acceleration, not a correctness requirement.
                    if "pin memory" not in str(error).lower() and "cuda" not in str(error).lower():
                        raise
                    source = parameter.detach().cpu()
                    cpu_tensor = torch.empty_strided(source.size(), source.stride(), dtype=source.dtype, device="cpu")
                    cpu_tensor.copy_(source)
                size_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
                self._cpu_weight_bytes += size_bytes
                if cpu_tensor.is_pinned():
                    self._pinned_weight_bytes += size_bytes
                block_records.append(_TensorRecord(parameter=parameter, cpu_tensor=cpu_tensor))
                parameter.data = torch.empty(1, dtype=parameter.dtype, device="cpu")
            self._records.append(block_records)

    def _register_hooks(self) -> None:
        for block_index, block in enumerate(self.blocks):
            self._handles.append(
                block.register_forward_pre_hook(lambda _module, _args, index=block_index: self._before_block(index))
            )
            self._handles.append(
                block.register_forward_hook(
                    lambda _module, _args, output, index=block_index: self._after_block(index, output)
                )
            )

    def _materialize(self, block_index: int) -> None:
        if block_index in self._loaded:
            return
        if self._copy_stream is None:
            raise RuntimeError("layerwise offload is not active")
        with torch.cuda.device(self.execution_device), torch.cuda.stream(self._copy_stream):
            for record in self._records[block_index]:
                source = record.cpu_tensor
                target = torch.empty_strided(
                    source.size(),
                    source.stride(),
                    dtype=source.dtype,
                    device=self.execution_device,
                )
                target.copy_(source, non_blocking=source.is_pinned())
                record.parameter.data = target
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        self._events[block_index] = event
        self._loaded.add(block_index)

    def _wait_for(self, block_index: int) -> None:
        event = self._events.get(block_index)
        if event is None:
            raise RuntimeError(f"Transformer block {block_index} was not prefetched")
        current = torch.cuda.current_stream(self.execution_device)
        current.wait_event(event)
        for record in self._records[block_index]:
            record.parameter.data.record_stream(current)

    def _next_streamed(self, block_index: int) -> Iterable[int]:
        stop = min(len(self.blocks), block_index + self.prefetch_blocks + 1)
        for candidate in range(block_index + 1, stop):
            if candidate < self.resident_blocks or candidate in self._loaded:
                continue
            yield candidate

    def _before_block(self, block_index: int) -> None:
        if not self._active:
            return
        self._materialize(block_index)
        self._wait_for(block_index)
        for candidate in self._next_streamed(block_index):
            self._materialize(candidate)

    def _release(self, block_index: int) -> None:
        if block_index not in self._loaded:
            return
        for record in self._records[block_index]:
            placeholder = self._gpu_placeholders.setdefault(
                record.parameter.dtype,
                torch.empty(1, dtype=record.parameter.dtype, device=self.execution_device),
            )
            record.parameter.data = placeholder
        self._events.pop(block_index, None)
        self._loaded.remove(block_index)

    def _after_block(self, block_index: int, output):
        if self._active and block_index >= self.resident_blocks:
            self._release(block_index)
        return output

    def activate(self) -> None:
        if self._active:
            return
        with torch.cuda.device(self.execution_device):
            self._copy_stream = torch.cuda.Stream(device=self.execution_device)
        # Only one-element block placeholders move here; non-block weights move normally.
        self.generator.to(self.execution_device)
        self._active = True
        for block_index in range(self.resident_blocks):
            self._materialize(block_index)
        if self.resident_blocks == 0:
            for block_index in range(min(len(self.blocks), self.prefetch_blocks)):
                self._materialize(block_index)
        for block_index in range(self.resident_blocks):
            self._wait_for(block_index)

    def deactivate(self) -> None:
        if not self._active:
            return
        torch.cuda.synchronize(self.execution_device)
        for block_index in list(self._loaded):
            self._release(block_index)
        self._active = False
        self._copy_stream = None
        self.generator.to("cpu")
        self._gpu_placeholders.clear()

    def close(self) -> None:
        self.deactivate()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
