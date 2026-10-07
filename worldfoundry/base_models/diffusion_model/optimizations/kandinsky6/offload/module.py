"""One resident module group at a time (text, DiT, VAE, vocoder)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.offload.common import OffloadStrategy, _iter_named, _module_to, _pin_module


class ModuleOffload:
    """Keep registered modules on CPU; bring named ones to CUDA for a stage."""

    strategy: OffloadStrategy = "module"

    def __init__(
        self,
        compute_device: torch.device,
        *,
        storage_device: torch.device | None = None,
        pin_memory: bool = True,
    ):
        if compute_device.type != "cuda":
            raise ValueError("ModuleOffload requires a CUDA compute device")
        self.compute_device = torch.device(compute_device)
        self.storage_device = storage_device or torch.device("cpu")
        self.pin_memory = pin_memory
        self._modules: dict[str, Any] = {}
        self._on_compute: set[str] = set()
        self._ready: dict[str, torch.cuda.Event | None] = {}
        self._xfer = torch.cuda.Stream(device=self.compute_device)

    def register(self, name: str, module: Any) -> None:
        _module_to(module, self.storage_device, non_blocking=False)
        if self.pin_memory:
            _pin_module(module)
        self._modules[name] = module
        self._on_compute.discard(name)
        self._ready[name] = None

    def prefetch(self, *names: str) -> None:
        for name, module in _iter_named(self._modules, names):
            if name in self._on_compute:
                continue
            with torch.cuda.stream(self._xfer):
                _module_to(module, self.compute_device, non_blocking=True)
                event = self._xfer.record_event()
            self._ready[name] = event
            self._on_compute.add(name)

    def _wait_ready(self, name: str) -> None:
        event = self._ready.get(name)
        if event is not None:
            torch.cuda.current_stream(self.compute_device).wait_event(event)
            self._ready[name] = None

    def _ensure(self, *names: str) -> None:
        missing = [n for n, _ in _iter_named(self._modules, names) if n not in self._on_compute]
        if missing:
            self.prefetch(*missing)
        for name, _ in _iter_named(self._modules, names):
            self._wait_ready(name)

    def release(self, *names: str) -> None:
        """Schedule async D2H for named modules after current compute finishes."""
        if not names:
            return
        done = torch.cuda.current_stream(self.compute_device).record_event()
        with torch.cuda.stream(self._xfer):
            self._xfer.wait_event(done)
            for name, module in _iter_named(self._modules, names):
                if name not in self._on_compute:
                    continue
                _module_to(module, self.storage_device, non_blocking=True)
                self._on_compute.discard(name)
                self._ready[name] = None
                if self.pin_memory:
                    _pin_module(module)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    @contextmanager
    def use(
        self,
        *names: str,
        prefetch: str | Sequence[str] | None = None,
    ) -> Iterator[None]:
        self._ensure(*names)
        if prefetch is not None:
            pref = (prefetch,) if isinstance(prefetch, str) else tuple(prefetch)
            # Overlap next H2D with this stage's compute.
            self.prefetch(*pref)
        try:
            yield
        finally:
            self.release(*names)
            self.empty_cache()
