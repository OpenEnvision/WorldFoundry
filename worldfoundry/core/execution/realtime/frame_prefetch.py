# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-to-host frame prefetch primitives for realtime presentation.

Responsibility: start a non-blocking D2H copy onto pinned host memory
so presentation can overlap with the next decode, then materialize a
contiguous numpy view only when transport asks.

This module is not a video decoder, display pipeline, or CUDA Graph
participant. Torch is imported lazily so CPU-only Studio and manifest
tooling do not acquire an accelerator dependency. Failed prefetch
returns ``False`` / falls back to a blocking ``.cpu()`` — it must not
raise into the decode loop.

Public surface:
- :class:`CudaHostPrefetch` — one tensor, one side-stream copy.
- :class:`LazyCudaFrame` — keep a decoded frame on CUDA until host
  materialization; implements ``__array__``.
- :func:`prefetch_to_numpy` — duck-typed kick for any lazy frame.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import numpy.typing as npt

# Presentation workers may prefetch from multiple threads. Keep lazy stream
# creation and cache reads under this lock so first access cannot create two
# streams with unrelated copy ordering for the same device.
_STREAMS_LOCK = threading.Lock()
_HOST_COPY_STREAMS: dict[int, Any] = {}


# ──────────────────────────────────────────────────────────────────────────
# Side-stream D2H — pin, copy, record; never raise into the decode loop
# ──────────────────────────────────────────────────────────────────────────


class CudaHostPrefetch:
    """Stage one CUDA tensor into pinned host memory on a reusable side stream."""

    def __init__(self, tensor: Any, *, source_event: Any | None = None) -> None:
        """Hold the source tensor; copy starts only on :meth:`start`."""
        self._tensor = tensor
        self._source_event = source_event
        self._host_tensor: Any | None = None
        self._done_event: Any | None = None
        self._started = False

    @property
    def started(self) -> bool:
        """True after :meth:`start` ran, even when the copy was declined."""
        return self._started

    def start(self) -> bool:
        """Begin a non-blocking D2H copy; ``False`` means stay on the fallback path.

        Missing torch, a non-CUDA tensor, or any copy exception is a
        declined prefetch, not a crash. ``record_stream`` keeps the source
        alive until the side stream finishes even if the producer frees it.
        """
        if self._started:
            return self._host_tensor is not None
        self._started = True
        try:
            import torch
        except ImportError:
            return False
        if not torch.is_tensor(self._tensor) or not self._tensor.is_cuda:
            return False
        try:
            host_tensor = torch.empty(
                tuple(self._tensor.shape),
                dtype=self._tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            copy_stream = _host_copy_stream(torch, self._tensor.device)
            with torch.cuda.device(self._tensor.device):
                if self._source_event is not None:
                    copy_stream.wait_event(self._source_event)
                with torch.cuda.stream(copy_stream):
                    host_tensor.copy_(self._tensor, non_blocking=True)
                    self._tensor.record_stream(copy_stream)
                    done_event = torch.cuda.Event()
                    done_event.record(copy_stream)
        except Exception:
            return False
        self._host_tensor = host_tensor
        self._done_event = done_event
        return True

    def to_numpy(self) -> np.ndarray:
        """Block on the copy event and return a contiguous host view."""
        if self._host_tensor is None:
            raise RuntimeError("CUDA host prefetch was not started successfully.")
        if self._done_event is not None:
            self._done_event.synchronize()
        return np.ascontiguousarray(self._host_tensor.numpy())


# ──────────────────────────────────────────────────────────────────────────
# Lazy frame — CUDA until transport; host materialization drops the source
# ──────────────────────────────────────────────────────────────────────────


class LazyCudaFrame:
    """Keep a decoded frame on CUDA until transport requests host materialization."""

    def __init__(
        self,
        frames_hwc_uint8: Any,
        frame_index: int,
        *,
        source_event: object | None = None,
    ) -> None:
        """Alias one index in a CUDA batch; do not copy until prefetch or ``to_numpy``."""
        self._frames_hwc_uint8: Any | None = frames_hwc_uint8
        self._frame_index = int(frame_index)
        self._source_event = source_event
        self._host: np.ndarray | None = None
        self._prefetch: CudaHostPrefetch | None = None

    def prefetch_to_numpy(self) -> None:
        """Kick a side-stream copy if the frame is still on CUDA and uncached."""
        if self._host is not None or self._prefetch is not None or self._frames_hwc_uint8 is None:
            return
        frame = self._frames_hwc_uint8[self._frame_index].detach()
        prefetch = CudaHostPrefetch(frame, source_event=self._source_event)
        if prefetch.start():
            self._prefetch = prefetch

    def to_numpy(self) -> np.ndarray:
        """Materialize host pixels once, then drop the CUDA batch alias.

        After this returns the CUDA source is cleared so a later
        :meth:`to_cuda_tensor` cannot observe a stale batch that the
        decoder already reused.
        """
        if self._host is not None:
            return self._host
        if self._prefetch is not None:
            self._host = self._prefetch.to_numpy()
        else:
            if self._frames_hwc_uint8 is None:
                raise RuntimeError("Lazy CUDA frame lost its source before materialization.")
            synchronize = getattr(self._source_event, "synchronize", None)
            if callable(synchronize):
                synchronize()
            self._host = np.ascontiguousarray(self._frames_hwc_uint8[self._frame_index].detach().cpu().numpy())
        self._frames_hwc_uint8 = None
        self._prefetch = None
        return self._host

    def to_cuda_tensor(self) -> Any:
        """Return the still-resident CUDA slice; refuse after host materialization."""
        if self._frames_hwc_uint8 is None:
            raise RuntimeError("Lazy CUDA frame was already materialized on the host.")
        return self._frames_hwc_uint8[self._frame_index]

    def to_cuda_event(self) -> object | None:
        """Producer event while the CUDA source lives; ``None`` after host materialization."""
        return self._source_event if self._frames_hwc_uint8 is not None else None

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """NumPy protocol: honor ``copy=False`` only when the dtype already matches."""
        array = self.to_numpy()
        if dtype is not None:
            target_dtype = np.dtype(dtype)
            if copy is False and target_dtype != array.dtype:
                raise ValueError("Cannot honor copy=False while converting the frame dtype.")
            array = array.astype(target_dtype, copy=False)
        return np.array(array, copy=True) if copy is True else array


def prefetch_to_numpy(frame: object) -> None:
    """Start host materialization when a frame exposes the lazy protocol."""

    prefetch = getattr(frame, "prefetch_to_numpy", None)
    if callable(prefetch):
        prefetch()


def _host_copy_stream(torch: Any, device: Any) -> Any:
    """Reuse one side stream per device index so copies stay ordered per GPU."""
    index = getattr(device, "index", None)
    key = 0 if index is None else int(index)
    with _STREAMS_LOCK:
        stream = _HOST_COPY_STREAMS.get(key)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
            _HOST_COPY_STREAMS[key] = stream
        return stream


__all__ = ["CudaHostPrefetch", "LazyCudaFrame", "prefetch_to_numpy"]
