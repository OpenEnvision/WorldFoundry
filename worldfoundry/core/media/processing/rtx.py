# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional NVIDIA RTX Video Super Resolution post-processing.

Responsibility: apply NVIDIA VFX super-resolution / deblur / denoise
while preserving the input layout, container, dtype, and numeric range
so the processor can sit in the shared Studio frame pipeline.

Not this module: the layout / session ABI lives in
:mod:`worldfoundry.core.media.processing.postprocess`. Encode / decode lives in
:mod:`worldfoundry.core.media.codecs.video`. The proprietary ``nvidia-vfx``
runtime is never imported unless an RTX processor is explicitly
selected or a probe is requested.

Public surface: :class:`RTXVideoSuperResolutionConfig`,
:class:`RTXVideoSuperResolutionPostProcessor`,
:class:`RTXVFXCapability`, :func:`inspect_rtx_vfx_capability`,
:func:`probe_rtx_vfx_runtime`, :func:`require_rtx_vfx_runtime`,
:func:`rtx_postprocessor_from_preset`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from ctypes import util as ctypes_util
from dataclasses import dataclass, replace
from importlib import metadata, util
from pathlib import Path
from typing import Any, Literal, get_args

from worldfoundry.core.media.processing.postprocess import (
    VideoChunk,
    VideoPostProcessor,
    VideoPostProcessorSession,
    VideoSpec,
    VideoTensorLayout,
)

RTXVideoSuperResolutionQuality = Literal[
    "BICUBIC",
    "LOW",
    "MEDIUM",
    "HIGH",
    "ULTRA",
    "DENOISE_LOW",
    "DENOISE_MEDIUM",
    "DENOISE_HIGH",
    "DENOISE_ULTRA",
    "DEBLUR_LOW",
    "DEBLUR_MEDIUM",
    "DEBLUR_HIGH",
    "DEBLUR_ULTRA",
    "HIGHBITRATE_LOW",
    "HIGHBITRATE_MEDIUM",
    "HIGHBITRATE_HIGH",
    "HIGHBITRATE_ULTRA",
]
RTXVideoFloatRange = Literal["minus-one-one", "zero-one"]
RTXVFXProbeStatus = Literal["not-run", "passed", "failed"]

_QUALITY_NAMES = tuple(get_args(RTXVideoSuperResolutionQuality))
_SAME_RESOLUTION_QUALITIES = frozenset(
    {
        "DENOISE_LOW",
        "DENOISE_MEDIUM",
        "DENOISE_HIGH",
        "DENOISE_ULTRA",
        "DEBLUR_LOW",
        "DEBLUR_MEDIUM",
        "DEBLUR_HIGH",
        "DEBLUR_ULTRA",
    }
)
RTX_POSTPROCESS_PRESET_NAMES = (
    "rtx-super-resolution",
    "rtx-super-resolution-ultra",
    "rtx-deblur-ultra",
)

# ──────────────────────────────────────────────────────────────────────────
# Config + factory — quality modes cap scale; nvidia-vfx stays unimported
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class RTXVideoSuperResolutionConfig:
    """Configuration for one RTX VSR processor factory.

    Floating-point input is interpreted using ``float_range`` and emitted in
    the same range and dtype. ``uint8`` input is interpreted and emitted as
    display-ready RGB in ``[0, 255]``.
    """

    scale: float = 2.0
    output_width: int | None = None
    output_height: int | None = None
    quality: RTXVideoSuperResolutionQuality = "HIGH"
    device: int = 0
    clamp_input: bool = True
    non_blocking: bool = False
    use_current_stream: bool = True
    float_range: RTXVideoFloatRange = "minus-one-one"

    def __post_init__(self) -> None:
        """Reject illegal quality / device / scale combinations before a session starts."""

        if self.quality not in _QUALITY_NAMES:
            available = ", ".join(_QUALITY_NAMES)
            raise ValueError(f"Unsupported RTX VSR quality {self.quality!r}. Supported values: {available}.")
        if self.device < 0:
            raise ValueError(f"RTX VSR CUDA device index must be non-negative; got {self.device}.")
        if self.float_range not in {"minus-one-one", "zero-one"}:
            raise ValueError(f"Unsupported RTX VSR floating-point range {self.float_range!r}.")
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("RTX VSR requires both output_width and output_height when either is configured.")
        if self.output_width is not None and self.output_width <= 0:
            raise ValueError("RTX VSR output_width must be positive.")
        if self.output_height is not None and self.output_height <= 0:
            raise ValueError("RTX VSR output_height must be positive.")
        if self.output_width is None and self.scale <= 0:
            raise ValueError(f"RTX VSR scale must be positive; got {self.scale}.")

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Return the layout-independent stream specification emitted by VSR."""

        if input_spec.channels != 3:
            raise ValueError(f"RTX VSR expects RGB input with 3 channels; got {input_spec.channels}.")
        output_height, output_width = self.output_dimensions(input_spec)
        return VideoSpec(
            height=output_height,
            width=output_width,
            fps=input_spec.fps,
            channels=3,
            dtype=input_spec.dtype,
        )

    def output_dimensions(self, input_spec: VideoSpec) -> tuple[int, int]:
        """Resolve explicit size or ``scale``; same-resolution qualities cannot resize.

        Standard / high-bitrate modes refuse downscale and cap at 4x so a
        mis-set Studio override cannot ask VFX for an unsupported geometry.
        """

        if self.output_width is not None and self.output_height is not None:
            output_height = self.output_height
            output_width = self.output_width
        else:
            output_height = int(round(input_spec.height * self.scale))
            output_width = int(round(input_spec.width * self.scale))
        if output_height <= 0 or output_width <= 0:  # guarded above, retained for rounded scales
            raise ValueError(f"RTX VSR output dimensions must be positive; got {output_height}x{output_width}.")
        if self.quality in _SAME_RESOLUTION_QUALITIES:
            if (output_height, output_width) != (input_spec.height, input_spec.width):
                raise ValueError(
                    f"RTX VSR quality {self.quality!r} is a same-resolution mode; use scale=1.0 or explicit "
                    f"dimensions {input_spec.width}x{input_spec.height}."
                )
        elif output_height < input_spec.height or output_width < input_spec.width:
            raise ValueError("RTX VSR standard and high-bitrate modes do not support downscaling.")
        if output_height > input_spec.height * 4 or output_width > input_spec.width * 4:
            raise ValueError("RTX VSR supports at most 4x spatial upscaling per dimension.")
        return output_height, output_width


class RTXVideoSuperResolutionPostProcessor(VideoPostProcessor):
    """Preserve WorldFoundry video contracts while applying NVIDIA RTX VSR."""

    def __init__(
        self,
        config: RTXVideoSuperResolutionConfig | None = None,
        **overrides: Any,
    ) -> None:
        """Accept either a frozen config or kwargs; mixing them would silently drop fields."""

        if config is not None and overrides:
            raise TypeError("Pass either an RTX VSR config or keyword overrides, not both.")
        self.config = config or RTXVideoSuperResolutionConfig(**overrides)

    @property
    def name(self) -> str:
        """Stable chain id; presets share this name so telemetry stays one series."""

        return "rtx-video-super-resolution"

    @property
    def processing_device(self) -> str:
        """Use the configured VFX device instead of the caller's current GPU."""

        return f"cuda:{self.config.device}"

    def output_spec(self, input_spec: VideoSpec) -> VideoSpec:
        """Delegate geometry to the config so the chain can size the sink first."""

        return self.config.output_spec(input_spec)

    def start(self, spec: VideoSpec) -> VideoPostProcessorSession:
        """Open a session; the native effect is created later in ``prepare``."""

        return _RTXVideoSuperResolutionSession(self.config, spec)


@dataclass(frozen=True, slots=True)
class RTXVFXCapability:
    """Read-only diagnosis for the optional proprietary runtime."""

    available: bool
    device: int
    reason: str
    package_version: str | None = None
    gpu_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    driver_library: str | None = None
    probe_status: RTXVFXProbeStatus = "not-run"
    backend_error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe diagnosis; ``compute_capability`` is a list so manifests need no encoder."""

        return {
            "available": self.available,
            "device": self.device,
            "reason": self.reason,
            "package_version": self.package_version,
            "gpu_name": self.gpu_name,
            "compute_capability": list(self.compute_capability) if self.compute_capability else None,
            "driver_library": self.driver_library,
            "probe_status": self.probe_status,
            "backend_error": self.backend_error,
        }


# ──────────────────────────────────────────────────────────────────────────
# Capability probes — static checks first; native effect only on request
# ──────────────────────────────────────────────────────────────────────────


def inspect_rtx_vfx_capability(*, device: int = 0) -> RTXVFXCapability:
    """Inspect static prerequisites without creating a VFX effect."""

    package_version = _installed_nvidia_vfx_version()
    try:
        package_spec = util.find_spec("nvvfx")
    except (ImportError, ModuleNotFoundError, ValueError):
        package_spec = None
    if package_spec is None:
        return RTXVFXCapability(
            available=False,
            device=device,
            reason="The optional nvidia-vfx package is not installed.",
            package_version=package_version,
        )
    try:
        torch = _import_torch()
    except RuntimeError as exc:
        return RTXVFXCapability(
            available=False,
            device=device,
            reason=str(exc),
            package_version=package_version,
        )
    if not torch.cuda.is_available():
        return RTXVFXCapability(
            available=False,
            device=device,
            reason="PyTorch cannot access CUDA; NVIDIA VFX requires a supported NVIDIA GPU.",
            package_version=package_version,
        )
    device_count = int(torch.cuda.device_count())
    if device < 0 or device >= device_count:
        return RTXVFXCapability(
            available=False,
            device=device,
            reason=f"CUDA device {device} is unavailable; PyTorch exposes {device_count} device(s).",
            package_version=package_version,
        )
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    gpu_name = str(torch.cuda.get_device_name(device))
    if capability < (7, 5):
        return RTXVFXCapability(
            available=False,
            device=device,
            reason=(
                f"GPU {gpu_name!r} has compute capability {capability[0]}.{capability[1]}; "
                "nvidia-vfx requires Turing-class (7.5) or newer Tensor Cores."
            ),
            package_version=package_version,
            gpu_name=gpu_name,
            compute_capability=capability,
        )
    driver_library, driver_error = _inspect_required_driver_library()
    if driver_error is not None:
        return RTXVFXCapability(
            available=False,
            device=device,
            reason=driver_error,
            package_version=package_version,
            gpu_name=gpu_name,
            compute_capability=capability,
            driver_library=driver_library,
        )
    return RTXVFXCapability(
        available=True,
        device=device,
        reason=(
            "Static nvidia-vfx, CUDA, GPU, and driver-library prerequisites are available; "
            "the native effect has not been probed."
        ),
        package_version=package_version,
        gpu_name=gpu_name,
        compute_capability=capability,
        driver_library=driver_library,
    )


def probe_rtx_vfx_runtime(
    *,
    device: int = 0,
    config: RTXVideoSuperResolutionConfig | None = None,
    input_spec: VideoSpec | None = None,
    run_inference: bool = True,
) -> RTXVFXCapability:
    """Load the native effect and optionally process one representative frame."""

    capability = inspect_rtx_vfx_capability(device=device)
    if not capability.available:
        return replace(capability, probe_status="failed")

    resolved_config = config or RTXVideoSuperResolutionConfig(device=device)
    if resolved_config.device != device:
        raise ValueError(
            f"RTX VFX probe device cuda:{device} does not match config device cuda:{resolved_config.device}."
        )
    resolved_spec = input_spec or VideoSpec(
        height=360,
        width=640,
        channels=3,
        dtype="uint8",
    )
    session = _RTXVideoSuperResolutionSession(resolved_config, resolved_spec)
    try:
        session.prepare()
        if run_inference:
            torch = _import_torch()
            frame = torch.zeros(
                (1, resolved_spec.height, resolved_spec.width, 3),
                dtype=torch.uint8,
            )
            session.process(VideoChunk(frames=frame, layout="thwc"))
    except Exception as exc:
        backend_error = _exception_chain_message(exc)
        return replace(
            capability,
            available=False,
            reason=_runtime_probe_failure_reason(
                device=device,
                backend_error=backend_error,
            ),
            probe_status="failed",
            backend_error=backend_error,
        )
    finally:
        session.close()

    action = "loaded and processed a CUDA frame" if run_inference else "loaded"
    return replace(
        capability,
        reason=f"The native nvidia-vfx effect {action} successfully on cuda:{device}.",
        probe_status="passed",
        backend_error=None,
    )


def require_rtx_vfx_runtime(
    *,
    device: int = 0,
    config: RTXVideoSuperResolutionConfig | None = None,
    input_spec: VideoSpec | None = None,
    run_inference: bool = True,
) -> RTXVFXCapability:
    """Fully probe the selected RTX runtime and fail with an actionable message."""

    capability = probe_rtx_vfx_runtime(
        device=device,
        config=config,
        input_spec=input_spec,
        run_inference=run_inference,
    )
    if not capability.available:
        raise RuntimeError(
            f"RTX video post-processing is unavailable: {capability.reason} "
            "Install `worldfoundry[rtx_postprocess]` and use a supported NVIDIA GPU/driver."
        )
    return capability


def rtx_postprocessor_from_preset(
    preset: str,
    *,
    scale: float | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    quality: RTXVideoSuperResolutionQuality | None = None,
    device: int | None = None,
    non_blocking: bool | None = None,
    use_current_stream: bool | None = None,
) -> RTXVideoSuperResolutionPostProcessor:
    """Resolve a stable named preset, then apply explicit operator overrides."""

    normalized = preset.strip().lower().replace("_", "-")
    configs = {
        "rtx-super-resolution": RTXVideoSuperResolutionConfig(),
        "rtx-super-resolution-ultra": RTXVideoSuperResolutionConfig(quality="ULTRA"),
        "rtx-deblur-ultra": RTXVideoSuperResolutionConfig(scale=1.0, quality="DEBLUR_ULTRA"),
    }
    try:
        config = configs[normalized]
    except KeyError as exc:
        available = ", ".join(RTX_POSTPROCESS_PRESET_NAMES)
        raise ValueError(f"Unknown RTX postprocess preset {preset!r}. Available presets: {available}.") from exc
    overrides: dict[str, Any] = {}
    for key, value in (
        ("scale", scale),
        ("output_width", output_width),
        ("output_height", output_height),
        ("quality", quality),
        ("device", device),
        ("non_blocking", non_blocking),
        ("use_current_stream", use_current_stream),
    ):
        if value is not None:
            overrides[key] = value
    return RTXVideoSuperResolutionPostProcessor(replace(config, **overrides))


# ──────────────────────────────────────────────────────────────────────────
# Session + layout canonicalize — BVTCHW unit float in, original domain out
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _CanonicalInput:
    """One chunk rewritten to BVTCHW plus enough metadata to invert the rewrite."""

    tensor: Any
    layout: VideoTensorLayout
    source_kind: Literal["numpy", "torch", "frame-list-numpy", "frame-list-torch"]
    source_device: Any
    source_dtype: Any
    uint8: bool


class _RTXVideoSuperResolutionSession(VideoPostProcessorSession):
    """Own one VFX effect for a stream; ``flush`` closes the native handle."""

    def __init__(self, config: RTXVideoSuperResolutionConfig, spec: VideoSpec) -> None:
        """Bind config and specs; the effect is created on first ``prepare`` / ``process``."""

        self._config = config
        self._input_spec = spec
        self._output_spec = config.output_spec(spec)
        self._effect: Any | None = None
        self._closed = False

    def prepare(self) -> None:
        """Load the native effect before the first measured chunk."""

        if self._closed:
            raise RuntimeError("cannot prepare RTX VSR after flush()")
        self._ensure_effect()

    def process(self, chunk: VideoChunk) -> list[VideoChunk]:
        """Upscale one chunk and restore the caller's layout / dtype / range."""

        if self._closed:
            raise RuntimeError("cannot process RTX VSR after flush()")
        torch = _import_torch()
        with torch.no_grad():
            canonical = _canonicalize_chunk(chunk, spec=self._input_spec)
            frames = int(canonical.tensor.shape[2])
            if frames:
                output_unit = self._run_vsr(canonical.tensor, self._ensure_effect())
            else:
                shape = (
                    int(canonical.tensor.shape[0]),
                    int(canonical.tensor.shape[1]),
                    0,
                    3,
                    self._output_spec.height,
                    self._output_spec.width,
                )
                output_unit = canonical.tensor.new_empty(shape, dtype=torch.float32)
            restored_domain = _restore_numeric_domain(output_unit, canonical=canonical, config=self._config)
            output = _restore_layout_and_container(restored_domain, canonical=canonical)
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "postprocess": "rtx-video-super-resolution",
                "rtx_quality": self._config.quality,
                "rtx_output_size": [self._output_spec.width, self._output_spec.height],
            }
        )
        return [VideoChunk(frames=output, layout=chunk.layout, metadata=metadata)]

    def flush(self) -> list[VideoChunk]:
        """Close the native effect; VFX does not buffer a tail."""

        if self._closed:
            return []
        self._closed = True
        if self._effect is not None:
            self._effect.close()
            self._effect = None
        return []

    def _ensure_effect(self) -> Any:
        """Create and ``load`` the VFX effect once; a failed load must close the handle."""

        if self._effect is not None:
            return self._effect
        video_super_res = _load_video_super_res_class()
        effect: Any | None = None
        try:
            effect = video_super_res(
                quality=_resolve_quality(video_super_res, self._config.quality),
                device=self._config.device,
            )
            effect.output_width = self._output_spec.width
            effect.output_height = self._output_spec.height
            effect.load()
        except Exception as exc:
            if effect is not None:
                try:
                    effect.close()
                except Exception:
                    pass
            raise RuntimeError(
                "Failed to initialize NVIDIA RTX VSR "
                f"on cuda:{self._config.device} at {self._output_spec.width}x{self._output_spec.height} "
                f"with quality {self._config.quality!r}. Backend error: {exc}. "
                "Check the nvidia-vfx platform and driver requirements."
            ) from exc
        self._effect = effect
        return effect

    def _run_vsr(self, canonical: Any, effect: Any) -> Any:
        """Run VFX one RGB frame at a time; clone DLPack so the next ``run`` can reuse the buffer."""

        torch = _import_torch()
        batch, views, frames, _, _, _ = canonical.shape
        device = _torch_device_for_nvvfx_device(self._config.device)
        stream_ptr = _current_cuda_stream_ptr(device=device, enabled=self._config.use_current_stream)
        outputs: list[Any] = []
        for batch_index in range(batch):
            for view_index in range(views):
                for frame_index in range(frames):
                    frame = _frame_to_unit_float(
                        canonical[batch_index, view_index, frame_index],
                        device=device,
                        config=self._config,
                    )
                    result = effect.run(
                        frame,
                        non_blocking=self._config.non_blocking,
                        stream_ptr=stream_ptr,
                    )
                    _synchronize_nonblocking_output(device=device, enabled=self._config.non_blocking)
                    output = torch.from_dlpack(result.image).clone()
                    expected_shape = (3, self._output_spec.height, self._output_spec.width)
                    if tuple(int(value) for value in output.shape) != expected_shape:
                        raise RuntimeError(
                            f"NVIDIA RTX VSR emitted shape {tuple(output.shape)}; expected {expected_shape}."
                        )
                    outputs.append(output)
        return torch.stack(outputs).reshape(
            batch,
            views,
            frames,
            3,
            self._output_spec.height,
            self._output_spec.width,
        )


def _canonicalize_chunk(chunk: VideoChunk, *, spec: VideoSpec) -> _CanonicalInput:
    """Rewrite a chunk to BVTCHW without mixing NumPy and torch frame-list entries."""

    torch = _import_torch()
    frames = chunk.frames
    if chunk.layout == "frame-list":
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes, bytearray)):
            raise TypeError("RTX VSR frame-list input must be a sequence of HWC arrays or tensors.")
        if not frames:
            tensor = torch.empty((1, 1, 0, 3, spec.height, spec.width), dtype=torch.uint8)
            return _CanonicalInput(tensor, chunk.layout, "frame-list-numpy", torch.device("cpu"), tensor.dtype, True)
        first = frames[0]
        if torch.is_tensor(first):
            source_kind: Literal["frame-list-numpy", "frame-list-torch"] = "frame-list-torch"
            source_device = first.device
            converted = [frame if torch.is_tensor(frame) else _raise_mixed_frame_list() for frame in frames]
        else:
            np = _import_numpy()
            if not isinstance(first, np.ndarray):
                raise TypeError("RTX VSR frame-list input supports NumPy arrays or PyTorch tensors.")
            source_kind = "frame-list-numpy"
            source_device = torch.device("cpu")
            converted = [
                torch.as_tensor(np.ascontiguousarray(frame))
                if isinstance(frame, np.ndarray)
                else _raise_mixed_frame_list()
                for frame in frames
            ]
        try:
            tensor = torch.stack(converted)
        except RuntimeError as exc:
            raise ValueError("RTX VSR frame-list entries must share shape, dtype, and device.") from exc
        canonical = tensor.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
    else:
        if torch.is_tensor(frames):
            source_kind = "torch"
            source_device = frames.device
            tensor = frames
        else:
            np = _import_numpy()
            if not isinstance(frames, np.ndarray):
                raise TypeError("RTX VSR tensor-layout input supports NumPy arrays or PyTorch tensors.")
            source_kind = "numpy"
            source_device = torch.device("cpu")
            tensor = torch.as_tensor(np.ascontiguousarray(frames))
        canonical = _to_bvtchw(tensor, layout=chunk.layout)
    _validate_canonical_input(canonical, spec=spec)
    return _CanonicalInput(
        tensor=canonical,
        layout=chunk.layout,
        source_kind=source_kind,
        source_device=source_device,
        source_dtype=canonical.dtype,
        uint8=canonical.dtype == torch.uint8,
    )


def _to_bvtchw(tensor: Any, *, layout: VideoTensorLayout) -> Any:
    """Permute a tensor layout into BVTCHW; ``frame-list`` is handled by the caller."""

    if layout == "thwc":
        return tensor.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
    if layout == "tchw":
        return tensor.unsqueeze(0).unsqueeze(0)
    if layout == "bthwc":
        return tensor.permute(0, 1, 4, 2, 3).unsqueeze(1)
    if layout == "btchw":
        return tensor.unsqueeze(1)
    if layout == "bcthw":
        return tensor.permute(0, 2, 1, 3, 4).unsqueeze(1)
    if layout == "bvthwc":
        return tensor.permute(0, 1, 2, 5, 3, 4)
    if layout == "bvtchw":
        return tensor
    raise ValueError(f"Unsupported RTX VSR layout {layout!r}.")


def _from_bvtchw(tensor: Any, *, layout: VideoTensorLayout) -> Any:
    """Invert :func:`_to_bvtchw` so the sink sees the same layout the generator sent."""

    if layout in {"frame-list", "thwc"}:
        return tensor[0, 0].permute(0, 2, 3, 1)
    if layout == "tchw":
        return tensor[0, 0]
    if layout == "bthwc":
        return tensor[:, 0].permute(0, 1, 3, 4, 2)
    if layout == "btchw":
        return tensor[:, 0]
    if layout == "bcthw":
        return tensor[:, 0].permute(0, 2, 1, 3, 4)
    if layout == "bvthwc":
        return tensor.permute(0, 1, 2, 4, 5, 3)
    if layout == "bvtchw":
        return tensor
    raise ValueError(f"Unsupported RTX VSR layout {layout!r}.")


def _validate_canonical_input(canonical: Any, *, spec: VideoSpec) -> None:
    """Require RGB BVTCHW whose H/W match the frozen stream spec."""

    torch = _import_torch()
    if canonical.ndim != 6:
        raise ValueError(f"RTX VSR canonical input must be BVTCHW; got shape {tuple(canonical.shape)}.")
    _, _, _, channels, height, width = canonical.shape
    if int(channels) != 3:
        raise ValueError(f"RTX VSR expects RGB input with 3 channels; got {channels}.")
    if (int(height), int(width)) != (spec.height, spec.width):
        raise ValueError(
            f"RTX VSR stream dimensions changed from {spec.width}x{spec.height} to {int(width)}x{int(height)}."
        )
    if canonical.dtype != torch.uint8 and not torch.is_floating_point(canonical):
        raise TypeError(f"RTX VSR supports uint8 or floating-point RGB input; got {canonical.dtype}.")


def _frame_to_unit_float(frame: Any, *, device: Any, config: RTXVideoSuperResolutionConfig) -> Any:
    """Map uint8 or signed-float RGB onto the ``[0, 1]`` domain VFX expects."""

    torch = _import_torch()
    if frame.dtype == torch.uint8:
        return frame.to(device=device, dtype=torch.float32, non_blocking=True).mul_(1.0 / 255.0).contiguous()
    converted = frame.to(device=device, dtype=torch.float32, non_blocking=True)
    if config.float_range == "minus-one-one":
        if config.clamp_input:
            converted = converted.clamp(-1.0, 1.0)
        converted = converted.add(1.0).mul(0.5)
    elif config.clamp_input:
        converted = converted.clamp(0.0, 1.0)
    return converted.contiguous()


def _restore_numeric_domain(
    output_unit: Any,
    *,
    canonical: _CanonicalInput,
    config: RTXVideoSuperResolutionConfig,
) -> Any:
    """Invert :func:`_frame_to_unit_float` so uint8 stays display-ready and signed float stays signed."""

    if canonical.uint8:
        return output_unit.clamp(0.0, 1.0).mul(255.0).round().to(dtype=canonical.source_dtype)
    if config.float_range == "minus-one-one":
        output_unit = output_unit.mul(2.0).sub(1.0)
    return output_unit.to(dtype=canonical.source_dtype)


def _restore_layout_and_container(output: Any, *, canonical: _CanonicalInput) -> Any:
    """Return NumPy / torch / frame-list in the original device so Studio does not recast."""

    restored = _from_bvtchw(output, layout=canonical.layout).to(device=canonical.source_device)
    if canonical.source_kind == "frame-list-torch":
        return [frame.contiguous() for frame in restored]
    if canonical.source_kind == "frame-list-numpy":
        return [frame.contiguous().cpu().numpy() for frame in restored]
    if canonical.source_kind == "numpy":
        return restored.contiguous().cpu().numpy()
    return restored.contiguous()


# ──────────────────────────────────────────────────────────────────────────
# Optional runtime — lazy nvvfx / torch / numpy; never import at module load
# ──────────────────────────────────────────────────────────────────────────


def _load_video_super_res_class() -> Any:
    """Import ``VideoSuperRes`` only when a session or probe actually needs it."""

    try:
        from nvvfx import VideoSuperRes  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "RTX video post-processing requires the optional proprietary `nvidia-vfx` package. "
            "Install `worldfoundry[rtx_postprocess]` and verify the NVIDIA driver requirements."
        ) from exc
    return VideoSuperRes


def _resolve_quality(video_super_res: Any, quality: str) -> Any:
    """Map a string quality onto the installed wheel's enum (nested or module-level)."""

    quality_level = getattr(video_super_res, "QualityLevel", None)
    if quality_level is None:
        quality_level = _load_quality_level_enum()
    try:
        return getattr(quality_level, quality)
    except AttributeError as exc:
        available = ", ".join(_QUALITY_NAMES)
        raise ValueError(f"Unsupported RTX VSR quality {quality!r}. Supported values: {available}.") from exc


def _load_quality_level_enum() -> Any:
    """Load the enum from its real 0.1.x location.

    NVIDIA's generated API page presents it as ``VideoSuperRes.QualityLevel``,
    while the published 0.1.0.1 wheel exports it from the effect module.
    Supporting both forms keeps the adapter compatible with either layout.
    """

    try:
        from nvvfx.effects.video_super_res import QualityLevel  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError("The installed nvidia-vfx package does not expose VideoSuperRes QualityLevel.") from exc
    return QualityLevel


def _inspect_required_driver_library() -> tuple[str | None, str | None]:
    """Find Linux NGX without preloading it ahead of the NVIDIA runtime.

    Loading the NGX core directly can change how it identifies its calling
    module. Let ``libnvngxruntime`` own the eventual ``dlopen`` operation.
    """

    if not sys.platform.startswith("linux"):
        return None, None
    library = "libnvidia-ngx.so.1"
    if ctypes_util.find_library("nvidia-ngx"):
        return library, None
    search_directories = [
        *(Path(item) for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item),
        Path("/lib"),
        Path("/lib/x86_64-linux-gnu"),
        Path("/usr/lib"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib"),
    ]
    if any((directory / library).is_file() for directory in search_directories):
        return library, None
    return library, (
        f"The NVIDIA NGX driver library {library!r} is not visible to the dynamic linker. "
        "Install the NGX/OpenGL userspace component from the exact same NVIDIA driver release "
        "(commonly the libnvidia-gl package) and expose it through the system linker or "
        "LD_LIBRARY_PATH before Python starts."
    )


def _exception_chain_message(exc: BaseException) -> str:
    """Flatten ``__cause__`` / ``__context__`` so a probe receipt shows the native NGX error."""

    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rendered = str(current).strip() or repr(current)
        messages.append(f"{type(current).__name__}: {rendered}")
        current = current.__cause__ or current.__context__
    return " <- ".join(messages)


def _runtime_probe_failure_reason(*, device: int, backend_error: str) -> str:
    """Append the Linux NGX hint when the native error looks like a missing driver library."""

    reason = f"The native nvidia-vfx probe failed on cuda:{device}. Backend error: {backend_error}."
    lowered = backend_error.lower()
    if sys.platform.startswith("linux") and ("initializ" in lowered or "code -12" in lowered):
        reason += (
            " Verify that libnvidia-ngx.so.1 comes from the exact installed driver release; "
            "a compute-only/headless driver image may omit this NGX component."
        )
    return reason


def _torch_device_for_nvvfx_device(device: int) -> Any:
    """Map the VFX integer device index onto ``torch.device('cuda:N')``."""

    return _import_torch().device(f"cuda:{device}")


def _current_cuda_stream_ptr(*, device: Any, enabled: bool) -> int:
    """Pass the current stream pointer so VFX writes land on the Studio stream, not a default."""

    if not enabled or getattr(device, "type", None) != "cuda":
        return 0
    return int(_import_torch().cuda.current_stream(device=device).cuda_stream)


def _synchronize_nonblocking_output(*, device: Any, enabled: bool) -> None:
    """Wait until a VFX async write is safe to clone through DLPack."""

    if enabled and getattr(device, "type", None) == "cuda":
        _import_torch().cuda.synchronize(device=device)


def _installed_nvidia_vfx_version() -> str | None:
    """Return the wheel version even when ``import nvvfx`` would still fail."""

    try:
        return metadata.version("nvidia-vfx")
    except metadata.PackageNotFoundError:
        return None


def _import_torch() -> Any:
    """Import torch at call time so CPU-only processes can import this module."""

    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "RTX video post-processing requires PyTorch for DLPack frame exchange. "
            "Install `worldfoundry[rtx_postprocess]`."
        ) from exc
    return torch


def _import_numpy() -> Any:
    """Import NumPy at call time; required only for ndarray / frame-list-numpy input."""

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - Studio and tests require NumPy
        raise RuntimeError("NumPy is required for RTX VSR NumPy frame input.") from exc
    return np


def _raise_mixed_frame_list() -> Any:
    """Fail in a list comprehension without a multi-line ``else`` branch."""

    raise TypeError("RTX VSR frame-list input cannot mix NumPy arrays and PyTorch tensors.")


__all__ = [
    "RTX_POSTPROCESS_PRESET_NAMES",
    "RTXVFXCapability",
    "RTXVFXProbeStatus",
    "RTXVideoFloatRange",
    "RTXVideoSuperResolutionConfig",
    "RTXVideoSuperResolutionPostProcessor",
    "RTXVideoSuperResolutionQuality",
    "inspect_rtx_vfx_capability",
    "probe_rtx_vfx_runtime",
    "require_rtx_vfx_runtime",
    "rtx_postprocessor_from_preset",
]
