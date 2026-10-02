# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit raw-E4M3 cuDNN Frontend SDPA, adapted from NVIDIA FlashDreams.

Plan construction and CUDA child-graph insertion follow FlashDreams commit
91c906b69992f63fb19907a25cd805fb783d314a, accelerated/multi_head_attention/
cudnn/native_fp8.py. Callers provide E4M3 inputs and their explicit dequantization
scales; outputs are dequantized to BF16. Callers own input quantization and model
quality validation. BF16/default attention is unchanged.
"""

from __future__ import annotations

import importlib
import math
import threading
import weakref

import torch

_PLANS: dict[tuple, object] = {}
_PLAN_LOCK = threading.Lock()
_MAX_PLANS = 64


def _positive_scale(name: str, value: float) -> float:
    result = float(value)
    limits = torch.finfo(torch.float32)
    if not math.isfinite(result) or result < limits.tiny or result > limits.max:
        raise ValueError(f"cudnn_fp8 {name} must be positive and finite in float32")
    return result


def _validate(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float | None) -> float:
    if any(tensor.device.type != "cuda" for tensor in (query, key, value)):
        raise ValueError("cudnn_fp8 requires CUDA Q/K/V tensors")
    if any(tensor.dtype != torch.float8_e4m3fn for tensor in (query, key, value)):
        raise ValueError("cudnn_fp8 requires raw torch.float8_e4m3fn Q/K/V; quantize explicitly")
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("cudnn_fp8 requires [B,H,L,D] Q/K/V tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("cudnn_fp8 Q/K/V must be on the same CUDA device")
    if query.shape[:2] != key.shape[:2] or key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("cudnn_fp8 requires matching batch/head/dim and K/V shapes; GQA is unsupported")
    if any(min(tensor.shape) <= 0 or tensor.stride(-1) != 1 for tensor in (query, key, value)):
        raise ValueError("cudnn_fp8 requires nonempty tensors with a contiguous head dimension")
    if any(tensor.data_ptr() % 16 for tensor in (query, key, value)):
        raise ValueError("cudnn_fp8 requires 16-byte aligned Q/K/V base pointers")
    if any(tensor.requires_grad for tensor in (query, key, value)):
        raise ValueError("cudnn_fp8 is inference-only and does not support autograd")
    if query.shape[-1] not in {64, 128}:
        raise ValueError("cudnn_fp8 supports head dimensions 64 or 128")
    if torch.cuda.get_device_capability(query.device)[0] < 9 or torch.version.hip:
        raise ValueError("cudnn_fp8 requires NVIDIA Hopper or newer")
    return _positive_scale("scale", query.shape[-1] ** -0.5 if scale is None else scale)


class _Plan:
    def __init__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_scale: float) -> None:
        try:
            self.cudnn = importlib.import_module("cudnn")
            self.runtime = importlib.import_module("cuda.bindings.runtime")
        except ImportError as exc:
            raise RuntimeError("cudnn_fp8 requires nvidia-cudnn-frontend and cuda-bindings") from exc
        cudnn = self.cudnn
        self.device = query.device
        self.graph = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self.descriptors = [
            self.graph.tensor(
                name=name, dim=list(tensor.shape), stride=list(tensor.stride()), data_type=cudnn.data_type.FP8_E4M3
            )
            for name, tensor in zip(("query", "key", "value"), (query, key, value))
        ]
        self.scales = [
            self.graph.tensor(name=name, dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
            for name in ("descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o")
        ]
        output, _, amax_s, amax_o = self.graph.sdpa_fp8(
            q=self.descriptors[0],
            k=self.descriptors[1],
            v=self.descriptors[2],
            descale_q=self.scales[0],
            descale_k=self.scales[1],
            descale_v=self.scales[2],
            descale_s=self.scales[3],
            scale_s=self.scales[4],
            scale_o=self.scales[5],
            is_inference=True,
            attn_scale=attention_scale,
        )
        self.output_shape = tuple(query.shape)
        b, h, length, dim = self.output_shape
        self.output_stride = (h * length * dim, length * dim, dim, 1)
        output.set_output(True).set_dim(list(query.shape)).set_stride(list(self.output_stride))
        for descriptor in (amax_s, amax_o):
            descriptor.set_output(False).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1])
        self.outputs = (output, amax_s, amax_o)
        self.graph.select_behavior_notes([cudnn.behavior_note.SUPPORTS_CUDA_GRAPH_NATIVE_API])
        self.graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        self.handle = cudnn.create_handle()
        weakref.finalize(self, cudnn.destroy_handle, self.handle)

    def _checked(self, operation: str, result):
        error, *values = result
        if error != self.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudnn_fp8 {operation} failed: {error}")
        return values

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        descale_q: float,
        descale_k: float,
        descale_v: float,
        scale_o: float,
    ) -> torch.Tensor:
        output = torch.empty_strided(self.output_shape, self.output_stride, device=self.device, dtype=query.dtype)
        # Capture owns these allocations. Cached plan eviction/other streams must
        # never invalidate or concurrently overwrite a graph's workspace/scales.
        workspace = torch.empty(self.graph.get_workspace_size(), device=self.device, dtype=torch.uint8)
        one = torch.ones((1, 1, 1, 1), device=self.device, dtype=torch.float32)
        amax_s, amax_o = torch.empty_like(one), torch.empty_like(one)
        pack = dict(zip(self.descriptors, (query, key, value)))
        pack.update(dict.fromkeys(self.scales, one))
        for descriptor, factor in zip(self.scales[:3], (descale_q, descale_k, descale_v)):
            pack[descriptor] = one * factor
        # Keep probabilities representable for long K/V sequences. The inverse
        # descale restores the original softmax before the value projection.
        pack[self.scales[3]] = one / 128
        pack[self.scales[4]] = one * 128
        pack[self.scales[5]] = one * scale_o
        pack.update(dict(zip(self.outputs, (output, amax_s, amax_o))))
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.cudnn.set_stream(self.handle, stream)
        if not torch.cuda.is_current_stream_capturing():
            self.graph.execute(pack, workspace, handle=self.handle)
            return (output.float() / scale_o).to(torch.bfloat16)
        runtime = self.runtime
        (child_graph,) = self._checked("cudaGraphCreate", runtime.cudaGraphCreate(0))
        try:
            self.graph.populate_cuda_graph(
                self.handle,
                {descriptor.get_uid(): tensor.data_ptr() for descriptor, tensor in pack.items()},
                workspace.data_ptr(),
                int(child_graph),
            )
            capture = self._checked("cudaStreamGetCaptureInfo", runtime.cudaStreamGetCaptureInfo(stream))
            status, _, parent_graph, dependencies, _, dependency_count = capture
            if status != runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
                raise RuntimeError("cudnn_fp8 requires active CUDA capture")
            (child_node,) = self._checked(
                "cudaGraphAddChildGraphNode",
                runtime.cudaGraphAddChildGraphNode(parent_graph, dependencies, dependency_count, child_graph),
            )
            self._checked(
                "cudaStreamUpdateCaptureDependencies",
                runtime.cudaStreamUpdateCaptureDependencies(
                    stream,
                    [child_node],
                    None,
                    1,
                    runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
                ),
            )
        finally:
            self._checked("cudaGraphDestroy", runtime.cudaGraphDestroy(child_graph))
        return (output.float() / scale_o).to(torch.bfloat16)


@torch.compiler.disable
def native_cudnn_fp8_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    descale_q: float = 1.0,
    descale_k: float = 1.0,
    descale_v: float = 1.0,
    scale_o: float | None = None,
) -> torch.Tensor:
    """Run actual cuDNN FP8 SDPA; unsupported requests fail without a dense fallback.

    Inputs reconstruct as ``query.float() * descale_q`` (likewise K/V).
    ``scale_o`` scales the internal E4M3 output and defaults to ``1/descale_v``
    so its range matches the quantized V input. The returned BF16 output is
    explicitly dequantized. No BF16 input is implicitly cast to FP8.

    Warm up the same shape/stride/attention scale on this
    thread before CUDA Graph capture. Plans are thread-local in identity and
    process-owned in lifetime; at most 64 plans are built, then new signatures
    fail explicitly. No plan/handle is removed while captured graphs may use it.
    """

    attention_scale = _validate(query, key, value, scale)
    descale_q = _positive_scale("descale_q", descale_q)
    descale_k = _positive_scale("descale_k", descale_k)
    descale_v = _positive_scale("descale_v", descale_v)
    scale_o = _positive_scale("scale_o", 1 / descale_v if scale_o is None else scale_o)
    signature = (
        threading.current_thread(),
        query.device,
        attention_scale,
        *((tuple(tensor.shape), tensor.stride()) for tensor in (query, key, value)),
    )
    with torch.cuda.device(query.device):
        plan = _PLANS.get(signature)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up cudnn_fp8 before CUDA graph capture")
            with _PLAN_LOCK:
                plan = _PLANS.get(signature)
                if plan is None:
                    if len(_PLANS) >= _MAX_PLANS:
                        raise RuntimeError("cudnn_fp8 plan capacity reached; reuse an existing shape/stride/scale")
                    plan = _Plan(query, key, value, attention_scale)
                    _PLANS[signature] = plan
        return plan(query, key, value, descale_q, descale_k, descale_v, scale_o)


__all__ = ["native_cudnn_fp8_sdpa"]
