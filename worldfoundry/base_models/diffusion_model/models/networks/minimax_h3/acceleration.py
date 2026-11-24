# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine-aligned inference acceleration stack for MiniMax H3 DiT models.

Composes orthogonal, auditable optimizations across canonical
:class:`MiniMaxH3DiTModel` and its variants (:class:`PiH3DiTModel`,
:class:`H3TinyDiTModel`):

1. **Projection Fusion** — collapses Diffusers-split ``HyperFlowQKVProjection``
   (3 separate matmuls + ``torch.cat``) into one fused ``qkv_proj`` linear and
   pre-swaps ``HyperFlowSwiGLUProjection`` ``[up | gate]`` rows into canonical
   ``[gate | up]`` order (bitwise lossless).
2. **Attention Backend Routing** — configures the
   :class:`AttentionBackendConfigurable` seam across ``torch`` (cuDNN9 SM90
   SDPA), ``cudnn_fp8``, and ``sol_attn`` with optional dense edge blocks.
3. **Hopper Rowwise FP8 W8A8 + Fused SwiGLU-FP8 FFN** — quantizes heavy DiT
   block projections via :class:`Float8Linear` while excluding sensitive
   patch/time/refiner/AdaLN/head layers, and fuses SwiGLU + rowwise FP8
   activation quantization directly into ``mlp.fc2``.
4. **CUDA Graph / Piecewise Block Graph Capture & Replay** — captures the
   zero-sync transformer block stack via :class:`InferenceCUDAGraphRunner` or
   :class:`PiecewiseBlockGraphRunner`.
5. **Regional ``torch.compile``** — optionally compiles individual
   :class:`MiniMaxH3DiTBlock` instances without unrolling 50 blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Literal

import torch
from torch import nn

from .dit import MiniMaxH3DiTModel
from .triton_fp8_swiglu import fuse_minimax_h3_fp8_swiglu_ffns

DEFAULT_MINIMAX_H3_FP8_EXCLUDES: tuple[str, ...] = (
    "video_patch_proj",
    "audio_patch_proj",
    "condition_proj",
    "time_embedder",
    "token_refiner",
    "adaln_proj",
    "final_layer",
    "dwd_norm_out",
    "dwd_pixel_refiner",
)


@dataclass(frozen=True)
class MiniMaxH3AccelerationConfig:
    """Declarative acceleration recipe for a MiniMax H3 DiT backbone."""

    fuse_projections: bool = True
    attention_backend: str | None = None
    attention_options: Mapping[str, Any] | None = None
    dense_attention_layers: tuple[int, ...] = ()
    dense_initial_steps: int = 0
    include_refiner_attention: bool = False
    quantization: Literal["none", "fp8"] = "none"
    fp8_scaling: Literal["rowwise", "tensorwise"] = "rowwise"
    fp8_use_fast_accum: bool = False
    fp8_keep_dense_fallback: bool = False
    fp8_min_features: int = 1024
    fp8_exclude_edge_blocks: bool = False
    fp8_exclude: tuple[str, ...] = DEFAULT_MINIMAX_H3_FP8_EXCLUDES
    fuse_fp8_ffn: bool = True
    cuda_graph: Literal["off", "whole", "piecewise"] = "off"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 16
    cuda_graph_clone_outputs: bool = True
    piecewise_head_blocks: int = 2
    piecewise_tail_blocks: int = 2
    compile: bool = False
    compile_mode: str = "default"
    skip_host_validation: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.attention_options is not None:
            payload["attention_options"] = dict(self.attention_options)
        return payload


def _fuse_hyperflow_qkv(module: Any) -> nn.Linear:
    """Fuse a ``HyperFlowQKVProjection`` into one contiguous ``nn.Linear``."""

    q_proj = module.q
    k_proj = module.k
    v_proj = module.v
    head_dim = getattr(module, "head_dim", None)
    bias_flags = (q_proj.bias is not None, k_proj.bias is not None, v_proj.bias is not None)
    if len(set(bias_flags)) != 1:
        raise ValueError("HyperFlowQKVProjection q/k/v linears must either all have bias or none")
    has_bias = bias_flags[0]
    req_grad = bool(q_proj.weight.requires_grad or k_proj.weight.requires_grad or v_proj.weight.requires_grad)

    with torch.no_grad():
        if head_dim is None:
            weight = torch.cat((q_proj.weight.data, k_proj.weight.data, v_proj.weight.data), dim=0).contiguous()
            bias = (
                torch.cat((q_proj.bias.data, k_proj.bias.data, v_proj.bias.data), dim=0).contiguous()
                if has_bias
                else None
            )
        else:
            hd = int(head_dim)
            in_features = int(q_proj.in_features)
            num_heads = int(q_proj.out_features) // hd
            weight = (
                torch.stack(
                    (
                        q_proj.weight.data.view(num_heads, hd, in_features),
                        k_proj.weight.data.view(num_heads, hd, in_features),
                        v_proj.weight.data.view(num_heads, hd, in_features),
                    ),
                    dim=1,
                )
                .reshape(3 * num_heads * hd, in_features)
                .contiguous()
            )
            bias = (
                torch.stack(
                    (
                        q_proj.bias.data.view(num_heads, hd),
                        k_proj.bias.data.view(num_heads, hd),
                        v_proj.bias.data.view(num_heads, hd),
                    ),
                    dim=1,
                )
                .reshape(3 * num_heads * hd)
                .contiguous()
                if has_bias
                else None
            )

    fused = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    fused.weight = nn.Parameter(weight, requires_grad=req_grad)
    if bias is not None:
        bias_req_grad = bool(q_proj.bias.requires_grad or k_proj.bias.requires_grad or v_proj.bias.requires_grad)
        fused.bias = nn.Parameter(bias, requires_grad=bias_req_grad)
    return fused


def _fuse_hyperflow_swiglu(module: Any) -> nn.Linear:
    """Replace ``HyperFlowSwiGLUProjection`` ``[up|gate]`` with canonical ``[gate|up]`` ``nn.Linear``."""

    proj = module.proj
    with torch.no_grad():
        up_w, gate_w = proj.weight.data.chunk(2, dim=0)
        weight = torch.cat((gate_w, up_w), dim=0).contiguous()
        bias = None
        if proj.bias is not None:
            up_b, gate_b = proj.bias.data.chunk(2, dim=0)
            bias = torch.cat((gate_b, up_b), dim=0).contiguous()

    fused = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    fused.weight = nn.Parameter(weight, requires_grad=proj.weight.requires_grad)
    if bias is not None:
        fused.bias = nn.Parameter(bias, requires_grad=proj.bias.requires_grad)
    return fused


def fuse_minimax_h3_projections(model: nn.Module) -> dict[str, int]:
    """Fuse Diffusers-split QKV and SwiGLU wrapper modules into single ``nn.Linear`` layers.

    Idempotent and mathematically lossless: eliminates per-block ``torch.cat``
    memory traffic on Q/K/V and ``[up|gate] -> [gate|up]`` reordering across
    :class:`PiH3DiTModel`, :class:`H3TinyDiTModel`, and HyperFlow-loaded VAEs.
    """

    from .variants.hyperflow.conditioning import HyperFlowQKVProjection, HyperFlowSwiGLUProjection

    qkv_fused = 0
    swiglu_fused = 0

    def _visit(parent: nn.Module) -> None:
        nonlocal qkv_fused, swiglu_fused
        for name, child in list(parent.named_children()):
            if isinstance(child, HyperFlowQKVProjection) and all(
                isinstance(getattr(child, attr), nn.Linear) for attr in ("q", "k", "v")
            ):
                setattr(parent, name, _fuse_hyperflow_qkv(child))
                qkv_fused += 1
                continue
            if isinstance(child, HyperFlowSwiGLUProjection) and isinstance(child.proj, nn.Linear):
                setattr(parent, name, _fuse_hyperflow_swiglu(child))
                swiglu_fused += 1
                continue
            _visit(child)

    _visit(model)
    invalidate = getattr(model, "invalidate_cuda_graphs", None)
    if callable(invalidate):
        invalidate()
    return {"qkv_fused": qkv_fused, "swiglu_fused": swiglu_fused}


def quantize_minimax_h3_fp8(
    model: nn.Module,
    *,
    scaling: Literal["rowwise", "tensorwise"] = "rowwise",
    use_fast_accum: bool = True,
    keep_dense_fallback: bool = False,
    min_features: int = 1024,
    exclude_edge_blocks: bool = False,
    exclude: Sequence[str] = DEFAULT_MINIMAX_H3_FP8_EXCLUDES,
    fuse_fp8_ffn: bool = True,
) -> dict[str, Any]:
    """Apply Hopper/Blackwell rowwise FP8 W8A8 + optional Fused SwiGLU-FP8 FFN."""

    from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8

    resolved_excludes = list(exclude)
    blocks = getattr(model, "blocks", None)
    if exclude_edge_blocks and blocks is not None and len(blocks) >= 2:
        resolved_excludes.extend(["blocks.0.", f"blocks.{len(blocks) - 1}."])
    exclude_tuple = tuple(dict.fromkeys(resolved_excludes))

    replaced = replace_linear_with_float8(
        model,
        min_features=int(min_features),
        keep_dense_fallback=bool(keep_dense_fallback),
        exclude=exclude_tuple,
        scaling=scaling,
        use_fast_accum=bool(use_fast_accum),
    )
    fused_ffns = 0
    if fuse_fp8_ffn and scaling == "rowwise":
        fused_ffns = fuse_minimax_h3_fp8_swiglu_ffns(model)
    invalidate = getattr(model, "invalidate_cuda_graphs", None)
    if callable(invalidate):
        invalidate()
    return {
        "fp8_linears": int(replaced),
        "fused_fp8_ffns": int(fused_ffns),
        "scaling": scaling,
        "use_fast_accum": bool(use_fast_accum),
        "keep_dense_fallback": bool(keep_dense_fallback),
        "exclude": exclude_tuple,
    }


def install_minimax_h3_cuda_graph(
    model: MiniMaxH3DiTModel,
    *,
    mode: Literal["off", "whole", "piecewise"] = "whole",
    warmup: int = 1,
    max_graphs: int = 16,
    clone_outputs: bool = True,
    head_blocks: int = 2,
    tail_blocks: int = 2,
    extra_key: str = "minimax-h3",
) -> dict[str, Any]:
    """Attach a whole-stack or piecewise block-stack CUDA Graph runner to ``model``."""

    model.invalidate_cuda_graphs()
    model._block_graph_runner = None
    model._piecewise_graph = None
    model._piecewise_segment_names = ()
    model._piecewise_segment_bounds = {}

    normalized = str(mode).strip().casefold()
    if normalized in {"off", "none", "false", "disabled"}:
        return {"mode": "off", "segments": ()}

    num_layers = len(model.blocks)
    if normalized == "whole":
        from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

        fn = partial(model._run_block_slice, 0, num_layers)
        model._block_graph_runner = InferenceCUDAGraphRunner(
            fn,
            warmup=int(warmup),
            max_graphs=int(max_graphs),
            clone_outputs=bool(clone_outputs),
            enabled=True,
            extra_key=f"{extra_key}:whole",
        )
        return {"mode": "whole", "segments": (("blocks", 0, num_layers),)}

    if normalized == "piecewise":
        from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

        runner = PiecewiseBlockGraphRunner(
            warmup=int(warmup),
            max_graphs_per_segment=int(max_graphs),
            clone_outputs=bool(clone_outputs),
            enabled=True,
        )
        hb = max(0, int(head_blocks))
        tb = max(0, int(tail_blocks))
        segments: list[tuple[str, int, int]] = []
        if num_layers > hb + tb and (hb > 0 or tb > 0):
            if hb > 0:
                segments.append(("head", 0, hb))
            segments.append(("body", hb, num_layers - tb))
            if tb > 0:
                segments.append(("tail", num_layers - tb, num_layers))
        else:
            segments.append(("blocks", 0, num_layers))

        for seg_name, start, stop in segments:
            runner.register_segment(
                seg_name,
                partial(model._run_block_slice, start, stop),
                extra_key=f"{extra_key}:piecewise:{seg_name}",
            )
        model._piecewise_graph = runner
        model._piecewise_segment_names = tuple(name for name, _, _ in segments)
        model._piecewise_segment_bounds = {name: (start, stop) for name, start, stop in segments}
        return {"mode": "piecewise", "segments": tuple(segments)}

    raise ValueError(f"unsupported MiniMax H3 cuda_graph mode {mode!r}; expected 'off', 'whole', or 'piecewise'")


def compile_minimax_h3_blocks(
    model: MiniMaxH3DiTModel,
    *,
    mode: str = "default",
    fullgraph: bool = False,
) -> dict[str, Any]:
    """Apply regional ``torch.compile`` to each DiT block in ``model.blocks``."""

    for block in model.blocks:
        if block.attention_backend == "sol_attn":
            raise ValueError("Sol-Attn cannot be combined with torch.compile; use CUDA Graph instead")
    compiled_count = 0
    for idx, block in enumerate(model.blocks):
        raw_block = getattr(block, "_orig_mod", block)
        compiled_block: Any = torch.compile(raw_block, mode=mode, fullgraph=fullgraph)
        model.blocks[idx] = compiled_block
        compiled_count += 1
    model.invalidate_cuda_graphs()
    return {"compiled_blocks": compiled_count, "mode": mode, "fullgraph": bool(fullgraph)}


def minimax_h3_sol_profile(
    tier: Literal["exact", "fast", "sol_high_fidelity", "sol_balanced", "sol_turbo"] | str,
    *,
    sm_major: int = 9,
    keep_dense_fallback: bool = False,
    cuda_graph: Literal["off", "whole", "piecewise"] | None = None,
) -> MiniMaxH3AccelerationConfig:
    """Return a validated SOL-Engine preset for MiniMax H3 DiT models.

    Tiers:
        - ``exact``: Projection fusion + zero-sync Triton kernels + cuDNN9 BF16
          SDPA + whole-stack CUDA Graph (lossless BF16 parity).
        - ``fast``: ``exact`` + Hopper rowwise FP8 W8A8 (edge blocks BF16) +
          Fused SwiGLU-FP8 FFN (excluding sensitive conditioning/AdaLN/head layers).
        - ``sol_high_fidelity``: ``fast`` + step-adaptive ``sol_attn``
          (``tau=0.25, thresh_type="exact"``, blocks ``0, 1, -1`` dense, first
          2 denoising steps dense, prefix-sink) + piecewise block CUDA Graph.
        - ``sol_balanced``: ``fast`` + step-adaptive ``sol_attn``
          (``tau=0.5, thresh_type="exact"``, blocks ``0, 1, -1`` dense, first
          1 denoising step dense, prefix-sink) + piecewise block CUDA Graph.
        - ``sol_turbo``: FP8 W8A8 + ``sol_attn`` (``tau=1.0, thresh_type="exact"``,
          blocks ``0, 1, -1`` dense, prefix-sink) + piecewise block CUDA Graph.
    """

    normalized = str(tier).strip().lower()
    fp8_ok = int(sm_major) >= 9
    if normalized == "exact":
        return MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="torch",
            quantization="none",
            cuda_graph=cuda_graph or "whole",
            notes=("lossless BF16 projection fusion + zero-sync Triton kernels + CUDA Graph",),
        )
    if normalized == "fast":
        return MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="torch",
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fp8_keep_dense_fallback=keep_dense_fallback,
            fp8_exclude_edge_blocks=True,
            fuse_fp8_ffn=True,
            cuda_graph=cuda_graph or "whole",
            notes=("Hopper rowwise FP8 W8A8 (edge blocks BF16) + Fused SwiGLU-FP8 FFN + whole CUDA Graph",),
        )
    if normalized == "sol_high_fidelity":
        return MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="sol_attn",
            attention_options={"tau": 0.25, "thresh_type": "exact", "compress_kv": True},
            dense_attention_layers=(0, 1, -1),
            dense_initial_steps=2,
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fp8_keep_dense_fallback=keep_dense_fallback,
            fp8_exclude_edge_blocks=True,
            fuse_fp8_ffn=True,
            cuda_graph=cuda_graph or "piecewise",
            piecewise_head_blocks=2,
            piecewise_tail_blocks=2,
            notes=(
                "Step-adaptive SOL-Attn(exact, tau=0.25, dense_steps=2, prefix-sink, dense 0/1/-1) "
                "+ FP8(edge-excl) + Piecewise Graph",
            ),
        )
    if normalized == "sol_balanced":
        return MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="sol_attn",
            attention_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True},
            dense_attention_layers=(0, 1, -1),
            dense_initial_steps=1,
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fp8_keep_dense_fallback=keep_dense_fallback,
            fp8_exclude_edge_blocks=True,
            fuse_fp8_ffn=True,
            cuda_graph=cuda_graph or "piecewise",
            piecewise_head_blocks=2,
            piecewise_tail_blocks=2,
            notes=(
                "Step-adaptive SOL-Attn(exact, tau=0.5, dense_steps=1, prefix-sink, dense 0/1/-1) "
                "+ FP8(edge-excl) + Piecewise Graph",
            ),
        )
    if normalized == "sol_turbo":
        return MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="sol_attn",
            attention_options={"tau": 1.0, "thresh_type": "exact", "compress_kv": True},
            dense_attention_layers=(0, 1, -1),
            dense_initial_steps=0,
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fp8_keep_dense_fallback=keep_dense_fallback,
            fp8_exclude_edge_blocks=False,
            fuse_fp8_ffn=True,
            cuda_graph=cuda_graph or "piecewise",
            piecewise_head_blocks=2,
            piecewise_tail_blocks=2,
            notes=("SOL-Attn(exact, tau=1.0, prefix-sink, dense 0/1/-1) + FP8 W8A8 + Piecewise Graph",),
        )
    raise ValueError(
        f"unsupported MiniMax H3 acceleration tier {tier!r}; "
        "expected one of: exact, fast, sol_high_fidelity, sol_balanced, sol_turbo"
    )


def accelerate_minimax_h3_dit(
    model: MiniMaxH3DiTModel,
    config: MiniMaxH3AccelerationConfig | str,
    *,
    sm_major: int | None = None,
) -> dict[str, Any]:
    """Apply a complete, conflict-free acceleration stack to a MiniMax H3 DiT.

    Validates all configuration invariants before mutating ``model`` so invalid
    combinations never leave the model partially quantized or reconfigured.
    Returns a JSON-serializable summary receipt of every installed pass.
    """

    if isinstance(config, str):
        if sm_major is None:
            param = next(model.parameters(), None)
            sm_major = torch.cuda.get_device_capability(param.device)[0] if param is not None and param.is_cuda else 0
        resolved = minimax_h3_sol_profile(config, sm_major=sm_major)
    else:
        resolved = config

    # Validate before mutating model so failed calls leave model untouched.
    normalized_backend: str | None = None
    if resolved.attention_backend is not None:
        from worldfoundry.core.attention.backends.probe import normalize_attention_backend

        normalized_backend = normalize_attention_backend(str(resolved.attention_backend))
    if resolved.compile and normalized_backend == "sol_attn":
        raise ValueError("Sol-Attn cannot be combined with torch.compile; use CUDA Graph instead")
    if (
        resolved.compile
        and resolved.attention_backend is None
        and any(block.attention_backend == "sol_attn" for block in model.blocks)
    ):
        raise ValueError("Sol-Attn cannot be combined with torch.compile; use CUDA Graph instead")
    normalized_graph = str(resolved.cuda_graph).strip().casefold()
    if normalized_graph not in {"off", "none", "false", "disabled", "whole", "piecewise"}:
        raise ValueError(
            f"unsupported MiniMax H3 cuda_graph mode {resolved.cuda_graph!r}; expected 'off', 'whole', or 'piecewise'"
        )
    if int(resolved.dense_initial_steps) < 0:
        raise ValueError(f"dense_initial_steps must be non-negative, got {resolved.dense_initial_steps}")

    receipt: dict[str, Any] = {"config": resolved.to_dict()}

    # 1. Projection fusion first so FP8 quantizes one fused QKV / SwiGLU linear.
    if resolved.fuse_projections:
        receipt["projection_fusion"] = fuse_minimax_h3_projections(model)

    # 2. Attention backend routing via AttentionBackendConfigurable seam.
    model._dense_initial_steps = int(resolved.dense_initial_steps)
    if normalized_backend is not None:
        model.set_attention_backend(
            normalized_backend,
            backend_options=resolved.attention_options,
            dense_layers=resolved.dense_attention_layers,
            include_refiner=resolved.include_refiner_attention,
        )
        receipt["attention"] = {
            "backend": model.attention_backend,
            "options": dict(resolved.attention_options) if resolved.attention_options else None,
            "dense_layers": list(resolved.dense_attention_layers),
            "dense_initial_steps": int(resolved.dense_initial_steps),
        }

    # 3. Hopper rowwise FP8 W8A8 + Fused SwiGLU-FP8 FFN.
    if resolved.quantization == "fp8":
        receipt["quantization"] = quantize_minimax_h3_fp8(
            model,
            scaling=resolved.fp8_scaling,
            use_fast_accum=resolved.fp8_use_fast_accum,
            keep_dense_fallback=resolved.fp8_keep_dense_fallback,
            min_features=resolved.fp8_min_features,
            exclude_edge_blocks=resolved.fp8_exclude_edge_blocks,
            exclude=resolved.fp8_exclude,
            fuse_fp8_ffn=resolved.fuse_fp8_ffn,
        )

    # 4. Zero-sync host validation flag for H3-Tiny MeanFlow embedder if present.
    if resolved.skip_host_validation and hasattr(model, "time_embedder"):
        time_embedder: Any = model.time_embedder
        time_embedder._skip_host_validation = True

    # 5. Optional regional torch.compile on individual DiT blocks.
    if resolved.compile:
        receipt["compile"] = compile_minimax_h3_blocks(model, mode=resolved.compile_mode)

    # 6. CUDA Graph / Piecewise Block Graph capture wrapper (always last).
    receipt["cuda_graph"] = install_minimax_h3_cuda_graph(
        model,
        mode=resolved.cuda_graph,
        warmup=resolved.cuda_graph_warmup,
        max_graphs=resolved.cuda_graph_max_graphs,
        clone_outputs=resolved.cuda_graph_clone_outputs,
        head_blocks=resolved.piecewise_head_blocks,
        tail_blocks=resolved.piecewise_tail_blocks,
    )

    model._acceleration_metadata = receipt
    return receipt


__all__ = [
    "DEFAULT_MINIMAX_H3_FP8_EXCLUDES",
    "MiniMaxH3AccelerationConfig",
    "accelerate_minimax_h3_dit",
    "compile_minimax_h3_blocks",
    "fuse_minimax_h3_projections",
    "install_minimax_h3_cuda_graph",
    "minimax_h3_sol_profile",
    "quantize_minimax_h3_fp8",
]
