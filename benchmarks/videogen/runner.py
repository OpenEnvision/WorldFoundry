# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible single- and multi-GPU benchmark runner for foundation video DiTs.

Supports:
- Single GPU (``python -m benchmarks.videogen.runner ...``)
- Multi-GPU NVSwitch scaling (``torchrun --nproc_per_node=2|4|8 -m benchmarks.videogen.runner ...``)
- Declarative acceleration presets via :func:`resolve_acceleration_profile`
  (``baseline``, ``exact``, ``fast``, ``turbo``) plus explicit overrides
- Synchronized per-stage latency accounting (``text_encode``, ``dit_denoise``,
  ``vae_decode``), separated warmup vs steady-state medians, peak VRAM, full
  execution receipts, and strict PSNR/SSIM/Rel-L2 quality parity comparison
  against a saved 1-GPU BF16 reference tensor bundle.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from benchmarks.inference.plugin_diagnostics import compare_generation
from worldfoundry.core.acceleration.profiles import resolve_acceleration_profile

DEFAULT_PROMPT = (
    "A golden retriever running happily along a sunny beach with gentle ocean waves, "
    "cinematic golden hour sunlight, crisp natural motion."
)
DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly."
)

_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "wan2.1-t2v-1.3b": {
        "num_heads": 12,
        "checkpoint_dir": "/data2/chaoliu/icml/weights/Wan2.1-T2V-1.3B",
        "supports_cfg": True,
    },
    "wan2.1-t2v-14b": {
        "num_heads": 40,
        "checkpoint_dir": "/data2/chaoliu/icml/weights/Wan2.1-T2V-14B",
        "supports_cfg": True,
    },
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")
    return 0, 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _instrument_pipeline_stages(native_pipeline: Any, device: torch.device) -> tuple[dict[str, list[float]], Any]:
    """Instrument conditioner.encode, denoiser.__call__, and decoder.decode cleanly."""
    timings: dict[str, list[float]] = {"text": [], "dit": [], "vae": []}
    comps = native_pipeline.components
    orig_cond = comps.conditioner
    orig_den = comps.denoiser
    orig_dec = comps.decoder

    orig_encode = getattr(orig_cond, "encode", None)
    orig_decode = getattr(orig_dec, "decode", None)

    class _TimedConditioner:
        def __getattr__(self, name: str) -> Any:
            return getattr(orig_cond, name)

        def encode(self, *args: Any, **kwargs: Any) -> Any:
            if not callable(orig_encode):
                raise TypeError("conditioner does not expose encode()")
            _sync(device)
            t0 = time.perf_counter()
            out = orig_encode(*args, **kwargs)
            _sync(device)
            timings["text"].append(time.perf_counter() - t0)
            return out

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            _sync(device)
            t0 = time.perf_counter()
            out = orig_cond(*args, **kwargs)
            _sync(device)
            timings["text"].append(time.perf_counter() - t0)
            return out

    class _TimedDenoiser:
        def __getattr__(self, name: str) -> Any:
            return getattr(orig_den, name)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            _sync(device)
            t0 = time.perf_counter()
            out = orig_den(*args, **kwargs)
            _sync(device)
            timings["dit"].append(time.perf_counter() - t0)
            return out

    class _TimedDecoder:
        def __getattr__(self, name: str) -> Any:
            return getattr(orig_dec, name)

        def decode(self, *args: Any, **kwargs: Any) -> Any:
            if not callable(orig_decode):
                raise TypeError("decoder does not expose decode()")
            _sync(device)
            t0 = time.perf_counter()
            out = orig_decode(*args, **kwargs)
            _sync(device)
            timings["vae"].append(time.perf_counter() - t0)
            return out

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            _sync(device)
            t0 = time.perf_counter()
            out = orig_dec(*args, **kwargs)
            _sync(device)
            timings["vae"].append(time.perf_counter() - t0)
            return out

    native_pipeline.runner.components = replace(
        comps,
        conditioner=_TimedConditioner(),
        denoiser=_TimedDenoiser(),
        decoder=_TimedDecoder(),
    )
    return timings, orig_den


def _build_pipeline_options(args: argparse.Namespace, world_size: int, sm_major: int) -> tuple[dict[str, Any], str]:
    meta = _MODEL_DEFAULTS[args.model]
    if args.profile == "baseline":
        opts: dict[str, Any] = {
            "torch_dtype": "bfloat16",
            "offload_mode": "resident",
            "attention_backend": "torch",
            "dit_weight_dtype": args.dit_weight_dtype or "bfloat16",
        }
        topo_label = "1GPU" if world_size == 1 else f"{world_size}GPU"
    else:
        spec = resolve_acceleration_profile(
            args.profile,
            world_size=world_size,
            num_heads=meta["num_heads"],
            height=args.height,
            supports_cfg=meta["supports_cfg"],
            sm_major=sm_major,
            enable_compile=args.compile,
            enable_cuda_graph=args.cuda_graph != "off",
            enable_vae_parallel=args.vae_parallel,
        )
        opts = spec.to_pipeline_kwargs()
        topo = spec.topology
        if world_size == 1:
            topo_label = "1GPU"
        else:
            parts = []
            if topo.cfg_parallel > 1:
                parts.append(f"CFG={topo.cfg_parallel}")
            if topo.ulysses_degree > 1:
                parts.append(f"SP={topo.ulysses_degree}")
            if topo.vae_parallel > 1:
                parts.append("VAE-P")
            topo_label = "x".join(parts) if parts else f"{world_size}GPU"

    # Apply explicit CLI overrides cleanly
    if args.dit_weight_dtype:
        opts["dit_weight_dtype"] = args.dit_weight_dtype
    if args.cfg_parallel is not None:
        if args.cfg_parallel > 1:
            opts["cfg_parallel"] = args.cfg_parallel
        else:
            opts.pop("cfg_parallel", None)
    if args.sp_degree is not None:
        if args.sp_degree > 1:
            opts["sequence_parallel"] = args.sp_degree
        else:
            opts.pop("sequence_parallel", None)
    if args.vae_parallel is False:
        opts.pop("vae_parallel", None)
        opts.pop("vae_spatial_tiling", None)
    elif world_size > 1 and opts.get("vae_parallel", 1) > 1:
        opts.setdefault("vae_tile_size", (30, 52) if args.height <= 480 else (34, 34))
        opts.setdefault("vae_tile_stride", (15, 26) if args.height <= 480 else (18, 18))
    if args.cuda_graph is not None:
        if args.cuda_graph == "off":
            opts.pop("cuda_graph", None)
        elif args.cuda_graph == "whole":
            opts["cuda_graph"] = True
        else:
            opts["cuda_graph"] = args.cuda_graph
    if args.teacache is not None:
        if args.teacache <= 0:
            opts.pop("teacache", None)
        else:
            opts["teacache"] = float(args.teacache)
            opts["inplace_residual"] = False
            if opts.get("cuda_graph") is True:
                opts["cuda_graph"] = "piecewise"
    if args.quantization == "none":
        opts.pop("quantization", None)
    elif args.quantization == "fp8":
        exclude = ("time_embedding", "time_projection", "text_embedding", "head", "img_emb", "cross_attn")
        if args.fp8_exclude_edge_blocks:
            last_idx = 29 if args.model == "wan2.1-t2v-1.3b" else 39
            exclude = (*exclude, "blocks.0.", f"blocks.{last_idx}.")
        opts["quantization"] = {
            "mode": "fp8",
            "exclude": exclude,
            "options": {
                "min_features": 1024,
                "scaling": "rowwise",
                "use_fast_accum": False,
                "fuse_fp8_ffn": not args.no_fuse_fp8_ffn,
                "keep_dense_fallback": False,
            },
        }
    if args.fp8_exact_accum and isinstance(opts.get("quantization"), Mapping):
        qcfg_acc: dict[str, Any] = dict(opts["quantization"])
        qopts = dict(qcfg_acc.get("options") or {})
        qopts["use_fast_accum"] = False
        qcfg_acc["options"] = qopts
        opts["quantization"] = qcfg_acc
    if args.fp8_exclude_edge_blocks and isinstance(opts.get("quantization"), Mapping):
        qcfg: dict[str, Any] = dict(opts["quantization"])
        last_idx = 29 if args.model == "wan2.1-t2v-1.3b" else 39
        existing_excl = tuple(qcfg.get("exclude", ()))
        qcfg["exclude"] = tuple(dict.fromkeys((*existing_excl, "blocks.0.", f"blocks.{last_idx}.")))
        opts["quantization"] = qcfg
    if args.attention_backend:
        opts["attention_backend"] = args.attention_backend
    if args.sol_attn_tau is not None:
        acc = dict(opts.get("accelerations") or {})
        sol_cfg: dict[str, Any] = {
            "backend": "sol_attn",
            "tau": float(args.sol_attn_tau),
            "thresh_type": args.sol_attn_thresh,
            "compress_kv": not args.sol_attn_no_compress,
        }
        if args.sol_attn_kv_splits > 1:
            sol_cfg["kv_splits"] = int(args.sol_attn_kv_splits)
            sol_cfg["allow_multi_split"] = True
        if args.sol_attn_dense_steps > 0:
            sol_cfg["dense_steps"] = int(args.sol_attn_dense_steps)
        if args.sol_attn_dense_blocks:
            sol_cfg["dense_blocks"] = tuple(int(x) for x in args.sol_attn_dense_blocks.split(",") if x.strip())
        if args.sol_attn_tile_3d:
            sol_cfg["tile_3d"] = tuple(int(x) for x in args.sol_attn_tile_3d.split("x") if x.strip())
        acc["attention_policy"] = {"self": sol_cfg}
        opts["accelerations"] = acc
        opts.pop("compile", None)
    elif args.sol_attn_kv_splits > 1 and isinstance(opts.get("accelerations"), Mapping):
        acc = dict(opts["accelerations"])
        attn_pol = acc.get("attention_policy")
        if isinstance(attn_pol, Mapping) and isinstance(attn_pol.get("self"), Mapping):
            self_cfg = dict(attn_pol["self"])
            if self_cfg.get("backend") == "sol_attn":
                self_cfg["kv_splits"] = int(args.sol_attn_kv_splits)
                self_cfg["allow_multi_split"] = True
                acc["attention_policy"] = {**dict(attn_pol), "self": self_cfg}
                opts["accelerations"] = acc
    if args.token_pruning is not None:
        acc = dict(opts.get("accelerations") or {})
        acc["token_pruning"] = {
            "keep_ratio": float(args.token_pruning),
            "method": args.token_pruning_method,
            "start_block": args.token_pruning_start_block,
            "end_block": args.token_pruning_end_block,
        }
        opts["accelerations"] = acc
        if opts.get("cuda_graph") is True:
            opts["cuda_graph"] = "piecewise"
    acc_map = opts.get("accelerations")
    if isinstance(acc_map, Mapping) and (
        len(acc_map) > 1
        or (bool(opts.get("teacache")) and ("attention_policy" in acc_map or "token_pruning" in acc_map))
    ):
        opts["allow_approximate_composition"] = True
    if args.compile:
        opts["compile"] = True
        opts["compile_scope"] = args.compile_scope
        if args.compile_mode:
            opts["compile_mode"] = args.compile_mode

    if world_size > 1:
        parts = []
        if opts.get("cfg_parallel", 1) > 1:
            parts.append(f"CFG={opts['cfg_parallel']}")
        if opts.get("sequence_parallel", 1) > 1:
            parts.append(f"SP={opts['sequence_parallel']}")
        if opts.get("vae_parallel", 1) > 1:
            parts.append("VAE-P")
        topo_label = "x".join(parts) if parts else f"{world_size}GPU"

    return opts, topo_label


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from worldfoundry.core.distributed.sequence_parallel.runtime import (
        initialize_cfg_parallel_group,
        initialize_sequence_parallel_state,
    )
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import (
        Wan2p1T2V14BPipeline,
        Wan2p1T2VPipeline,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank, local_rank, world_size, device = _init_distributed()
    sm_major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    opts, topo_label = _build_pipeline_options(args, world_size, sm_major)

    cfg_deg = int(opts.get("cfg_parallel", 1))
    sp_deg = int(opts.get("sequence_parallel", 1))
    if world_size > 1:
        if cfg_deg * sp_deg != world_size:
            raise ValueError(
                f"cfg_parallel ({cfg_deg}) * sequence_parallel ({sp_deg}) must equal world_size ({world_size})"
            )
        initialize_sequence_parallel_state(sp_deg)
        if cfg_deg > 1:
            initialize_cfg_parallel_group(sp_deg, cfg_deg)
        dist.barrier()

    ckpt_dir = args.checkpoint_dir or _MODEL_DEFAULTS[args.model]["checkpoint_dir"]
    if rank == 0:
        print(
            f"\n>>> [{args.model}] label={args.label} world={world_size} ({topo_label}) "
            f"{args.width}x{args.height}x{args.frames}f@{args.steps}s opts={opts}",
            flush=True,
        )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    pipeline_cls = Wan2p1T2VPipeline if args.model == "wan2.1-t2v-1.3b" else Wan2p1T2V14BPipeline
    t_load0 = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(
        model_path=ckpt_dir,
        device=str(device),
        **opts,
    )
    _sync(device)
    if world_size > 1:
        dist.barrier()
    load_s = round(time.perf_counter() - t_load0, 2)

    timings, orig_den = _instrument_pipeline_stages(pipe.native_pipeline, device)
    req_kwargs: dict[str, Any] = {
        "prompt": DEFAULT_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg_scale,
        "seed": args.seed,
        "return_dict": True,
    }

    # Warmup pass (absorbs graph capture, Triton JIT, inductor compile)
    t_cold0 = time.perf_counter()
    warm_out = pipe(**req_kwargs)
    _sync(device)
    if world_size > 1:
        dist.barrier()
    cold_s = round(time.perf_counter() - t_cold0, 3)
    del warm_out

    # Steady-state repeats
    runs: list[dict[str, Any]] = []
    last_out: SimpleNamespace | None = None
    for _ in range(args.repeats):
        for bucket in timings.values():
            bucket.clear()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if world_size > 1:
            dist.barrier()
        _sync(device)
        t0 = time.perf_counter()
        res = pipe(**req_kwargs)
        _sync(device)
        if world_size > 1:
            dist.barrier()
        e2e_s = time.perf_counter() - t0
        text_s = sum(timings["text"])
        dit_s = sum(timings["dit"])
        vae_s = sum(timings["vae"])
        peak_alloc = round(torch.cuda.max_memory_allocated(device) / 1024**3, 2) if device.type == "cuda" else 0.0
        peak_res = round(torch.cuda.max_memory_reserved(device) / 1024**3, 2) if device.type == "cuda" else 0.0
        runs.append(
            {
                "e2e_s": round(e2e_s, 3),
                "text_s": round(text_s, 3),
                "dit_s": round(dit_s, 3),
                "dit_calls": len(timings["dit"]),
                "vae_s": round(vae_s, 3),
                "other_s": round(max(0.0, e2e_s - text_s - dit_s - vae_s), 3),
                "peak_alloc_gib": peak_alloc,
                "peak_reserved_gib": peak_res,
            }
        )
        if rank == 0:
            last_out = SimpleNamespace(
                sample=res["sample"].detach().cpu(),
                latents=res["latents"].detach().cpu(),
            )
        del res

    receipt = orig_den.runtime_optimization_report() if hasattr(orig_den, "runtime_optimization_report") else {}
    quality: dict[str, Any] = {}
    if rank == 0 and last_out is not None:
        if args.save_ref:
            ref_path = Path(args.save_ref)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"sample": last_out.sample, "latents": last_out.latents}, ref_path)
            quality = {"reference": True, "saved_ref": str(ref_path)}
        elif args.compare_ref:
            ref_blob = torch.load(args.compare_ref, map_location="cpu", weights_only=True)
            ref_ns = SimpleNamespace(sample=ref_blob["sample"], latents=ref_blob["latents"])
            quality = compare_generation(ref_ns, last_out)

    result: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "profile": args.profile,
        "geometry": f"{args.width}x{args.height}x{args.frames}f@{args.steps}s",
        "world_size": world_size,
        "topology_label": topo_label,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "load_s": load_s,
        "cold_first_run_s": cold_s,
        "median_e2e_s": round(statistics.median(r["e2e_s"] for r in runs), 3),
        "median_text_s": round(statistics.median(r["text_s"] for r in runs), 3),
        "median_dit_s": round(statistics.median(r["dit_s"] for r in runs), 3),
        "median_vae_s": round(statistics.median(r["vae_s"] for r in runs), 3),
        "peak_alloc_gib": max(r["peak_alloc_gib"] for r in runs),
        "quality": quality,
        "options": opts,
        "receipt": {
            "quality_tier": receipt.get("quality_tier") if isinstance(receipt, Mapping) else None,
            "fallbacks": receipt.get("fallbacks") if isinstance(receipt, Mapping) else None,
            "effective": receipt.get("effective") if isinstance(receipt, Mapping) else None,
            "cuda_graph": (receipt.get("runtime") or {}).get("cuda_graph") if isinstance(receipt, Mapping) else None,
            "fused_rope": (receipt.get("runtime") or {}).get("fused_rope") if isinstance(receipt, Mapping) else None,
            "qkv_fusion": (receipt.get("runtime") or {}).get("qkv_fusion") if isinstance(receipt, Mapping) else None,
        },
        "runs": runs,
    }

    if rank == 0:
        print(
            f"  DONE {args.label}: e2e={result['median_e2e_s']}s "
            f"(text={result['median_text_s']}s, dit={result['median_dit_s']}s, vae={result['median_vae_s']}s) "
            f"cold={cold_s}s peak={result['peak_alloc_gib']}GiB quality={quality} "
            f"fallbacks={result['receipt']['fallbacks']}",
            flush=True,
        )
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible H200 VideoGen acceleration benchmark runner.")
    parser.add_argument("--model", choices=sorted(_MODEL_DEFAULTS), required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--profile", choices=("baseline", "exact", "fast", "turbo", "sol"), default="exact")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dit-weight-dtype", type=str, default=None)
    parser.add_argument("--cfg-parallel", type=int, default=None)
    parser.add_argument("--sp-degree", type=int, default=None)
    parser.add_argument("--no-vae-parallel", dest="vae_parallel", action="store_false", default=True)
    parser.add_argument("--cuda-graph", choices=("off", "whole", "piecewise"), default=None)
    parser.add_argument("--teacache", type=float, default=None)
    parser.add_argument("--quantization", choices=("none", "fp8"), default=None)
    parser.add_argument("--fp8-exclude-edge-blocks", action="store_true")
    parser.add_argument("--fp8-exact-accum", action="store_true")
    parser.add_argument("--no-fuse-fp8-ffn", action="store_true")
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--sol-attn-tau", type=float, default=None)
    parser.add_argument("--sol-attn-thresh", choices=("diag", "exact"), default="diag")
    parser.add_argument("--sol-attn-kv-splits", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--sol-attn-no-compress", action="store_true")
    parser.add_argument("--sol-attn-dense-steps", type=int, default=0)
    parser.add_argument(
        "--sol-attn-dense-blocks", type=str, default=None, help="Comma-separated block indices e.g. 0,1,-1"
    )
    parser.add_argument("--sol-attn-tile-3d", type=str, default=None, help="3D tile shape e.g. 1x8x8")
    parser.add_argument("--token-pruning", type=float, default=None)
    parser.add_argument("--token-pruning-method", type=str, default="feat_norm")
    parser.add_argument("--token-pruning-start-block", type=int, default=2)
    parser.add_argument("--token-pruning-end-block", type=int, default=-2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-scope", choices=("regional", "full"), default="regional")
    parser.add_argument("--compile-mode", type=str, default=None)
    parser.add_argument("--save-ref", type=str, default=None)
    parser.add_argument("--compare-ref", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run_benchmark(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
