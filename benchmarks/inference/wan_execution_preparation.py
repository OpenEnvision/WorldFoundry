# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Public-path Wan execution correctness replay; timings are diagnostic only.

Run with the unified GPU environment::

    python -m benchmarks.inference.wan_execution_preparation \
      --checkpoint ../ckpts/Wan-AI--Wan2.1-T2V-1.3B --output-dir tmp/wan-preparation

Uses full checkpoint graphs, original samplers, fixed seeds, and the existing
native generation quality gate. This is not an isolated performance benchmark;
use wan22_multigpu_e2e for deployment timing and device-isolation evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Compare dense, static-KV/Graph and regional-compile/Graph A/B/A requests."""
    import torch

    from benchmarks.inference.plugin_diagnostics import BUDGET, compare_generation
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline
    from worldfoundry.pipelines.wan.pipeline_wan_2p2 import (
        Wan2p2I2VA14BPipeline,
        Wan2p2Pipeline,
        Wan2p2T2VA14BPipeline,
    )
    from worldfoundry.runtime.performance import capture_runtime_fingerprint

    pipeline_class = {
        "wan2.1-t2v-1.3b": Wan2p1T2VPipeline,
        "wan2.2-ti2v-5b": Wan2p2Pipeline,
        "wan2.2-t2v-a14b": Wan2p2T2VA14BPipeline,
        "wan2.2-i2v-a14b": Wan2p2I2VA14BPipeline,
    }[args.model_id]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = [
        {"prompt": "A small boat crossing a calm lake at sunrise.", "seed": 11},
        {"prompt": "A dog running through a green field in daylight.", "seed": 29},
    ]
    if args.images is not None:
        if not pipeline_class.ACCEPTS_IMAGES:
            raise ValueError("image replay requires an image-conditioned Wan pipeline")
        for case, path in zip(cases, args.images, strict=True):
            if not path.is_file():
                raise FileNotFoundError(path)
            case["images"] = str(path.resolve())
    elif pipeline_class.REQUIRES_IMAGES:
        raise ValueError(f"{args.model_id} requires --images IMAGE_A IMAGE_B")
    common: dict[str, Any] = dict(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=5.0,
        negative_prompt="",
    )
    warmup = [
        {
            "prompt": cases[0]["prompt"],
            "negative_prompt": "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "sampling": {
                "num_inference_steps": args.steps,
                "guidance_scale": 5.0,
                "seed": 11,
                "scheduler_options": dict(pipeline_class.DEFAULT_SCHEDULER_OPTIONS),
            },
            "inputs": {
                **pipeline_class.REQUEST_INPUT_DEFAULTS,
                "fps": pipeline_class.DEFAULT_FPS,
                "return_latent": False,
                **({"images": cases[0]["images"]} if args.images is not None else {}),
            },
        }
    ]
    native_warmup = warmup_requests_to_native(warmup)
    batch_requests = tuple(
        replace(
            native_warmup[0],
            prompt=case["prompt"],
            sampling=replace(native_warmup[0].sampling, seed=case["seed"]),
            inputs={**native_warmup[0].inputs, **({"images": case["images"]} if "images" in case else {})},
        )
        for case in cases
    )
    configurations: dict[str, dict[str, Any]] = {
        "dense": {},
        "static-graph": {"static_cross_kv": True, "cuda_graph": True},
        "regional-static-graph": {
            "static_cross_kv": True,
            "cuda_graph": True,
            "compile": True,
            "compile_scope": "regional",
            "compile_fullgraph": True,
        },
    }
    references: list[SimpleNamespace] = []
    report: dict[str, Any] = {
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint.resolve()),
        "geometry_and_sampling": common,
        "cases": cases,
        "compute_dtype": "bfloat16",
        "dit_weight_dtype_requested": args.dit_weight_dtype,
        "request_input_defaults": dict(pipeline_class.REQUEST_INPUT_DEFAULTS),
        "requested_configurations": ["dense", *args.configurations],
        "quality_budget": BUDGET,
        "runtime": capture_runtime_fingerprint().to_dict(),
        "performance_validated": False,
        "request_batching": args.request_batching,
        "complete": False,
        "configurations": {},
        "passed": True,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for name in report["requested_configurations"]:
        options = dict(configurations[name])
        if args.dit_weight_dtype is not None:
            options["dit_weight_dtype"] = args.dit_weight_dtype
        if name != "dense" and args.inplace_residual:
            options["inplace_residual"] = True
        started = perf_counter()
        pipeline = pipeline_class.from_pretrained(
            model_path=str(args.checkpoint.resolve()),
            device="cuda",
            torch_dtype="bfloat16",
            offload_mode="resident",
            attention_backend="torch",
            **options,
        )
        torch.cuda.synchronize()
        load_seconds = perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        warmup_report = pipeline.warmup([*native_warmup, batch_requests] if args.request_batching else native_warmup)
        torch.cuda.synchronize()
        denoiser = pipeline.native_pipeline.components.denoiser
        experts = (denoiser.high_noise, denoiser.low_noise) if args.model_id.endswith("a14b") else (denoiser,)
        entry: dict[str, Any] = {
            "options": options,
            "load_seconds": load_seconds,
            "dit_parameter_dtypes": sorted(
                {str(parameter.dtype) for expert in experts for parameter in expert.model.parameters()}
            ),
            "warmup_seconds": perf_counter() - started,
            "warmup_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "warmup_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "warmup": warmup_report,
            "requests": [],
        }
        report["configurations"][name] = entry
        for index in (0, 1, 0):
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            result = pipeline(**common, **cases[index], return_dict=True)
            torch.cuda.synchronize()
            seconds = perf_counter() - started
            output = SimpleNamespace(sample=result["sample"].cpu(), latents=result["latents"].cpu())
            receipt = pipeline.native_pipeline.components.denoiser.runtime_optimization_report()
            if name == "dense" and len(references) < 2:
                references.append(output)
            quality = compare_generation(references[index], output)
            execution_passed = _execution_passed(receipt, options)
            if name == "static-graph":
                quality["passed"] = quality["passed"] and quality["latent_bitwise_equal"]
            request_report = {
                "case": index,
                "seconds": seconds,
                "quality": quality,
                "execution_passed": bool(execution_passed),
                "denoiser": receipt,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            entry["requests"].append(request_report)
            report["passed"] = report["passed"] and quality["passed"] and execution_passed
            torch.save(vars(output), output_dir / f"{name}-{len(entry['requests'])}.pt")
            (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
            print(
                json.dumps(
                    {"configuration": name, "case": index, "quality": quality, "execution_passed": execution_passed}
                ),
                flush=True,
            )
            del result, output
        if args.request_batching:
            entry["batches"] = []
            for order in ((0, 1), (1, 0), (0, 1)):
                torch.cuda.reset_peak_memory_stats()
                started = perf_counter()
                results = pipeline.run_batch(tuple(batch_requests[index] for index in order))
                torch.cuda.synchronize()
                seconds = perf_counter() - started
                receipt = denoiser.runtime_optimization_report()
                execution_passed = _execution_passed(receipt, options)
                batch_entry = {
                    "order": order,
                    "seconds": seconds,
                    "denoiser": receipt,
                    "execution_passed": execution_passed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "requests": [],
                }
                entry["batches"].append(batch_entry)
                if len(results) != len(order):
                    raise RuntimeError("batch replay must return one output per request")
                for position, (index, result) in enumerate(zip(order, results, strict=True)):
                    output = SimpleNamespace(sample=result.sample.cpu(), latents=result.latents.cpu())
                    quality = compare_generation(references[index], output)
                    batching = dict(cast(Mapping[str, Any], result.metadata["request_batching"]))
                    request_execution = (
                        execution_passed
                        and batching["execution"] == "batched"
                        and batching["request_count"] == batching["sample_count"] == len(order)
                        and batching["request_index"] == position
                        and batching["denoiser_calls"] > 0
                        and result.metadata["seed"] == cases[index]["seed"]
                    )
                    batch_entry["requests"].append(
                        {
                            "case": index,
                            "quality": quality,
                            "batching": batching,
                            "execution_passed": request_execution,
                        }
                    )
                    report["passed"] = report["passed"] and quality["passed"] and request_execution
                    torch.save(vars(output), output_dir / f"{name}-batch-{len(entry['batches'])}-{position}.pt")
                    print(
                        json.dumps(
                            {
                                "configuration": name,
                                "batch": order,
                                "case": index,
                                "quality": quality,
                                "execution_passed": request_execution,
                            }
                        ),
                        flush=True,
                    )
                (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
                del results, result, output
            entry["batch_comparison"] = _batch_comparison(entry)
        del pipeline, denoiser, experts
        gc.collect()
        torch.cuda.empty_cache()
    report["complete"] = True
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def _batch_comparison(entry: dict[str, Any]) -> dict[str, Any]:
    """Compare the same two-request workload, without certifying shared-GPU time."""
    singleton_pair = sum(
        statistics.median(value["seconds"] for value in entry["requests"] if value["case"] == case) for case in (0, 1)
    )
    batch_seconds = [value["seconds"] for value in entry["batches"]]
    median_batch = statistics.median(batch_seconds)
    return {
        "scope": "diagnostic synchronized wall time; encode, sample and decode; no export or queue wait",
        "aggregation": "sum of per-case singleton medians versus median of three two-request batches",
        "singleton_pair_seconds": singleton_pair,
        "batch_seconds": batch_seconds,
        "median_batch_seconds": median_batch,
        "serial_requests_per_second": 2 / singleton_pair if singleton_pair > 0 else None,
        "batched_requests_per_second": 2 / median_batch if median_batch > 0 else None,
        "throughput_ratio": singleton_pair / median_batch if median_batch > 0 else None,
        "warmup_seconds": entry["warmup_seconds"],
        "peak_batch_allocated_bytes": max(value["peak_allocated_bytes"] for value in entry["batches"]),
        "performance_validated": False,
    }


def _execution_passed(receipt: dict[str, Any], options: dict[str, Any]) -> bool:
    """Require current-request execution from every routed expert."""
    dual = receipt.get("runtime", {}).get("dual_expert")
    reports = [receipt]
    if dual is not None:
        names = ("high-noise", "low-noise")
        if not all(dual["route_calls"][name] > 0 for name in names):
            return False
        reports = [dual["experts"][name] for name in names]
    for report in reports:
        effective = report.get("effective", {})
        if options.get("cuda_graph"):
            graph = effective.get("cuda_graph_report", {})
            if not (
                graph.get("replay", 0) > 0
                and graph.get("capture") == graph.get("capture_failed") == graph.get("eager") == 0
                and graph.get("capture_on_miss") is False
            ):
                return False
        if options.get("compile") and effective.get("compile") != "compiled-cuda-graph-replayed":
            return False
        if options.get("inplace_residual") and effective.get("inplace_residual") != "in-place-executed":
            return False
    return True


def warmup_requests_to_native(values: list[dict[str, Any]]) -> list[DiffusionRequest]:
    """Construct the explicit native warmup cases used by the public API."""
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig

    return [
        DiffusionRequest(
            **{key: value for key, value in case.items() if key != "sampling"},
            sampling=SamplingConfig(**case["sampling"]),
        )
        for case in values
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        choices=("wan2.1-t2v-1.3b", "wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.2-i2v-a14b"),
        default="wan2.1-t2v-1.3b",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--images", type=Path, nargs=2, metavar=("IMAGE_A", "IMAGE_B"))
    parser.add_argument("--inplace-residual", action="store_true", help="Enable native in-place gates on candidates.")
    parser.add_argument(
        "--request-batching",
        action="store_true",
        help="Also compare Wan T2V/I2V A/B, B/A, A/B batches with independent dense requests.",
    )
    parser.add_argument(
        "--dit-weight-dtype",
        choices=("float32", "bfloat16"),
        help="DiT weight dtype for every configuration, including dense; compute remains BF16.",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=("static-graph", "regional-static-graph"),
        default=("static-graph", "regional-static-graph"),
        help="Candidates to compare against dense with the same inputs and precision.",
    )
    args = parser.parse_args()
    return 0 if run(args)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
