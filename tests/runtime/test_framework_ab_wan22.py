from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.inference import framework_ab_wan22


def _live_wan_math_requested(
    *,
    rope_precision: str = "fp32",
    rms_norm_precision: str = "input",
) -> dict[str, object]:
    return {
        "rope_precision": rope_precision,
        "rms_norm_precision": rms_norm_precision,
    }


def _live_wan_math_effective(
    *,
    fused_rope: bool = False,
    rope_precision: str = "fp32",
    rms_norm_precision: str = "input",
) -> dict[str, object]:
    return {
        "rope_precision": (
            f"fused-{rope_precision}"
            if fused_rope
            else f"complex-{rope_precision}"
        ),
        "rms_norm_precision": {
            "mode": rms_norm_precision,
            "configured_modules": 60,
        },
        "wan_timestep_mode": "global-explicit-all-ones",
    }


def _live_qkv_static_runtime(*, compiled: bool = False) -> dict[str, object]:
    """Return truthful request receipts used by positive gate fixtures."""

    return {
        "qkv_fusion": {
            "fused_blocks": 30,
            "eager_projection_calls": 0 if compiled else 120,
            "compiled_graph_traces": 2 if compiled else 0,
            "strategy": "auto",
            "split_threshold": 8192,
            "eager_packed_projection_calls": 0,
            "eager_split_projection_calls": 0 if compiled else 120,
            "compiled_packed_graph_traces": 0,
            "compiled_split_graph_traces": 2 if compiled else 0,
            "lifetime_eager_packed_projection_calls": 0,
            "lifetime_eager_split_projection_calls": 0 if compiled else 120,
            "lifetime_compiled_packed_graph_traces": 0,
            "lifetime_compiled_split_graph_traces": 2 if compiled else 0,
            "execution": (
                "compiled-graph-traced (execution-pending-wrapper-receipt)"
                if compiled
                else "eager-projection-executed"
            ),
        },
        "static_cross_kv": {
            "wrapped_blocks": 30,
            "effective": "kv-reuse",
            "hits": 2940,
            "misses": 60,
            "condition_hits": 98,
            "condition_misses": 2,
            "bypasses": 0,
            "bypass_kwargs": [],
        },
        "wan_timestep": {
            "calls": 100,
            "global_calls": 100,
            "per_token_calls": 0,
            "explicit_all_ones_calls": 100,
        },
    }


def _vmoba_profile() -> dict[str, object]:
    return {
        "temporal_chunk_size": 1,
        "temporal_topk": 3,
        "spatial_chunk_size": [2, 5],
        "spatial_topk": 20,
        "st_chunk_size": [1, 2, 5],
        "st_topk": 15,
        "moba_select_mode": "threshold",
        "moba_threshold": 0.25,
        "moba_threshold_type": "query_head",
        "first_full_step": 12,
        "first_full_layer": 0,
        "temporal_layer": 1,
        "spatial_layer": 1,
        "st_layer": 1,
    }


def _live_vmoba_optimization_report() -> dict[str, object]:
    profile = _vmoba_profile()
    provider = "fastvideo_kernel.moba_attn_varlen"
    provider_symbols = [
        "moba_attn_varlen",
        "process_moba_input",
        "process_moba_output",
    ]
    grid = [31, 22, 40]
    tokens = 31 * 22 * 40
    request_id = "vmoba-request-current"
    request_epoch = 7
    steps = 50
    blocks = 30
    branches = ("positive", "negative")
    events: list[dict[str, object]] = []
    chunk_calls = {"temporal": 0, "spatial": 0, "spatiotemporal": 0}
    routes = (
        ("temporal", 1, 1 * 22 * 40, 3),
        ("spatial", [2, 5], 31 * 2 * 5, 20),
        ("spatiotemporal", [1, 2, 5], 1 * 2 * 5, 15),
    )
    tensor = {
        "shape": [tokens, 1, 64],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    packed = {
        "shape": [1, tokens, 1, 64],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    for branch in branches:
        for step in range(steps):
            for layer in range(blocks):
                sparse = step >= int(profile["first_full_step"])
                event: dict[str, object] = {
                    "algorithm": "vmoba",
                    "request_id": request_id,
                    "request_epoch": request_epoch,
                    "request_local": True,
                    "branch": branch,
                    "step": step,
                    "step_index": step,
                    "total_steps": steps,
                    "layer_idx": layer,
                    "module_path": f"blocks.{layer}.self_attn",
                    "execution": "sparse" if sparse else "scheduled_dense",
                    "provider_attempted": sparse,
                    "provider_path": provider if sparse else None,
                    "fallback_reason": None,
                    "dense_fallback_executed": False,
                    "grid_size": grid,
                    "input": tensor,
                    "output": tensor,
                    "input_shape": tensor["shape"],
                    "input_device": tensor["device"],
                    "input_dtype": tensor["dtype"],
                    "output_shape": tensor["shape"],
                    "output_device": tensor["device"],
                    "output_dtype": tensor["dtype"],
                }
                if sparse:
                    route, layout, provider_chunk_size, requested_topk = routes[
                        layer % len(routes)
                    ]
                    chunk_calls[route] += 1
                    event.update(
                        {
                            "chunk_kind": route,
                            "chunk_layout": layout,
                            "provider_chunk_size": provider_chunk_size,
                            "requested_topk": requested_topk,
                            "effective_topk": requested_topk,
                            "select_mode": profile["moba_select_mode"],
                            "threshold": profile["moba_threshold"],
                            "threshold_type": profile["moba_threshold_type"],
                            "provider_symbols": provider_symbols,
                            "validated_grid": grid,
                            "validated_profile": {
                                "temporal_chunk_size": 1,
                                "spatial_chunk_size": [2, 5],
                                "st_chunk_size": [1, 2, 5],
                            },
                            "packed_input": packed,
                            "restored_output": packed,
                        }
                    )
                events.append(event)
    sparse_events = [event for event in events if event["execution"] == "sparse"]
    scheduled_dense = len(events) - len(sparse_events)
    branch_reports = {
        branch: {
            "last_step": steps - 1,
            "observed_steps": list(range(steps)),
            "expected_steps": list(range(steps)),
            "completed_steps": list(range(steps)),
            "steps_contiguous": True,
            "routed_steps": False,
            "all_steps_complete": True,
            "layer_events": steps * blocks,
            "missing_layers": {},
        }
        for branch in branches
    }
    coverage = {
        "complete": True,
        "routed_steps": False,
        "expected_layers": list(range(blocks)),
        "expected_layer_count": blocks,
        "expected_calls": len(events),
        "observed_calls": len(events),
        "event_totals_match_counters": True,
        "provider_contract_complete": True,
        "request_local": True,
        "branches": branch_reports,
        "execution_counts": {
            "scheduled_dense": scheduled_dense,
            "sparse": len(sparse_events),
        },
    }
    receipt = {
        "kind": "vmoba",
        "request_id": request_id,
        "request_epoch": request_epoch,
        "request_local": True,
        "routed_steps": False,
        "finalized": True,
        "completed": True,
        "release_reason": "completed",
        "wrapped_blocks": blocks,
        "expected_layers": list(range(blocks)),
        "expected_calls": len(events),
        "runtime_effective": True,
        "kernel_attempts": len(sparse_events),
        "sparse_calls": len(sparse_events),
        "provider_dense_calls": 0,
        "kernel_fallbacks": 0,
        "dense_fallback_calls": 0,
        "scheduled_dense_calls": scheduled_dense,
        "effective_kernel": "vmoba",
        "provider_path": provider,
        "provider_paths": [provider],
        "grid_size": grid,
        "branches": branch_reports,
        "events": events,
        "event_count": len(events),
        "coverage": coverage,
        "notes": [],
        "lifecycle": {
            "live_requests": 0,
            "receipt_snapshots": 1,
            "max_receipt_snapshots": 32,
        },
        "vmoba": {
            "first_full_step": profile["first_full_step"],
            "first_full_layer": profile["first_full_layer"],
            "layer_cycle": {
                "temporal": 1,
                "spatial": 1,
                "spatiotemporal": 1,
            },
            "provider_symbols": provider_symbols,
            "chunk_calls": chunk_calls,
            "events": sparse_events,
        },
    }
    return {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "approximate_attention_kernel": "vmoba",
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "approximate_attention": receipt,
        },
    }


def _live_lightx2v_optimization_report() -> dict[str, object]:
    kind = "dynamic_sparse"
    operator = "triton"
    provider_family = "lightx2v/dynamic-sparse"
    provider = "lightx2v.common.ops.attn.kernels.sla_kernel._attention.apply"
    commit = "6fb7c1362b89d4908a9ea197bac4fbd7482ee2d5"
    source_fingerprint = hashlib.sha256(
        f"{commit}\0True\0".encode()
    ).hexdigest()
    grid = [31, 22, 40]
    tokens = 31 * 22 * 40
    steps = 2
    blocks = 30
    branches = ("positive", "negative")
    request_id = "lightx2v-request-current"
    request_epoch = 9
    tensor = {
        "shape": [1, tokens, 1, 64],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    events: list[dict[str, object]] = []
    for branch in branches:
        for step in range(steps):
            for layer in range(blocks):
                provider_fields = {
                    "provider_family": provider_family,
                    "operator": operator,
                    "adapter_path": (
                        "lightx2v.common.ops.attn.synthetic.Provider.apply"
                    ),
                    "provider_calls": 1,
                    "reference_lightx2v_commit": commit,
                    "provider_source_commit": commit,
                    "provider_source_clean": True,
                    "provider_source_fingerprint": source_fingerprint,
                    "provider_source_root": "/opt/lightx2v",
                    "reference_parity_verified": True,
                    "sparsity_ratio": 0.9,
                    "canonical_sparse_provider_path": provider,
                    "sparse_kernel_executed": True,
                    "provider_dense_executed": False,
                    "provider_config": {
                        "sparsity_ratio": 0.9,
                        "operator": operator,
                        "nbhd_coefficient": [1.0, 0.5, 0.056],
                        "nbhd_min_width": 1.0,
                        "attnmap_frame_num": grid[0],
                        "per_block_mean": False,
                        "pool_size": 128,
                        "skip_timesteps": -1,
                        "dense_attn_type": "flash_attn3",
                        "svg_sample_mse_max_row": 10000,
                        "svg_num_sampled_rows": 64,
                        "svg_context_length": 0,
                    },
                    "runtime_effective": True,
                }
                events.append(
                    {
                        "algorithm": kind,
                        "request_id": request_id,
                        "request_epoch": request_epoch,
                        "request_local": True,
                        "branch": branch,
                        "step": step,
                        "step_index": step,
                        "total_steps": steps,
                        "layer_idx": layer,
                        "module_path": f"blocks.{layer}.self_attn",
                        "execution": "sparse",
                        "provider_attempted": True,
                        "provider_path": provider,
                        "fallback_reason": None,
                        "dense_fallback_executed": False,
                        "grid_size": grid,
                        "input": tensor,
                        "output": tensor,
                        "input_shape": tensor["shape"],
                        "input_device": tensor["device"],
                        "input_dtype": tensor["dtype"],
                        "output_shape": tensor["shape"],
                        "output_device": tensor["device"],
                        "output_dtype": tensor["dtype"],
                        **provider_fields,
                    }
                )
    branch_reports = {
        branch: {
            "last_step": steps - 1,
            "observed_steps": list(range(steps)),
            "expected_steps": list(range(steps)),
            "completed_steps": list(range(steps)),
            "steps_contiguous": True,
            "routed_steps": False,
            "all_steps_complete": True,
            "layer_events": steps * blocks,
            "missing_layers": {},
        }
        for branch in branches
    }
    coverage = {
        "complete": True,
        "routed_steps": False,
        "expected_layers": list(range(blocks)),
        "expected_layer_count": blocks,
        "expected_calls": len(events),
        "observed_calls": len(events),
        "event_totals_match_counters": True,
        "provider_contract_complete": True,
        "request_local": True,
        "branches": branch_reports,
        "execution_counts": {"sparse": len(events)},
    }
    receipt = {
        "kind": kind,
        "request_id": request_id,
        "request_epoch": request_epoch,
        "request_local": True,
        "routed_steps": False,
        "finalized": True,
        "completed": True,
        "release_reason": "completed",
        "wrapped_blocks": blocks,
        "expected_layers": list(range(blocks)),
        "expected_calls": len(events),
        "runtime_effective": True,
        "kernel_attempts": len(events),
        "sparse_calls": len(events),
        "provider_dense_calls": 0,
        "scheduled_dense_calls": 0,
        "dense_fallback_calls": 0,
        "kernel_fallbacks": 0,
        "effective_kernel": kind,
        "provider_path": provider,
        "provider_paths": [provider],
        "grid_size": grid,
        "branches": branch_reports,
        "events": events,
        "event_count": len(events),
        "coverage": coverage,
        "vmoba": None,
        "notes": [],
        "lifecycle": {
            "live_requests": 0,
            "receipt_snapshots": 1,
            "max_receipt_snapshots": 32,
        },
        "lightx2v": {
            "algorithm": kind,
            "provider_family": provider_family,
            "provider_families": [provider_family],
            "operator": operator,
            "canonical_provider_symbol": provider,
            "canonical_provider_executed": True,
            "reference_lightx2v_commit": commit,
            "provider_source_commit": commit,
            "provider_source_commits": [commit],
            "provider_source_fingerprints": [source_fingerprint],
            "provider_commit_complete": True,
            "provider_source_clean": True,
            "reference_parity_verified": True,
            "provider_calls": len(events),
            "receipt_count": len(events),
            "sparse_event_count": len(events),
            "provider_dense_event_count": 0,
            "provider_dense_events": [],
            "events": events,
        },
    }
    return {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "approximate_attention_kernel": kind,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "approximate_attention": receipt,
        },
    }


def _live_block_feature_cache_runtime(
    algorithm: str = "dynamicblock",
    *,
    request_epoch: int = 3,
    request_id: str = "request-current",
    steps: int = 50,
) -> dict[str, object]:
    branches: dict[str, object] = {}
    for branch in ("positive", "negative"):
        event_receipts = []
        for step in range(steps):
            hit = step > 0
            if algorithm == "blocktaylorseer":
                hit = step % 4 != 0
            if not hit:
                dense_blocks = list(range(30))
                skipped_blocks: list[int] = []
            elif algorithm == "firstblock":
                dense_blocks = [0]
                skipped_blocks = list(range(1, 30))
            elif algorithm == "dualblock":
                dense_blocks = list(range(5)) + list(range(25, 30))
                skipped_blocks = list(range(5, 25))
            else:
                dense_blocks = []
                skipped_blocks = list(range(30))
            event_receipts.append(
                {
                    "algorithm": algorithm,
                    "step": step,
                    "hit": hit,
                    "reason": (
                        "seed"
                        if step == 0
                        else "taylor-extrapolation"
                        if hit and algorithm == "blocktaylorseer"
                        else "scheduled-dense"
                        if algorithm == "blocktaylorseer"
                        else "below-threshold"
                    ),
                    "dense_blocks": dense_blocks,
                    "skipped_blocks": skipped_blocks,
                    "dense_block_calls": len(dense_blocks),
                    "skipped_block_calls": len(skipped_blocks),
                    "relative_l1": 0.0 if hit else None,
                }
            )
        dense_calls = sum(event["dense_block_calls"] for event in event_receipts)
        skipped_calls = sum(
            event["skipped_block_calls"] for event in event_receipts
        )
        hits = sum(int(event["hit"]) for event in event_receipts)
        receipt = {
            "algorithm": algorithm,
            "branch": branch,
            "request_id": request_id,
            "request_epoch": request_epoch,
            "residual_diff_threshold": 0.1,
            "downsample_factor": 1,
            "dense_first": 1,
            "dense_last": 0,
            "events": steps,
            "hits": hits,
            "dense_block_calls": dense_calls,
            "skipped_block_calls": skipped_calls,
            "event_receipts": event_receipts,
        }
        if algorithm == "blocktaylorseer":
            receipt.update(
                prediction_scope="per-block-self-cross-ffn",
                taylor_order=1,
                dense_pattern=[True, False, False, False],
            )
        branches[branch] = {
            "algorithm": algorithm,
            "request_id": request_id,
            "request_epoch": request_epoch,
            "request_local": True,
            "events": steps,
            "hits": hits,
            "misses": steps - hits,
            "hit_rate": hits / steps,
            "reasons": {},
            "dense_block_calls": dense_calls,
            "skipped_block_calls": skipped_calls,
            "receipt": receipt,
        }
    total_hits = sum(branch["hits"] for branch in branches.values())
    total_dense = sum(branch["dense_block_calls"] for branch in branches.values())
    total_skipped = sum(
        branch["skipped_block_calls"] for branch in branches.values()
    )
    return {
        "enabled": True,
        "algorithm": algorithm,
        "threshold": 0.1,
        "request_id": request_id,
        "request_epoch": request_epoch,
        "request_local": True,
        "finalized": True,
        "release_reason": "completed",
        "effective": "residual-reuse",
        "events": steps * 2,
        "hits": total_hits,
        "misses": steps * 2 - total_hits,
        "hit_rate": total_hits / (steps * 2),
        "dense_block_calls": total_dense,
        "skipped_block_calls": total_skipped,
        "branches": branches,
    }


def _live_custom_feature_cache_runtime(
    *,
    request_epoch: int = 4,
    request_id: str = "custom-request-current",
    steps: int = 50,
) -> dict[str, object]:
    branches = {}
    for branch in ("positive", "negative"):
        events = [
            {
                "step": step,
                "hit": step >= 5 and step % 4 != 0,
                "reason": (
                    "seed"
                    if step == 0
                    else "dense-boundary"
                    if step < 5
                    else "polynomial-taylor-extrapolation"
                    if step % 4 != 0
                    else "polynomial-threshold"
                ),
                "accumulated_change": 0.01,
            }
            for step in range(steps)
        ]
        hits = sum(int(event["hit"]) for event in events)
        receipt = {
            "algorithm": "custom",
            "branch": branch,
            "request_id": request_id,
            "request_epoch": request_epoch,
            "decision": "teacache-polynomial",
            "prediction": "first-order-stack-residual",
            "events": steps,
            "hits": hits,
            "event_receipts": events,
        }
        branches[branch] = {
            "algorithm": "custom",
            "request_id": request_id,
            "request_epoch": request_epoch,
            "request_local": True,
            "events": steps,
            "hits": hits,
            "misses": steps - hits,
            "hit_rate": hits / steps,
            "reasons": {},
            "dense_block_calls": 0,
            "skipped_block_calls": 0,
            "receipt": receipt,
        }
    total_hits = sum(branch["hits"] for branch in branches.values())
    return {
        "enabled": True,
        "algorithm": "custom",
        "threshold": 0.26,
        "request_id": request_id,
        "request_epoch": request_epoch,
        "request_local": True,
        "finalized": True,
        "release_reason": "completed",
        "effective": "residual-reuse",
        "events": steps * 2,
        "hits": total_hits,
        "misses": steps * 2 - total_hits,
        "hit_rate": total_hits / (steps * 2),
        "dense_block_calls": 0,
        "skipped_block_calls": 0,
        "branches": branches,
    }


def _record(
    *,
    framework: str,
    pair_id: str,
    gpu: int,
    order: str,
    generation_s: float,
    tag: str = "test",
) -> dict:
    digits = "".join(character for character in pair_id if character.isdigit())
    pair_index = int(digits or 0)
    first_framework = "worldfoundry" if order == "AB" else (
        "fastvideo" if framework in {"worldfoundry", "fastvideo"} else "lightx2v"
    )
    lane_index = 0 if framework == first_framework else 1
    started = datetime(2026, 8, 31, tzinfo=timezone.utc) + timedelta(
        minutes=pair_index * 10 + lane_index * 2
    )
    completed = started + timedelta(minutes=1)
    gpu_uuid = f"GPU-test-{gpu}"
    audit_started = 100.0 + pair_index * 10 + lane_index
    timestamps = [audit_started + 0.01, audit_started + 0.11]
    audit_stopped = audit_started + 0.20
    isolation = {
        "audit_version": framework_ab_wan22.ISOLATION_AUDIT_VERSION,
        "monitored_gpu_uuid": gpu_uuid,
        "window_policy": "post-idle-gate-through-completed-output",
        "window_started_at": started.isoformat(),
        "window_stopped_at": completed.isoformat(),
        "window_started_monotonic_seconds": audit_started,
        "window_stopped_monotonic_seconds": audit_stopped,
        "compute_apps_before": [],
        "compute_apps_after": [{"pid": 900, "used_memory_mib": 512.0}],
        "added_compute_apps": [],
        "removed_compute_apps": [],
        "monitor_errors": [],
        "samples": 2,
        "poll_interval_seconds": 0.1,
        "sample_timestamps_monotonic_seconds": timestamps,
        "max_sample_gap_seconds": 0.1,
        "query_latency_seconds": {
            "raw": [0.001, 0.001],
            "max": 0.001,
            "median": 0.001,
        },
        "allowed_root_pids": [100],
        "allowed_compute_pids": [100],
        "allowed_process_identities": [
            {
                "pid": 100,
                "starttime_ticks": 123,
                "nspid": [900, 100],
                "host_pid": 900,
                "host_pid_mapping_proven": True,
            }
        ],
        "bound_host_compute_pids": [900],
        "host_pid_bindings": [
            {
                "host_pid": 900,
                "local_pid": 100,
                "local_starttime_ticks": 123,
                "proof": "exclusive-process-temporal-binding",
            }
        ],
        "identity_proof": "exclusive-process-compute-mode",
        "exclusive_compute_mode": True,
        "thread_stopped": True,
        "bounded_polling_complete": True,
        "certifying": True,
        "contaminated": False,
    }
    return {
        "schema_version": framework_ab_wan22.SCHEMA_VERSION,
        "status": "completed",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "semantic_config_sha256": "compatible-config",
        "semantic_config": {
            "protocol_id": framework_ab_wan22.PROTOCOL_ID,
            "isolation_policy": {
                "require_idle_at_start": True,
                "fail_on_generation_contamination": True,
                "proof_mode": "compute-exclusive",
                "poll_interval_seconds": 0.1,
                "allow_unproven_pid_binding": False,
            },
            "worldfoundry_optimization_profile": {"require_effective": True},
        },
        "run": {
            "framework": framework,
            "pair_id": pair_id,
            "physical_gpu": gpu,
            "order": order,
            "run_id": f"{pair_id}-{framework}",
            "tag": tag,
        },
        "measurements": {
            "framework_import_s": 1.0,
            "model_load_s": 2.0,
            "generation_s": generation_s,
            "gpu_generation": {"peak_memory_used_mib": 1024.0},
            "gpu_at_generation_start": {"uuid": gpu_uuid},
            "generation_isolation": isolation,
        },
        "idle_gate": [{"uuid": gpu_uuid, "compute_mode": "Exclusive_Process"}],
        "cuda_device_mapping": {
            "cuda_visible_devices": str(gpu),
            "framework_local_device": 0,
            "current_cuda_device": 0,
            "selected_visible_device": str(gpu),
            "monitored_physical_gpu": gpu,
            "monitored_gpu_uuid": gpu_uuid,
            "current_cuda_uuid": gpu_uuid,
            "checks": {
                "visible_token_selects_monitored_gpu": True,
                "current_cuda_uuid_matches_monitored_gpu": True,
            },
            "single_device_mapping_proven": True,
        },
    }


def _write_record(directory: Path, payload: dict) -> None:
    path = directory / "records" / f"{payload['run']['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_uses_same_gpu_paired_ratios(tmp_path: Path) -> None:
    samples = (
        _record(
            framework="worldfoundry",
            pair_id="r0-gpu0",
            gpu=0,
            order="AB",
            generation_s=5.0,
        ),
        _record(
            framework="fastvideo",
            pair_id="r0-gpu0",
            gpu=0,
            order="AB",
            generation_s=10.0,
        ),
        _record(
            framework="worldfoundry",
            pair_id="r1-gpu0",
            gpu=0,
            order="BA",
            generation_s=8.0,
        ),
        _record(
            framework="fastvideo",
            pair_id="r1-gpu0",
            gpu=0,
            order="BA",
            generation_s=12.0,
        ),
    )
    for sample in samples:
        _write_record(tmp_path, sample)

    args = argparse.Namespace(
        output_dir=tmp_path,
        tag="test",
        min_samples=2,
        bootstrap_samples=200,
        allow_insufficient=False,
    )
    assert framework_ab_wan22.summarize(args) == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["paired_sample_count"] == 2
    assert summary["worldfoundry_speedup_vs_fastvideo"] == pytest.approx(1.75)
    assert summary["worldfoundry_ratio_of_framework_medians"] == pytest.approx(11.0 / 6.5)
    assert summary["worldfoundry_speedup_bootstrap_95ci"][0] >= 1.5
    assert summary["certification"]["certifying"] is False
    assert summary["claim"] == "non_certifying_diagnostic"
    assert any(
        "requires at least 6 complete pairs" in reason
        for reason in summary["certification"]["noncertifying_reasons"]
    )
    assert summary["video_quality_gate"]["enabled"] is False
    assert summary["video_quality_gate"]["certification_role"] == "diagnostic-only"
    assert summary["video_quality_gate"]["disabled_reason"]


def test_formal_certification_requires_six_pairs_and_three_per_order(
    tmp_path: Path,
) -> None:
    for pair_index in range(6):
        order = "AB" if pair_index % 2 == 0 else "BA"
        for framework, generation_s in (
            ("worldfoundry", 5.0 + pair_index / 10),
            ("fastvideo", 10.0 + pair_index / 10),
        ):
            _write_record(
                tmp_path,
                _record(
                    framework=framework,
                    pair_id=f"r{pair_index}-gpu0",
                    gpu=0,
                    order=order,
                    generation_s=generation_s,
                ),
            )
    args = argparse.Namespace(
        output_dir=tmp_path,
        tag="test",
        # A low CLI threshold must not weaken the fixed certification floor.
        min_samples=1,
        bootstrap_samples=100,
        allow_insufficient=False,
        reference_framework="fastvideo",
    )

    assert framework_ab_wan22.summarize(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["certification"]["certifying"] is True
    assert summary["certification"]["order_counts"] == {"AB": 3, "BA": 3}
    assert summary["certification"]["minimum_formal_pairs"] == 6
    assert summary["claim"] == "worldfoundry_faster"

    args.allow_insufficient = True
    assert framework_ab_wan22.summarize(args) == 0
    diagnostic = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert diagnostic["certification"]["certifying"] is False
    assert "--allow-insufficient was enabled" in diagnostic["certification"][
        "noncertifying_reasons"
    ]


def test_pairing_excludes_cross_gpu_samples() -> None:
    worldfoundry = _record(
        framework="worldfoundry",
        pair_id="bad-pair",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    fastvideo = _record(
        framework="fastvideo",
        pair_id="bad-pair",
        gpu=1,
        order="AB",
        generation_s=6.0,
    )
    pairs, exclusions = framework_ab_wan22._paired_records((worldfoundry, fastvideo))
    assert not pairs
    assert exclusions == [
        {
            "reason": "physical_gpu_mismatch",
            "pair_id": "bad-pair",
            "worldfoundry_gpu": 0,
            "fastvideo_gpu": 1,
        }
    ]


def test_benchmark_source_fingerprint_hashes_runner_and_launcher() -> None:
    fingerprint = framework_ab_wan22._benchmark_source_fingerprint()

    assert set(fingerprint["files"]) == {"runner", "launcher"}
    for item in fingerprint["files"].values():
        path = Path(item["path"])
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["size_bytes"] == path.stat().st_size
        assert item["dirty"] is bool(item["git_status"])
    assert len(fingerprint["identity_sha256"]) == 64


def test_git_state_records_dirty_content_fingerprint(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], *, check: bool = True):
        del check
        if "rev-parse" in command:
            stdout = "abc123\n"
        elif "status" in command:
            stdout = " M worldfoundry/runtime.py\n?? benchmarks/inference/runner.py\n"
        elif "diff" in command:
            stdout = "diff --git a/runtime.py b/runtime.py\n+changed\n"
        else:  # pragma: no cover - protects the command contract
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(framework_ab_wan22, "_run", fake_run)
    state = framework_ab_wan22._git_state(tmp_path)

    assert state["commit"] == "abc123"
    assert state["dirty"] is True
    assert state["dirty_entry_count"] == 2
    assert len(state["tracked_diff_sha256"]) == 64
    assert len(state["dirty_fingerprint_sha256"]) == 64


def test_source_tree_fingerprint_changes_with_file_content(tmp_path: Path) -> None:
    package = tmp_path / "runtime"
    package.mkdir()
    source = package / "kernel.py"
    source.write_text("provider = 'dense'\n", encoding="utf-8")
    first = framework_ab_wan22._source_tree_fingerprint(
        repository=tmp_path,
        roots=(package,),
    )

    source.write_text("provider = 'int8'\n", encoding="utf-8")
    second = framework_ab_wan22._source_tree_fingerprint(
        repository=tmp_path,
        roots=(package,),
    )

    assert first["file_count"] == second["file_count"] == 1
    assert first["sha256"] != second["sha256"]


@pytest.mark.parametrize(
    ("completed", "expected"),
    (
        (
            subprocess.CompletedProcess(
                ["nvidia-smi"],
                17,
                stdout="",
                stderr="driver unavailable",
            ),
            "exit 17",
        ),
        (
            subprocess.CompletedProcess(
                ["nvidia-smi"],
                0,
                stdout="not-a-pid, 1024\n",
                stderr="",
            ),
            "malformed",
        ),
    ),
)
def test_gpu_compute_apps_fails_closed_on_query_error(
    monkeypatch,
    completed: subprocess.CompletedProcess[str],
    expected: str,
) -> None:
    monkeypatch.setattr(
        framework_ab_wan22,
        "_run",
        lambda *_args, **_kwargs: completed,
    )

    with pytest.raises(RuntimeError, match=expected):
        framework_ab_wan22._gpu_compute_apps(0)


def test_gpu_compute_apps_fails_closed_on_timeout(monkeypatch) -> None:
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["nvidia-smi"], timeout=10)

    monkeypatch.setattr(framework_ab_wan22, "_run", timed_out)

    with pytest.raises(RuntimeError, match="timed out"):
        framework_ab_wan22._gpu_compute_apps(0)


def test_cuda_visibility_requires_live_runtime_uuid(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    snapshot = {"uuid": "GPU-card-3"}

    evidence = framework_ab_wan22._cuda_visibility_evidence(
        3,
        snapshot,
        current_cuda_uuid="GPU-card-3",
    )

    assert evidence["single_device_mapping_proven"] is True
    assert evidence["current_cuda_uuid"] == "GPU-card-3"
    with pytest.raises(RuntimeError, match="live CUDA UUID"):
        framework_ab_wan22._cuda_visibility_evidence(
            3,
            snapshot,
            current_cuda_uuid="GPU-other-card",
        )


def test_gpu_uuid_normalizes_unprefixed_torch_2p9_value() -> None:
    bare_uuid = "aa5247c3-5298-d7e4-7a4d-903559cbe65b"

    assert framework_ab_wan22._normalize_gpu_uuid(
        bare_uuid,
        source="torch.cuda device properties",
    ) == f"GPU-{bare_uuid}"
    assert framework_ab_wan22._normalize_gpu_uuid(
        f"GPU-{bare_uuid}",
        source="nvidia-smi",
    ) == f"GPU-{bare_uuid}"


def test_generation_isolation_detects_new_compute_app_and_enforces_policy() -> None:
    report = framework_ab_wan22._generation_isolation_report(
        [{"pid": 20, "used_memory_mib": 1024}],
        [
            {"pid": 20, "used_memory_mib": 2048},
            {"pid": 30, "used_memory_mib": 4096},
        ],
    )

    assert report["contaminated"] is True
    assert report["added_compute_apps"] == [{"pid": 30, "used_memory_mib": 4096.0}]
    with pytest.raises(RuntimeError, match="isolation certification failed"):
        framework_ab_wan22._raise_for_generation_contamination(
            {"generation_isolation": report},
            enabled=True,
        )
    framework_ab_wan22._raise_for_generation_contamination(
        {"generation_isolation": report},
        enabled=False,
    )


def test_compute_app_monitor_retains_transient_pid(monkeypatch) -> None:
    snapshots = iter(
        (
            [
                {"pid": 20, "used_memory_mib": 1024.0},
                {"pid": 30, "used_memory_mib": 4096.0},
            ],
            [{"pid": 20, "used_memory_mib": 1024.0}],
        )
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: next(snapshots),
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(
        0,
        [{"pid": 20, "used_memory_mib": 1024.0}],
    )

    monitor._sample()
    report = monitor.stop()

    assert report["compute_apps_after"] == [
        {"pid": 20, "used_memory_mib": 1024.0}
    ]
    assert report["added_compute_apps"] == [
        {"pid": 20, "used_memory_mib": 1024.0},
        {"pid": 30, "used_memory_mib": 4096.0},
    ]
    assert report["samples"] == 2
    assert report["contaminated"] is True


def test_compute_app_monitor_rejects_baseline_and_unproven_numeric_descendant(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: [
            {"pid": 20, "used_memory_mib": 1024.0},
            {"pid": 30, "used_memory_mib": 4096.0},
        ],
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_is_descendant_process",
        lambda pid, roots: pid == 30 and roots == {10},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_descendant_process_pids",
        lambda roots: {10, 30} if roots == {10} else set(roots),
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(
        0,
        [{"pid": 20, "used_memory_mib": 1024.0}],
        allowed_root_pids={10},
        allow_descendants=True,
    )

    monitor.start()
    report = monitor.stop()

    assert report["allowed_root_pids"] == [10]
    assert report["allowed_compute_pids"] == [10, 30]
    assert report["added_compute_apps"] == [
        {"pid": 20, "used_memory_mib": 1024.0},
        {"pid": 30, "used_memory_mib": 4096.0},
    ]
    assert report["contaminated"] is True


def test_compute_app_monitor_binds_descendant_across_pid_namespace(
    monkeypatch,
) -> None:
    snapshots = iter(
        (
            [{"pid": 900, "used_memory_mib": 1024.0}],
            [
                {"pid": 900, "used_memory_mib": 1024.0},
                {"pid": 901, "used_memory_mib": 2048.0},
            ],
            [
                {"pid": 900, "used_memory_mib": 1024.0},
                {"pid": 901, "used_memory_mib": 2048.0},
            ],
        )
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: next(snapshots),
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_device_client_pids",
        lambda _physical_gpu, _candidate_pids=None: {30},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_is_descendant_process",
        lambda pid, roots: pid == 30 and roots == {10},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_descendant_process_pids",
        lambda roots: {10, 30} if roots == {10} else set(roots),
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_process_identity",
        lambda pid: {
            "pid": pid,
            "starttime_ticks": 123 if pid == 30 else 100,
            "nspid": [900, 30] if pid == 30 else [800, 10],
            "host_pid": 900 if pid == 30 else 800,
            "host_pid_mapping_proven": True,
        },
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(
        0,
        [],
        allowed_root_pids={10},
        allow_descendants=True,
    )

    monitor._sample()
    monitor._sample()
    report = monitor.stop()

    assert report["allowed_compute_pids"] == [10, 30]
    assert report["bound_host_compute_pids"] == [900]
    assert report["added_compute_apps"] == [
        {"pid": 901, "used_memory_mib": 2048.0}
    ]
    assert report["contaminated"] is True


def test_compute_app_monitor_does_not_reuse_host_pid_binding_slot(
    monkeypatch,
) -> None:
    """One local CUDA client cannot authorize sequential host-namespace PIDs."""

    snapshots = iter(
        (
            [{"pid": 900, "used_memory_mib": 1024.0}],
            [{"pid": 901, "used_memory_mib": 2048.0}],
            [{"pid": 901, "used_memory_mib": 2048.0}],
        )
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: next(snapshots),
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_device_client_pids",
        lambda _physical_gpu, _candidate_pids=None: {30},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_is_descendant_process",
        lambda _pid, _roots: False,
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_descendant_process_pids",
        lambda roots: {10, 30} if roots == {10} else set(roots),
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_process_identity",
        lambda pid: {
            "pid": pid,
            "starttime_ticks": 123 if pid == 30 else 100,
            "nspid": [pid],
            "host_pid": None,
            "host_pid_mapping_proven": False,
        },
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(
        0,
        [],
        allowed_root_pids={10},
        allow_descendants=True,
        allow_unproven_pid_binding=True,
    )

    monitor._sample()
    monitor._sample()
    report = monitor.stop()

    assert report["bound_host_compute_pids"] == [900]
    assert report["added_compute_apps"] == [
        {"pid": 901, "used_memory_mib": 2048.0}
    ]
    assert report["contaminated"] is True


def test_compute_app_monitor_does_not_bind_a_stale_device_client(
    monkeypatch,
) -> None:
    """A PID seen holding the GPU in an earlier poll cannot authorize a later app."""

    app_snapshots = iter(
        (
            [],
            [{"pid": 900, "used_memory_mib": 2048.0}],
            [{"pid": 900, "used_memory_mib": 2048.0}],
        )
    )
    device_clients = iter(({10}, set(), set()))
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: next(app_snapshots),
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_device_client_pids",
        lambda _physical_gpu, _candidate_pids=None: next(device_clients),
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_process_identity",
        lambda pid: {
            "pid": pid,
            "starttime_ticks": 123,
            "nspid": [pid],
            "host_pid": None,
            "host_pid_mapping_proven": False,
        },
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(
        0,
        [],
        allowed_root_pids={10},
        exclusive_compute_mode=True,
        monitored_gpu_uuid="GPU-card-0",
    )

    monitor._sample()
    monitor._sample()
    report = monitor.stop()

    assert report["bound_host_compute_pids"] == []
    assert report["added_compute_apps"] == [
        {"pid": 900, "used_memory_mib": 2048.0}
    ]
    assert report["contaminated"] is True


def test_compute_app_monitor_does_not_trust_host_local_numeric_collision(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: [{"pid": 30, "used_memory_mib": 1024.0}],
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_device_client_pids",
        lambda _physical_gpu, _candidate_pids=None: {30},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_process_identity",
        lambda pid: {
            "pid": pid,
            "starttime_ticks": 123,
            "nspid": [pid],
            "host_pid": None,
            "host_pid_mapping_proven": False,
        },
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(
        0,
        [],
        allowed_root_pids={30},
    )

    monitor.start()
    report = monitor.stop()

    assert report["bound_host_compute_pids"] == []
    assert report["added_compute_apps"] == [
        {"pid": 30, "used_memory_mib": 1024.0}
    ]
    assert report["identity_proof"] == "linux-nspid"
    assert report["contaminated"] is True


def test_compute_app_monitor_fails_closed_when_thread_does_not_stop(
    monkeypatch,
) -> None:
    monkeypatch.setattr(framework_ab_wan22, "_gpu_compute_apps", lambda _gpu: [])
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_device_client_pids",
        lambda _gpu, _candidate_pids=None: set(),
    )
    monitor = framework_ab_wan22.ComputeAppMonitor(0, [], allowed_root_pids={10})
    monitor._sample()

    class StuckThread:
        def join(self, timeout: float) -> None:
            assert timeout >= framework_ab_wan22.GPU_QUERY_TIMEOUT_SECONDS

        def is_alive(self) -> bool:
            return True

    monitor._thread = StuckThread()  # type: ignore[assignment]
    report = monitor.stop()

    assert report["thread_stopped"] is False
    assert report["contaminated"] is True
    assert any("did not stop" in error for error in report["monitor_errors"])


def test_run_one_fails_contaminated_release_but_allows_explicit_smoke(
    monkeypatch,
    tmp_path: Path,
) -> None:
    contaminated_measurements = {
        "output_path": str(tmp_path / "video.mp4"),
        "generation_isolation": framework_ab_wan22._generation_isolation_report(
            [{"pid": 10, "used_memory_mib": 1024.0}],
            [
                {"pid": 10, "used_memory_mib": 1024.0},
                {"pid": 20, "used_memory_mib": 2048.0},
            ],
        ),
    }
    monkeypatch.setattr(
        framework_ab_wan22,
        "_require_idle_gpu",
        lambda *_args, **_kwargs: [
            {
                "compute_apps": [],
                "uuid": "GPU-test-0",
                "compute_mode": "Exclusive_Process",
            }
        ],
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_cuda_visibility_evidence",
        lambda physical_gpu, snapshot: {
            "framework_local_device": 0,
            "current_cuda_device": 0,
            "monitored_physical_gpu": physical_gpu,
            "monitored_gpu_uuid": snapshot["uuid"],
            "current_cuda_uuid": snapshot["uuid"],
            "single_device_mapping_proven": True,
        },
    )

    contaminated_audit = {
        **_record(
            framework="worldfoundry",
            pair_id="fake",
            gpu=0,
            order="AB",
            generation_s=1.0,
        )["measurements"]["generation_isolation"],
        "added_compute_apps": [{"pid": 20, "used_memory_mib": 2048.0}],
        "contaminated": True,
        "certifying": False,
    }

    class FakeMonitor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> dict:
            return dict(contaminated_audit)

    monkeypatch.setattr(framework_ab_wan22, "ComputeAppMonitor", FakeMonitor)
    monkeypatch.setattr(
        framework_ab_wan22,
        "_checkpoint_fingerprint",
        lambda path: {"identity_sha256": str(path)},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_git_state",
        lambda path, **_kwargs: {
            "path": str(path),
            "commit": "commit",
            "dirty_fingerprint_sha256": "dirty",
        },
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_runtime_source_fingerprints",
        lambda **_kwargs: {
            name: {"sha256": f"{name}-source"}
            for name in ("worldfoundry", "fastvideo", "lightx2v")
        },
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_benchmark_source_fingerprint",
        lambda: {"files": {}, "dirty": False, "identity_sha256": "sources"},
    )
    monkeypatch.setattr(framework_ab_wan22, "_package_versions", lambda: {})
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 20, "used_memory_mib": 2048.0}],
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_run_worldfoundry",
        lambda _args, _path: contaminated_measurements,
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_validate_video",
        lambda _path, _args: {"full_decode_ok": True},
    )

    common = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--output-dir",
        str(tmp_path),
        "--physical-gpu",
        "0",
    ]
    parser = framework_ab_wan22.build_parser()
    release = parser.parse_args([*common, "--run-id", "contaminated-release"])
    assert framework_ab_wan22.run_one(release) == 1
    release_record = json.loads((tmp_path / "records" / "contaminated-release.json").read_text(encoding="utf-8"))
    assert release_record["status"] == "failed"
    assert release_record["measurements"]["generation_isolation"]["contaminated"] is True
    assert "isolation certification failed" in release_record["error"]["message"]

    smoke = parser.parse_args(
        [
            *common,
            "--run-id",
            "contaminated-smoke",
            "--no-fail-on-contamination",
        ]
    )
    assert framework_ab_wan22.run_one(smoke) == 0
    smoke_record = json.loads((tmp_path / "records" / "contaminated-smoke.json").read_text(encoding="utf-8"))
    assert smoke_record["status"] == "completed"
    assert smoke_record["measurements"]["generation_isolation"]["contaminated"] is True


def test_summary_filter_excludes_contaminated_and_unaudited_records() -> None:
    clean = _record(
        framework="worldfoundry",
        pair_id="clean",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    contaminated = _record(
        framework="fastvideo",
        pair_id="contaminated",
        gpu=0,
        order="AB",
        generation_s=6.0,
    )
    contaminated["measurements"]["generation_isolation"].update(
        contaminated=True,
        added_compute_apps=[{"pid": 30, "used_memory_mib": 4096.0}],
    )
    unaudited = _record(
        framework="fastvideo",
        pair_id="legacy",
        gpu=0,
        order="AB",
        generation_s=7.0,
    )
    del unaudited["measurements"]["generation_isolation"]

    accepted, exclusions = framework_ab_wan22._filter_generation_isolation_records((clean, contaminated, unaudited))

    assert accepted == [clean]
    assert [item["reason"] for item in exclusions] == [
        "generation_compute_app_contamination",
        "missing_generation_isolation_audit",
    ]


def test_summary_filter_rejects_endpoint_only_pseudo_v4_audit() -> None:
    record = _record(
        framework="worldfoundry",
        pair_id="pseudo-v4",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    record["measurements"]["generation_isolation"] = {
        "compute_apps_before": [],
        "compute_apps_after": [],
        "added_compute_apps": [],
        "contaminated": False,
    }

    accepted, exclusions = framework_ab_wan22._filter_generation_isolation_records(
        (record,)
    )

    assert accepted == []
    assert exclusions[0]["reason"] == "invalid_generation_isolation_audit"
    assert "missing fields" in exclusions[0]["issues"][0]


def test_summary_filter_rejects_live_cuda_uuid_mismatch() -> None:
    record = _record(
        framework="worldfoundry",
        pair_id="uuid-mismatch",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    record["cuda_device_mapping"]["current_cuda_uuid"] = "GPU-other-card"

    accepted, exclusions = framework_ab_wan22._filter_generation_isolation_records(
        (record,)
    )

    assert accepted == []
    assert exclusions[0]["reason"] == "invalid_generation_isolation_audit"
    assert any("GPU UUID evidence disagrees" in issue for issue in exclusions[0]["issues"])


def test_loader_excludes_old_framework_schema(tmp_path: Path) -> None:
    record = _record(
        framework="worldfoundry",
        pair_id="old-schema",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    record["schema_version"] = "worldfoundry-framework-ab-v3"
    _write_record(tmp_path, record)

    assert framework_ab_wan22._load_completed_records(tmp_path / "records") == []


def test_summary_escape_hatch_is_always_noncertifying(tmp_path: Path) -> None:
    for framework, generation_s in (("worldfoundry", 5.0), ("fastvideo", 10.0)):
        _write_record(
            tmp_path,
            _record(
                framework=framework,
                pair_id="escape",
                gpu=0,
                order="AB",
                generation_s=generation_s,
            ),
        )
    args = argparse.Namespace(
        output_dir=tmp_path,
        tag="test",
        min_samples=1,
        bootstrap_samples=100,
        allow_insufficient=False,
        allow_contaminated=True,
        allow_unaudited_generation_isolation=False,
        reference_framework="fastvideo",
    )

    assert framework_ab_wan22.summarize(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["certification"]["certifying"] is False
    assert summary["claim"] == "non_certifying_diagnostic"


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    (
        ("require_idle_at_start", False, "--no-require-idle"),
        (
            "fail_on_generation_contamination",
            False,
            "--no-fail-on-contamination",
        ),
        (
            "allow_unproven_pid_binding",
            True,
            "--allow-unproven-pid-binding",
        ),
        ("proof_mode", "bounded-polling", "compute-exclusive"),
    ),
)
def test_record_smoke_switches_are_always_noncertifying(
    field: str,
    value: object,
    expected_reason: str,
) -> None:
    record = _record(
        framework="worldfoundry",
        pair_id="smoke-policy",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    record["semantic_config"]["isolation_policy"][field] = value

    reasons = framework_ab_wan22._record_smoke_mode_reasons(record)

    assert any(expected_reason in reason for reason in reasons)


def test_clean_smoke_record_is_accepted_only_as_noncertifying() -> None:
    record = _record(
        framework="worldfoundry",
        pair_id="clean-smoke",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    record["semantic_config"]["isolation_policy"][
        "fail_on_generation_contamination"
    ] = False

    accepted, exclusions = framework_ab_wan22._filter_generation_isolation_records(
        (record,)
    )

    assert accepted == [record]
    assert exclusions == []
    assert "run used --no-fail-on-contamination" in record[
        "_noncertifying_reasons"
    ]


def test_summary_rejects_cross_card_repeats(tmp_path: Path) -> None:
    records = (
        _record(
            framework="worldfoundry",
            pair_id="r0-gpu0",
            gpu=0,
            order="AB",
            generation_s=5.0,
        ),
        _record(
            framework="fastvideo",
            pair_id="r0-gpu0",
            gpu=0,
            order="AB",
            generation_s=10.0,
        ),
        _record(
            framework="worldfoundry",
            pair_id="r1-gpu1",
            gpu=1,
            order="BA",
            generation_s=6.0,
        ),
        _record(
            framework="fastvideo",
            pair_id="r1-gpu1",
            gpu=1,
            order="BA",
            generation_s=11.0,
        ),
    )
    for record in records:
        _write_record(tmp_path, record)
    args = argparse.Namespace(
        output_dir=tmp_path,
        tag="test",
        min_samples=2,
        bootstrap_samples=100,
        allow_insufficient=False,
        reference_framework="fastvideo",
    )

    with pytest.raises(RuntimeError, match="one physical GPU/UUID"):
        framework_ab_wan22.summarize(args)


def test_summary_rejects_unbalanced_order_repeats(tmp_path: Path) -> None:
    for pair_id, base in (("r0-gpu0", 5.0), ("r1-gpu0", 6.0)):
        for framework, generation_s in (
            ("worldfoundry", base),
            ("fastvideo", base + 5.0),
        ):
            _write_record(
                tmp_path,
                _record(
                    framework=framework,
                    pair_id=pair_id,
                    gpu=0,
                    order="AB",
                    generation_s=generation_s,
                ),
            )
    args = argparse.Namespace(
        output_dir=tmp_path,
        tag="test",
        min_samples=2,
        bootstrap_samples=100,
        allow_insufficient=False,
        reference_framework="fastvideo",
    )

    with pytest.raises(RuntimeError, match="order-balanced"):
        framework_ab_wan22.summarize(args)


def test_run_parser_exposes_auditable_feature_cache_selection() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "cache-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            "magcache",
            "--worldfoundry-feature-cache-options-json",
            '{"ratios":[[1.0],[1.0]]}',
        ]
    )
    assert args.worldfoundry_feature_cache == "magcache"
    assert json.loads(args.worldfoundry_feature_cache_options_json)["ratios"] == [
        [1.0],
        [1.0],
    ]
    assert args.fail_on_contamination is True
    assert args.worldfoundry_require_effective is True

    smoke_args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "smoke-contamination-contract",
            "--physical-gpu",
            "0",
            "--no-fail-on-contamination",
        ]
    )
    assert smoke_args.fail_on_contamination is False


@pytest.mark.parametrize(
    "algorithm",
    ("firstblock", "dualblock", "dynamicblock"),
)
def test_run_parser_maps_block_feature_cache_to_generic_validated_option(
    algorithm: str,
) -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            f"{algorithm}-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            algorithm,
            "--worldfoundry-feature-cache-options-json",
            (
                '{"residual_diff_threshold":0.1,"downsample_factor":2,'
                '"dense_first":2,"dense_last":1}'
            ),
        ]
    )

    assert args.worldfoundry_feature_cache == algorithm
    assert framework_ab_wan22._worldfoundry_feature_cache_request(args) == {
        "algorithm": algorithm,
        "residual_diff_threshold": 0.1,
        "downsample_factor": 2,
        "dense_first": 2,
        "dense_last": 1,
    }


@pytest.mark.parametrize(
    ("algorithm", "options", "expected"),
    (
        (
            "blocktaylorseer",
            '{"dense_pattern":[true,false,false,false]}',
            {"dense_pattern": [True, False, False, False]},
        ),
        (
            "custom",
            '{"threshold":0.26}',
            {"threshold": 0.26},
        ),
    ),
)
def test_run_parser_exposes_lightx2v_cache_algorithms(
    algorithm: str,
    options: str,
    expected: dict[str, object],
) -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            f"{algorithm}-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            algorithm,
            "--worldfoundry-feature-cache-options-json",
            options,
        ]
    )
    assert framework_ab_wan22._worldfoundry_feature_cache_request(args) == {
        "algorithm": algorithm,
        **expected,
    }


def test_feature_cache_cli_mapping_rejects_conflicting_or_orphaned_json() -> None:
    parser = framework_ab_wan22.build_parser()
    common = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        "/checkpoint",
        "--output-dir",
        "/output",
        "--run-id",
        "invalid-cache-contract",
        "--physical-gpu",
        "0",
    ]
    conflicting = parser.parse_args(
        [
            *common,
            "--worldfoundry-feature-cache",
            "dynamicblock",
            "--worldfoundry-feature-cache-options-json",
            '{"algorithm":"firstblock","residual_diff_threshold":0.1}',
        ]
    )
    with pytest.raises(ValueError, match="must not override"):
        framework_ab_wan22._worldfoundry_feature_cache_request(conflicting)

    orphaned = parser.parse_args(
        [
            *common,
            "--worldfoundry-feature-cache-options-json",
            '{"residual_diff_threshold":0.1}',
        ]
    )
    with pytest.raises(ValueError, match="requires a selected cache"):
        framework_ab_wan22._worldfoundry_feature_cache_request(orphaned)


def test_run_parser_maps_complete_inline_and_file_vmoba_profiles(
    tmp_path: Path,
) -> None:
    parser = framework_ab_wan22.build_parser()
    profile = _vmoba_profile()
    common = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        "/checkpoint",
        "--output-dir",
        "/output",
        "--run-id",
        "vmoba-contract",
        "--physical-gpu",
        "0",
        "--worldfoundry-approximate-attention",
        "vmoba",
    ]
    inline = parser.parse_args(
        [
            *common,
            "--worldfoundry-approximate-attention-profile-json",
            json.dumps(profile),
        ]
    )
    assert framework_ab_wan22._worldfoundry_approximate_attention_request(
        inline
    ) == {"kind": "vmoba", **profile}

    profile_path = tmp_path / "vmoba.json"
    profile_path.write_text(
        json.dumps({"kind": "vmoba", **profile}),
        encoding="utf-8",
    )
    from_file = parser.parse_args(
        [
            *common,
            "--worldfoundry-approximate-attention-profile",
            str(profile_path),
        ]
    )
    assert framework_ab_wan22._worldfoundry_approximate_attention_request(
        from_file
    ) == {"kind": "vmoba", **profile}


def test_vmoba_profile_boundary_rejects_defaults_conflicts_and_orphans(
    tmp_path: Path,
) -> None:
    parser = framework_ab_wan22.build_parser()
    common = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        "/checkpoint",
        "--output-dir",
        "/output",
        "--run-id",
        "invalid-vmoba-contract",
        "--physical-gpu",
        "0",
    ]
    bare = parser.parse_args(
        [*common, "--worldfoundry-approximate-attention", "vmoba"]
    )
    with pytest.raises(ValueError, match="missing fields"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(bare)

    conflict = parser.parse_args(
        [
            *common,
            "--worldfoundry-approximate-attention",
            "vmoba",
            "--worldfoundry-approximate-attention-profile-json",
            json.dumps({"kind": "vsa", **_vmoba_profile()}),
        ]
    )
    with pytest.raises(ValueError, match="kind conflicts"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(conflict)

    orphaned = parser.parse_args(
        [
            *common,
            "--worldfoundry-approximate-attention-profile-json",
            "{}",
        ]
    )
    with pytest.raises(ValueError, match="requires.*approximate-attention"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(orphaned)

    profile_path = tmp_path / "vmoba.json"
    profile_path.write_text(json.dumps(_vmoba_profile()), encoding="utf-8")
    duplicate = parser.parse_args(
        [
            *common,
            "--worldfoundry-approximate-attention",
            "vmoba",
            "--worldfoundry-approximate-attention-profile-json",
            json.dumps(_vmoba_profile()),
            "--worldfoundry-approximate-attention-profile",
            str(profile_path),
        ]
    )
    with pytest.raises(ValueError, match="only one"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(duplicate)


@pytest.mark.parametrize(
    ("updates", "match"),
    (
        ({"temporal_chunk_size": 2}, "temporal grid 31"),
        ({"spatial_chunk_size": [3, 5]}, "spatial grid"),
        ({"st_chunk_size": [2, 2, 5]}, "patch grid"),
        ({"first_full_step": 50}, "leaves no sparse step"),
        ({"first_full_layer": 30}, "leaves no sparse layer"),
        ({"first_full_layer": 29}, "do not exercise every route"),
    ),
)
def test_vmoba_profile_rejects_incompatible_benchmark_geometry(
    updates: dict[str, object],
    match: str,
) -> None:
    profile = {**_vmoba_profile(), **updates}
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "invalid-vmoba-geometry",
            "--physical-gpu",
            "0",
            "--worldfoundry-approximate-attention",
            "vmoba",
            "--worldfoundry-approximate-attention-profile-json",
            json.dumps(profile),
        ]
    )

    with pytest.raises(ValueError, match=match):
        framework_ab_wan22._worldfoundry_approximate_attention_request(args)


def test_run_parser_exposes_gguf_storage_without_implying_a_kernel() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint/model.gguf",
            "--output-dir",
            "/output",
            "--run-id",
            "gguf-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-quantization",
            "gguf",
        ]
    )

    assert args.worldfoundry_quantization == "gguf"


def test_run_parser_exposes_quantization_config_for_hybrid_ab() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "int8-hybrid-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-quantization",
            "int8",
            "--worldfoundry-quantization-config-json",
            '{"exclude":["time_embedding"],"options":{"min_features":2048}}',
        ]
    )

    assert json.loads(args.worldfoundry_quantization_config_json) == {
        "exclude": ["time_embedding"],
        "options": {"min_features": 2048},
    }


def test_worldfoundry_kernel_request_rejects_invalid_combinations() -> None:
    common = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        "/checkpoint",
        "--output-dir",
        "/output",
        "--run-id",
        "kernel-request-contract",
        "--physical-gpu",
        "0",
    ]
    parser = framework_ab_wan22.build_parser()
    defaults = parser.parse_args(common)
    assert defaults.worldfoundry_fuse_qkv is False
    assert defaults.worldfoundry_qkv_strategy == "auto"
    assert defaults.worldfoundry_qkv_split_threshold == 8192
    assert defaults.worldfoundry_inplace_residual is False
    assert defaults.worldfoundry_static_cross_kv is True
    assert defaults.worldfoundry_fused_rope is False
    assert defaults.worldfoundry_rope_precision == "fp32"
    assert defaults.worldfoundry_rms_norm_precision == "input"

    invalid_threshold = parser.parse_args(
        [*common, "--worldfoundry-qkv-split-threshold", "0"]
    )
    with pytest.raises(ValueError, match="qkv-split-threshold"):
        framework_ab_wan22._validate_worldfoundry_kernel_request(
            invalid_threshold
        )

    fused_fp64 = parser.parse_args(
        [
            *common,
            "--worldfoundry-fused-rope",
            "--worldfoundry-rope-precision",
            "fp64",
            "--worldfoundry-rms-norm-precision",
            "fp32",
        ]
    )
    with pytest.raises(ValueError, match="rope-precision=fp32"):
        framework_ab_wan22._validate_worldfoundry_kernel_request(fused_fp64)

    fused_input_norm = parser.parse_args(
        [*common, "--worldfoundry-fused-rope"]
    )
    with pytest.raises(ValueError, match="rms-norm-precision=fp32"):
        framework_ab_wan22._validate_worldfoundry_kernel_request(
            fused_input_norm
        )

    valid_fused = parser.parse_args(
        [
            *common,
            "--worldfoundry-fused-rope",
            "--worldfoundry-rms-norm-precision",
            "fp32",
        ]
    )
    framework_ab_wan22._validate_worldfoundry_kernel_request(valid_fused)


def test_worldfoundry_effectiveness_gate_rejects_fallback_and_accepts_live_defaults() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "effective-contract",
            "--physical-gpu",
            "0",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            }
        },
    }
    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    report["fallbacks"] = ["attention: unavailable; fell back to SDPA"]
    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(report, args)


def test_worldfoundry_qkv_gate_requires_the_configured_workload_path() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "qkv-workload-path-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-fuse-qkv",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
        },
    }

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True
    assert verification["checks"]["qkv_fusion.eager_path"]["expected"] == (
        "split projection calls > 0"
    )

    qkv = report["runtime"]["qkv_fusion"]
    qkv["eager_packed_projection_calls"] = 120
    qkv["eager_split_projection_calls"] = 0
    with pytest.raises(RuntimeError, match="qkv_fusion.eager_path"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            report,
            args,
        )


def test_worldfoundry_effectiveness_gate_requires_live_inplace_receipt() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "inplace-effective-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-inplace-residual",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "inplace_residual": "in-place-executed",
            "static_cross_kv_blocks": 30,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "inplace_residual": {
                "calls": 100,
                "inplace_calls": 100,
                "functional_calls": 0,
                "autograd_fallback_calls": 0,
                "feature_cache_fallback_calls": 0,
            },
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
        },
    }

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    report["runtime"]["inplace_residual"]["functional_calls"] = 1
    with pytest.raises(RuntimeError, match="inplace_residual.functional_calls"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(report, args)


def _vmoba_args() -> argparse.Namespace:
    return framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "vmoba-effective-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-approximate-attention",
            "vmoba",
            "--worldfoundry-approximate-attention-profile-json",
            json.dumps(_vmoba_profile()),
        ]
    )


def _lightx2v_sparse_args(
    kind: str = "dynamic_sparse",
) -> argparse.Namespace:
    return framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            f"{kind}-effective-contract",
            "--physical-gpu",
            "0",
            "--steps",
            "2",
            "--worldfoundry-approximate-attention",
            kind,
        ]
    )


def test_vmoba_effectiveness_gate_accepts_complete_provider_receipts() -> None:
    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        _live_vmoba_optimization_report(),
        _vmoba_args(),
    )

    assert verification["passed"] is True
    assert verification["issues"] == []


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    (
        ("runtime-kind", "approximate_attention.kind"),
        ("provider", "approximate_attention.provider_path"),
        ("symbols", "vmoba.provider_symbols"),
        ("missing-route", "chunk_calls.spatial"),
        ("events", "vmoba.events"),
        ("event-provider", "vmoba.event_receipts"),
        ("totals", "vmoba.receipt_totals"),
        ("fallback", "kernel_fallbacks"),
    ),
)
def test_vmoba_effectiveness_gate_rejects_false_or_incomplete_receipts(
    mutation: str,
    issue_fragment: str,
) -> None:
    report = json.loads(json.dumps(_live_vmoba_optimization_report()))
    approximate = report["runtime"]["approximate_attention"]
    vmoba = approximate["vmoba"]
    if mutation == "runtime-kind":
        approximate["kind"] = "vsa"
    elif mutation == "provider":
        approximate["provider_path"] = "fastvideo_kernel.fake"
    elif mutation == "symbols":
        vmoba["provider_symbols"] = ["moba_attn_varlen"]
    elif mutation == "missing-route":
        vmoba["chunk_calls"]["spatial"] = 0
    elif mutation == "events":
        vmoba["events"] = []
    elif mutation == "event-provider":
        vmoba["events"][0]["provider_path"] = "fastvideo_kernel.fake"
    elif mutation == "totals":
        approximate["kernel_attempts"] = 4
        approximate["sparse_calls"] = 4
    else:
        approximate["kernel_fallbacks"] = 1

    with pytest.raises(RuntimeError, match=issue_fragment):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            report,
            _vmoba_args(),
        )


@pytest.mark.parametrize(
    "kind",
    framework_ab_wan22.WORLD_FOUNDRY_LIGHTX2V_SPARSE_KINDS,
)
def test_parser_exposes_canonical_lightx2v_sparse_kinds(kind: str) -> None:
    args = _lightx2v_sparse_args(kind)

    request = framework_ab_wan22._worldfoundry_approximate_attention_request(
        args
    )

    assert request is not None
    assert request["kind"] == kind
    from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
        parse_approximate_attention,
    )

    config = parse_approximate_attention(request)
    expectation = framework_ab_wan22._lightx2v_sparse_expectation(config)
    assert expectation["commit"] == (
        "6fb7c1362b89d4908a9ea197bac4fbd7482ee2d5"
    )
    assert expectation["operator"]
    assert expectation["provider_symbol"]


def test_sta_default_grid_fails_fast_but_tuned_grid_is_accepted() -> None:
    default_args = _lightx2v_sparse_args("sta")
    with pytest.raises(ValueError, match="no tuned provider plan"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(
            default_args
        )

    tuned = _lightx2v_sparse_args("sta")
    tuned.frames = 69
    tuned.height = 1536
    tuned.width = 2560
    request = framework_ab_wan22._worldfoundry_approximate_attention_request(
        tuned
    )
    assert request is not None
    assert request["kind"] == "sta"


def test_nbhd_frame_profile_fails_before_cuda_on_grid_mismatch() -> None:
    args = _lightx2v_sparse_args("nbhd")
    args.worldfoundry_approximate_attention_profile_json = json.dumps(
        {"attnmap_frame_num": 30}
    )

    with pytest.raises(ValueError, match="attnmap_frame_num"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(args)


def test_lightx2v_effectiveness_gate_accepts_pinned_complete_receipts() -> None:
    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        _live_lightx2v_optimization_report(),
        _lightx2v_sparse_args(),
    )

    assert verification["passed"] is True
    assert verification["issues"] == []


@pytest.mark.parametrize("mutation", ("provider", "commit", "receipt"))
def test_lightx2v_effectiveness_gate_rejects_false_parity(
    mutation: str,
) -> None:
    report = json.loads(json.dumps(_live_lightx2v_optimization_report()))
    approximate = report["runtime"]["approximate_attention"]
    sparse_event = approximate["events"][0]
    if mutation == "provider":
        approximate["provider_path"] = f"{approximate['provider_path']}.fake"
    elif mutation == "commit":
        sparse_event["provider_source_commit"] = "0" * 40
    else:
        sparse_event.pop("provider_source_fingerprint")

    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            report,
            _lightx2v_sparse_args(),
        )


def test_block_feature_cache_effectiveness_gate_requires_live_per_branch_skips() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "dynamicblock-effective-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            "dynamicblock",
            "--worldfoundry-feature-cache-options-json",
            '{"residual_diff_threshold":0.1}',
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "feature_cache": "residual-reuse",
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "feature_cache": _live_block_feature_cache_runtime(),
        },
    }

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    wrong_algorithm = json.loads(json.dumps(report))
    wrong_algorithm["runtime"]["feature_cache"]["algorithm"] = "firstblock"
    with pytest.raises(RuntimeError, match="feature_cache.algorithm"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            wrong_algorithm,
            args,
        )

    stack_hit_without_block_skip = json.loads(json.dumps(report))
    stack_hit_without_block_skip["runtime"]["feature_cache"][
        "skipped_block_calls"
    ] = 0
    with pytest.raises(RuntimeError, match="skipped_block_calls"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            stack_hit_without_block_skip,
            args,
        )

    branch_without_skip = json.loads(json.dumps(report))
    branch_without_skip["runtime"]["feature_cache"]["branches"]["negative"][
        "skipped_block_calls"
    ] = 0
    with pytest.raises(RuntimeError, match="negative.*skipped_block_calls"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            branch_without_skip,
            args,
        )

    missing_cfg_branch = json.loads(json.dumps(report))
    missing_cfg_branch["runtime"]["feature_cache"]["branches"].pop("negative")
    with pytest.raises(RuntimeError, match="feature_cache.branches"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            missing_cfg_branch,
            args,
        )

    duplicate_block = json.loads(json.dumps(report))
    duplicate_block["runtime"]["feature_cache"]["branches"]["positive"][
        "receipt"
    ]["event_receipts"][0]["dense_blocks"][1] = 0
    with pytest.raises(RuntimeError, match="event_consistency"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            duplicate_block,
            args,
        )

    duplicate_step = json.loads(json.dumps(report))
    duplicate_step["runtime"]["feature_cache"]["branches"]["positive"][
        "receipt"
    ]["event_receipts"][1]["step"] = 0
    with pytest.raises(RuntimeError, match="event_consistency"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            duplicate_step,
            args,
        )

    non_seed_step_zero = json.loads(json.dumps(report))
    non_seed_step_zero["runtime"]["feature_cache"]["branches"]["positive"][
        "receipt"
    ]["event_receipts"][0]["reason"] = "dense"
    with pytest.raises(RuntimeError, match="event_consistency"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            non_seed_step_zero,
            args,
        )


def test_block_taylor_effectiveness_gate_requires_phase_skip_receipts() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "block-taylor-effective-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            "blocktaylorseer",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "feature_cache": "residual-reuse",
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "feature_cache": _live_block_feature_cache_runtime(
                "blocktaylorseer"
            ),
        },
    }

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    no_real_phase_skip = json.loads(json.dumps(report))
    no_real_phase_skip["runtime"]["feature_cache"]["branches"]["positive"][
        "receipt"
    ]["event_receipts"][1]["skipped_blocks"] = []
    with pytest.raises(RuntimeError, match="event_consistency"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            no_real_phase_skip,
            args,
        )


def test_custom_cache_gate_requires_current_request_tea_taylor_receipts() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "custom-effective-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            "custom",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "feature_cache": "residual-reuse",
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "feature_cache": _live_custom_feature_cache_runtime(),
        },
    }
    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    stale = json.loads(json.dumps(report))
    stale["runtime"]["feature_cache"]["branches"]["negative"]["receipt"][
        "request_id"
    ] = "previous-request"
    with pytest.raises(RuntimeError, match="negative.*receipt"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(stale, args)


def test_block_feature_cache_gate_rejects_a_stale_previous_request_receipt() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "stale-dynamicblock-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-feature-cache",
            "dynamicblock",
            "--worldfoundry-feature-cache-options-json",
            '{"residual_diff_threshold":0.1}',
        ]
    )
    cache = _live_block_feature_cache_runtime(request_epoch=9)
    cache["branches"]["negative"]["receipt"]["request_epoch"] = 8
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "effective": {
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "feature_cache": cache,
        },
    }

    with pytest.raises(RuntimeError, match="receipt.request_epoch"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(report, args)


@pytest.mark.parametrize(
    ("counter_name", "counter_value"),
    (
        ("missing", None),
        ("successes", 0),
        ("attempts", 121),
        ("fallbacks", 1),
        ("errors", 1),
        ("quarantined_skips", 1),
    ),
)
def test_worldfoundry_effectiveness_gate_rejects_unproven_attention_provider_calls(
    counter_name: str,
    counter_value: int | None,
) -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "provider-counter-contract",
            "--physical-gpu",
            "0",
        ]
    )
    provider_counters = {
        "attempts": 120,
        "successes": 120,
        "fallbacks": 0,
        "errors": 0,
        "quarantined_skips": 0,
    }
    provider_calls = {"flash_attention_3": provider_counters}
    if counter_name == "missing":
        provider_calls.clear()
    else:
        provider_counters[counter_name] = counter_value
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "effective": {
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
        },
        "runtime": {
            "attention_dispatch": {
                "provider_calls": provider_calls,
            }
        },
    }

    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(report, args)


def test_worldfoundry_compile_gate_requires_provider_graph_receipt_and_executed_wrapper() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "compiled-provider-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-compile",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "compile": "compile-wrapper-executed",
        },
        "runtime": {
            **_live_qkv_static_runtime(compiled=True),
            "compile": {
                "wrapper_installed": True,
                "calls": 50,
                "failures": 0,
                "last_error": None,
                "attention_provider_graph_traces": {"flash_attention_3": 2},
            },
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 0,
                        "successes": 0,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                        "compiled_graph_traces": 0,
                    }
                }
            },
        },
    }

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True
    assert "attention_dispatch.provider.successes" not in verification["checks"]

    rejected_paths = (
        ("scoped", "flash_attention_3", 0),
        ("compile", "calls", 0),
        ("compile", "wrapper_installed", False),
        ("compile", "failures", 1),
        ("compile", "last_error", "BackendCompilerFailed"),
        ("provider", "attempts", 1),
        ("provider", "fallbacks", 1),
    )
    for section, field, rejected_value in rejected_paths:
        rejected_report = json.loads(json.dumps(report))
        if section == "provider":
            rejected_report["runtime"]["attention_dispatch"]["provider_calls"][
                "flash_attention_3"
            ][field] = rejected_value
        elif section == "scoped":
            rejected_report["runtime"]["compile"][
                "attention_provider_graph_traces"
            ][field] = rejected_value
        else:
            rejected_report["runtime"]["compile"][field] = rejected_value
        with pytest.raises(RuntimeError, match="effectiveness gate failed"):
            framework_ab_wan22._verify_worldfoundry_optimization_report(
                rejected_report,
                args,
            )

    rejected_report = json.loads(json.dumps(report))
    rejected_report["effective"]["compile"] = "compile-wrapper-installed (lazy)"
    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            rejected_report,
            args,
        )


def test_worldfoundry_cuda_graph_gate_requires_capture_and_replay() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "cuda-graph-runtime-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-cuda-graph",
            "--no-worldfoundry-static-cross-kv",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "cuda_graph": "captured-and-replayed",
        },
        "runtime": {
            "qkv_fusion": _live_qkv_static_runtime()["qkv_fusion"],
            "wan_timestep": _live_qkv_static_runtime()["wan_timestep"],
            "cuda_graph": {
                "capture": 1,
                "replay": 49,
                "graphs": 1,
                "eager": 0,
                "capture_failed": 0,
            },
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
        },
    }

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    reused_report = json.loads(json.dumps(report))
    reused_report["runtime"]["cuda_graph"]["capture"] = 0
    reused_report["effective"]["cuda_graph"] = "replayed-existing-graph"
    assert framework_ab_wan22._verify_worldfoundry_optimization_report(
        reused_report,
        args,
    )["passed"] is True

    for section, field, rejected_value in (
        ("runtime", "replay", 0),
        ("runtime", "eager", 1),
        ("runtime", "capture_failed", 1),
        ("effective", "cuda_graph", "eager"),
    ):
        rejected_report = json.loads(json.dumps(report))
        if section == "runtime":
            rejected_report["runtime"]["cuda_graph"][field] = rejected_value
        else:
            rejected_report["effective"][field] = rejected_value
        with pytest.raises(RuntimeError, match="effectiveness gate failed"):
            framework_ab_wan22._verify_worldfoundry_optimization_report(
                rejected_report,
                args,
            )

    rejected_report = json.loads(json.dumps(report))
    rejected_report["runtime"]["cuda_graph"]["capture"] = 0
    rejected_report["runtime"]["cuda_graph"]["graphs"] = 0
    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            rejected_report,
            args,
        )


def test_worldfoundry_fused_rope_gate_requires_real_provider_calls() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "fused-rope-runtime-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-fused-rope",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(fused_rope=True),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "fused_rope": "hidden_qk_rmsnorm_rope_3d:fp32",
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "fused_rope": {
                "installed_blocks": 30,
                "effective": "accelerated-provider-executed",
                "eager_calls": 1500,
                "compiled_graph_traces": 0,
                "provider_calls": 1500,
                "torch_fallback_calls": 0,
                "provider_failures": 0,
                "quarantined_skips": 0,
                "malformed_receipts": 0,
                "provider_paths": ["triton_hidden_qk_rmsnorm_rope_3d"],
            },
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 1500,
                        "successes": 1500,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
        },
    }

    assert framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )["passed"] is True

    rejected_values = {
        "provider_calls": 0,
        "torch_fallback_calls": 1,
        "provider_failures": 1,
        "quarantined_skips": 1,
        "malformed_receipts": 1,
        "provider_paths": ["torch"],
    }
    for field, value in rejected_values.items():
        rejected = json.loads(json.dumps(report))
        rejected["runtime"]["fused_rope"][field] = value
        with pytest.raises(RuntimeError, match="effectiveness gate failed"):
            framework_ab_wan22._verify_worldfoundry_optimization_report(
                rejected,
                args,
            )


def test_worldfoundry_quantization_gate_requires_only_low_precision_kernel_calls() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "quant-runtime-contract",
            "--physical-gpu",
            "0",
            "--worldfoundry-quantization",
            "int8",
        ]
    )
    report = {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "quantization": {
                "low_precision_kernel_calls": 90,
                "packed_weight_calls": 0,
                "dense_policy_calls": 0,
                "dense_fallback_calls": 0,
                "fallback_reasons": [],
            },
        },
    }
    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        report,
        args,
    )
    assert verification["passed"] is True

    rejected_values = {
        "low_precision_kernel_calls": 0,
        "packed_weight_calls": 1,
        "dense_policy_calls": 1,
        "dense_fallback_calls": 1,
        "fallback_reasons": ["dense retry"],
    }
    for field, rejected_value in rejected_values.items():
        rejected_report = json.loads(json.dumps(report))
        rejected_report["runtime"]["quantization"][field] = rejected_value
        with pytest.raises(RuntimeError, match="effectiveness gate failed"):
            framework_ab_wan22._verify_worldfoundry_optimization_report(
                rejected_report,
                args,
            )


def test_worldfoundry_runner_resets_attention_counters_before_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import torch

    from worldfoundry.core import attention as attention_module
    from worldfoundry.pipelines.wan import pipeline_wan_2p2

    events: list[str] = []

    class FakeModel:
        _worldfoundry_dit_weight_dtype = torch.bfloat16
        _worldfoundry_applied_optimizations = SimpleNamespace(
            requested={
                **_live_wan_math_requested(),
                "attention": "flash_attention",
            },
            effective={
                **_live_wan_math_effective(),
                "attention": "flash_attention_3",
                "fuse_qkv_blocks": 30,
                "static_cross_kv_blocks": 30,
            },
            fallbacks=[],
            quality_tier="exact",
        )

        def parameters(self):
            return iter(())

    runtime_report = {
        "requested": {
            **_live_wan_math_requested(),
            "attention": "flash_attention",
        },
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
        },
        "fallbacks": [],
        "quality_tier": "exact",
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            }
        },
    }

    class FakePipeline:
        def __init__(self) -> None:
            model = FakeModel()
            denoiser = SimpleNamespace(
                model=model,
                runtime_optimization_report=lambda: runtime_report,
            )
            decoder = SimpleNamespace(
                vae=SimpleNamespace(decode_autocast_dtype=torch.bfloat16)
            )
            self.native_pipeline = SimpleNamespace(
                runner=SimpleNamespace(
                    components=SimpleNamespace(
                        denoiser=denoiser,
                        decoder=decoder,
                    )
                )
            )

        @classmethod
        def from_pretrained(cls, **_kwargs):
            events.append("loaded")
            return cls()

        def __call__(self, **kwargs):
            events.append("generated")
            Path(kwargs["output_path"]).write_bytes(b"video")
            return {"artifact_path": str(kwargs["output_path"])}

    class FakeSampler:
        def __init__(self, physical_gpu: int) -> None:
            self.physical_gpu = physical_gpu

        def start(self) -> None:
            events.append("sampler-started")

        def stop(self) -> None:
            events.append("sampler-stopped")

        def summary(self):
            return {"sample_count": 1}

    monkeypatch.setattr(pipeline_wan_2p2, "Wan2p2Pipeline", FakePipeline)
    monkeypatch.setattr(
        attention_module,
        "reset_attention_provider_runtime",
        lambda: events.append("attention-counters-reset"),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024**2)
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_query",
        lambda physical_gpu: {"index": physical_gpu},
    )
    monkeypatch.setattr(framework_ab_wan22, "GpuSampler", FakeSampler)
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "reset-runtime-contract",
            "--physical-gpu",
            "0",
        ]
    )
    args._compute_app_monitor = object()
    result = framework_ab_wan22._run_worldfoundry(args, tmp_path / "output.mp4")

    assert events.index("loaded") < events.index("attention-counters-reset")
    assert events.index("attention-counters-reset") < events.index("generated")
    assert result["optimization_verification"]["passed"] is True


def test_fastvideo_reference_probe_proves_one_shot_role_calls_and_rejects_fallbacks(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    class FakeImplementation:
        pass

    class FakeAttention(torch.nn.Module):
        def __init__(self, backend: str) -> None:
            super().__init__()
            self.backend = SimpleNamespace(name=backend)
            self.attn_impl = FakeImplementation()

        def forward(self, value):
            return value

    def make_worker(backend: str):
        blocks = []
        for _ in range(2):
            blocks.append(
                SimpleNamespace(
                    attn1=FakeAttention(backend),
                    attn2=SimpleNamespace(attn=FakeAttention(backend)),
                )
            )
        worker = SimpleNamespace(
            pipeline=SimpleNamespace(
                modules={"transformer": SimpleNamespace(blocks=blocks)}
            )
        )
        return worker, blocks

    worker, blocks = make_worker("FLASH_ATTN")
    installed = framework_ab_wan22._install_fastvideo_reference_probe(worker)
    preflight = framework_ab_wan22._verify_fastvideo_reference_probe(
        [installed],
        require_successful_calls=False,
    )
    assert preflight["passed"] is True
    blocks[0].attn1(torch.ones(1))
    blocks[0].attn1(torch.ones(1))
    blocks[0].attn2.attn(torch.ones(1))
    blocks[0].attn2.attn(torch.ones(1))
    collected = framework_ab_wan22._collect_fastvideo_reference_probe(worker)
    verification = framework_ab_wan22._verify_fastvideo_reference_probe(
        [collected],
        require_successful_calls=True,
    )
    assert verification["passed"] is True
    assert collected["roles"]["self"]["successful_calls"] == 1
    assert collected["roles"]["cross1"]["successful_calls"] == 1
    assert collected["roles"]["self"]["hook_removed_after_success"] is True
    assert collected["roles"]["cross1"]["hook_removed_after_success"] is True

    fallback_worker, _ = make_worker("TORCH_SDPA")
    fallback_probe = framework_ab_wan22._install_fastvideo_reference_probe(
        fallback_worker
    )
    with pytest.raises(RuntimeError, match="FastVideo runtime effectiveness gate failed"):
        framework_ab_wan22._verify_fastvideo_reference_probe(
            [fallback_probe],
            require_successful_calls=False,
        )

    uncalled_worker, uncalled_blocks = make_worker("FLASH_ATTN")
    framework_ab_wan22._install_fastvideo_reference_probe(uncalled_worker)
    uncalled_blocks[0].attn1(torch.ones(1))
    uncalled_probe = framework_ab_wan22._collect_fastvideo_reference_probe(
        uncalled_worker
    )
    with pytest.raises(RuntimeError, match="cross1.successful_calls"):
        framework_ab_wan22._verify_fastvideo_reference_probe(
            [uncalled_probe],
            require_successful_calls=True,
        )


def _fake_fastvideo_flash_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    execute_kernels: bool,
) -> tuple[SimpleNamespace, list[SimpleNamespace], dict[str, object]]:
    """Build a worker whose FlashAttention globals are observable call sites."""

    import torch

    default_module_name = "fastvideo.attention.utils.flash_attn_default"
    no_pad_module_name = "fastvideo.attention.utils.flash_attn_no_pad"
    default_module = ModuleType(default_module_name)
    no_pad_module = ModuleType(no_pad_module_name)

    def default_fa3(value, *_args, **_kwargs):
        return value

    def masked_self_fa2(value, *_args, **_kwargs):
        return value

    def varlen_fa3(value, *_args, **_kwargs):
        return value

    default_module.fa_version = "3"
    default_module._fa_default = default_fa3
    no_pad_module._FA_VARLEN_VERSION = "3"
    no_pad_module.flash_attn_varlen_qkvpacked_func = masked_self_fa2
    no_pad_module.flash_attn_varlen_func_impl = varlen_fa3
    monkeypatch.setitem(sys.modules, default_module_name, default_module)
    monkeypatch.setitem(sys.modules, no_pad_module_name, no_pad_module)

    class FakeFlashAttentionImpl:
        pass

    class FakeAttention(torch.nn.Module):
        def __init__(self, module: ModuleType, attribute: str) -> None:
            super().__init__()
            self.backend = SimpleNamespace(name="FLASH_ATTN")
            self.attn_impl = FakeFlashAttentionImpl()
            self.kernel_module = module
            self.kernel_attribute = attribute

        def forward(self, value):
            if execute_kernels:
                return getattr(self.kernel_module, self.kernel_attribute)(value)
            return value

    blocks = [
        SimpleNamespace(
            attn1=FakeAttention(default_module, "_fa_default"),
            attn2=SimpleNamespace(
                attn=FakeAttention(no_pad_module, "flash_attn_varlen_func_impl")
            ),
        )
    ]
    worker = SimpleNamespace(
        pipeline=SimpleNamespace(
            modules={"transformer": SimpleNamespace(blocks=blocks)}
        )
    )
    originals = {
        "default": default_fa3,
        "masked_self": masked_self_fa2,
        "varlen": varlen_fa3,
        "default_module": default_module,
        "no_pad_module": no_pad_module,
    }
    return worker, blocks, originals


def test_fastvideo_flash_probe_proves_real_self_and_cross_kernel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    worker, blocks, originals = _fake_fastvideo_flash_runtime(
        monkeypatch,
        execute_kernels=True,
    )

    installed = framework_ab_wan22._install_fastvideo_reference_probe(worker)
    preflight = framework_ab_wan22._verify_fastvideo_reference_probe(
        [installed],
        require_successful_calls=False,
    )
    assert preflight["passed"] is True
    assert installed["flash_attention_versions"] == {
        "default": "3",
        "varlen": "3",
    }
    assert {
        candidate["entrypoint"]
        for candidate in installed["flash_kernel_candidates"]
    } == {"default", "masked_self_qkvpacked", "varlen_qkv"}

    blocks[0].attn1(torch.ones(1))
    blocks[0].attn2.attn(torch.ones(1))
    collected = framework_ab_wan22._collect_fastvideo_reference_probe(worker)
    verification = framework_ab_wan22._verify_fastvideo_reference_probe(
        [collected],
        require_successful_calls=True,
    )

    assert verification["passed"] is True
    assert collected["roles"]["self"]["kernel_calls"] == 1
    assert collected["roles"]["self"]["kernel_entrypoints"] == ["default"]
    assert collected["roles"]["cross1"]["kernel_calls"] == 1
    assert collected["roles"]["cross1"]["kernel_entrypoints"] == [
        "varlen_qkv"
    ]
    assert collected["kernel_wrappers_restored"] is True
    assert originals["default_module"]._fa_default is originals["default"]
    assert (
        originals["no_pad_module"].flash_attn_varlen_qkvpacked_func
        is originals["masked_self"]
    )
    assert (
        originals["no_pad_module"].flash_attn_varlen_func_impl
        is originals["varlen"]
    )


def test_fastvideo_flash_probe_rejects_outer_forward_without_kernel_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    worker, blocks, originals = _fake_fastvideo_flash_runtime(
        monkeypatch,
        execute_kernels=False,
    )
    framework_ab_wan22._install_fastvideo_reference_probe(worker)

    blocks[0].attn1(torch.ones(1))
    blocks[0].attn2.attn(torch.ones(1))
    collected = framework_ab_wan22._collect_fastvideo_reference_probe(worker)

    assert collected["roles"]["self"]["successful_calls"] == 1
    assert collected["roles"]["cross1"]["successful_calls"] == 1
    assert collected["roles"]["self"]["kernel_calls"] == 0
    assert collected["roles"]["cross1"]["kernel_calls"] == 0
    assert collected["kernel_wrappers_restored"] is True
    assert originals["default_module"]._fa_default is originals["default"]
    assert (
        originals["no_pad_module"].flash_attn_varlen_func_impl
        is originals["varlen"]
    )
    with pytest.raises(RuntimeError, match="concrete_kernel_calls"):
        framework_ab_wan22._verify_fastvideo_reference_probe(
            [collected],
            require_successful_calls=True,
        )


def test_lightx2v_reference_probe_requires_kernel_symbols_and_both_t2v_roles(
    monkeypatch,
) -> None:
    kernel_module_name = "lightx2v.common.ops.attn.flash_attn"
    kernel_module = ModuleType(kernel_module_name)
    kernel_module.flash_attn_func_v2 = lambda *_args, **_kwargs: None
    kernel_module.flash_attn_varlen_func_v2 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, kernel_module_name, kernel_module)

    class FlashAttn2Weight:
        def apply(self, value):
            return value

    FlashAttn2Weight.__module__ = kernel_module_name
    FlashAttn2Weight.__qualname__ = "FlashAttn2Weight"

    def make_pipeline():
        self_attention = FlashAttn2Weight()
        cross_attention = FlashAttn2Weight()
        pipeline = SimpleNamespace(
            runner=SimpleNamespace(
                model=SimpleNamespace(
                    transformer_weights=SimpleNamespace(
                        blocks=[
                            SimpleNamespace(
                                compute_phases=[
                                    SimpleNamespace(self_attn_1=self_attention),
                                    SimpleNamespace(cross_attn_1=cross_attention),
                                ]
                            )
                        ]
                    )
                )
            )
        )
        return pipeline, self_attention, cross_attention

    pipeline, self_attention, cross_attention = make_pipeline()
    installed = framework_ab_wan22._install_lightx2v_reference_probe(
        pipeline,
        "flash_attn2",
    )
    framework_ab_wan22._verify_lightx2v_reference_probe(
        installed,
        require_successful_calls=False,
    )
    assert self_attention.apply("self") == "self"
    assert self_attention.apply("self-again") == "self-again"
    assert cross_attention.apply("cross1") == "cross1"
    assert cross_attention.apply("cross1-again") == "cross1-again"
    collected = framework_ab_wan22._collect_lightx2v_reference_probe(pipeline)
    verification = framework_ab_wan22._verify_lightx2v_reference_probe(
        collected,
        require_successful_calls=True,
    )
    assert verification["passed"] is True
    assert collected["roles"]["self"]["successful_calls"] == 1
    assert collected["roles"]["cross1"]["successful_calls"] == 1
    assert "apply" not in vars(self_attention)
    assert "apply" not in vars(cross_attention)

    uncalled_pipeline, uncalled_self, _ = make_pipeline()
    framework_ab_wan22._install_lightx2v_reference_probe(
        uncalled_pipeline,
        "flash_attn2",
    )
    uncalled_self.apply("self")
    uncalled_probe = framework_ab_wan22._collect_lightx2v_reference_probe(
        uncalled_pipeline
    )
    with pytest.raises(RuntimeError, match="cross1.successful_calls"):
        framework_ab_wan22._verify_lightx2v_reference_probe(
            uncalled_probe,
            require_successful_calls=True,
        )

    kernel_module.flash_attn_func_v2 = None
    missing_kernel_pipeline, _, _ = make_pipeline()
    missing_kernel_probe = framework_ab_wan22._install_lightx2v_reference_probe(
        missing_kernel_pipeline,
        "flash_attn2",
    )
    with pytest.raises(RuntimeError, match="kernel symbols are unavailable"):
        framework_ab_wan22._verify_lightx2v_reference_probe(
            missing_kernel_probe,
            require_successful_calls=False,
        )


def test_lightx2v_role_probe_proves_dynamic_sparse_self_and_dense_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    sparse_module_name = "lightx2v.common.ops.attn.dynamic_sparse_attn"
    sparse_module = ModuleType(sparse_module_name)

    def get_block_map(*_args, **_kwargs):
        return (
            torch.tensor([[True, False], [True, True]]),
            torch.tensor([0]),
            1,
        )

    class _Attention:
        @staticmethod
        def apply(value=None, *_args, **_kwargs):
            return value

    sparse_module.get_block_map = get_block_map
    sparse_module._attention = _Attention
    monkeypatch.setitem(sys.modules, sparse_module_name, sparse_module)

    class DynamicSparseAttnWeight:
        def __init__(self, *, call_final_kernel: bool = True) -> None:
            self.sparsity_ratio = 0.8
            self.operator = "triton"
            self.per_block_mean = False
            self.topk = 1.0 - self.sparsity_ratio
            self.BLKQ = 64
            self.BLKK = 64
            self.arch = "sm90"
            self.call_final_kernel = call_final_kernel
            self.apply_func = self.apply_triton

        def apply_triton(self, value=None):
            sparse_module.get_block_map(value, value)
            if self.call_final_kernel:
                return sparse_module._attention.apply(value)
            return value

        def apply(self, value=None):
            return self.apply_func(value)

    DynamicSparseAttnWeight.__module__ = sparse_module_name
    DynamicSparseAttnWeight.__qualname__ = "DynamicSparseAttnWeight"
    DynamicSparseAttnWeight.apply_triton.__module__ = sparse_module_name
    DynamicSparseAttnWeight.apply_triton.__qualname__ = (
        "DynamicSparseAttnWeight.apply_triton"
    )

    dense_module_name = "lightx2v.common.ops.attn.flash_attn"
    dense_module = ModuleType(dense_module_name)
    dense_module.flash_attn_func_v3 = lambda *_args, **_kwargs: None
    dense_module.flash_attn_varlen_func_v3 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, dense_module_name, dense_module)

    class FlashAttn3Weight:
        def apply(self, value=None):
            return value

    FlashAttn3Weight.__module__ = dense_module_name
    FlashAttn3Weight.__qualname__ = "FlashAttn3Weight"

    self_attention = DynamicSparseAttnWeight()
    cross_attention = FlashAttn3Weight()
    pipeline = SimpleNamespace(
        runner=SimpleNamespace(
            model=SimpleNamespace(
                transformer_weights=SimpleNamespace(
                    blocks=[
                        SimpleNamespace(
                            compute_phases=[
                                SimpleNamespace(self_attn_1=self_attention),
                                SimpleNamespace(cross_attn_1=cross_attention),
                            ]
                        )
                    ]
                )
            )
        )
    )
    request = {
        "self_mode": "dynamic_sparse_attn",
        "cross_mode": "flash_attn3",
        "self_family": "sparse",
        "self_settings": {
            "sparsity_ratio": 0.8,
            "operator": "triton",
        },
    }

    installed = framework_ab_wan22._install_lightx2v_reference_probe(
        pipeline,
        request,
    )
    framework_ab_wan22._verify_lightx2v_reference_probe(
        installed,
        require_successful_calls=False,
    )
    assert self_attention.apply("self") == "self"
    assert cross_attention.apply("cross") == "cross"
    collected = framework_ab_wan22._collect_lightx2v_reference_probe(pipeline)
    verification = framework_ab_wan22._verify_lightx2v_reference_probe(
        collected,
        require_successful_calls=True,
    )

    assert verification["passed"] is True
    assert collected["requested_roles"] == {
        "self": "dynamic_sparse_attn",
        "cross1": "flash_attn3",
    }
    sparse_role = collected["roles"]["self"]
    assert sparse_role["sparse_dispatch_successful_calls"] == 1
    assert sparse_role["sparse_final_kernel_successful_calls"] == 1
    assert sparse_role["sparse_mask_successful_calls"] == 1
    assert sparse_role["sparse_mask"]["strictly_sparse"] is True
    assert sparse_role["dense_provider_successful_calls"] == 0
    assert sparse_role["sparse_provider"]["kernel_available"] is True
    assert sparse_role["sparse_provider"]["kernel_attribute"] == (
        "lightx2v.common.ops.attn.dynamic_sparse_attn._attention.apply"
    )
    assert collected["roles"]["cross1"]["attention_family"] == "dense"
    assert collected["roles"]["cross1"]["successful_calls"] == 1

    no_kernel_self = DynamicSparseAttnWeight(call_final_kernel=False)
    no_kernel_cross = FlashAttn3Weight()
    no_kernel_pipeline = SimpleNamespace(
        runner=SimpleNamespace(
            model=SimpleNamespace(
                transformer_weights=SimpleNamespace(
                    blocks=[
                        SimpleNamespace(
                            compute_phases=[
                                SimpleNamespace(self_attn_1=no_kernel_self),
                                SimpleNamespace(cross_attn_1=no_kernel_cross),
                            ]
                        )
                    ]
                )
            )
        )
    )
    framework_ab_wan22._install_lightx2v_reference_probe(
        no_kernel_pipeline,
        request,
    )
    no_kernel_self.apply("self")
    no_kernel_cross.apply("cross")
    no_kernel_collected = framework_ab_wan22._collect_lightx2v_reference_probe(
        no_kernel_pipeline
    )
    with pytest.raises(RuntimeError, match="sparse_final_kernel_successful_calls"):
        framework_ab_wan22._verify_lightx2v_reference_probe(
            no_kernel_collected,
            require_successful_calls=True,
        )


@pytest.mark.parametrize(
    ("arch", "sage2pp_enabled", "expected_attribute"),
    [
        (
            "sm80",
            False,
            "qattn."
            "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_"
            "with_pv_threshold",
        ),
        (
            "sm90",
            False,
            "qattn."
            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_"
            "fuse_v_scale_sm90",
        ),
        (
            "sm89",
            True,
            "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_"
            "fuse_v_scale_with_pv_threshold",
        ),
        (
            "sm120",
            False,
            "qattn."
            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_"
            "fuse_v_scale_with_pv_threshold",
        ),
    ],
)
def test_lightx2v_dynamic_sage2_resolves_arch_specific_final_kernel(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    sage2pp_enabled: bool,
    expected_attribute: str,
) -> None:
    sparse_module_name = "lightx2v.common.ops.attn.dynamic_sparse_attn"
    sparge_module_name = "lightx2v.common.ops.attn.utils.sparge_util"
    sparse_module = ModuleType(sparse_module_name)
    sparge_module = ModuleType(sparge_module_name)

    def sage2_block_sparse_attn(value=None, *_args, **_kwargs):
        return value

    sage2_block_sparse_attn.__module__ = sparge_module_name
    sparse_module.sage2_block_sparse_attn = sage2_block_sparse_attn
    sparge_module.sage2_block_sparse_attn = sage2_block_sparse_attn
    sparge_module.SAGE2PP_ENABLED = sage2pp_enabled
    sparge_module.qattn = SimpleNamespace()

    def kernel(value=None, *_args, **_kwargs):
        return value

    kernel_owner = sparge_module
    kernel_attribute = expected_attribute
    if expected_attribute.startswith("qattn."):
        kernel_owner = sparge_module.qattn
        kernel_attribute = expected_attribute.removeprefix("qattn.")
    setattr(kernel_owner, kernel_attribute, kernel)
    monkeypatch.setitem(sys.modules, sparse_module_name, sparse_module)
    monkeypatch.setitem(sys.modules, sparge_module_name, sparge_module)

    class DynamicSparseAttnWeight:
        pass

    DynamicSparseAttnWeight.__module__ = sparse_module_name
    instance = DynamicSparseAttnWeight()
    instance.arch = arch

    owner, attribute, path, resolved = (
        framework_ab_wan22._lightx2v_dynamic_sparse_kernel_target(
            instance,
            "sage2",
        )
    )

    assert owner is kernel_owner
    assert attribute == kernel_attribute
    assert path == f"{sparge_module_name}.{expected_attribute}"
    assert resolved is kernel


def test_lightx2v_role_probe_proves_sage2_final_cuda_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    sparse_module_name = "lightx2v.common.ops.attn.dynamic_sparse_attn"
    sparge_module_name = "lightx2v.common.ops.attn.utils.sparge_util"
    sparse_module = ModuleType(sparse_module_name)
    sparge_module = ModuleType(sparge_module_name)
    final_kernel_name = (
        "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90"
    )

    def final_kernel(value=None, *_args, **_kwargs):
        return value

    qattn = SimpleNamespace(**{final_kernel_name: final_kernel})
    sparge_module.qattn = qattn
    sparge_module.SAGE2PP_ENABLED = False

    def sage2_block_sparse_attn(value=None, *_args, **_kwargs):
        return getattr(sparge_module.qattn, final_kernel_name)(value)

    sage2_block_sparse_attn.__module__ = sparge_module_name
    sparge_module.sage2_block_sparse_attn = sage2_block_sparse_attn
    sparse_module.sage2_block_sparse_attn = sage2_block_sparse_attn
    sparse_module.get_block_map = lambda *_args, **_kwargs: (
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([0]),
        1,
    )
    monkeypatch.setitem(sys.modules, sparse_module_name, sparse_module)
    monkeypatch.setitem(sys.modules, sparge_module_name, sparge_module)

    class DynamicSparseAttnWeight:
        def __init__(self) -> None:
            self.sparsity_ratio = 0.9
            self.operator = "sage2"
            self.per_block_mean = False
            self.topk = 1.0 - self.sparsity_ratio
            self.BLKQ = 64
            self.BLKK = 128
            self.arch = "sm90"
            self.apply_func = self.apply_sage2

        def apply_sage2(self, value=None):
            sparse_module.get_block_map(value, value)
            return sparse_module.sage2_block_sparse_attn(value)

        def apply(self, value=None):
            return self.apply_func(value)

    DynamicSparseAttnWeight.__module__ = sparse_module_name
    DynamicSparseAttnWeight.__qualname__ = "DynamicSparseAttnWeight"
    DynamicSparseAttnWeight.apply_sage2.__module__ = sparse_module_name
    DynamicSparseAttnWeight.apply_sage2.__qualname__ = (
        "DynamicSparseAttnWeight.apply_sage2"
    )

    dense_module_name = "lightx2v.common.ops.attn.flash_attn"
    dense_module = ModuleType(dense_module_name)
    dense_module.flash_attn_func_v3 = lambda *_args, **_kwargs: None
    dense_module.flash_attn_varlen_func_v3 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, dense_module_name, dense_module)

    class FlashAttn3Weight:
        def apply(self, value=None):
            return value

    FlashAttn3Weight.__module__ = dense_module_name
    FlashAttn3Weight.__qualname__ = "FlashAttn3Weight"

    self_attention = DynamicSparseAttnWeight()
    cross_attention = FlashAttn3Weight()
    pipeline = SimpleNamespace(
        runner=SimpleNamespace(
            model=SimpleNamespace(
                transformer_weights=SimpleNamespace(
                    blocks=[
                        SimpleNamespace(
                            compute_phases=[
                                SimpleNamespace(self_attn_1=self_attention),
                                SimpleNamespace(cross_attn_1=cross_attention),
                            ]
                        )
                    ]
                )
            )
        )
    )
    request = {
        "self_mode": "dynamic_sparse_attn",
        "cross_mode": "flash_attn3",
        "self_family": "sparse",
        "self_settings": {
            "sparsity_ratio": 0.9,
            "operator": "sage2",
        },
    }

    installed = framework_ab_wan22._install_lightx2v_reference_probe(
        pipeline,
        request,
    )
    framework_ab_wan22._verify_lightx2v_reference_probe(
        installed,
        require_successful_calls=False,
    )
    self_attention.apply("self")
    cross_attention.apply("cross")
    collected = framework_ab_wan22._collect_lightx2v_reference_probe(pipeline)
    verification = framework_ab_wan22._verify_lightx2v_reference_probe(
        collected,
        require_successful_calls=True,
    )

    sparse_role = collected["roles"]["self"]
    provider = sparse_role["sparse_provider"]
    assert verification["passed"] is True
    assert provider["dispatcher_available"] is True
    assert provider["dispatcher_attribute"] == (
        f"{sparse_module_name}.sage2_block_sparse_attn"
    )
    assert provider["kernel_attribute"] == (
        f"{sparge_module_name}.qattn.{final_kernel_name}"
    )
    assert sparse_role["sparse_dispatch_successful_calls"] == 1
    assert sparse_role["sparse_final_kernel_successful_calls"] == 1
    assert sparse_role["sparse_mask"]["strictly_sparse"] is True
    assert collected["roles"]["cross1"]["successful_calls"] == 1


def test_lightx2v_role_probe_rejects_dense_mask_as_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    sparse_module_name = "lightx2v.common.ops.attn.dynamic_sparse_attn"
    sparse_module = ModuleType(sparse_module_name)
    sparse_module.get_block_map = lambda *_args, **_kwargs: (
        torch.ones((2, 2), dtype=torch.bool),
        torch.tensor([0]),
        2,
    )

    class _Attention:
        @staticmethod
        def apply(value=None, *_args, **_kwargs):
            return value

    sparse_module._attention = _Attention
    monkeypatch.setitem(sys.modules, sparse_module_name, sparse_module)

    class DynamicSparseAttnWeight:
        def __init__(self) -> None:
            self.sparsity_ratio = 0.8
            self.operator = "triton"
            self.per_block_mean = False
            self.topk = 1.0 - self.sparsity_ratio
            self.BLKQ = 64
            self.BLKK = 64
            self.arch = "sm90"
            self.apply_func = self.apply_triton

        def apply_triton(self, value=None):
            sparse_module.get_block_map(value, value)
            return sparse_module._attention.apply(value)

        def apply(self, value=None):
            return self.apply_func(value)

    DynamicSparseAttnWeight.__module__ = sparse_module_name
    DynamicSparseAttnWeight.__qualname__ = "DynamicSparseAttnWeight"
    DynamicSparseAttnWeight.apply_triton.__module__ = sparse_module_name
    DynamicSparseAttnWeight.apply_triton.__qualname__ = (
        "DynamicSparseAttnWeight.apply_triton"
    )

    dense_module_name = "lightx2v.common.ops.attn.flash_attn"
    dense_module = ModuleType(dense_module_name)
    dense_module.flash_attn_func_v3 = lambda *_args, **_kwargs: None
    dense_module.flash_attn_varlen_func_v3 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, dense_module_name, dense_module)

    class FlashAttn3Weight:
        def apply(self, value=None):
            return value

    FlashAttn3Weight.__module__ = dense_module_name
    FlashAttn3Weight.__qualname__ = "FlashAttn3Weight"
    self_attention = DynamicSparseAttnWeight()
    cross_attention = FlashAttn3Weight()
    pipeline = SimpleNamespace(
        runner=SimpleNamespace(
            model=SimpleNamespace(
                transformer_weights=SimpleNamespace(
                    blocks=[
                        SimpleNamespace(
                            compute_phases=[
                                SimpleNamespace(self_attn_1=self_attention),
                                SimpleNamespace(cross_attn_1=cross_attention),
                            ]
                        )
                    ]
                )
            )
        )
    )
    framework_ab_wan22._install_lightx2v_reference_probe(
        pipeline,
        {
            "self_mode": "dynamic_sparse_attn",
            "cross_mode": "flash_attn3",
            "self_family": "sparse",
            "self_settings": {
                "sparsity_ratio": 0.8,
                "operator": "triton",
            },
        },
    )
    self_attention.apply("self")
    cross_attention.apply("cross")
    collected = framework_ab_wan22._collect_lightx2v_reference_probe(pipeline)

    with pytest.raises(RuntimeError, match="strictly_sparse_mask"):
        framework_ab_wan22._verify_lightx2v_reference_probe(
            collected,
            require_successful_calls=True,
        )


def test_lightx2v_sparse_provider_report_rejects_dense_substitution() -> None:
    class FakeDynamicSparse:
        sparsity_ratio = 0.8
        operator = "triton"
        per_block_mean = False
        topk = 1.0 - sparsity_ratio
        BLKQ = 64
        BLKK = 64
        arch = "sm90"

        def dense_attention(self, value=None):
            return value

        def __init__(self) -> None:
            self.apply_func = self.dense_attention

    report, issues = framework_ab_wan22._lightx2v_sparse_provider_report(
        [FakeDynamicSparse()],
        requested_mode="dynamic_sparse_attn",
        settings={"sparsity_ratio": 0.8, "operator": "triton"},
    )

    assert report["settings_match"] is True
    assert report["provider_paths"] != [report["expected_provider_path"]]
    assert any("wrong provider method" in issue for issue in issues)


def test_lightx2v_role_probe_proves_general_sparse_operator_and_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    operator_module_name = "lightx2v.common.ops.attn.sparse_operator"
    mask_module_name = "lightx2v.common.ops.attn.sparse_mask_generator"

    class SlaTritonOperator:
        def __init__(self) -> None:
            self.operator_setting = {}

        def __call__(self, q, _k, _v, _mask):
            return q

    SlaTritonOperator.__module__ = operator_module_name
    SlaTritonOperator.__qualname__ = "SlaTritonOperator"

    class SlaMaskGenerator:
        def __init__(self) -> None:
            self.sparse_setting = {"sparsity_ratio": 0.8}

        def __call__(self, _q, _k):
            return torch.tensor([[True, False], [True, True]])

        def reorg(self, q, k, v):
            return q, k, v

        def restore(self, value):
            return value

    SlaMaskGenerator.__module__ = mask_module_name
    SlaMaskGenerator.__qualname__ = "SlaMaskGenerator"

    sparse_module_name = "lightx2v.common.ops.attn.general_sparse_attn"
    sparse_module = ModuleType(sparse_module_name)
    monkeypatch.setitem(sys.modules, sparse_module_name, sparse_module)

    class GeneralSparseAttnWeight:
        def __init__(self) -> None:
            self.sparse_operator = "sla_triton_operator"
            self.sparse_mask_generator = "sla_mask_generator"
            self.sparse_setting = {"sparsity_ratio": 0.8}
            self.operator_setting = {}
            self.operator = SlaTritonOperator()
            self.mask_generator = SlaMaskGenerator()

        def apply(self, value=None):
            mask = self.mask_generator(value, value)
            q, k, v = self.mask_generator.reorg(value, value, value)
            return self.mask_generator.restore(self.operator(q, k, v, mask))

    GeneralSparseAttnWeight.__module__ = sparse_module_name
    GeneralSparseAttnWeight.__qualname__ = "GeneralSparseAttnWeight"

    dense_module_name = "lightx2v.common.ops.attn.flash_attn"
    dense_module = ModuleType(dense_module_name)
    dense_module.flash_attn_func_v3 = lambda *_args, **_kwargs: None
    dense_module.flash_attn_varlen_func_v3 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, dense_module_name, dense_module)

    class FlashAttn3Weight:
        def apply(self, value=None):
            return value

    FlashAttn3Weight.__module__ = dense_module_name
    FlashAttn3Weight.__qualname__ = "FlashAttn3Weight"
    self_attention = GeneralSparseAttnWeight()
    cross_attention = FlashAttn3Weight()
    pipeline = SimpleNamespace(
        runner=SimpleNamespace(
            model=SimpleNamespace(
                transformer_weights=SimpleNamespace(
                    blocks=[
                        SimpleNamespace(
                            compute_phases=[
                                SimpleNamespace(self_attn_1=self_attention),
                                SimpleNamespace(cross_attn_1=cross_attention),
                            ]
                        )
                    ]
                )
            )
        )
    )
    request = {
        "self_mode": "general_sparse_attn",
        "cross_mode": "flash_attn3",
        "self_family": "sparse",
        "self_settings": {
            "sparse_mask_generator": "sla_mask_generator",
            "sparse_operator": "sla_triton_operator",
            "sparse_setting": {"sparsity_ratio": 0.8},
            "operator_setting": {},
        },
    }

    installed = framework_ab_wan22._install_lightx2v_reference_probe(
        pipeline,
        request,
    )
    framework_ab_wan22._verify_lightx2v_reference_probe(
        installed,
        require_successful_calls=False,
    )
    self_attention.apply("self")
    cross_attention.apply("cross")
    collected = framework_ab_wan22._collect_lightx2v_reference_probe(pipeline)
    verification = framework_ab_wan22._verify_lightx2v_reference_probe(
        collected,
        require_successful_calls=True,
    )

    assert verification["passed"] is True
    sparse_role = collected["roles"]["self"]
    assert sparse_role["sparse_provider"]["provider_paths"] == [
        "lightx2v.common.ops.attn.sparse_operator.SlaTritonOperator"
    ]
    assert sparse_role["sparse_provider"]["mask_generator_paths"] == [
        "lightx2v.common.ops.attn.sparse_mask_generator.SlaMaskGenerator"
    ]
    assert sparse_role["sparse_mask"]["strictly_sparse"] is True


def test_semantic_digest_separates_resident_and_vae_precision() -> None:
    common = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        "/checkpoint",
        "--output-dir",
        "/output",
        "--run-id",
        "precision-contract",
        "--physical-gpu",
        "0",
    ]
    parser = framework_ab_wan22.build_parser()
    bf16 = parser.parse_args(common)
    fp32 = parser.parse_args(
        [
            *common,
            "--worldfoundry-dit-weight-dtype",
            "fp32",
            "--worldfoundry-vae-decode-autocast",
            "none",
        ]
    )
    checkpoints = {name: {"identity_sha256": name} for name in ("worldfoundry", "fastvideo", "lightx2v")}

    bf16_config = framework_ab_wan22._semantic_config(bf16, checkpoints)
    fp32_config = framework_ab_wan22._semantic_config(fp32, checkpoints)

    assert bf16_config["worldfoundry_optimization_profile"]["dit_weight_dtype"] == "bf16"
    assert bf16_config["worldfoundry_optimization_profile"]["vae_decode_autocast"] == "bf16"
    assert fp32_config["worldfoundry_optimization_profile"]["dit_weight_dtype"] == "fp32"
    assert fp32_config["worldfoundry_optimization_profile"]["vae_decode_autocast"] == "none"
    assert framework_ab_wan22._canonical_digest(bf16_config) != framework_ab_wan22._canonical_digest(fp32_config)

    source_a = framework_ab_wan22._semantic_config(
        bf16,
        checkpoints,
        execution_fingerprint_sha256="source-a",
    )
    source_b = framework_ab_wan22._semantic_config(
        bf16,
        checkpoints,
        execution_fingerprint_sha256="source-b",
    )
    assert framework_ab_wan22._canonical_digest(source_a) != framework_ab_wan22._canonical_digest(source_b)


def test_run_parser_exposes_lightx2v_public_pipeline_contract() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "lightx2v",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "lightx2v-contract",
            "--physical-gpu",
            "0",
            "--lightx2v-attention-mode",
            "sage_attn2",
            "--lightx2v-sample-shift",
            "5.0",
        ]
    )
    assert args.framework == "lightx2v"
    assert args.lightx2v_attention_mode == "sage_attn2"
    assert args.lightx2v_self_attention_mode is None
    assert args.lightx2v_cross_attention_mode is None
    assert args.lightx2v_self_attention_settings_json == "{}"
    assert args.lightx2v_sample_shift == 5.0
    assert args.lightx2v_double_precision_rope is True


def test_lightx2v_role_request_matches_upstream_sparse_self_dense_cross() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "lightx2v",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "lightx2v-sparse-contract",
            "--physical-gpu",
            "0",
            "--lightx2v-attention-mode",
            "flash_attn3",
            "--lightx2v-self-attention-mode",
            "dynamic_sparse_attn",
            "--lightx2v-cross-attention-mode",
            "flash_attn3",
            "--lightx2v-self-attention-settings-json",
            '{"sparsity_ratio": 0.8, "operator": "triton"}',
        ]
    )

    request = framework_ab_wan22._lightx2v_attention_request(args)

    assert request == {
        "legacy_attention_mode": "flash_attn3",
        "self_mode": "dynamic_sparse_attn",
        "cross_mode": "flash_attn3",
        "self_family": "sparse",
        "self_settings": {
            "sparsity_ratio": 0.8,
            "operator": "triton",
        },
        "self_setting_key": "dynamic_sparse_attn_setting",
    }
    checkpoints = {
        name: {"identity_sha256": name}
        for name in ("worldfoundry", "fastvideo", "lightx2v")
    }
    semantic_profile = framework_ab_wan22._semantic_config(
        args,
        checkpoints,
    )["lightx2v_optimization_profile"]
    assert semantic_profile["self_attention_mode"] == "dynamic_sparse_attn"
    assert semantic_profile["cross_attention_mode"] == "flash_attn3"
    assert semantic_profile["self_attention_family"] == "sparse"
    assert semantic_profile["self_attention_settings"] == {
        "sparsity_ratio": 0.8,
        "operator": "triton",
    }


@pytest.mark.parametrize(
    ("self_mode", "cross_mode", "settings", "match"),
    (
        (
            "dynamic_sparse_attn",
            None,
            {"sparsity_ratio": 0.8, "operator": "triton"},
            "requires an explicit dense",
        ),
        (
            "draft_attn",
            "flash_attn3",
            {},
            "no fail-closed dense or sparse runtime contract",
        ),
        (
            "dynamic_sparse_attn",
            "dynamic_sparse_attn",
            {"sparsity_ratio": 0.8, "operator": "triton"},
            "cross attention must use a certifying dense provider",
        ),
    ),
)
def test_lightx2v_role_request_rejects_uncertifiable_sparse_profiles(
    self_mode: str,
    cross_mode: str | None,
    settings: dict[str, object],
    match: str,
) -> None:
    args = SimpleNamespace(
        lightx2v_attention_mode=self_mode,
        lightx2v_self_attention_mode=None,
        lightx2v_cross_attention_mode=cross_mode,
        lightx2v_self_attention_settings_json=settings,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        framework_ab_wan22._lightx2v_attention_request(args)


@pytest.mark.parametrize("role", ("self", "cross"))
def test_lightx2v_prequantized_kv_attention_fails_closed_without_lifecycle(
    role: str,
) -> None:
    mode = "sage_attn2_k_int8_v_fp8"
    args = SimpleNamespace(
        lightx2v_attention_mode="flash_attn3",
        lightx2v_self_attention_mode=(mode if role == "self" else None),
        lightx2v_cross_attention_mode=(mode if role == "cross" else "flash_attn3"),
        lightx2v_self_attention_settings_json={},
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"{role} attention.*fail-closed.*pre-quantized.*"
            r"no audited per-layer quantize/cache/invalidation lifecycle"
        ),
    ):
        framework_ab_wan22._lightx2v_attention_request(args)


def test_summary_supports_same_gpu_lightx2v_pairs(tmp_path: Path) -> None:
    samples = (
        _record(
            framework="worldfoundry",
            pair_id="light-pair",
            gpu=0,
            order="AB",
            generation_s=8.0,
        ),
        _record(
            framework="lightx2v",
            pair_id="light-pair",
            gpu=0,
            order="AB",
            generation_s=10.0,
        ),
    )
    for sample in samples:
        _write_record(tmp_path, sample)

    args = argparse.Namespace(
        output_dir=tmp_path,
        tag="test",
        min_samples=1,
        bootstrap_samples=100,
        allow_insufficient=False,
        reference_framework="lightx2v",
    )
    assert framework_ab_wan22.summarize(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["reference_framework"] == "lightx2v"
    assert summary["worldfoundry_speedup_vs_lightx2v"] == pytest.approx(1.25)
    assert summary["claim"] == "non_certifying_diagnostic"


def test_video_quality_gate_requires_all_three_metrics() -> None:
    passed = framework_ab_wan22._video_quality_gate(
        {
            "psnr_db_mean": 35.0,
            "ssim_mean": 0.98,
            "lpips_alex_mean": 0.04,
        },
        min_psnr=30.0,
        min_ssim=0.95,
        max_lpips=0.10,
    )
    assert passed["passed"] is True

    failed = framework_ab_wan22._video_quality_gate(
        {
            "psnr_db_mean": "infinity",
            "ssim_mean": 0.94,
            "lpips_alex_mean": 0.04,
        },
        min_psnr=30.0,
        min_ssim=0.95,
        max_lpips=0.10,
    )
    assert failed["passed"] is False
    assert failed["checks"]["ssim_mean"]["passed"] is False


def test_video_validation_rejects_frame_rate_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")

    def fake_run(command, **_kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "codec_name": "h264",
                                "width": 1280,
                                "height": 704,
                                "r_frame_rate": "30/1",
                                "nb_read_frames": "121",
                            }
                        ]
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(framework_ab_wan22, "_run", fake_run)
    expected = argparse.Namespace(width=1280, height=704, frames=121, fps=24)
    with pytest.raises(RuntimeError, match="frame rate mismatch"):
        framework_ab_wan22._validate_video(video, expected)

    expected.fps = 30
    report = framework_ab_wan22._validate_video(video, expected)
    assert report["decoded_fps"] == 30.0
    assert report["full_decode_ok"] is True


def test_validation_ffmpeg_prefers_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WAN22_AB_VALIDATION_FFMPEG", "/opt/ffmpeg-pinned")

    assert framework_ab_wan22._validation_ffmpeg_executable() == "/opt/ffmpeg-pinned"


def test_validation_ffmpeg_rejects_empty_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WAN22_AB_VALIDATION_FFMPEG", "   ")

    with pytest.raises(ValueError, match="must not be empty"):
        framework_ab_wan22._validation_ffmpeg_executable()


def test_summary_parser_exposes_quality_gate() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "summarize",
            "--output-dir",
            "/output",
            "--reference-framework",
            "lightx2v",
            "--quality",
            "--performance-gate",
            "--min-video-psnr",
            "32",
        ]
    )
    assert args.quality is True
    assert args.performance_gate is True
    assert args.reference_framework == "lightx2v"
    assert args.min_video_psnr == 32.0
    assert args.allow_contaminated is False
    assert args.allow_unaudited_generation_isolation is False


def test_performance_gate_requires_samples_median_and_ci() -> None:
    passed = framework_ab_wan22._performance_gate(
        sample_sufficient=True,
        paired_speedup_median=1.05,
        bootstrap_ci_lower=0.99,
        min_paired_speedup=1.0,
        min_ci_lower=0.97,
    )
    assert passed["passed"] is True

    failed = framework_ab_wan22._performance_gate(
        sample_sufficient=True,
        paired_speedup_median=1.05,
        bootstrap_ci_lower=0.94,
        min_paired_speedup=1.0,
        min_ci_lower=0.97,
    )
    assert failed["passed"] is False
    assert failed["checks"]["bootstrap_95ci_lower"]["passed"] is False


def test_lightx2v_runner_uses_public_pipeline_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    kernel_module_name = "lightx2v.common.ops.attn.sage_attn"
    kernel_module = ModuleType(kernel_module_name)
    kernel_module.sageattn = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, kernel_module_name, kernel_module)
    sageattention = ModuleType("sageattention")
    sageattention.sageattn = lambda *_args, **_kwargs: None
    sageattention.__file__ = str(tmp_path / "sageattention.py")
    monkeypatch.setitem(sys.modules, "sageattention", sageattention)

    class SageAttn2Weight:
        def apply(self, value=None):
            return value

    SageAttn2Weight.__module__ = kernel_module_name
    SageAttn2Weight.__qualname__ = "SageAttn2Weight"

    class FakePipeline:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def enable_offload(self, **kwargs) -> None:
            calls["offload"] = kwargs

        def create_generator(self, **kwargs) -> None:
            calls["generator"] = kwargs
            calls["bound_config_json"] = self.config_json
            config = json.loads(Path(kwargs["config_json"]).read_text(encoding="utf-8"))
            self.self_attn_1_type = config["self_attn_1_type"]
            self.cross_attn_1_type = config["cross_attn_1_type"]
            self.cross_attn_2_type = config["cross_attn_2_type"]
            self.rope_type = config["rope_type"]
            self.double_precision_rope = config["double_precision_rope"]
            self_attention = SageAttn2Weight()
            cross_attention = SageAttn2Weight()
            calls["self_attention"] = self_attention
            calls["cross_attention"] = cross_attention
            self.runner = SimpleNamespace(
                model=SimpleNamespace(
                    transformer_weights=SimpleNamespace(
                        blocks=[
                            SimpleNamespace(
                                compute_phases=[
                                    SimpleNamespace(
                                        self_attn_1=self_attention,
                                        rope=SimpleNamespace(
                                            compute_dtype=torch.float32
                                        ),
                                        attn_rms_norm_type="sgl-kernel",
                                        self_attn_norm_q=SimpleNamespace(
                                            infer_dtype=torch.bfloat16,
                                            sensitive_layer_dtype=torch.bfloat16,
                                        ),
                                        self_attn_norm_k=SimpleNamespace(
                                            infer_dtype=torch.bfloat16,
                                            sensitive_layer_dtype=torch.bfloat16,
                                        ),
                                    ),
                                    SimpleNamespace(cross_attn_1=cross_attention),
                                ]
                            )
                        ]
                    )
                )
            )

        def generate(self, **kwargs) -> None:
            calls["generate"] = kwargs
            calls["self_attention"].apply("self")
            calls["cross_attention"].apply("cross1")
            Path(kwargs["save_result_path"]).write_bytes(b"video")

    module = ModuleType("lightx2v")
    module.LightX2VPipeline = FakePipeline
    module.__file__ = str(tmp_path / "lightx2v" / "__init__.py")
    monkeypatch.setitem(sys.modules, "lightx2v", module)

    import torch

    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024**2)
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_query",
        lambda physical_gpu: {"index": physical_gpu},
    )
    compute_app_snapshots = iter(
        (
            [{"pid": 100, "used_memory_mib": 1024.0}],
            [{"pid": 100, "used_memory_mib": 1024.0}],
            [{"pid": 100, "used_memory_mib": 1024.0}],
        )
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_compute_apps",
        lambda _physical_gpu: next(compute_app_snapshots),
    )

    class FakeSampler:
        def __init__(self, physical_gpu: int) -> None:
            self.physical_gpu = physical_gpu

        def start(self) -> None:
            calls["sampler_started"] = self.physical_gpu

        def stop(self) -> None:
            calls["sampler_stopped"] = self.physical_gpu

        def summary(self):
            return {"sample_count": 1}

    monkeypatch.setattr(framework_ab_wan22, "GpuSampler", FakeSampler)
    output_path = tmp_path / "output.mp4"
    args = SimpleNamespace(
        lightx2v_checkpoint=None,
        checkpoint=tmp_path / "checkpoint",
        lightx2v_attention_mode="sage_attn2",
        lightx2v_sample_shift=5.0,
        lightx2v_rope_type="torch_complex_rope",
        lightx2v_double_precision_rope=True,
        steps=50,
        height=704,
        width=1280,
        frames=121,
        fps=24,
        guidance=5.0,
        seed=42,
        prompt="prompt",
        negative_prompt="negative",
        physical_gpu=3,
    )
    result = framework_ab_wan22._run_lightx2v(args, output_path)

    assert calls["init"] == {
        "model_path": str(args.checkpoint),
        "model_cls": "wan2.2",
        "task": "t2v",
    }
    assert calls["offload"] == {
        "cpu_offload": False,
        "text_encoder_offload": False,
        "image_encoder_offload": False,
        "vae_offload": False,
    }
    generator_config_path = output_path.with_suffix(".lightx2v-config.json")
    assert calls["generator"] == {"config_json": str(generator_config_path)}
    assert calls["bound_config_json"] == str(generator_config_path)
    assert json.loads(generator_config_path.read_text(encoding="utf-8")) == {
        "infer_steps": 50,
        "target_video_length": 121,
        "text_len": 512,
        "target_height": 704,
        "target_width": 1280,
        "num_channels_latents": 48,
        "vae_stride": [4, 16, 16],
        "self_attn_1_type": "sage_attn2",
        "cross_attn_1_type": "sage_attn2",
        "cross_attn_2_type": "sage_attn2",
        "sample_guide_scale": 5.0,
        "sample_shift": 5.0,
        "enable_cfg": True,
        "cpu_offload": False,
        "offload_granularity": "model",
        "t5_cpu_offload": False,
        "vae_cpu_offload": False,
        "fps": 24,
        "rope_type": "torch_complex_rope",
        "double_precision_rope": True,
    }
    assert calls["generate"] == {
        "seed": 42,
        "prompt": "prompt",
        "negative_prompt": "negative",
        "save_result_path": str(output_path),
        "return_result_tensor": False,
    }
    assert result["output_path"] == str(output_path.resolve())
    assert result["optimization_report"]["quality_tier"] == "numerically-approximate"
    assert result["optimization_report"]["fallbacks"] == []
    optimization_report = result["optimization_report"]
    assert optimization_report["effective"][
        "double_precision_rope_configured"
    ] is True
    assert optimization_report["effective"]["double_precision_rope"] is False
    assert optimization_report["effective"]["rope_compute_dtype"] == "float32"
    assert optimization_report["effective"]["rms_norm_compute_mode"] == "input"
    assert {
        item["compute_mode"]
        for item in optimization_report["wan_math_runtime"]["rms_norm"].values()
    } == {"input"}
    runtime_probe = optimization_report["runtime_probe"]
    assert runtime_probe["roles"]["self"]["successful_calls"] == 1
    assert runtime_probe["roles"]["cross1"]["successful_calls"] == 1
    assert result["generation_isolation"]["contaminated"] is True
    assert result["generation_isolation"]["certifying"] is False


def test_lightx2v_runner_uses_sparse_self_and_dense_cross_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    calls: dict[str, object] = {}
    sparse_module_name = "lightx2v.common.ops.attn.dynamic_sparse_attn"
    sparse_module = ModuleType(sparse_module_name)
    sparse_module.get_block_map = lambda *_args, **_kwargs: (
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([0]),
        1,
    )

    class _Attention:
        @staticmethod
        def apply(value=None, *_args, **_kwargs):
            return value

    sparse_module._attention = _Attention
    monkeypatch.setitem(sys.modules, sparse_module_name, sparse_module)

    class DynamicSparseAttnWeight:
        def __init__(self, setting: dict[str, object]) -> None:
            self.sparsity_ratio = setting["sparsity_ratio"]
            self.operator = setting["operator"]
            self.per_block_mean = setting.get("per_block_mean", False)
            self.topk = 1.0 - self.sparsity_ratio
            self.BLKQ = 64
            self.BLKK = 64
            self.arch = "sm90"
            self.apply_func = self.apply_triton

        def apply_triton(self, value=None):
            sparse_module.get_block_map(value, value)
            return sparse_module._attention.apply(value)

        def apply(self, value=None):
            return self.apply_func(value)

    DynamicSparseAttnWeight.__module__ = sparse_module_name
    DynamicSparseAttnWeight.__qualname__ = "DynamicSparseAttnWeight"
    DynamicSparseAttnWeight.apply_triton.__module__ = sparse_module_name
    DynamicSparseAttnWeight.apply_triton.__qualname__ = (
        "DynamicSparseAttnWeight.apply_triton"
    )

    dense_module_name = "lightx2v.common.ops.attn.flash_attn"
    dense_module = ModuleType(dense_module_name)
    dense_module.flash_attn_func_v3 = lambda *_args, **_kwargs: None
    dense_module.flash_attn_varlen_func_v3 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, dense_module_name, dense_module)

    class FlashAttn3Weight:
        def apply(self, value=None):
            return value

    FlashAttn3Weight.__module__ = dense_module_name
    FlashAttn3Weight.__qualname__ = "FlashAttn3Weight"

    external_fa3 = ModuleType("flash_attn_interface")
    external_fa3.flash_attn_func = lambda *_args, **_kwargs: None
    external_fa3.flash_attn_varlen_func = lambda *_args, **_kwargs: None
    external_fa3.__file__ = str(tmp_path / "flash_attn_interface.py")
    monkeypatch.setitem(sys.modules, "flash_attn_interface", external_fa3)

    class FakePipeline:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def enable_offload(self, **kwargs) -> None:
            calls["offload"] = kwargs

        def create_generator(self, **kwargs) -> None:
            config = json.loads(
                Path(kwargs["config_json"]).read_text(encoding="utf-8")
            )
            calls["generator_config"] = config
            self.self_attn_1_type = config["self_attn_1_type"]
            self.cross_attn_1_type = config["cross_attn_1_type"]
            self.cross_attn_2_type = config["cross_attn_2_type"]
            self.rope_type = config["rope_type"]
            self.double_precision_rope = config["double_precision_rope"]
            self_attention = DynamicSparseAttnWeight(
                config["dynamic_sparse_attn_setting"]
            )
            cross_attention = FlashAttn3Weight()
            calls["self_attention"] = self_attention
            calls["cross_attention"] = cross_attention
            self.runner = SimpleNamespace(
                model=SimpleNamespace(
                    transformer_weights=SimpleNamespace(
                        blocks=[
                            SimpleNamespace(
                                compute_phases=[
                                    SimpleNamespace(
                                        self_attn_1=self_attention,
                                        rope=SimpleNamespace(
                                            compute_dtype=torch.float32
                                        ),
                                        attn_rms_norm_type="sgl-kernel",
                                        self_attn_norm_q=SimpleNamespace(
                                            infer_dtype=torch.bfloat16,
                                            sensitive_layer_dtype=torch.bfloat16,
                                        ),
                                        self_attn_norm_k=SimpleNamespace(
                                            infer_dtype=torch.bfloat16,
                                            sensitive_layer_dtype=torch.bfloat16,
                                        ),
                                    ),
                                    SimpleNamespace(cross_attn_1=cross_attention),
                                ]
                            )
                        ]
                    )
                )
            )

        def generate(self, **kwargs) -> None:
            calls["self_attention"].apply("self")
            calls["cross_attention"].apply("cross")
            Path(kwargs["save_result_path"]).write_bytes(b"video")

    lightx2v_module = ModuleType("lightx2v")
    lightx2v_module.LightX2VPipeline = FakePipeline
    lightx2v_module.__file__ = str(tmp_path / "lightx2v" / "__init__.py")
    monkeypatch.setitem(sys.modules, "lightx2v", lightx2v_module)

    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024**2)
    monkeypatch.setattr(
        framework_ab_wan22,
        "_gpu_query",
        lambda physical_gpu: {"index": physical_gpu},
    )

    class FakeSampler:
        def __init__(self, physical_gpu: int) -> None:
            self.physical_gpu = physical_gpu

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def summary(self):
            return {"sample_count": 1}

    monkeypatch.setattr(framework_ab_wan22, "GpuSampler", FakeSampler)
    output_path = tmp_path / "sparse-output.mp4"
    args = SimpleNamespace(
        lightx2v_checkpoint=None,
        checkpoint=tmp_path / "checkpoint",
        lightx2v_attention_mode="flash_attn3",
        lightx2v_self_attention_mode="dynamic_sparse_attn",
        lightx2v_cross_attention_mode="flash_attn3",
        lightx2v_self_attention_settings_json={
            "sparsity_ratio": 0.8,
            "operator": "triton",
        },
        lightx2v_sample_shift=5.0,
        lightx2v_rope_type="torch_complex_rope",
        lightx2v_double_precision_rope=True,
        steps=50,
        height=480,
        width=832,
        frames=81,
        fps=16,
        guidance=5.0,
        seed=42,
        prompt="prompt",
        negative_prompt="negative",
        physical_gpu=0,
        _compute_app_monitor=SimpleNamespace(),
    )

    result = framework_ab_wan22._run_lightx2v(args, output_path)

    generator_config = calls["generator_config"]
    assert generator_config["self_attn_1_type"] == "dynamic_sparse_attn"
    assert generator_config["cross_attn_1_type"] == "flash_attn3"
    assert generator_config["cross_attn_2_type"] == "flash_attn3"
    assert generator_config["dynamic_sparse_attn_setting"] == {
        "sparsity_ratio": 0.8,
        "operator": "triton",
    }
    report = result["optimization_report"]
    assert report["requested"]["self_attention_family"] == "sparse"
    assert report["effective"]["self_attention_mode"] == "dynamic_sparse_attn"
    assert report["effective"]["cross1_attention_mode"] == "flash_attn3"
    assert report["effective"]["double_precision_rope_configured"] is True
    assert report["effective"]["double_precision_rope"] is False
    assert report["effective"]["rope_compute_dtype"] == "float32"
    assert report["effective"]["rms_norm_compute_mode"] == "input"
    assert report["backend_capability"]["roles"]["self"][
        "runtime_resolved_request"
    ] is True
    assert report["runtime_probe"]["roles"]["self"]["sparse_mask"][
        "strictly_sparse"
    ] is True


def _fastvideo_sla_args(
    kind: str = "fastvideo_sla",
    *,
    backend: str | None = None,
    profile: dict[str, object] | None = None,
) -> argparse.Namespace:
    expected_backend = {
        "fastvideo_sla": "SLA_ATTN",
        "fastvideo_sagesla": "SAGE_SLA_ATTN",
    }[kind]
    argv = [
        "run",
        "--framework",
        "worldfoundry",
        "--checkpoint",
        "/checkpoint",
        "--output-dir",
        "/output",
        "--run-id",
        f"{kind}-contract",
        "--physical-gpu",
        "0",
        "--steps",
        "1",
        "--worldfoundry-approximate-attention",
        kind,
        "--fastvideo-attention-backend",
        backend or expected_backend,
    ]
    if profile is not None:
        argv.extend(
            [
                "--worldfoundry-approximate-attention-profile-json",
                json.dumps(profile),
            ]
        )
    return framework_ab_wan22.build_parser().parse_args(argv)


def _fastvideo_sla_contract(kind: str) -> dict[str, object]:
    configured = framework_ab_wan22.FASTVIDEO_SLA_REFERENCE_CONTRACTS[kind]
    layer_fingerprints = {
        str(layer): hashlib.sha256(f"projection:{layer}".encode()).hexdigest()
        for layer in range(30)
    }
    source_fingerprint = hashlib.sha256(
        b"pinned-fastvideo-sla-source"
    ).hexdigest()
    return {
        "provider_path": configured["provider_path"],
        "provider_family": configured["provider_family"],
        "commit": framework_ab_wan22.WORLD_FOUNDRY_PINNED_FASTVIDEO_COMMIT,
        "source_fingerprint": source_fingerprint,
        "checkpoint_layout": framework_ab_wan22.FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
        "projection_fingerprint": (
            framework_ab_wan22._fastvideo_projection_set_fingerprint(
                layer_fingerprints,
                checkpoint_layout=(
                    framework_ab_wan22.FASTVIDEO_SLA_CHECKPOINT_LAYOUT
                ),
            )
        ),
        "layer_projection_fingerprints": layer_fingerprints,
    }


def _live_fastvideo_sla_optimization_report(
    kind: str = "fastvideo_sla",
) -> dict[str, object]:
    contract = _fastvideo_sla_contract(kind)
    configured = framework_ab_wan22.FASTVIDEO_SLA_REFERENCE_CONTRACTS[kind]
    provider = str(contract["provider_path"])
    provider_family = str(contract["provider_family"])
    commit = str(contract["commit"])
    source_fingerprint = str(contract["source_fingerprint"])
    checkpoint_layout = str(contract["checkpoint_layout"])
    projection_fingerprint = str(contract["projection_fingerprint"])
    layer_fingerprints = dict(contract["layer_projection_fingerprints"])
    topk_ratio = float(configured["topk_ratio"])
    source_root = "/opt/FastVideo"
    source_file = "/opt/FastVideo/fastvideo/attention/backends/sla.py"
    provider_fingerprint = hashlib.sha256(
        f"{provider}:runtime".encode()
    ).hexdigest()
    request_id = f"{kind}-request-current"
    request_epoch = 13
    blocks = 30
    steps = 1
    grid = [31, 22, 40]
    tensor = {
        "shape": [1, 31 * 22 * 40, 1, 64],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    branches = ("positive", "negative")
    events: list[dict[str, object]] = []
    layer_call_indices = {layer: 0 for layer in range(blocks)}
    call_index = 0
    for branch in branches:
        for step in range(steps):
            for layer in range(blocks):
                call_index += 1
                layer_call_indices[layer] += 1
                prefix = f"blocks.{layer}.attn1.attn_impl.proj_l"
                events.append(
                    {
                        "algorithm": kind,
                        "request_id": request_id,
                        "request_epoch": request_epoch,
                        "request_local": True,
                        "branch": branch,
                        "step": step,
                        "step_index": step,
                        "total_steps": steps,
                        "layer_idx": layer,
                        "module_path": f"blocks.{layer}.self_attn",
                        "execution": "sparse",
                        "provider_attempted": True,
                        "provider_path": provider,
                        "fallback_reason": None,
                        "dense_fallback_executed": False,
                        "grid_size": grid,
                        "input": tensor,
                        "output": tensor,
                        "input_shape": tensor["shape"],
                        "input_device": tensor["device"],
                        "input_dtype": tensor["dtype"],
                        "output_shape": tensor["shape"],
                        "output_device": tensor["device"],
                        "output_dtype": tensor["dtype"],
                        "provider_family": provider_family,
                        "reference_provider_path": provider,
                        "provider_fingerprint": provider_fingerprint,
                        "injected_test_provider": False,
                        "provider_calls": 1,
                        "reference_fastvideo_commit": commit,
                        "provider_source_commit": commit,
                        "provider_source_clean": True,
                        "provider_source_fingerprint": source_fingerprint,
                        "provider_source_root": source_root,
                        "provider_source_file": source_file,
                        "reference_parity_verified": True,
                        "checkpoint_layout": checkpoint_layout,
                        "projection_source_keys": [
                            f"{prefix}.weight",
                            f"{prefix}.bias",
                        ],
                        "projection_weight_fingerprint": layer_fingerprints[
                            str(layer)
                        ],
                        "all_projection_weights_fingerprint": (
                            projection_fingerprint
                        ),
                        "layer_prefix": f"blocks.{layer}.attn1",
                        "call_index": call_index,
                        "layer_call_index": layer_call_indices[layer],
                        "current_timestep": step,
                        "topk_ratio": topk_ratio,
                        "feature_map": "softmax",
                        "runtime_effective": True,
                    }
                )
    branch_reports = {
        branch: {
            "last_step": 0,
            "observed_steps": [0],
            "expected_steps": [0],
            "completed_steps": [0],
            "steps_contiguous": True,
            "routed_steps": False,
            "all_steps_complete": True,
            "layer_events": blocks,
            "missing_layers": {},
        }
        for branch in branches
    }
    projection_source_keys = {
        str(layer): [
            f"blocks.{layer}.attn1.attn_impl.proj_l.weight",
            f"blocks.{layer}.attn1.attn_impl.proj_l.bias",
        ]
        for layer in range(blocks)
    }
    receipt = {
        "kind": kind,
        "request_id": request_id,
        "request_epoch": request_epoch,
        "request_local": True,
        "routed_steps": False,
        "finalized": True,
        "completed": True,
        "release_reason": "completed",
        "wrapped_blocks": blocks,
        "expected_layers": list(range(blocks)),
        "expected_calls": len(events),
        "runtime_effective": True,
        "kernel_attempts": len(events),
        "sparse_calls": len(events),
        "provider_dense_calls": 0,
        "scheduled_dense_calls": 0,
        "dense_fallback_calls": 0,
        "kernel_fallbacks": 0,
        "effective_kernel": kind,
        "provider_path": provider,
        "provider_paths": [provider],
        "grid_size": grid,
        "branches": branch_reports,
        "events": events,
        "event_count": len(events),
        "coverage": {
            "complete": True,
            "routed_steps": False,
            "expected_layers": list(range(blocks)),
            "expected_layer_count": blocks,
            "expected_calls": len(events),
            "observed_calls": len(events),
            "event_totals_match_counters": True,
            "provider_contract_complete": True,
            "request_local": True,
            "branches": branch_reports,
            "execution_counts": {"sparse": len(events)},
        },
        "notes": [],
        "lifecycle": {
            "live_requests": 0,
            "receipt_snapshots": 1,
            "max_receipt_snapshots": 32,
        },
        "fastvideo_sla": {
            "algorithm": kind,
            "provider_family": provider_family,
            "provider_families": [provider_family],
            "expected_layers": list(range(blocks)),
            "canonical_provider_path": provider,
            "canonical_provider_executed": True,
            "reference_fastvideo_commit": commit,
            "provider_source_commit": commit,
            "provider_source_commits": [commit],
            "provider_source_fingerprints": [source_fingerprint],
            "provider_fingerprints": [provider_fingerprint],
            "provider_source_root": source_root,
            "provider_source_roots": [source_root],
            "provider_source_file": source_file,
            "provider_source_files": [source_file],
            "provider_commit_complete": True,
            "provider_source_clean": True,
            "reference_parity_verified": True,
            "checkpoint_layout": checkpoint_layout,
            "checkpoint_layouts": [checkpoint_layout],
            "all_projection_weights_fingerprint": projection_fingerprint,
            "projection_fingerprints": [projection_fingerprint],
            "layer_projection_fingerprints": layer_fingerprints,
            "projection_source_keys": projection_source_keys,
            "topk_ratio": topk_ratio,
            "feature_map": "softmax",
            "provider_calls": len(events),
            "receipt_count": len(events),
            "sparse_event_count": len(events),
            "events": events,
        },
    }
    return {
        "available": True,
        "dit_weight_dtype": "bfloat16",
        "vae_decode_autocast": "bfloat16",
        "fallbacks": [],
        "requested": _live_wan_math_requested(),
        "effective": {
            **_live_wan_math_effective(),
            "attention": "flash_attention_3",
            "fuse_qkv_blocks": 30,
            "static_cross_kv_blocks": 30,
            "approximate_attention_kernel": kind,
        },
        "runtime": {
            **_live_qkv_static_runtime(),
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 120,
                        "successes": 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                    }
                }
            },
            "approximate_attention": receipt,
        },
    }


def _stub_framework_fastvideo_sla_expectation(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    contract = _fastvideo_sla_contract(kind)
    checkpoint_expectation = {
        "checkpoint_path": "/checkpoint",
        "checkpoint_layout": contract["checkpoint_layout"],
        "projection_fingerprint": contract["projection_fingerprint"],
        "canonical_projection_fingerprint": (
            framework_ab_wan22._fastvideo_projection_set_fingerprint(
                contract["layer_projection_fingerprints"],
                checkpoint_layout=None,
            )
        ),
        "layer_projection_fingerprints": contract[
            "layer_projection_fingerprints"
        ],
    }
    monkeypatch.setattr(
        framework_ab_wan22,
        "_fastvideo_sla_checkpoint_expectation_for_args",
        lambda _args: checkpoint_expectation,
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_fastvideo_sla_expectation",
        lambda _config, *, fastvideo_repository, checkpoint_expectation=None: {
            "provider_path": contract["provider_path"],
            "provider_family": contract["provider_family"],
            "commit": contract["commit"],
            "source_fingerprint": contract["source_fingerprint"],
            **(
                checkpoint_expectation
                if checkpoint_expectation is not None
                else {}
            ),
        },
    )


def _write_fastvideo_sla_safetensors(
    tmp_path: Path,
    *,
    layout: str = "fastvideo_diffusers",
    missing_bias_layer: int | None = None,
    zero_weight_layer: int | None = None,
) -> tuple[Path, dict[str, object]]:
    import torch
    from safetensors.torch import save_file

    state = {}
    layer_tensors: dict[str, tuple[object, object]] = {}
    for layer in range(30):
        if layout == "fastvideo_diffusers":
            prefix = f"blocks.{layer}.attn1.attn_impl.proj_l"
        else:
            prefix = (
                f"blocks.{layer}.self_attn.attn_op.local_attn.proj_l"
            )
        weight = torch.full((128, 128), float(layer + 1))
        bias = torch.full((128,), float(layer) / 10.0)
        if layer == zero_weight_layer:
            weight.zero_()
        state[f"{prefix}.weight"] = weight
        if layer != missing_bias_layer:
            state[f"{prefix}.bias"] = bias
        layer_tensors[str(layer)] = (weight, bias)
    checkpoint = tmp_path / f"{layout}.safetensors"
    save_file(state, checkpoint)
    return checkpoint, layer_tensors


def test_fastvideo_sla_checkpoint_preflight_hashes_exact_diffusers_projections(
    tmp_path: Path,
) -> None:
    checkpoint, tensors = _write_fastvideo_sla_safetensors(tmp_path)
    expected_layers = {
        layer: framework_ab_wan22._fastvideo_projection_fingerprint(
            int(layer),
            weight,
            bias,
        )
        for layer, (weight, bias) in tensors.items()
    }

    expectation = framework_ab_wan22._fastvideo_sla_checkpoint_expectation(
        checkpoint
    )

    assert expectation["checkpoint_path"] == str(checkpoint.resolve())
    assert expectation["checkpoint_layout"] == "fastvideo_diffusers"
    assert expectation["layer_projection_fingerprints"] == expected_layers
    assert expectation["projection_fingerprint"] == (
        framework_ab_wan22._fastvideo_projection_set_fingerprint(
            expected_layers,
            checkpoint_layout="fastvideo_diffusers",
        )
    )
    assert expectation["canonical_projection_fingerprint"] == (
        framework_ab_wan22._fastvideo_projection_set_fingerprint(
            expected_layers,
            checkpoint_layout=None,
        )
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "missing"),
        ("zero", "all zero"),
        ("wrong-layout", "upstream Diffusers"),
    ),
)
def test_fastvideo_sla_checkpoint_preflight_rejects_unusable_projection_sets(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    checkpoint, _ = _write_fastvideo_sla_safetensors(
        tmp_path,
        layout=(
            "turbodiffusion_original"
            if mutation == "wrong-layout"
            else "fastvideo_diffusers"
        ),
        missing_bias_layer=17 if mutation == "missing" else None,
        zero_weight_layer=9 if mutation == "zero" else None,
    )

    with pytest.raises(RuntimeError, match=match):
        framework_ab_wan22._fastvideo_sla_checkpoint_expectation(checkpoint)


@pytest.mark.parametrize(
    ("kind", "backend", "topk_ratio"),
    (
        ("fastvideo_sla", "SLA_ATTN", 0.1),
        ("fastvideo_sagesla", "SAGE_SLA_ATTN", 0.5),
    ),
)
def test_fastvideo_sla_cli_freezes_fair_reference_profile_and_pairing(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    backend: str,
    topk_ratio: float,
) -> None:
    source_checks: list[Path] = []
    checkpoint_checks: list[argparse.Namespace] = []
    monkeypatch.setattr(
        framework_ab_wan22,
        "_fastvideo_sla_source_identity",
        lambda repository: source_checks.append(repository) or {},
    )
    monkeypatch.setattr(
        framework_ab_wan22,
        "_fastvideo_sla_checkpoint_expectation_for_args",
        lambda parsed: checkpoint_checks.append(parsed) or {},
    )
    args = _fastvideo_sla_args(
        kind,
        backend=backend,
        profile={
            "dense_steps": 0,
            "fastvideo_topk_ratio": topk_ratio,
            "fastvideo_feature_map": "softmax",
        },
    )

    request = framework_ab_wan22._worldfoundry_approximate_attention_request(
        args
    )
    framework_ab_wan22._validate_fastvideo_sla_pairing_request(args)

    assert request == {
        "kind": kind,
        "sparsity": 0.9,
        "dense_steps": 0,
        "fastvideo_topk_ratio": topk_ratio,
        "fastvideo_feature_map": "softmax",
    }
    assert args.fastvideo_attention_backend == backend
    assert source_checks == [args.fastvideo_source]
    assert checkpoint_checks == [args]


@pytest.mark.parametrize(
    ("kind", "backend", "match"),
    (
        ("fastvideo_sla", "SAGE_SLA_ATTN", "requires.*SLA_ATTN"),
        ("fastvideo_sagesla", "SLA_ATTN", "requires.*SAGE_SLA_ATTN"),
    ),
)
def test_fastvideo_sla_pairing_rejects_mismatched_backends(
    kind: str,
    backend: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        framework_ab_wan22._validate_fastvideo_sla_pairing_request(
            _fastvideo_sla_args(kind, backend=backend)
        )


@pytest.mark.parametrize(
    "profile",
    (
        {"dense_steps": 1},
        {"fastvideo_topk_ratio": 0.2},
        {"fastvideo_feature_map": "relu"},
    ),
)
def test_fastvideo_sla_profile_rejects_reference_divergence(
    profile: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="pinned FastVideo runtime profile"):
        framework_ab_wan22._worldfoundry_approximate_attention_request(
            _fastvideo_sla_args(profile=profile)
        )


def test_fastvideo_sla_backend_is_rejected_without_matching_worldfoundry_lane() -> None:
    args = framework_ab_wan22.build_parser().parse_args(
        [
            "run",
            "--framework",
            "worldfoundry",
            "--checkpoint",
            "/checkpoint",
            "--output-dir",
            "/output",
            "--run-id",
            "orphaned-sla-backend",
            "--physical-gpu",
            "0",
            "--fastvideo-attention-backend",
            "SLA_ATTN",
        ]
    )

    with pytest.raises(ValueError, match="requires matching"):
        framework_ab_wan22._validate_fastvideo_sla_pairing_request(args)


def _remove_imported_fastvideo_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in tuple(sys.modules):
        if module_name == "fastvideo" or module_name.startswith("fastvideo."):
            monkeypatch.delitem(sys.modules, module_name)


def test_fastvideo_backend_environment_is_set_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _remove_imported_fastvideo_modules(monkeypatch)
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)

    evidence = framework_ab_wan22._configure_fastvideo_attention_environment(
        SimpleNamespace(fastvideo_attention_backend="SLA_ATTN")
    )

    assert os.environ["FASTVIDEO_ATTENTION_BACKEND"] == "SLA_ATTN"
    assert evidence == {
        "name": "FASTVIDEO_ATTENTION_BACKEND",
        "value": "SLA_ATTN",
        "configured_before_import": True,
        "ambient_value": None,
        "preexisting_fastvideo_modules": [],
    }


def test_fastvideo_backend_environment_rejects_ambient_conflict_and_preimport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _remove_imported_fastvideo_modules(monkeypatch)
    args = SimpleNamespace(fastvideo_attention_backend="SLA_ATTN")
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    with pytest.raises(RuntimeError, match="ambient.*conflicts"):
        framework_ab_wan22._configure_fastvideo_attention_environment(args)

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SLA_ATTN")
    monkeypatch.setitem(
        sys.modules,
        "fastvideo.attention.backends.sla",
        ModuleType("fastvideo.attention.backends.sla"),
    )
    with pytest.raises(RuntimeError, match="before importing FastVideo"):
        framework_ab_wan22._configure_fastvideo_attention_environment(args)


def _fake_fastvideo_sla_worker(
    *,
    backend: str = "SLA_ATTN",
    blocks: int = 30,
    projection_dtype: object | None = None,
    zero_layer: int | None = None,
    changed_layer: int | None = None,
    wrong_provider: bool = False,
) -> tuple[SimpleNamespace, list[object]]:
    import torch

    dtype = projection_dtype or torch.float32
    expected_name = (
        "SLAAttentionImpl"
        if backend == "SLA_ATTN"
        else "SageSLAAttentionImpl"
    )
    provider_name = "FakeSLAAttentionImpl" if wrong_provider else expected_name
    provider_type = type(
        provider_name,
        (torch.nn.Module,),
        {
            "__module__": "fastvideo.attention.backends.sla",
        },
    )

    class FakeAttention(torch.nn.Module):
        def __init__(self, implementation: object) -> None:
            super().__init__()
            self.backend = SimpleNamespace(name=backend)
            self.attn_impl = implementation

        def forward(self, value):
            return value

    implementations: list[object] = []
    transformer_blocks = []
    head_size = framework_ab_wan22.WAN22_TI2V_ATTENTION_HEAD_SIZE
    for layer in range(blocks):
        implementation = provider_type()
        implementation.topk_ratio = 0.1 if backend == "SLA_ATTN" else 0.5
        implementation.proj_l = torch.nn.Linear(
            head_size,
            head_size,
            dtype=dtype,
        )
        with torch.no_grad():
            implementation.proj_l.weight.fill_(float(layer + 1))
            implementation.proj_l.bias.fill_(float(layer + 1) / 10.0)
            if layer == zero_layer:
                implementation.proj_l.weight.zero_()
            if layer == changed_layer:
                implementation.proj_l.weight[0, 0].add_(0.5)
        implementations.append(implementation)
        # Learned SLA is a self-attention optimization; intentionally expose no
        # attn2/cross-attention object so the probe cannot require one.
        transformer_blocks.append(
            SimpleNamespace(attn1=FakeAttention(implementation))
        )
    worker = SimpleNamespace(
        pipeline=SimpleNamespace(
            modules={
                "transformer": SimpleNamespace(blocks=transformer_blocks)
            }
        )
    )
    return worker, implementations


def _checkpoint_expectation_from_probe_role(
    role: dict[str, object],
) -> dict[str, object]:
    return {
        "checkpoint_path": "/checkpoint",
        "checkpoint_layout": role["checkpoint_layout"],
        "projection_fingerprint": role[
            "all_projection_weights_fingerprint"
        ],
        "canonical_projection_fingerprint": role[
            "canonical_projection_fingerprint"
        ],
        "layer_projection_fingerprints": role[
            "layer_projection_fingerprints"
        ],
    }


@pytest.mark.parametrize(
    "backend",
    ("SLA_ATTN", "SAGE_SLA_ATTN"),
)
def test_fastvideo_sla_probe_proves_all_self_attention_projections(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import torch

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", backend)
    worker, implementations = _fake_fastvideo_sla_worker(backend=backend)

    installed = framework_ab_wan22._install_fastvideo_reference_probe(worker)
    checkpoint_expectation = _checkpoint_expectation_from_probe_role(
        installed["roles"]["self"]
    )
    preflight = framework_ab_wan22._verify_fastvideo_reference_probe(
        [installed],
        require_successful_calls=False,
        expected_projection=checkpoint_expectation,
    )
    assert preflight["passed"] is True
    assert set(installed["roles"]) == {"self"}
    assert installed["roles"]["self"]["layer_count"] == 30

    for implementation in implementations:
        implementation.proj_l(torch.ones(1, 128))
        implementation.proj_l(torch.ones(1, 128))
    collected = framework_ab_wan22._collect_fastvideo_reference_probe(worker)
    verification = framework_ab_wan22._verify_fastvideo_reference_probe(
        [collected],
        require_successful_calls=True,
        expected_projection=checkpoint_expectation,
    )

    assert verification["passed"] is True
    role = collected["roles"]["self"]
    assert role["successful_calls"] == 30
    assert role["projection_forward_complete"] is True
    assert all(
        layer["projection_forward_calls"] == 1
        and layer["projection_hook_removed_after_success"] is True
        for layer in role["layers"]
    )


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    (
        ("fp16", "torch.float32"),
        ("zero", "all zero"),
        ("wrong-shape", "geometry is malformed"),
        ("provider", "wrong learned-SLA implementation"),
        ("missing-layer", "requires every Wan2.2 TI2V block"),
    ),
)
def test_fastvideo_sla_probe_rejects_invalid_learned_projections(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    issue_fragment: str,
) -> None:
    import torch

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SLA_ATTN")
    worker, _ = _fake_fastvideo_sla_worker(
        projection_dtype=torch.float16 if mutation == "fp16" else None,
        zero_layer=0 if mutation == "zero" else None,
        wrong_provider=mutation == "provider",
        blocks=29 if mutation == "missing-layer" else 30,
    )
    if mutation == "wrong-shape":
        worker.pipeline.modules["transformer"].blocks[0].attn1.attn_impl.proj_l = (
            torch.nn.Linear(4, 4)
        )

    installed = framework_ab_wan22._install_fastvideo_reference_probe(worker)

    assert installed["phase"] == "installation-failed"
    assert any(issue_fragment in issue for issue in installed["issues"])
    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_fastvideo_reference_probe(
            [installed],
            require_successful_calls=False,
        )


def test_fastvideo_sla_probe_rejects_one_projection_without_forward_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SLA_ATTN")
    worker, implementations = _fake_fastvideo_sla_worker()
    framework_ab_wan22._install_fastvideo_reference_probe(worker)
    for implementation in implementations[:-1]:
        implementation.proj_l(torch.ones(1, 128))
    collected = framework_ab_wan22._collect_fastvideo_reference_probe(worker)

    assert collected["roles"]["self"]["successful_calls"] == 29
    assert collected["roles"]["self"]["layers"][-1][
        "projection_cleanup_removed_uncalled_hook"
    ] is True
    with pytest.raises(RuntimeError, match="successful_calls"):
        framework_ab_wan22._verify_fastvideo_reference_probe(
            [collected],
            require_successful_calls=True,
        )


def test_fastvideo_sla_projection_fingerprint_binds_each_layer_and_global_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SLA_ATTN")
    baseline_worker, _ = _fake_fastvideo_sla_worker()
    changed_worker, _ = _fake_fastvideo_sla_worker(changed_layer=7)

    baseline = framework_ab_wan22._install_fastvideo_reference_probe(
        baseline_worker
    )["roles"]["self"]
    changed = framework_ab_wan22._install_fastvideo_reference_probe(
        changed_worker
    )["roles"]["self"]

    assert baseline["layer_projection_fingerprints"]["6"] == changed[
        "layer_projection_fingerprints"
    ]["6"]
    assert baseline["layer_projection_fingerprints"]["7"] != changed[
        "layer_projection_fingerprints"
    ]["7"]
    assert baseline["all_projection_weights_fingerprint"] != changed[
        "all_projection_weights_fingerprint"
    ]
    assert baseline["canonical_projection_fingerprint"] != changed[
        "canonical_projection_fingerprint"
    ]


def test_fastvideo_sla_probe_rejects_runtime_that_disagrees_with_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SLA_ATTN")
    runtime_worker, _ = _fake_fastvideo_sla_worker()
    checkpoint_worker, _ = _fake_fastvideo_sla_worker(changed_layer=11)
    runtime_probe = framework_ab_wan22._install_fastvideo_reference_probe(
        runtime_worker
    )
    checkpoint_probe = framework_ab_wan22._install_fastvideo_reference_probe(
        checkpoint_worker
    )
    checkpoint_expectation = _checkpoint_expectation_from_probe_role(
        checkpoint_probe["roles"]["self"]
    )

    with pytest.raises(RuntimeError, match="checkpoint_preflight"):
        framework_ab_wan22._verify_fastvideo_reference_probe(
            [runtime_probe],
            require_successful_calls=False,
            expected_projection=checkpoint_expectation,
        )


@pytest.mark.parametrize(
    ("weight_shape", "bias_shape"),
    (((), (1,)), ((4,), (4,))),
)
def test_fastvideo_sla_projection_geometry_fails_closed_without_index_error(
    weight_shape: tuple[int, ...],
    bias_shape: tuple[int, ...],
) -> None:
    import torch

    projection = torch.nn.Linear(4, 4)
    projection.weight = torch.nn.Parameter(torch.ones(weight_shape))
    projection.bias = torch.nn.Parameter(torch.ones(bias_shape))

    report, target, issues = framework_ab_wan22._fastvideo_sla_projection_layer(
        SimpleNamespace(proj_l=projection),
        layer_index=0,
    )

    assert target is projection
    assert report["projection_weight_fingerprint"] is None
    assert issues == [
        "proj_l geometry is malformed: "
        f"weight={weight_shape}, bias={bias_shape}"
    ]


@pytest.mark.parametrize(
    "kind",
    ("fastvideo_sla", "fastvideo_sagesla"),
)
def test_worldfoundry_fastvideo_sla_gate_accepts_complete_strict_receipt(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _stub_framework_fastvideo_sla_expectation(monkeypatch, kind)

    verification = framework_ab_wan22._verify_worldfoundry_optimization_report(
        _live_fastvideo_sla_optimization_report(kind),
        _fastvideo_sla_args(kind),
    )

    assert verification["passed"] is True
    assert verification["issues"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "provider",
        "source-fingerprint",
        "global-projection",
        "layer-projection",
        "nested-receipt",
        "summary-layers",
    ),
)
def test_worldfoundry_fastvideo_sla_gate_rejects_forged_receipts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    kind = "fastvideo_sla"
    _stub_framework_fastvideo_sla_expectation(monkeypatch, kind)
    report = json.loads(
        json.dumps(_live_fastvideo_sla_optimization_report(kind))
    )
    approximate = report["runtime"]["approximate_attention"]
    event = approximate["events"][0]
    if mutation == "provider":
        event["reference_provider_path"] += ".Fake"
    elif mutation == "source-fingerprint":
        event.pop("provider_source_fingerprint")
    elif mutation == "global-projection":
        event["all_projection_weights_fingerprint"] = "0" * 64
    elif mutation == "layer-projection":
        event["projection_weight_fingerprint"] = "not-a-sha256"
    elif mutation == "nested-receipt":
        event["provider_receipt"] = {"synthetic": True}
    else:
        approximate["fastvideo_sla"][
            "layer_projection_fingerprints"
        ].pop("29")

    with pytest.raises(RuntimeError, match="effectiveness gate failed"):
        framework_ab_wan22._verify_worldfoundry_optimization_report(
            report,
            _fastvideo_sla_args(kind),
        )


def _projection_contract_from_layers(
    layers: dict[str, str],
) -> dict[str, object]:
    canonical = framework_ab_wan22._fastvideo_projection_set_fingerprint(
        layers,
        checkpoint_layout=None,
    )
    layout_bound = framework_ab_wan22._fastvideo_projection_set_fingerprint(
        layers,
        checkpoint_layout=framework_ab_wan22.FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
    )
    return {
        "checkpoint_layout": framework_ab_wan22.FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
        "projection_fingerprint": layout_bound,
        "canonical_projection_fingerprint": canonical,
        "layer_projection_fingerprints": dict(layers),
    }


def _projection_runtime_evidence(
    layers: dict[str, str],
    *,
    framework: str,
) -> dict[str, object]:
    contract = _projection_contract_from_layers(layers)
    evidence = {
        "layer_projection_fingerprints": dict(layers),
        "all_projection_weights_fingerprint": contract[
            "projection_fingerprint"
        ],
    }
    if framework == "fastvideo":
        evidence["canonical_projection_fingerprint"] = contract[
            "canonical_projection_fingerprint"
        ]
    return evidence


def _fastvideo_sla_pair_records(
    *,
    runtime_layers: dict[str, str] | None = None,
    checkpoint_layers: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline_layers = {
        str(layer): hashlib.sha256(
            f"paired-projection:{layer}".encode()
        ).hexdigest()
        for layer in range(30)
    }
    runtime_layers = dict(runtime_layers or baseline_layers)
    checkpoint_layers = dict(checkpoint_layers or baseline_layers)
    checkpoint_contract = _projection_contract_from_layers(checkpoint_layers)
    worldfoundry = _record(
        framework="worldfoundry",
        pair_id="sla-pair",
        gpu=0,
        order="AB",
        generation_s=5.0,
    )
    fastvideo = _record(
        framework="fastvideo",
        pair_id="sla-pair",
        gpu=0,
        order="AB",
        generation_s=6.0,
    )
    for record in (worldfoundry, fastvideo):
        record["semantic_config"]["fastvideo_optimization_profile"] = {
            "learned_sla_contract": json.loads(
                json.dumps(checkpoint_contract)
            )
        }
    worldfoundry["semantic_config"]["worldfoundry_optimization_profile"][
        "approximate_attention"
    ] = "fastvideo_sla"
    worldfoundry["measurements"]["optimization_report"] = {
        "runtime": {
            "approximate_attention": {
                "fastvideo_sla": _projection_runtime_evidence(
                    runtime_layers,
                    framework="worldfoundry",
                )
            }
        }
    }
    fastvideo["measurements"]["optimization_report"] = {
        "runtime_probe": [
            {
                "roles": {
                    "self": _projection_runtime_evidence(
                        runtime_layers,
                        framework="fastvideo",
                    )
                }
            }
        ]
    }
    return worldfoundry, fastvideo


def test_pairing_accepts_only_complete_runtime_and_checkpoint_projection_match() -> None:
    worldfoundry, fastvideo = _fastvideo_sla_pair_records()

    pairs, exclusions = framework_ab_wan22._paired_records(
        (worldfoundry, fastvideo)
    )

    assert len(pairs) == 1
    assert pairs[0]["pair_id"] == "sla-pair"
    assert exclusions == []


@pytest.mark.parametrize(
    "mutation",
    ("missing-layer", "reported-global", "different-runtime-layer"),
)
def test_pairing_rejects_incomplete_or_mismatched_projection_evidence(
    mutation: str,
) -> None:
    worldfoundry, fastvideo = _fastvideo_sla_pair_records()
    wf_projection = worldfoundry["measurements"]["optimization_report"][
        "runtime"
    ]["approximate_attention"]["fastvideo_sla"]
    fv_projection = fastvideo["measurements"]["optimization_report"][
        "runtime_probe"
    ][0]["roles"]["self"]
    if mutation == "missing-layer":
        wf_projection["layer_projection_fingerprints"].pop("29")
    elif mutation == "reported-global":
        fv_projection["all_projection_weights_fingerprint"] = "0" * 64
    else:
        changed_layers = dict(fv_projection["layer_projection_fingerprints"])
        changed_layers["7"] = hashlib.sha256(b"changed-layer-7").hexdigest()
        fv_projection.update(
            _projection_runtime_evidence(
                changed_layers,
                framework="fastvideo",
            )
        )

    pairs, exclusions = framework_ab_wan22._paired_records(
        (worldfoundry, fastvideo)
    )

    assert pairs == []
    assert [item["reason"] for item in exclusions] == [
        "learned_sla_projection_fingerprint_mismatch"
    ]


def test_pairing_rejects_two_matching_runtimes_when_checkpoint_preflight_differs() -> None:
    checkpoint_layers = {
        str(layer): hashlib.sha256(
            f"paired-projection:{layer}".encode()
        ).hexdigest()
        for layer in range(30)
    }
    runtime_layers = dict(checkpoint_layers)
    runtime_layers["11"] = hashlib.sha256(b"wrong-but-shared").hexdigest()
    worldfoundry, fastvideo = _fastvideo_sla_pair_records(
        runtime_layers=runtime_layers,
        checkpoint_layers=checkpoint_layers,
    )

    pairs, exclusions = framework_ab_wan22._paired_records(
        (worldfoundry, fastvideo)
    )

    assert pairs == []
    assert exclusions[0]["reason"] == (
        "learned_sla_projection_fingerprint_mismatch"
    )
    assert exclusions[0]["worldfoundry_projection"] == exclusions[0][
        "fastvideo_projection"
    ]
    assert exclusions[0]["worldfoundry_projection"] != exclusions[0][
        "worldfoundry_preflight"
    ]


def test_non_sla_pairing_does_not_require_projection_fingerprints() -> None:
    worldfoundry, fastvideo = _fastvideo_sla_pair_records()
    worldfoundry["semantic_config"]["worldfoundry_optimization_profile"][
        "approximate_attention"
    ] = "vsa"
    worldfoundry["measurements"].pop("optimization_report")
    fastvideo["measurements"].pop("optimization_report")

    pairs, exclusions = framework_ab_wan22._paired_records(
        (worldfoundry, fastvideo)
    )

    assert len(pairs) == 1
    assert exclusions == []


def test_fastvideo_sla_pairing_rejects_lightx2v_as_reference() -> None:
    worldfoundry, _ = _fastvideo_sla_pair_records()
    lightx2v = _record(
        framework="lightx2v",
        pair_id="sla-pair",
        gpu=0,
        order="AB",
        generation_s=6.0,
    )

    pairs, exclusions = framework_ab_wan22._paired_records(
        (worldfoundry, lightx2v),
        reference_framework="lightx2v",
    )

    assert pairs == []
    assert exclusions == [
        {
            "reason": "learned_sla_requires_fastvideo_reference",
            "pair_id": "sla-pair",
            "reference_framework": "lightx2v",
        }
    ]


def test_shell_launcher_help_is_side_effect_free() -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarks/inference/run_framework_ab_wan22.sh"

    completed = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0
    assert "takes no positional arguments" in completed.stdout
    assert "WAN22_AB_ROUNDS" in completed.stdout
    assert "WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE_JSON" in completed.stdout
    assert "fastvideo_sla" in completed.stdout
    assert "fastvideo_sagesla" in completed.stdout
    assert "WAN22_AB_FV_ATTENTION_BACKEND" in completed.stdout
    assert "WAN22_AB_GPU_ISOLATION_PROOF" in completed.stdout
    assert "WAN22_AB_FRAMES/STEPS" in completed.stdout
    assert "WAN22_AB_WF_INPLACE_RESIDUAL" in completed.stdout
    assert "WAN22_AB_WF_FUSE_QKV" in completed.stdout
    assert "WAN22_AB_WF_QKV_STRATEGY" in completed.stdout
    assert "WAN22_AB_WF_ROPE_PRECISION" in completed.stdout
    assert "WAN22_AB_WF_RMS_NORM_PRECISION" in completed.stdout


def test_shell_launcher_uses_measured_production_kernel_defaults() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/inference/run_framework_ab_wan22.sh"
    )
    source = script.read_text(encoding="utf-8")

    for assignment in (
        'WF_FUSE_QKV="${WAN22_AB_WF_FUSE_QKV:-0}"',
        'WF_QKV_STRATEGY="${WAN22_AB_WF_QKV_STRATEGY:-auto}"',
        'WF_QKV_SPLIT_THRESHOLD="${WAN22_AB_WF_QKV_SPLIT_THRESHOLD:-8192}"',
        'WF_INPLACE_RESIDUAL="${WAN22_AB_WF_INPLACE_RESIDUAL:-0}"',
        'WF_STATIC_CROSS_KV="${WAN22_AB_WF_STATIC_CROSS_KV:-1}"',
        'WF_FUSED_ROPE="${WAN22_AB_WF_FUSED_ROPE:-0}"',
        'WF_ROPE_PRECISION="${WAN22_AB_WF_ROPE_PRECISION:-fp32}"',
        'WF_RMS_NORM_PRECISION="${WAN22_AB_WF_RMS_NORM_PRECISION:-input}"',
    ):
        assert assignment in source


def test_shell_launcher_uses_active_python_and_rejects_stale_transformers() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/inference/run_framework_ab_wan22.sh"
    )
    source = script.read_text(encoding="utf-8")

    assert 'WF_PYTHON="${WAN22_AB_WF_PYTHON:-}"' in source
    assert 'WF_PYTHON="$(command -v python3 || true)"' in source
    assert "transformers>=4.57,<5 with transformers.masking_utils" in source


def test_shell_launcher_preserves_lightx2v_settings_json() -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarks/inference/run_framework_ab_wan22.sh"
    settings = '{"operator":"sage2","sparsity_ratio":0.9}'

    completed = subprocess.run(
        ["bash", "-x", str(script)],
        env=_clean_launcher_environment(
            WAN22_AB_LX_SELF_ATTENTION_SETTINGS_JSON=settings,
            WAN22_AB_ROUNDS="1",
            WAN22_AB_MIN_SAMPLES="2",
        ),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert f"LX_SELF_ATTENTION_SETTINGS_JSON='{settings}'" in completed.stderr
    assert "is below WAN22_AB_MIN_SAMPLES" in completed.stderr


def test_shell_launcher_exposes_pinned_fastvideo_provider_and_kernel_paths() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/inference/run_framework_ab_wan22.sh"
    )
    source = script.read_text(encoding="utf-8")

    expected_path = (
        'python_path="${FV_SOURCE}:${FV_SOURCE}/fastvideo-kernel/python:'
        '${REPO_ROOT}"'
    )
    # One assignment is for FastVideo itself and one is for the WorldFoundry
    # learned-SLA lane, which imports the exact same pinned provider/kernel.
    assert source.count(expected_path) == 2


def test_shell_launcher_exposes_pinned_lightx2v_and_sparge_provider_paths() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/inference/run_framework_ab_wan22.sh"
    )
    source = script.read_text(encoding="utf-8")

    assert (
        'SPARGE_SOURCE="${WAN22_AB_SPARGE_SOURCE:-'
        '${REPO_ROOT}/tmp/refs/SpargeAttn}"'
    ) in source
    # One assignment serves the LightX2V reference and one serves
    # WorldFoundry's LightX2V-provider sparse lanes.
    assert source.count(
        'python_path="${SPARGE_SOURCE}:${LX_SOURCE}:${REPO_ROOT}"'
    ) == 2
    for kind in (
        "dynamic_sparse",
        "sparge",
        "nbhd",
        "lightx2v_sla_mask",
        "flexblock",
        "lightx2v_spas_sage",
        "draft_attn",
        "radial_attn",
        "rainfusion_attn",
        "svg_attn",
        "svg2_attn",
        "lightx2v_svg_mask",
    ):
        assert kind in source


def _clean_launcher_environment(**updates: str) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WAN22_AB_")
    }
    environment.update(updates)
    return environment


@pytest.mark.parametrize(
    ("kind", "expected_backend"),
    (
        ("fastvideo_sla", "SLA_ATTN"),
        ("fastvideo_sagesla", "SAGE_SLA_ATTN"),
    ),
)
def test_shell_launcher_automatically_maps_fastvideo_sla_backend(
    kind: str,
    expected_backend: str,
) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/inference/run_framework_ab_wan22.sh"
    )

    completed = subprocess.run(
        ["bash", "-x", str(script)],
        env=_clean_launcher_environment(
            WAN22_AB_WF_APPROXIMATE_ATTENTION=kind,
            WAN22_AB_ROUNDS="1",
            WAN22_AB_MIN_SAMPLES="2",
        ),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert f"FV_ATTENTION_BACKEND={expected_backend}" in completed.stderr
    assert "is below WAN22_AB_MIN_SAMPLES" in completed.stderr


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        (
            {
                "WAN22_AB_WF_APPROXIMATE_ATTENTION": "fastvideo_sla",
                "WAN22_AB_FV_ATTENTION_BACKEND": "FLASH_ATTN",
            },
            "fastvideo_sla requires WAN22_AB_FV_ATTENTION_BACKEND=SLA_ATTN",
        ),
        (
            {
                "WAN22_AB_WF_APPROXIMATE_ATTENTION": "fastvideo_sagesla",
                "WAN22_AB_FV_ATTENTION_BACKEND": "SLA_ATTN",
            },
            (
                "fastvideo_sagesla requires "
                "WAN22_AB_FV_ATTENTION_BACKEND=SAGE_SLA_ATTN"
            ),
        ),
        (
            {"WAN22_AB_FV_ATTENTION_BACKEND": "SLA_ATTN"},
            "SLA_ATTN requires matching WAN22_AB_WF_APPROXIMATE_ATTENTION",
        ),
        (
            {
                "WAN22_AB_WF_APPROXIMATE_ATTENTION": "fastvideo_sla",
                "WAN22_AB_FV_ATTENTION_BACKEND": "SLA_ATTN",
                "WAN22_AB_REFERENCE_FRAMEWORK": "lightx2v",
            },
            (
                "FastVideo learned-SLA comparison requires "
                "WAN22_AB_REFERENCE_FRAMEWORK=fastvideo"
            ),
        ),
    ),
)
def test_shell_launcher_rejects_fastvideo_sla_pairing_conflicts(
    environment: dict[str, str],
    message: str,
) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/inference/run_framework_ab_wan22.sh"
    )

    completed = subprocess.run(
        ["bash", str(script)],
        env=_clean_launcher_environment(**environment),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_shell_launcher_requires_explicit_vmoba_profile() -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarks/inference/run_framework_ab_wan22.sh"

    completed = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "WAN22_AB_WF_APPROXIMATE_ATTENTION": "vmoba",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert "VMoBA requires" in completed.stderr


def test_shell_launcher_rejects_positional_arguments_before_running() -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarks/inference/run_framework_ab_wan22.sh"

    completed = subprocess.run(
        ["bash", str(script), "unexpected"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert "unexpected positional argument: unexpected" in completed.stderr


def _fake_capability_module(
    name: str,
    symbols: tuple[str, ...],
) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"/opt/capabilities/{name.replace('.', '/')}.py"
    module.__version__ = "test"
    for symbol in symbols:
        owner: object = module
        components = symbol.split(".")
        for component in components[:-1]:
            nested = getattr(owner, component, None)
            if nested is None:
                nested = SimpleNamespace()
                setattr(owner, component, nested)
            owner = nested
        setattr(owner, components[-1], lambda: None)
    return module


def test_backend_capability_receipt_records_exact_imports_and_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fake_capability_module(
        "flash_attn_interface",
        ("flash_attn_func", "flash_attn_varlen_func"),
    )
    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        lambda name: module
        if name == "flash_attn_interface"
        else pytest.fail(f"unexpected capability import: {name}"),
    )

    receipt = framework_ab_wan22._require_backend_capability(
        "worldfoundry",
        "flash3",
        cuda_capability=(9, 0),
    )

    assert receipt["passed"] is True
    assert receipt["certification_role"] == "diagnostic-only"
    assert receipt["requested_backend"] == "flash3"
    assert receipt["cuda_supported"] is True
    requirement = receipt["requirements"][0]
    assert requirement["module"] == "flash_attn_interface"
    assert requirement["required_symbols"] == [
        "flash_attn_func",
        "flash_attn_varlen_func",
    ]
    assert requirement["imported"] is True
    assert requirement["module_file"] == (
        "/opt/capabilities/flash_attn_interface.py"
    )
    assert requirement["module_version"] == "test"
    assert [symbol["name"] for symbol in requirement["symbols"]] == [
        "flash_attn_func",
        "flash_attn_varlen_func",
    ]
    assert all(
        symbol["available"] is True
        and symbol["callable_identity"].endswith(
            "_fake_capability_module.<locals>.<lambda>"
        )
        for symbol in requirement["symbols"]
    )
    assert requirement["error"] is None


@pytest.mark.parametrize(
    ("framework", "backend", "missing_module"),
    (
        ("lightx2v", "flash_attn3", "flash_attn_interface"),
        ("lightx2v", "flash_attn4", "flash_attn.cute"),
        ("lightx2v", "sage_attn2", "sageattention"),
        ("lightx2v", "sage_attn3", "sageattn3"),
        ("fastvideo", "SAGE_ATTN", "sageattention"),
        ("fastvideo", "SAGE_ATTN_THREE", "sageattn3"),
        ("fastvideo", "SAGE_SLA_ATTN", "spas_sage_attn._qattn"),
    ),
)
def test_backend_capability_gate_rejects_missing_acceleration_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    backend: str,
    missing_module: str,
) -> None:
    contract = framework_ab_wan22._BACKEND_CAPABILITY_CONTRACTS[
        framework
    ][backend.casefold()]
    modules = {
        module_name: _fake_capability_module(module_name, symbols)
        for module_name, symbols in contract["requirements"]
    }

    def import_module(name: str) -> ModuleType:
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return modules[name]

    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        import_module,
    )

    with pytest.raises(
        framework_ab_wan22.BackendCapabilityError,
        match="backend capability gate failed",
    ) as captured:
        framework_ab_wan22._require_backend_capability(
            framework,
            backend,
            cuda_capability=(9, 0),
        )

    receipt = captured.value.receipt
    assert receipt["passed"] is False
    assert receipt["requested_backend"] == backend.casefold()
    assert any(
        requirement["module"] == missing_module
        and requirement["imported"] is False
        and "ModuleNotFoundError" in requirement["error"]
        for requirement in receipt["requirements"]
    )


@pytest.mark.parametrize(
    ("framework", "backend", "cuda_capability"),
    (
        ("worldfoundry", "flash4", (9, 0)),
        ("worldfoundry", "sage3", (9, 0)),
        ("lightx2v", "flash_attn4", (9, 0)),
        ("lightx2v", "sage_attn3", (9, 0)),
        ("fastvideo", "SAGE_ATTN_THREE", (9, 0)),
    ),
)
def test_backend_capability_gate_rejects_unsupported_gpu_architecture(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    backend: str,
    cuda_capability: tuple[int, int],
) -> None:
    contract = framework_ab_wan22._BACKEND_CAPABILITY_CONTRACTS[
        framework
    ][backend.casefold()]
    modules = {
        module_name: _fake_capability_module(module_name, symbols)
        for module_name, symbols in contract["requirements"]
    }
    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        modules.__getitem__,
    )

    receipt = framework_ab_wan22._backend_capability_receipt(
        framework,
        backend,
        cuda_capability=cuda_capability,
    )

    assert receipt["passed"] is False
    assert receipt["cuda_supported"] is False
    assert any("below required major" in issue for issue in receipt["issues"])


def test_lightx2v_fa3_capability_is_hopper_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fake_capability_module(
        "flash_attn_interface",
        ("flash_attn_func", "flash_attn_varlen_func"),
    )
    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        lambda name: module
        if name == "flash_attn_interface"
        else pytest.fail(f"unexpected capability import: {name}"),
    )

    receipt = framework_ab_wan22._backend_capability_receipt(
        "lightx2v",
        "flash_attn3",
        cuda_capability=(12, 0),
    )

    assert receipt["passed"] is False
    assert receipt["cuda_supported"] is False
    assert receipt["accepted_cuda_capabilities"] == [[9, 0]]
    assert any("not one of the accepted" in issue for issue in receipt["issues"])


@pytest.mark.parametrize(
    ("cuda_capability", "expected_symbol"),
    (
        ((8, 9), "sageattn_qk_int8_pv_fp16_triton"),
        ((9, 0), "sageattn"),
        ((12, 0), "sageattn_qk_int8_pv_fp16_triton"),
    ),
)
def test_lightx2v_sage2_capability_selects_architecture_specific_symbol(
    monkeypatch: pytest.MonkeyPatch,
    cuda_capability: tuple[int, int],
    expected_symbol: str,
) -> None:
    module = _fake_capability_module("sageattention", (expected_symbol,))
    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        lambda name: module
        if name == "sageattention"
        else pytest.fail(f"unexpected capability import: {name}"),
    )

    receipt = framework_ab_wan22._require_backend_capability(
        "lightx2v",
        "sage_attn2",
        cuda_capability=cuda_capability,
    )

    assert receipt["passed"] is True
    selected_symbols = [
        symbol
        for requirement in receipt["requirements"]
        for symbol in requirement["required_symbols"]
    ]
    assert selected_symbols == [expected_symbol]


def test_fastvideo_sla_capability_resolves_nested_apply_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fake_capability_module(
        "fastvideo_kernel.triton_kernels.sla_triton",
        ("_attention.apply",),
    )
    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        lambda name: module
        if name == module.__name__
        else pytest.fail(f"unexpected capability import: {name}"),
    )

    receipt = framework_ab_wan22._require_backend_capability(
        "fastvideo",
        "SLA_ATTN",
        cuda_capability=(9, 0),
    )

    assert receipt["passed"] is True
    assert receipt["requirements"][0]["symbols"] == [
        {
            "name": "_attention.apply",
            "available": True,
            "callable_identity": (
                "test_framework_ab_wan22."
                "_fake_capability_module.<locals>.<lambda>"
            ),
        }
    ]


@pytest.mark.parametrize(
    ("cuda_capability", "qattn_symbol", "selected_candidate"),
    (
        (
            (9, 0),
            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90",
            None,
        ),
        (
            (8, 9),
            "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
            0,
        ),
        (
            (8, 9),
            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
            1,
        ),
    ),
)
def test_fastvideo_sagesla_capability_selects_architecture_kernel(
    monkeypatch: pytest.MonkeyPatch,
    cuda_capability: tuple[int, int],
    qattn_symbol: str,
    selected_candidate: int | None,
) -> None:
    modules = {
        "fastvideo_kernel.triton_kernels.sla_triton": _fake_capability_module(
            "fastvideo_kernel.triton_kernels.sla_triton",
            ("_attention.apply",),
        ),
        "spas_sage_attn._fused": _fake_capability_module(
            "spas_sage_attn._fused",
            ("transpose_pad_permute_cuda", "scale_fuse_quant_cuda"),
        ),
        "spas_sage_attn.utils": _fake_capability_module(
            "spas_sage_attn.utils",
            ("get_vanilla_qk_quant", "block_map_lut_triton"),
        ),
        "spas_sage_attn._qattn": _fake_capability_module(
            "spas_sage_attn._qattn",
            (qattn_symbol,),
        ),
    }
    monkeypatch.setattr(
        framework_ab_wan22.importlib,
        "import_module",
        modules.__getitem__,
    )

    receipt = framework_ab_wan22._require_backend_capability(
        "fastvideo",
        "SAGE_SLA_ATTN",
        cuda_capability=cuda_capability,
    )

    assert receipt["passed"] is True
    if cuda_capability == (9, 0):
        assert receipt["one_of_requirements"] == []
        assert receipt["requirements"][-1]["required_symbols"] == [qattn_symbol]
    else:
        assert len(receipt["one_of_requirements"]) == 1
        assert receipt["one_of_requirements"][0]["selected_candidate"] == (
            selected_candidate
        )
