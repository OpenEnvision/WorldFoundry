from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.inference import wan22_multigpu_e2e as benchmark
from worldfoundry.base_models.diffusion_model.contracts import (
    DenoiserInput,
    DenoiserOutput,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    Wan22DualExpertDenoiser,
)


def _identity(
    local_pid: int,
    *,
    host_pid: int | None = None,
    start_time_ticks: int = 1234,
) -> dict:
    host_pid = local_pid if host_pid is None else host_pid
    namespace_pids = (
        [local_pid]
        if host_pid == local_pid
        else [host_pid, local_pid]
    )
    return {
        "local_pid": local_pid,
        "host_pid": host_pid,
        "namespace_pids": namespace_pids,
        "start_time_ticks": start_time_ticks,
    }


def _args(tmp_path, *extra: str):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    return benchmark.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ]
    )


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


def _vmoba_optimization_report() -> dict[str, object]:
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
        "requested": {"approximate_attention": "vmoba"},
        "effective": {"approximate_attention_kernel": "vmoba"},
        "fallbacks": [],
        "runtime": {
            "approximate_attention": receipt
        },
    }


def _vmoba_expected_workload() -> dict[str, object]:
    return {
        "kind": "vmoba",
        "profile": {"kind": "vmoba", **_vmoba_profile()},
        "grid": [31, 22, 40],
        "wrapped_blocks": 30,
        "total_steps": 50,
        "expected_routed_steps": False,
        "dual_expert": False,
        "branch_steps": {
            "positive": tuple(range(50)),
            "negative": tuple(range(50)),
        },
    }


def _lightx2v_optimization_report(
    workload: dict[str, object],
    *,
    branch_steps: dict[str, tuple[int, ...]] | None = None,
    request_id: str = "lightx2v-request-current",
) -> dict[str, object]:
    """Build a complete request-local receipt from a trusted benchmark workload."""

    kind = str(workload["kind"])
    profile = dict(workload["profile"])
    lightx2v = dict(workload["lightx2v"])
    provider = str(lightx2v["provider_symbol"])
    provider_family = str(lightx2v["provider_family"])
    operator = str(lightx2v["operator"])
    commit = str(lightx2v["commit"])
    fingerprint = str(lightx2v["source_fingerprint"])
    grid = [int(value) for value in workload["grid"]]
    blocks = int(workload["wrapped_blocks"])
    total_steps = int(workload["total_steps"])
    routed_steps = bool(workload["expected_routed_steps"])
    normalized_branch_steps = branch_steps or {
        str(branch): tuple(int(step) for step in steps)
        for branch, steps in dict(workload["branch_steps"]).items()
    }
    tokens = grid[0] * grid[1] * grid[2]
    tensor = {
        "shape": [1, tokens, 1, 64],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    events: list[dict[str, object]] = []
    for branch, steps in normalized_branch_steps.items():
        for step in steps:
            for layer in range(blocks):
                sparse = not (
                    step < int(profile["dense_steps"])
                    or step >= total_steps - int(profile["dense_steps"])
                )
                event: dict[str, object] = {
                    "algorithm": kind,
                    "request_id": request_id,
                    "request_epoch": 7,
                    "request_local": True,
                    "branch": branch,
                    "step": step,
                    "step_index": step,
                    "total_steps": total_steps,
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
                    provider_fields = {
                        "operator": operator,
                        "provider_family": provider_family,
                        "adapter_path": "lightx2v.common.ops.attn.adapter.apply",
                        "provider_calls": 1,
                        "reference_lightx2v_commit": commit,
                        "provider_source_commit": commit,
                        "provider_source_clean": True,
                        "provider_source_fingerprint": fingerprint,
                        "provider_source_root": "/opt/lightx2v",
                        "reference_parity_verified": True,
                        "runtime_effective": True,
                        "sparsity_ratio": profile["sparsity"],
                        "canonical_sparse_provider_path": provider,
                        "sparse_kernel_executed": True,
                        "provider_dense_executed": False,
                        "provider_config": {
                            "sparsity_ratio": profile["sparsity"],
                            "operator": operator,
                            "nbhd_coefficient": profile["nbhd_coefficient"],
                            "nbhd_min_width": profile["nbhd_min_width"],
                            "attnmap_frame_num": profile["attnmap_frame_num"]
                            or grid[0],
                            "per_block_mean": profile[
                                "lightx2v_per_block_mean"
                            ],
                            "pool_size": profile["lightx2v_pool_size"],
                            "skip_timesteps": profile[
                                "lightx2v_skip_timesteps"
                            ],
                            "dense_attn_type": profile[
                                "lightx2v_dense_attn_type"
                            ],
                            "svg_sample_mse_max_row": profile[
                                "svg_sample_mse_max_row"
                            ],
                            "svg_num_sampled_rows": profile[
                                "svg_num_sampled_rows"
                            ],
                            "svg_context_length": profile[
                                "svg_context_length"
                            ],
                        },
                    }
                    if kind == "nbhd":
                        provider_fields.update(
                            {
                                "attnmap_frame_num": profile["attnmap_frame_num"]
                                or grid[0],
                                "nbhd_coefficient": profile["nbhd_coefficient"],
                                "nbhd_min_width": profile["nbhd_min_width"],
                            }
                        )
                    event.update(provider_fields)
                events.append(event)

    sparse_events = [event for event in events if event["execution"] == "sparse"]
    scheduled_dense = len(events) - len(sparse_events)
    branches = {
        branch: {
            "last_step": steps[-1],
            "observed_steps": list(steps),
            "expected_steps": list(steps),
            "completed_steps": list(steps),
            "steps_contiguous": True,
            "routed_steps": routed_steps,
            "all_steps_complete": True,
            "layer_events": len(steps) * blocks,
            "missing_layers": {},
        }
        for branch, steps in normalized_branch_steps.items()
    }
    execution_counts = {"sparse": len(sparse_events)}
    if scheduled_dense:
        execution_counts["scheduled_dense"] = scheduled_dense
    receipt = {
        "kind": kind,
        "request_id": request_id,
        "request_epoch": 7,
        "request_local": True,
        "routed_steps": routed_steps,
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
        "effective_kernel": kind,
        "provider_path": provider,
        "provider_paths": [provider],
        "grid_size": grid,
        "branches": branches,
        "events": events,
        "event_count": len(events),
        "coverage": {
            "complete": True,
            "routed_steps": routed_steps,
            "expected_layers": list(range(blocks)),
            "expected_layer_count": blocks,
            "expected_calls": len(events),
            "observed_calls": len(events),
            "event_totals_match_counters": True,
            "provider_contract_complete": True,
            "request_local": True,
            "branches": branches,
            "execution_counts": execution_counts,
        },
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
            "provider_source_fingerprints": [fingerprint],
            "provider_commit_complete": True,
            "provider_source_clean": True,
            "reference_parity_verified": True,
            "provider_calls": len(sparse_events),
            "receipt_count": len(sparse_events),
            "sparse_event_count": len(sparse_events),
            "provider_dense_event_count": 0,
            "provider_dense_events": [],
            "events": sparse_events,
        },
    }
    return {
        "requested": {"approximate_attention": kind},
        "effective": {"approximate_attention_kernel": kind},
        "fallbacks": [],
        "runtime": {"approximate_attention": receipt},
    }


def _fastvideo_sla_source_expectation(kind: str) -> dict[str, object]:
    provider_class = (
        "SLAAttentionImpl"
        if kind == "fastvideo_sla"
        else "SageSLAAttentionImpl"
    )
    family = (
        "fastvideo/sparse-linear-attention"
        if kind == "fastvideo_sla"
        else "fastvideo/sage-sparse-linear-attention"
    )
    return {
        "kind": kind,
        "provider_path": (
            "fastvideo.attention.backends.sla." + provider_class
        ),
        "provider_family": family,
        "commit": "1b2b2a0161bc6b3b80158d1fa6380a051c6530c7",
        "source_fingerprint": "a" * 64,
        "source_root": "/opt/FastVideo",
        "source_file": "/opt/FastVideo/fastvideo/attention/backends/sla.py",
    }


def _stub_fastvideo_sla_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    layout: str = "fastvideo_diffusers",
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_fastvideo_sla_expectation",
        lambda config: _fastvideo_sla_source_expectation(str(config.kind)),
    )
    monkeypatch.setattr(
        benchmark,
        "_fastvideo_sla_checkpoint_expectation",
        lambda _checkpoint, *, model_id: {
            "checkpoint_layout": layout,
            "projection_fingerprint": "b" * 64,
        },
    )


def _fastvideo_sla_optimization_report(
    workload: dict[str, object],
    *,
    branch_steps: dict[str, tuple[int, ...]] | None = None,
    request_id: str = "fastvideo-sla-request-current",
) -> dict[str, object]:
    """Build a complete learned-SLA request receipt for audit tests."""

    kind = str(workload["kind"])
    profile = dict(workload["profile"])
    expectation = dict(workload["fastvideo_sla"])
    provider = str(expectation["provider_path"])
    provider_family = str(expectation["provider_family"])
    commit = str(expectation["commit"])
    source_fingerprint = str(expectation["source_fingerprint"])
    source_root = str(expectation["source_root"])
    source_file = str(expectation["source_file"])
    checkpoint_layout = str(expectation["checkpoint_layout"])
    projection_fingerprint = str(expectation["projection_fingerprint"])
    provider_fingerprint = "c" * 64
    grid = [int(value) for value in workload["grid"]]
    blocks = int(workload["wrapped_blocks"])
    total_steps = int(workload["total_steps"])
    routed_steps = bool(workload["expected_routed_steps"])
    normalized_branch_steps = branch_steps or {
        str(branch): tuple(int(step) for step in steps)
        for branch, steps in dict(workload["branch_steps"]).items()
    }
    tensor = {
        "shape": [1, grid[0] * grid[1] * grid[2], 24, 128],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    layer_projection_fingerprints = {
        layer: f"{layer + 1:064x}" for layer in range(blocks)
    }
    layer_source_keys: dict[int, list[str]] = {}
    for layer in range(blocks):
        if checkpoint_layout == "fastvideo_diffusers":
            prefix = f"blocks.{layer}.attn1.attn_impl.proj_l"
        else:
            prefix = (
                f"blocks.{layer}.self_attn.attn_op.local_attn.proj_l"
            )
        layer_source_keys[layer] = [f"{prefix}.weight", f"{prefix}.bias"]

    events: list[dict[str, object]] = []
    call_index = 0
    layer_call_indices = {layer: 0 for layer in range(blocks)}
    for branch, steps in normalized_branch_steps.items():
        for step in steps:
            for layer in range(blocks):
                dense_steps = int(profile["dense_steps"])
                sparse = not (
                    step < dense_steps
                    or step >= total_steps - dense_steps
                )
                event: dict[str, object] = {
                    "algorithm": kind,
                    "request_id": request_id,
                    "request_epoch": 7,
                    "request_local": True,
                    "branch": branch,
                    "step": step,
                    "step_index": step,
                    "total_steps": total_steps,
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
                    call_index += 1
                    layer_call_indices[layer] += 1
                    event.update(
                        {
                            "provider_family": provider_family,
                            "reference_provider_path": provider,
                            "provider_fingerprint": provider_fingerprint,
                            "injected_test_provider": False,
                            "reference_fastvideo_commit": commit,
                            "provider_source_commit": commit,
                            "provider_source_clean": True,
                            "provider_source_fingerprint": source_fingerprint,
                            "provider_source_root": source_root,
                            "provider_source_file": source_file,
                            "reference_parity_verified": True,
                            "checkpoint_layout": checkpoint_layout,
                            "projection_source_keys": layer_source_keys[layer],
                            "projection_weight_fingerprint": (
                                layer_projection_fingerprints[layer]
                            ),
                            "all_projection_weights_fingerprint": (
                                projection_fingerprint
                            ),
                            "layer_prefix": f"blocks.{layer}.attn1",
                            "call_index": call_index,
                            "layer_call_index": layer_call_indices[layer],
                            "provider_calls": 1,
                            "current_timestep": step,
                            "topk_ratio": profile["fastvideo_topk_ratio"],
                            "feature_map": profile["fastvideo_feature_map"],
                            "runtime_effective": True,
                        }
                    )
                events.append(event)

    sparse_events = [event for event in events if event["execution"] == "sparse"]
    scheduled_dense = len(events) - len(sparse_events)
    branches = {
        branch: {
            "last_step": steps[-1],
            "observed_steps": list(steps),
            "expected_steps": list(steps),
            "completed_steps": list(steps),
            "steps_contiguous": True,
            "routed_steps": routed_steps,
            "all_steps_complete": True,
            "layer_events": len(steps) * blocks,
            "missing_layers": {},
        }
        for branch, steps in normalized_branch_steps.items()
    }
    execution_counts = {"sparse": len(sparse_events)}
    if scheduled_dense:
        execution_counts["scheduled_dense"] = scheduled_dense
    receipt = {
        "kind": kind,
        "request_id": request_id,
        "request_epoch": 7,
        "request_local": True,
        "routed_steps": routed_steps,
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
        "effective_kernel": kind,
        "provider_path": provider,
        "provider_paths": [provider],
        "grid_size": grid,
        "branches": branches,
        "events": events,
        "event_count": len(events),
        "coverage": {
            "complete": True,
            "routed_steps": routed_steps,
            "expected_layers": list(range(blocks)),
            "expected_layer_count": blocks,
            "expected_calls": len(events),
            "observed_calls": len(events),
            "event_totals_match_counters": True,
            "provider_contract_complete": True,
            "request_local": True,
            "branches": branches,
            "execution_counts": execution_counts,
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
            "layer_projection_fingerprints": {
                str(layer): fingerprint
                for layer, fingerprint in layer_projection_fingerprints.items()
            },
            "projection_source_keys": {
                str(layer): keys for layer, keys in layer_source_keys.items()
            },
            "topk_ratio": profile["fastvideo_topk_ratio"],
            "feature_map": profile["fastvideo_feature_map"],
            "provider_calls": len(sparse_events),
            "receipt_count": len(sparse_events),
            "sparse_event_count": len(sparse_events),
            "events": sparse_events,
        },
    }
    return {
        "requested": {"approximate_attention": kind},
        "effective": {"approximate_attention_kernel": kind},
        "fallbacks": [],
        "runtime": {"approximate_attention": receipt},
    }


class _DualExpertAuditLeaf:
    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        return DenoiserOutput(sample=model_input.latents)

    def _reset_request_optimization_state(self) -> None:
        return None

    def runtime_optimization_report(self) -> dict[str, object]:
        return {
            "requested": {},
            "effective": {},
            "fallbacks": [],
            "quality_tier": "exact",
            "runtime": {},
        }


def _dual_expert_input(timestep: float, *, step_index: int) -> DenoiserInput:
    return DenoiserInput(
        latents=torch.zeros(1, 16, 2, 2, 2),
        timestep=torch.tensor(timestep),
        next_timestep=torch.tensor(0.0),
        conditioning={},
        step_index=step_index,
        total_steps=2,
    )


def _finalized_dual_expert_report() -> dict[str, object]:
    route_calls = {"high-noise": 2, "low-noise": 2}
    branch_calls = {"positive": 2, "negative": 2}
    events = [
        {"expert": "high-noise", "branch": branch, "step_index": 0}
        for branch in ("positive", "negative")
    ] + [
        {"expert": "low-noise", "branch": branch, "step_index": 1}
        for branch in ("positive", "negative")
    ]
    receipt = {
        "request_id": "request-current",
        "request_epoch": 7,
        "request_local": True,
        "finalized": True,
        "release_reason": "completed",
        "last_expert": "low-noise",
        "route_calls": route_calls,
        "branch_calls": branch_calls,
        "events": events,
    }
    expert_report = {
        "requested": {},
        "effective": {},
        "fallbacks": [],
        "runtime": {},
    }
    return {
        "requested": {},
        "effective": {"dual_expert_route_receipt": receipt},
        "fallbacks": [],
        "runtime": {
            "dual_expert": {
                "request_id": "request-current",
                "last_expert": "low-noise",
                "route_calls": route_calls,
                "route_receipt": receipt,
                "lifecycle": {
                    "live_requests": 0,
                    "receipt_snapshots": 1,
                    "max_receipt_snapshots": 32,
                },
                "experts": {
                    "high-noise": expert_report,
                    "low-noise": expert_report,
                },
            }
        },
    }


def _certifying_monitor(
    baseline_apps=None,
    *,
    allowed_pids=None,
    **kwargs,
):
    return benchmark._GpuIsolationMonitor(
        "1",
        [] if baseline_apps is None else baseline_apps,
        allowed_pids={100} if allowed_pids is None else allowed_pids,
        proof_mode="compute-exclusive",
        exclusive_compute_mode=True,
        idle_gate_passed=True,
        monitored_gpu_uuid="GPU-uuid",
        idle_gate_completed_monotonic=time.monotonic(),
        **kwargs,
    )


def _mark_complete(monitor) -> None:
    monitor.mark_cuda_initialization_started()
    monitor.set_cuda_affinity(
        {
            "passed": True,
            "one_rank_per_physical_gpu": True,
            "physical_gpu_uuids": ["GPU-uuid"],
        }
    )
    monitor.mark_output_encoding_completed()


def _idle_report(*, mode="Exclusive_Process", apps=None, passed=True) -> dict:
    return {
        "gpu_identifier": "1",
        "index": 1,
        "uuid": "GPU-uuid",
        "compute_mode": mode,
        "compute_apps": [] if apps is None else apps,
        "passed": passed,
    }


def test_parser_exposes_repeated_performance_and_quality_gates(tmp_path) -> None:
    args = _args(
        tmp_path,
        "--warmup-runs",
        "2",
        "--measured-runs",
        "5",
        "--min-video-psnr",
        "31.5",
        "--min-video-ssim",
        "0.97",
        "--max-video-lpips",
        "0.08",
        "--fail-on-fallback",
    )
    assert args.warmup_runs == 2
    assert args.measured_runs == 5
    assert args.min_video_psnr == 31.5
    assert args.min_video_ssim == 0.97
    assert args.max_video_lpips == 0.08
    assert args.fail_on_unaccounted is True
    assert args.fail_on_fallback is True
    assert args.require_idle_gpus is True
    assert args.max_idle_memory_mib == 128.0
    assert args.max_idle_utilization_percent == 0.0
    assert args.gpu_isolation_proof == "compute-exclusive"
    assert args.gpu_isolation_poll_interval_seconds == 0.1
    assert args.fused_rope is False
    assert args.rope_precision == "fp64"
    benchmark._validate_args(args)


def test_parser_accepts_complete_inline_and_file_vmoba_profiles(tmp_path) -> None:
    profile = _vmoba_profile()
    inline = _args(
        tmp_path,
        "--approximate-attention",
        "vmoba",
        "--approximate-attention-profile-json",
        json.dumps(profile),
    )

    request = benchmark._approximate_attention_request(inline)

    assert request == {"kind": "vmoba", **profile}
    assert benchmark._expected_requests(inline)["denoiser"][
        "approximate_attention"
    ] == "vmoba"
    assert "approximate_attention=vmoba" in benchmark._candidate_reference_reasons(
        inline
    )

    profile_path = tmp_path / "vmoba.json"
    profile_path.write_text(json.dumps({"kind": "vmoba", **profile}))
    from_file = _args(
        tmp_path,
        "--approximate-attention",
        "vmoba",
        "--approximate-attention-profile",
        str(profile_path),
    )
    assert benchmark._approximate_attention_request(from_file) == {
        "kind": "vmoba",
        **profile,
    }


@pytest.mark.parametrize(
    ("extra", "match"),
    (
        (("--approximate-attention", "vmoba"), "complete grid-specific profile"),
        (
            (
                "--approximate-attention",
                "vmoba",
                "--approximate-attention-profile-json",
                "{}",
            ),
            "missing fields",
        ),
        (
            (
                "--approximate-attention-profile-json",
                "{}",
            ),
            "requires --approximate-attention",
        ),
        (
            (
                "--approximate-attention",
                "vmoba",
                "--approximate-attention-profile-json",
                "[]",
            ),
            "JSON object",
        ),
    ),
)
def test_vmoba_profile_boundary_fails_closed(tmp_path, extra, match) -> None:
    args = _args(tmp_path, *extra)

    with pytest.raises(ValueError, match=match):
        benchmark._validate_args(args)


def test_vmoba_profile_rejects_kind_conflict_and_duplicate_sources(tmp_path) -> None:
    profile = _vmoba_profile()
    conflict = _args(
        tmp_path,
        "--approximate-attention",
        "vmoba",
        "--approximate-attention-profile-json",
        json.dumps({"kind": "vsa", **profile}),
    )
    with pytest.raises(ValueError, match="kind conflicts"):
        benchmark._approximate_attention_request(conflict)

    path = tmp_path / "vmoba.json"
    path.write_text(json.dumps(profile))
    duplicate = _args(
        tmp_path,
        "--approximate-attention",
        "vmoba",
        "--approximate-attention-profile-json",
        json.dumps(profile),
        "--approximate-attention-profile",
        str(path),
    )
    with pytest.raises(ValueError, match="only one"):
        benchmark._approximate_attention_request(duplicate)


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
    tmp_path: Path,
    updates: dict[str, object],
    match: str,
) -> None:
    profile = {**_vmoba_profile(), **updates}
    args = _args(
        tmp_path,
        "--approximate-attention",
        "vmoba",
        "--approximate-attention-profile-json",
        json.dumps(profile),
    )

    with pytest.raises(ValueError, match=match):
        benchmark._approximate_attention_request(args)


@pytest.mark.parametrize("kind", benchmark._LIGHTX2V_SPARSE_KINDS)
def test_parser_exposes_canonical_lightx2v_sparse_kinds(
    tmp_path: Path,
    kind: str,
) -> None:
    args = _args(tmp_path, "--approximate-attention", kind)

    request = benchmark._approximate_attention_request(args)
    workload = benchmark._expected_approximate_workload(args)

    assert request is not None
    assert request["kind"] == kind
    assert workload is not None
    assert workload["kind"] == kind
    assert workload["lightx2v"]["commit"] == (
        "6fb7c1362b89d4908a9ea197bac4fbd7482ee2d5"
    )
    assert workload["provider_paths"] == [
        workload["lightx2v"]["provider_symbol"]
    ]


@pytest.mark.parametrize(
    ("kind", "profile", "expected_topk", "expected_feature_map"),
    (
        ("fastvideo_sla", {}, 0.1, "softmax"),
        (
            "fastvideo_sagesla",
            {
                "fastvideo_topk_ratio": 0.35,
                "fastvideo_feature_map": "elu",
            },
            0.35,
            "elu",
        ),
    ),
)
def test_parser_exposes_fastvideo_learned_sla_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    profile: dict[str, object],
    expected_topk: float,
    expected_feature_map: str,
) -> None:
    _stub_fastvideo_sla_preflight(monkeypatch)
    extra = (
        (
            "--approximate-attention-profile-json",
            json.dumps(profile),
        )
        if profile
        else ()
    )
    args = _args(
        tmp_path,
        "--approximate-attention",
        kind,
        *extra,
    )

    request = benchmark._approximate_attention_request(args)
    workload = benchmark._expected_approximate_workload(args)

    assert request == {"kind": kind, **profile}
    assert workload is not None
    assert workload["kind"] == kind
    assert workload["profile"]["fastvideo_topk_ratio"] == expected_topk
    assert (
        workload["profile"]["fastvideo_feature_map"]
        == expected_feature_map
    )
    assert workload["fastvideo_sla"]["commit"] == (
        "1b2b2a0161bc6b3b80158d1fa6380a051c6530c7"
    )
    assert workload["fastvideo_sla"]["checkpoint_layout"] == (
        "fastvideo_diffusers"
    )
    assert workload["fastvideo_sla"]["projection_fingerprint"] == "b" * 64
    assert workload["provider_paths"] == [
        workload["fastvideo_sla"]["provider_path"]
    ]


def test_fastvideo_sla_profile_rejects_lightx2v_only_fields(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--approximate-attention",
        "fastvideo_sla",
        "--approximate-attention-profile-json",
        json.dumps({"lightx2v_operator": "triton"}),
    )

    with pytest.raises(ValueError, match="unknown.*lightx2v_operator"):
        benchmark._approximate_attention_request(args)


def test_fastvideo_sla_expectation_uses_pinned_runtime_source_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations import (
        sparse_linear_attention,
    )

    observed = []
    expected = _fastvideo_sla_source_expectation("fastvideo_sla")

    def resolve(config):
        observed.append(config)
        return expected

    monkeypatch.setattr(
        sparse_linear_attention,
        "fastvideo_sla_runtime_expectation",
        resolve,
    )
    config = SimpleNamespace(
        kind="fastvideo_sla",
        fastvideo_topk_ratio=0.2,
        fastvideo_feature_map="elu",
    )

    actual = benchmark._fastvideo_sla_expectation(config)

    assert actual == expected
    assert len(observed) == 1
    assert observed[0].kind == "fastvideo_sla"
    assert observed[0].topk_ratio == 0.2
    assert observed[0].feature_map == "elu"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("commit", "0" * 40),
        ("source_fingerprint", "not-sha256"),
        ("provider_path", ""),
    ),
)
def test_fastvideo_sla_expectation_rejects_untrusted_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    from worldfoundry.base_models.diffusion_model.optimizations import (
        sparse_linear_attention,
    )

    expectation = _fastvideo_sla_source_expectation("fastvideo_sla")
    expectation[field] = value
    monkeypatch.setattr(
        sparse_linear_attention,
        "fastvideo_sla_runtime_expectation",
        lambda _config: expectation,
    )
    config = SimpleNamespace(
        kind="fastvideo_sla",
        fastvideo_topk_ratio=0.1,
        fastvideo_feature_map="softmax",
    )

    with pytest.raises(RuntimeError, match="mismatch|incomplete"):
        benchmark._fastvideo_sla_expectation(config)


@pytest.mark.parametrize(
    "layout",
    ("fastvideo_diffusers", "turbodiffusion_original"),
)
def test_fastvideo_sla_checkpoint_preflight_hashes_exact_projection_set(
    tmp_path: Path,
    layout: str,
) -> None:
    from safetensors.torch import save_file

    checkpoint = tmp_path / "learned-sla.safetensors"
    state: dict[str, torch.Tensor] = {}
    for layer in range(30):
        if layout == "fastvideo_diffusers":
            prefix = f"blocks.{layer}.attn1.attn_impl.proj_l"
        else:
            prefix = (
                f"blocks.{layer}.self_attn.attn_op.local_attn.proj_l"
            )
        state[f"{prefix}.weight"] = torch.full(
            (128, 128),
            float(layer + 1),
        )
        state[f"{prefix}.bias"] = torch.full((128,), float(layer))
    save_file(state, checkpoint)

    expectation = benchmark._fastvideo_sla_checkpoint_expectation(
        checkpoint,
        model_id="wan2.2-ti2v-5b",
    )

    assert expectation["checkpoint_layout"] == layout
    assert len(expectation["projection_fingerprint"]) == 64
    assert set(expectation["projection_fingerprint"]) <= set(
        "0123456789abcdef"
    )


def test_fastvideo_sla_checkpoint_preflight_rejects_dense_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_fastvideo_sla_expectation",
        lambda config: _fastvideo_sla_source_expectation(str(config.kind)),
    )
    args = _args(
        tmp_path,
        "--approximate-attention",
        "fastvideo_sla",
    )

    with pytest.raises(RuntimeError, match="safetensors checkpoint"):
        benchmark._expected_approximate_workload(args)

    with pytest.raises(RuntimeError, match="safetensors checkpoint"):
        benchmark._validate_args(args)


def test_sta_default_grid_fails_fast_but_tuned_grid_is_accepted(
    tmp_path: Path,
) -> None:
    default_args = _args(tmp_path, "--approximate-attention", "sta")
    with pytest.raises(ValueError, match="no tuned provider plan"):
        benchmark._approximate_attention_request(default_args)

    tuned = _args(
        tmp_path,
        "--approximate-attention",
        "sta",
        "--frames",
        "69",
        "--height",
        "1536",
        "--width",
        "2560",
    )
    assert benchmark._approximate_attention_request(tuned) == {"kind": "sta"}


def test_nbhd_frame_profile_fails_before_cuda_on_grid_mismatch(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--approximate-attention",
        "nbhd",
        "--approximate-attention-profile-json",
        json.dumps({"attnmap_frame_num": 30}),
    )

    with pytest.raises(ValueError, match="attnmap_frame_num"):
        benchmark._approximate_attention_request(args)


@pytest.mark.parametrize(
    ("rank", "branch"),
    ((0, "positive"), (1, "positive"), (2, "negative"), (3, "negative")),
)
def test_cfg_parallel_expected_sparse_workload_uses_replica_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    branch: str,
) -> None:
    args = _args(
        tmp_path,
        "--model-id",
        "wan2.2-t2v-a14b",
        "--approximate-attention",
        "dynamic_sparse",
        "--sp-degree",
        "2",
        "--cfg-degree",
        "2",
    )
    monkeypatch.setenv("RANK", str(rank))

    workload = benchmark._expected_approximate_workload(args)

    assert workload is not None
    assert set(workload["branch_steps"]) == {branch}
    assert workload["expected_routed_steps"] is True
    assert workload["dual_expert"] is True


def test_visible_gpu_identifier_maps_local_rank(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")

    assert benchmark._visible_gpu_identifier(0) == "3"
    assert benchmark._visible_gpu_identifier(1) == "1"

    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        benchmark._visible_gpu_identifier(2)


def test_gpu_idle_report_requires_no_apps_and_thresholds(monkeypatch) -> None:
    def fake_nvidia_smi(*arguments: str) -> str:
        query = next(item for item in arguments if item.startswith("--query"))
        if query.startswith("--query-gpu"):
            return "1, GPU-uuid, NVIDIA H100, 4, 0, P0, 70.0, Exclusive_Process"
        return ""

    monkeypatch.setattr(benchmark, "_nvidia_smi", fake_nvidia_smi)

    report = benchmark._gpu_idle_report(
        "1",
        max_memory_mib=128.0,
        max_utilization_percent=0.0,
    )

    assert report["passed"] is True
    assert report["compute_apps"] == []
    assert report["normalized_compute_mode"] == "exclusive_process"


def test_compute_exclusive_preconditions_reject_default_compute_mode(tmp_path) -> None:
    args = _args(tmp_path)

    preconditions = benchmark._gpu_isolation_preconditions(
        args,
        _idle_report(mode="Default"),
    )

    assert preconditions["run_allowed"] is False
    assert preconditions["certifying_preconditions_passed"] is False
    assert preconditions["checks"]["exclusive_process_compute_mode"] is False


def test_compute_exclusive_requires_empty_idle_gate(tmp_path) -> None:
    args = _args(tmp_path)

    preconditions = benchmark._gpu_isolation_preconditions(
        args,
        _idle_report(
            apps=[{"pid": 77, "used_memory_mib": 1.0}],
            passed=False,
        ),
    )

    assert preconditions["run_allowed"] is False
    assert preconditions["checks"]["empty_compute_app_gate"] is False


def test_compute_exclusive_validation_rejects_disabled_idle_gate(tmp_path) -> None:
    args = _args(tmp_path, "--no-require-idle-gpus")

    with pytest.raises(ValueError, match="requires --require-idle-gpus"):
        benchmark._validate_args(args)


def test_bounded_polling_preconditions_are_diagnostic_only(tmp_path) -> None:
    args = _args(
        tmp_path,
        "--gpu-isolation-proof",
        "bounded-polling",
        "--no-require-idle-gpus",
    )

    benchmark._validate_args(args)
    preconditions = benchmark._gpu_isolation_preconditions(
        args,
        _idle_report(mode="Default", passed=False),
    )

    assert preconditions["run_allowed"] is True
    assert preconditions["certifying_preconditions_passed"] is False


def test_cuda_device_affinity_rejects_physical_uuid_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    evidence = benchmark._cuda_device_affinity_evidence(
        rank=0,
        local_rank=0,
        idle_gpu_report=_idle_report(),
        current_cuda_uuid="GPU-other",
    )

    assert evidence["checks"]["visible_token_selects_monitored_gpu"] is True
    assert evidence["checks"]["current_cuda_uuid_matches_monitored_gpu"] is False
    assert evidence["passed"] is False


def test_rank_gpu_affinity_rejects_duplicate_physical_uuid() -> None:
    evidence = [
        {
            "rank": rank,
            "passed": True,
            "monitored_gpu_uuid": "GPU-shared",
        }
        for rank in range(2)
    ]

    report = benchmark._rank_gpu_affinity_report(
        evidence,
        expected_world_size=2,
    )

    assert report["one_rank_per_physical_gpu"] is False
    assert report["passed"] is False
    assert any("same physical GPU UUID" in issue for issue in report["issues"])


def test_generation_isolation_monitor_detects_added_pid(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [
            {"pid": 900, "used_memory_mib": 1000.0},
            {"pid": 200, "used_memory_mib": 2000.0},
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor(
        "1",
        [{"pid": 900, "used_memory_mib": 1000.0}],
        allowed_pids={100},
    )

    monitor._sample()
    report = monitor.stop()

    assert report["contaminated"] is True
    assert report["added_compute_apps"] == [
        {"pid": 200, "used_memory_mib": 2000.0},
        {"pid": 900, "used_memory_mib": 1000.0},
    ]


def test_generation_isolation_does_not_grandfather_external_baseline_pid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 200, "used_memory_mib": 2000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor(
        "1",
        [{"pid": 200, "used_memory_mib": 2000.0}],
        allowed_pids={100},
    )

    monitor.start()
    report = monitor.stop()

    assert report["allowed_compute_pids"] == []
    assert report["contaminated"] is True
    assert report["added_compute_apps"] == [
        {"pid": 200, "used_memory_mib": 2000.0}
    ]


def test_generation_isolation_binds_host_pid_then_rejects_extra_pid(
    monkeypatch,
) -> None:
    snapshots = iter(
        (
            [{"pid": 900, "used_memory_mib": 1000.0}],
            [
                {"pid": 900, "used_memory_mib": 1000.0},
                {"pid": 901, "used_memory_mib": 2000.0},
            ],
            [
                {"pid": 900, "used_memory_mib": 1000.0},
                {"pid": 901, "used_memory_mib": 2000.0},
            ],
        )
    )
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: next(snapshots),
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor(
        "1",
        [],
        allowed_pids={100},
    )

    monitor._sample()
    monitor._sample()
    report = monitor.stop()

    assert report["bound_host_compute_pids"] == [900]
    assert report["added_compute_apps"] == [
        {"pid": 901, "used_memory_mib": 2000.0}
    ]
    assert report["contaminated"] is True


def test_generation_isolation_does_not_reuse_host_pid_binding_slot(
    monkeypatch,
) -> None:
    """One local CUDA client cannot authorize sequential host-namespace PIDs."""

    snapshots = iter(
        (
            [{"pid": 900, "used_memory_mib": 1000.0}],
            [{"pid": 901, "used_memory_mib": 2000.0}],
            [{"pid": 901, "used_memory_mib": 2000.0}],
        )
    )
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: next(snapshots),
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor(
        "1",
        [],
        allowed_pids={100},
    )

    monitor._sample()
    monitor._sample()
    report = monitor.stop()

    assert report["bound_host_compute_pids"] == [900]
    assert report["added_compute_apps"] == [
        {"pid": 901, "used_memory_mib": 2000.0}
    ]
    assert report["contaminated"] is True


def test_compute_exclusive_temporal_binding_certifies_hidden_nspid(
    monkeypatch,
) -> None:
    snapshots = iter(
        (
            [],
            [{"pid": 900, "used_memory_mib": 1000.0}],
            [{"pid": 900, "used_memory_mib": 1000.0}],
        )
    )
    monkeypatch.setattr(benchmark, "_gpu_compute_apps", lambda _gpu: next(snapshots))
    monkeypatch.setattr(benchmark, "_process_identity", lambda pid: _identity(pid))
    monkeypatch.setattr(
        benchmark,
        "_gpu_device_client_pids",
        lambda _uuid, candidates: set(candidates),
    )
    monitor = _certifying_monitor()

    monitor._sample()
    monitor.mark_cuda_initialization_started()
    monitor.set_cuda_affinity(
        {"passed": True, "one_rank_per_physical_gpu": True}
    )
    monitor._sample()
    monitor.mark_output_encoding_completed()
    report = monitor.stop()

    assert report["namespace_mapping_proven"] is False
    assert report["host_pid_bindings"] == [
        {
            "host_pid": 900,
            "local_pid": 100,
            "local_start_time_ticks": 1234,
            "proof": "exclusive-process-temporal-binding",
        }
    ]
    assert report["certifying"] is True
    assert benchmark._gpu_isolation_audit_issues(report) == []


def test_compute_exclusive_temporal_binding_is_never_reused(monkeypatch) -> None:
    snapshots = iter(
        (
            [],
            [{"pid": 900, "used_memory_mib": 1000.0}],
            [{"pid": 901, "used_memory_mib": 1000.0}],
            [{"pid": 901, "used_memory_mib": 1000.0}],
        )
    )
    monkeypatch.setattr(benchmark, "_gpu_compute_apps", lambda _gpu: next(snapshots))
    monkeypatch.setattr(benchmark, "_process_identity", lambda pid: _identity(pid))
    monkeypatch.setattr(
        benchmark,
        "_gpu_device_client_pids",
        lambda _uuid, candidates: set(candidates),
    )
    monitor = _certifying_monitor()

    monitor._sample()
    monitor.mark_cuda_initialization_started()
    monitor.set_cuda_affinity(
        {"passed": True, "one_rank_per_physical_gpu": True}
    )
    monitor._sample()
    monitor._sample()
    monitor.mark_output_encoding_completed()
    report = monitor.stop()

    assert report["bound_host_compute_pids"] == [900]
    assert report["retired_host_compute_pids"] == [900]
    assert report["added_compute_apps"] == [
        {"pid": 901, "used_memory_mib": 1000.0}
    ]
    assert any("disappeared" in error for error in report["monitor_errors"])
    assert report["certifying"] is False


def test_generation_isolation_rejects_bare_pid_numeric_collision(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 100, "used_memory_mib": 2000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor("1", [], allowed_pids={100})

    monitor._sample()
    report = monitor.stop()

    assert report["allowed_local_pids"] == [100]
    assert report["allowed_compute_pids"] == []
    assert report["added_compute_apps"] == [
        {"pid": 100, "used_memory_mib": 2000.0}
    ]
    assert report["contaminated"] is True


def test_generation_isolation_rejects_single_namespace_nspid_as_unproven(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 100, "used_memory_mib": 1000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid),
    )
    monitor = benchmark._GpuIsolationMonitor("1", [], allowed_pids={100})

    monitor._sample()
    report = monitor.stop()

    assert report["namespace_mapping_proven"] is False
    assert any("unproven NSpid" in error for error in report["namespace_mapping_errors"])
    assert report["added_compute_apps"] == [
        {"pid": 100, "used_memory_mib": 1000.0}
    ]
    assert report["contaminated"] is True
    assert report["certifying"] is False


def test_generation_isolation_rejects_changed_pid_identity(monkeypatch) -> None:
    identities = iter(
        (
            _identity(100, host_pid=900, start_time_ticks=1234),
            _identity(100, host_pid=900, start_time_ticks=5678),
        )
    )
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 900, "used_memory_mib": 1000.0}],
    )
    monkeypatch.setattr(benchmark, "_process_identity", lambda _pid: next(identities))
    monitor = benchmark._GpuIsolationMonitor("1", [], allowed_pids={100})

    monitor._sample()
    report = monitor.stop()

    assert report["namespace_mapping_proven"] is False
    assert any("changed identity" in error for error in report["namespace_mapping_errors"])
    assert report["contaminated"] is True
    assert report["certifying"] is False


def test_generation_isolation_fails_closed_on_query_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor("1", [], allowed_pids={100})

    monitor._sample()
    report = monitor.stop()

    assert report["namespace_mapping_proven"] is False
    assert report["monitor_error_count"] == 2
    assert any("TimeoutExpired" in error for error in report["monitor_errors"])
    assert report["contaminated"] is True
    assert report["certifying"] is False


@pytest.mark.parametrize(
    "output",
    (
        "malformed",
        "12, N/A",
        "12, 1, unexpected",
        "12, 1\n12, 2",
    ),
)
def test_gpu_compute_apps_rejects_malformed_output(monkeypatch, output: str) -> None:
    monkeypatch.setattr(benchmark, "_nvidia_smi", lambda *_arguments: output)

    with pytest.raises(RuntimeError, match="malformed|duplicate"):
        benchmark._gpu_compute_apps("1")


def test_generation_isolation_fails_closed_when_sampler_thread_is_alive(
    monkeypatch,
) -> None:
    class StuckThread:
        def join(self, timeout: float) -> None:
            assert timeout == pytest.approx(0.01)

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 900, "used_memory_mib": 1000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor(
        "1",
        [],
        allowed_pids={100},
        stop_timeout_s=0.01,
    )
    monitor._thread = StuckThread()  # type: ignore[assignment]

    report = monitor.stop()

    assert report["thread_stopped"] is False
    assert any("did not stop" in error for error in report["monitor_errors"])
    assert report["contaminated"] is True
    assert report["certifying"] is False


def test_bounded_polling_is_always_non_certifying(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 900, "used_memory_mib": 1000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = benchmark._GpuIsolationMonitor(
        "1",
        [],
        allowed_pids={100},
        proof_mode="bounded-polling",
        exclusive_compute_mode=True,
        idle_gate_passed=True,
        monitored_gpu_uuid="GPU-uuid",
        idle_gate_completed_monotonic=time.monotonic(),
    )

    monitor._sample()
    _mark_complete(monitor)
    report = monitor.stop()

    assert report["identity_proof"] == "bounded-polling-diagnostic"
    assert report["contaminated"] is False
    assert report["certifying"] is False
    assert any("diagnostic-only" in reason for reason in report["non_certifying_reasons"])
    assert any(
        "diagnostic-only" in issue
        for issue in benchmark._gpu_isolation_audit_issues(report)
    )


def test_generation_isolation_requires_output_completion_marker(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 900, "used_memory_mib": 1000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = _certifying_monitor()

    monitor._sample()
    monitor.mark_cuda_initialization_started()
    monitor.set_cuda_affinity(
        {"passed": True, "one_rank_per_physical_gpu": True}
    )
    report = monitor.stop()

    assert report["window"]["full_window_covered"] is False
    assert report["certifying"] is False


@pytest.mark.parametrize(
    ("timings", "window_ended", "failed_check"),
    (
        (
            [
                {
                    "sample_started_monotonic_seconds": 0.0,
                    "query_started_monotonic_seconds": 0.0,
                    "query_completed_monotonic_seconds": 0.01,
                    "sample_completed_monotonic_seconds": 0.01,
                    "query_latency_seconds": 0.01,
                },
                {
                    "sample_started_monotonic_seconds": 0.1,
                    "query_started_monotonic_seconds": 0.1,
                    "query_completed_monotonic_seconds": 0.11,
                    "sample_completed_monotonic_seconds": 0.11,
                    "query_latency_seconds": 0.01,
                },
            ],
            2.0,
            "sample_gap_within_certifying_limit",
        ),
        (
            [
                {
                    "sample_started_monotonic_seconds": 0.0,
                    "query_started_monotonic_seconds": 0.0,
                    "query_completed_monotonic_seconds": 1.5,
                    "sample_completed_monotonic_seconds": 1.5,
                    "query_latency_seconds": 1.5,
                },
                {
                    "sample_started_monotonic_seconds": 1.6,
                    "query_started_monotonic_seconds": 1.6,
                    "query_completed_monotonic_seconds": 1.61,
                    "sample_completed_monotonic_seconds": 1.61,
                    "query_latency_seconds": 0.01,
                },
            ],
            1.61,
            "query_latency_within_certifying_limit",
        ),
    ),
)
def test_polling_coverage_rejects_excessive_gap_or_query_latency(
    timings,
    window_ended,
    failed_check,
) -> None:
    coverage = benchmark._polling_coverage(
        timings,
        window_started=0.0,
        window_ended=window_ended,
        interval_s=0.1,
    )

    assert coverage[failed_check] is False
    assert coverage["bounded_polling_complete"] is False


def test_generation_isolation_clean_audit_records_timing_window(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_gpu_compute_apps",
        lambda _gpu: [{"pid": 900, "used_memory_mib": 1000.0}],
    )
    monkeypatch.setattr(
        benchmark,
        "_process_identity",
        lambda pid: _identity(pid, host_pid=900),
    )
    monitor = _certifying_monitor(interval_s=0.1)

    monitor._sample()
    _mark_complete(monitor)
    report = monitor.stop()

    assert report["audit_version"] == benchmark.GPU_ISOLATION_AUDIT_VERSION
    assert report["namespace_mapping_proven"] is True
    assert report["thread_stopped"] is True
    assert report["samples"] == 2
    assert len(report["sample_timings"]) == 2
    assert report["max_query_latency_seconds"] >= 0.0
    assert report["max_sample_start_gap_seconds"] >= 0.0
    assert report["bounded_polling_complete"] is True
    assert report["window"]["duration_seconds"] >= 0.0
    assert report["window"]["full_window_covered"] is True
    assert report["contaminated"] is False
    assert report["certifying"] is True
    assert benchmark._gpu_isolation_audit_issues(report) == []


def test_failure_report_retains_generation_isolation(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    benchmark._write_failure(
        output_dir,
        rank=0,
        stage="measured-0",
        failures=["rank 0: OutOfMemoryError"],
    )
    generation = {
        "samples": 10,
        "added_compute_apps": [{"pid": 200, "used_memory_mib": 2000.0}],
        "monitor_errors": [],
        "contaminated": True,
    }

    benchmark._write_failure_isolation(
        output_dir,
        rank=0,
        idle_gpu_report={"passed": True},
        generation_isolation=generation,
    )

    failure = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["gpu_isolation"]["pre_load_idle"]["passed"] is True
    assert failure["gpu_isolation"]["generation"] == generation
    audit = json.loads(
        (output_dir / "failure-isolation-rank-0.json").read_text(encoding="utf-8")
    )
    assert audit == failure["gpu_isolation"]


def test_cfg_parallel_is_part_of_expected_runtime_audit(tmp_path) -> None:
    args = _args(tmp_path, "--cfg-degree", "2")
    expected = benchmark._expected_requests(args)
    assert expected["runner"] == {"cfg_parallel": 2}
    assert expected["denoiser"]["attention"] == "flash_attention_3"
    assert expected["denoiser"]["offload"] == "none"


def test_async_block_offload_is_public_and_part_of_runtime_audit(tmp_path) -> None:
    args = _args(tmp_path, "--offload-mode", "block")

    assert args.offload_mode == "block"
    assert benchmark._expected_requests(args)["denoiser"]["offload"] == "block"
    assert "offload=async-double-buffer" in (
        benchmark._candidate_reference_reasons(args)
    )


@pytest.mark.parametrize(
    "conflicting_args",
    [
        ("--quantization", "fp8"),
        ("--sp-degree", "2"),
    ],
)
def test_async_block_offload_rejects_uncertified_compositions(
    tmp_path,
    conflicting_args,
) -> None:
    args = _args(tmp_path, "--offload-mode", "block", *conflicting_args)

    with pytest.raises(ValueError, match="block offload cannot be combined"):
        benchmark._validate_args(args)


def test_parser_exposes_gguf_runtime_storage_mode(tmp_path) -> None:
    args = _args(tmp_path, "--quantization", "gguf")

    assert args.quantization == "gguf"
    assert benchmark._expected_requests(args)["denoiser"]["quantization"] == "gguf"


def test_quantization_request_exposes_qualified_excludes_and_storage_policy(
    tmp_path,
) -> None:
    args = _args(
        tmp_path,
        "--quantization",
        "int8",
        "--quantization-exclude",
        "time_embedding",
        "--quantization-exclude",
        "blocks.0.",
        "--quantization-min-features",
        "2048",
        "--no-quantization-keep-dense-fallback",
    )

    assert benchmark._quantization_request(args) == {
        "mode": "int8",
        "exclude": ["time_embedding", "blocks.0."],
        "options": {
            "min_features": 2048,
            "keep_dense_fallback": False,
        },
    }


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ((), 1),
        (("--sp-degree", "4"), 4),
        (("--cfg-degree", "2"), 2),
        (("--vae-parallel-degree", "4"), 4),
        (
            (
                "--sp-degree",
                "2",
                "--cfg-degree",
                "2",
                "--vae-parallel-degree",
                "4",
            ),
            4,
        ),
    ],
)
def test_parallel_configuration_resolves_required_world_size(
    tmp_path, extra, expected
) -> None:
    assert benchmark._expected_world_size(_args(tmp_path, *extra)) == expected


def test_parallel_configuration_rejects_partial_vae_mesh(tmp_path) -> None:
    args = _args(
        tmp_path,
        "--sp-degree",
        "2",
        "--cfg-degree",
        "2",
        "--vae-parallel-degree",
        "2",
    )
    with pytest.raises(ValueError, match="complete SP×CFG world size"):
        benchmark._expected_world_size(args)


@pytest.mark.parametrize(
    "candidate_args",
    (
        ("--sp-degree", "2"),
        ("--cfg-degree", "2"),
        ("--vae-parallel-degree", "2"),
        ("--vae-spatial-tiling",),
        ("--vae-temporal-chunk-size", "4"),
        ("--vae-decode-autocast", "fp16"),
        ("--quantization", "int8"),
        ("--offload-mode", "block"),
        ("--fused-rope", "--rope-precision", "fp32"),
        ("--fused-residual-adaln",),
    ),
)
def test_candidate_optimizations_require_latent_and_video_references(
    tmp_path,
    candidate_args,
) -> None:
    args = _args(tmp_path, *candidate_args)

    with pytest.raises(ValueError, match="require both frozen.*reference-latents.*reference-video"):
        benchmark._validate_args(args)


def test_candidate_optimization_accepts_both_reference_artifacts(tmp_path) -> None:
    latents = tmp_path / "reference.pt"
    video = tmp_path / "reference.mp4"
    latents.touch()
    video.touch()
    args = _args(
        tmp_path,
        "--sp-degree",
        "2",
        "--reference-latents",
        str(latents),
        "--reference-video",
        str(video),
    )

    benchmark._validate_args(args)


@pytest.mark.parametrize("override", [None, {"calls": 0}, {"accelerated_calls": 0}, {"fallback_calls": 1}, {"blocks": False}, {"calls": "30"}])
def test_residual_adaln_requires_real_accelerated_execution(tmp_path, override):
    args = _args(tmp_path, "--fused-residual-adaln")
    assert benchmark._expected_requests(args)["denoiser"]["fused_residual_adaln"] is True
    counters = {"blocks": 30, "calls": 60, "accelerated_calls": 60, "fallback_calls": 0}
    counters.update(override or {})
    report = {"requested": {"fused_residual_adaln": True},
              "effective": {"fused_residual_adaln_runtime": counters},
              "runtime": {"fused_residual_adaln": counters}, "fallbacks": []}
    audit = benchmark._optimization_audit(
        {"denoiser": report}, expected_requests={"denoiser": {"fused_residual_adaln": True}},
    )
    assert audit["passed"] is (override is None)


def test_vae_parallel_audit_accounts_for_required_spatial_tiling(tmp_path) -> None:
    args = _args(tmp_path, "--vae-parallel-degree", "4")

    expected = benchmark._expected_requests(args)

    assert expected["decoder"]["vae_parallel_degree"] == 4
    assert expected["decoder"]["vae_spatial_tiling"] is True


def test_preview_decoder_is_part_of_expected_runtime_audit(tmp_path) -> None:
    preview = tmp_path / "taew2_2.safetensors"
    preview.touch()
    reference_latents = tmp_path / "reference.pt"
    reference_video = tmp_path / "reference.mp4"
    reference_latents.touch()
    reference_video.touch()
    args = _args(
        tmp_path,
        "--vae-preview-decoder-path",
        str(preview),
        "--reference-latents",
        str(reference_latents),
        "--reference-video",
        str(reference_video),
    )

    benchmark._validate_args(args)

    expected = benchmark._expected_requests(args)
    assert expected["decoder"]["vae_preview_decoder_path"] == str(preview)


def test_validation_rejects_missing_preview_decoder_checkpoint(tmp_path) -> None:
    missing = tmp_path / "missing-taew2_2.pth"
    args = _args(tmp_path, "--vae-preview-decoder-path", str(missing))

    with pytest.raises(FileNotFoundError, match="missing-taew2_2"):
        benchmark._validate_args(args)


@pytest.mark.parametrize(
    "conflicting_args",
    [
        ("--vae-spatial-tiling",),
        ("--vae-temporal-chunk-size", "4"),
        ("--vae-parallel-degree", "2"),
    ],
)
def test_validation_rejects_preview_with_official_vae_acceleration(
    tmp_path, conflicting_args
) -> None:
    preview = tmp_path / "taew2_2.pth"
    preview.touch()
    args = _args(
        tmp_path,
        "--vae-preview-decoder-path",
        str(preview),
        *conflicting_args,
    )

    with pytest.raises(ValueError, match="preview cannot be combined"):
        benchmark._validate_args(args)


def test_validation_rejects_i2v_without_an_image(tmp_path) -> None:
    args = _args(tmp_path, "--model-id", "wan2.2-i2v-a14b")
    with pytest.raises(ValueError, match="requires --image"):
        benchmark._validate_args(args)


def test_timing_summary_reports_repeat_distribution() -> None:
    report = benchmark._timing_summary([3.0, 1.0, 2.0])
    assert report == {
        "count": 3,
        "values_seconds": [3.0, 1.0, 2.0],
        "min_seconds": 1.0,
        "median_seconds": 2.0,
        "mean_seconds": 2.0,
        "p90_seconds": pytest.approx(2.8),
        "max_seconds": 3.0,
        "stdev_seconds": 1.0,
    }


def test_latent_metrics_clamps_roundoff_cosine_to_valid_range() -> None:
    latents = torch.full((8192,), 0.125, dtype=torch.float32)

    metrics = benchmark._latent_metrics(latents, latents.clone())

    assert metrics["cosine"] == 1.0


def _fingerprint(source: str = "source", workload: str = "workload") -> dict:
    return {
        "version": benchmark.FINGERPRINT_VERSION,
        "source": {"sha256": source},
        "checkpoint": {"sha256": "checkpoint"},
        "workload": {"sha256": workload},
    }


def _write_reference_report(tmp_path, fingerprint: dict) -> tuple:
    latents = tmp_path / "final-latents.pt"
    video = tmp_path / "output.mp4"
    latents.write_bytes(b"current latent artifact")
    video.write_bytes(b"current video artifact")
    report = {
        "passed": True,
        "execution_fingerprint": fingerprint,
        "artifacts": {
            "latents": {
                "path": str(latents.resolve()),
                "sha256": benchmark._sha256_file(latents),
            },
            "video": {
                "path": str(video.resolve()),
                "sha256": benchmark._sha256_file(video),
            },
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return latents, video


def test_reference_provenance_accepts_same_source_workload_and_artifacts(tmp_path) -> None:
    fingerprint = _fingerprint()
    latents, video = _write_reference_report(tmp_path, fingerprint)
    args = _args(
        tmp_path,
        "--reference-latents",
        str(latents),
        "--reference-video",
        str(video),
    )

    provenance = benchmark._validate_reference_provenance(args, fingerprint)

    assert provenance["reports"] == [str((tmp_path / "report.json").resolve())]
    assert set(provenance["artifacts"]) == {"latents", "video"}


def test_reference_provenance_rejects_legacy_report_without_fingerprint(tmp_path) -> None:
    latents = tmp_path / "final-latents.pt"
    latents.write_bytes(b"legacy")
    (tmp_path / "report.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    args = _args(tmp_path, "--reference-latents", str(latents))

    with pytest.raises(ValueError, match="no execution fingerprint.*stale"):
        benchmark._validate_reference_provenance(args, _fingerprint())


def test_reference_provenance_rejects_changed_source_or_workload(tmp_path) -> None:
    fingerprint = _fingerprint()
    latents, _ = _write_reference_report(tmp_path, fingerprint)
    args = _args(tmp_path, "--reference-latents", str(latents))

    with pytest.raises(ValueError, match="incompatible source fingerprint"):
        benchmark._validate_reference_provenance(
            args,
            _fingerprint(source="new-source"),
        )
    with pytest.raises(ValueError, match="incompatible workload fingerprint"):
        benchmark._validate_reference_provenance(
            args,
            _fingerprint(workload="new-workload"),
        )


def test_reference_provenance_rejects_modified_artifact(tmp_path) -> None:
    fingerprint = _fingerprint()
    latents, _ = _write_reference_report(tmp_path, fingerprint)
    latents.write_bytes(b"modified after report")
    args = _args(tmp_path, "--reference-latents", str(latents))

    with pytest.raises(ValueError, match="artifact is stale or was modified"):
        benchmark._validate_reference_provenance(args, fingerprint)


def test_quality_gate_consumes_all_latent_and_video_metrics() -> None:
    thresholds = {
        "min_latent_cosine": 0.999,
        "max_latent_mse": 1e-4,
        "min_latent_psnr": 30.0,
        "max_latent_error": 0.05,
        "min_video_psnr": 30.0,
        "min_video_ssim": 0.95,
        "max_video_lpips": 0.10,
    }
    report = benchmark._quality_gate(
        rank_consistency={"cosine_min": 0.9999, "max_abs_error": 0.01},
        reference_latents={
            "cosine": 0.9998,
            "mse": 1e-5,
            "psnr_db": 40.0,
            "max_abs_error": 0.02,
        },
        reference_video={
            "psnr_db_mean": 35.0,
            "ssim_mean": 0.98,
            "lpips_alex_mean": 0.03,
        },
        thresholds=thresholds,
    )
    assert report["passed"] is True
    assert set(report["checks"]) == {
        "rank_consistency.cosine_min",
        "rank_consistency.max_abs_error",
        "reference_latents.cosine",
        "reference_latents.mse",
        "reference_latents.psnr_db",
        "reference_latents.max_abs_error",
        "reference_video.psnr_db_mean",
        "reference_video.ssim_mean",
        "reference_video.lpips_alex_mean",
    }

    failed = benchmark._quality_gate(
        rank_consistency={"cosine_min": 1.0, "max_abs_error": 0.0},
        reference_latents=None,
        reference_video={
            "psnr_db_mean": 29.0,
            "ssim_mean": 0.98,
            "lpips_alex_mean": 0.03,
        },
        thresholds=thresholds,
    )
    assert failed["passed"] is False
    assert failed["checks"]["reference_video.psnr_db_mean"]["passed"] is False


@pytest.mark.parametrize(
    ("next_request_timestep", "expected_routes", "missing_expert"),
    (
        (900.0, {"high-noise": 1, "low-noise": 0}, "low-noise"),
        (100.0, {"high-noise": 0, "low-noise": 1}, "high-noise"),
    ),
)
def test_dual_expert_request_reset_prevents_previous_route_false_positive(
    next_request_timestep: float,
    expected_routes: dict[str, int],
    missing_expert: str,
) -> None:
    denoiser = Wan22DualExpertDenoiser(
        _DualExpertAuditLeaf(),
        _DualExpertAuditLeaf(),
        boundary_ratio=0.875,
    )
    # The previous request exercises both experts.  Those receipts must not be
    # allowed to make the following one-expert request pass the E2E audit.
    denoiser(_dual_expert_input(900.0, step_index=0))
    denoiser(_dual_expert_input(100.0, step_index=1))
    assert denoiser.runtime_optimization_report()["runtime"]["dual_expert"][
        "route_calls"
    ] == {"high-noise": 1, "low-noise": 1}

    denoiser(_dual_expert_input(next_request_timestep, step_index=0))
    report = denoiser.runtime_optimization_report()
    assert report["runtime"]["dual_expert"]["route_calls"] == expected_routes

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {}},
    )
    assert audit["passed"] is False
    assert any(
        f"denoiser.{missing_expert}: dual-expert route_calls must be > 0"
        in issue
        for issue in audit["issues"]
    )


def test_dual_expert_audit_accepts_finalized_request_owned_route_receipt() -> None:
    audit = benchmark._optimization_audit(
        {"denoiser": _finalized_dual_expert_report()},
        expected_requests={"denoiser": {}},
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    (
        ("not-finalized", "finalized, completed, request-local"),
        ("release-error", "finalized, completed, request-local"),
        ("request-mismatch", "agree with runtime and effective"),
        ("route-mismatch", "agree with runtime and effective"),
        ("event-mismatch", "fully account for route_calls"),
        ("live-request", "zero live requests"),
        ("unbounded", "bounded finalized receipt snapshot"),
    ),
)
def test_dual_expert_audit_rejects_stale_or_incomplete_route_receipt(
    mutation,
    issue_fragment,
) -> None:
    report = json.loads(json.dumps(_finalized_dual_expert_report()))
    dual = report["runtime"]["dual_expert"]
    receipt = dual["route_receipt"]
    if mutation == "not-finalized":
        receipt["finalized"] = False
    elif mutation == "release-error":
        receipt["release_reason"] = "error"
    elif mutation == "request-mismatch":
        dual["request_id"] = "previous-request"
    elif mutation == "route-mismatch":
        dual["route_calls"]["high-noise"] = 3
    elif mutation == "event-mismatch":
        receipt["events"].pop()
    elif mutation == "live-request":
        dual["lifecycle"]["live_requests"] = 1
    else:
        dual["lifecycle"]["receipt_snapshots"] = 33

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {}},
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


@pytest.mark.parametrize(
    ("events", "issue_fragment"),
    (
        (
            [
                {"expert": "high-noise", "branch": "positive", "step_index": 0},
                {"expert": "low-noise", "branch": "negative", "step_index": 0},
                {"expert": "low-noise", "branch": "positive", "step_index": 1},
                {"expert": "high-noise", "branch": "negative", "step_index": 1},
            ],
            "different experts",
        ),
        (
            [
                {"expert": "high-noise", "branch": "positive", "step_index": 0},
                {"expert": "high-noise", "branch": "negative", "step_index": 0},
                {"expert": "low-noise", "branch": "positive", "step_index": 1},
                {"expert": "low-noise", "branch": "negative", "step_index": 1},
                {"expert": "high-noise", "branch": "positive", "step_index": 2},
                {"expert": "high-noise", "branch": "negative", "step_index": 2},
            ],
            "at most one high-noise to low-noise transition",
        ),
        (
            [
                {"expert": "high-noise", "branch": "positive", "step_index": 0},
                {"expert": "high-noise", "branch": "negative", "step_index": 0},
                {"expert": "low-noise", "branch": "positive", "step_index": 1},
                {"expert": "low-noise", "branch": "positive", "step_index": 1},
            ],
            "unique and exactly cover",
        ),
    ),
)
def test_dual_expert_sparse_route_binding_rejects_impossible_schedules(
    events: list[dict[str, object]],
    issue_fragment: str,
) -> None:
    steps = tuple(range(3 if len(events) == 6 else 2))
    _, issues = benchmark._dual_expert_approximate_branch_steps(
        events,
        expected_branch_steps={"positive": steps, "negative": steps},
    )

    assert any(issue_fragment in issue for issue in issues)


@pytest.mark.parametrize("expected_dual", (False, True))
def test_sparse_audit_rejects_runtime_model_topology_spoofing(
    expected_dual: bool,
) -> None:
    report = (
        _finalized_dual_expert_report()
        if not expected_dual
        else {
            "requested": {},
            "effective": {},
            "fallbacks": [],
            "runtime": {},
        }
    )
    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {}},
        expected_approximate_workload={
            "dual_expert": expected_dual,
            "expected_routed_steps": expected_dual,
            "branch_steps": {
                "positive": (0, 1),
                "negative": (0, 1),
            },
        },
    )

    assert audit["passed"] is False
    assert any("trusted model architecture" in issue for issue in audit["issues"])


def test_optimization_audit_accepts_exercised_runtime_paths() -> None:
    reports = {
        "denoiser": {
            "requested": {
                "fuse_qkv": True,
                "static_cross_kv": True,
                "quantization": "fp8",
                "sequence_parallel": 2,
            },
            "effective": {
                "fuse_qkv_blocks": 30,
                "static_cross_kv_blocks": 30,
                "quantization": "fp8-kernel",
                "sequence_parallel_degree": 2,
            },
            "fallbacks": [],
            "quality_tier": "numerically-approximate",
            "runtime": {
                "qkv_fusion": {
                    "fused_blocks": 30,
                    "eager_projection_calls": 1500,
                    "compiled_graph_traces": 0,
                    "execution": "eager-projection-executed",
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
                "quantization": {
                    "low_precision_kernel_calls": 100,
                    "packed_weight_calls": 0,
                    "dense_policy_calls": 0,
                    "dense_fallback_calls": 0,
                    "fallback_reasons": [],
                },
                "sequence_parallel": {
                    "sp_degree": 2,
                    "backend": "native-ulysses",
                    "wrapped_blocks": 30,
                    "head_parallel": True,
                    "fused_rope_calls": 0,
                    "complex_rope_calls": 1500,
                },
            },
        },
        "decoder": {
            "requested": {
                "vae_decode_autocast": "torch.bfloat16",
                "vae_spatial_tiling": True,
                "vae_temporal_chunk_size": 4,
                "vae_parallel_degree": 2,
            },
            "effective": {
                "vae_decode_autocast": "torch.bfloat16",
                "vae_decode": "parallel-spatial-tiled",
                "vae_spatial_tiles": 4,
                "vae_parallel_degree": 2,
            },
            "fallbacks": [],
            "quality_tier": "numerically-approximate",
            "runtime": {
                "spatial_tiled_decode_calls": 0,
                "single_spatial_tile_calls": 0,
                "temporal_chunked_decode_calls": 1,
                "last_temporal_chunk_count": 3,
                "single_temporal_chunk_calls": 0,
                "parallel_tiled_decode_calls": 1,
                "last_spatial_tile_count": 4,
            },
        },
    }
    audit = benchmark._optimization_audit(
        reports,
        expected_requests={
            "denoiser": {
                "fuse_qkv": True,
                "static_cross_kv": True,
                "quantization": "fp8",
                "sequence_parallel": 2,
            },
            "decoder": {
                "vae_decode_autocast": "torch.bfloat16",
                "vae_spatial_tiling": True,
                "vae_temporal_chunk_size": 4,
                "vae_parallel_degree": 2,
            },
        },
        collective_counters={
            "all_to_all_calls": 3000,
            "fused_multi_tensor_all_to_all_calls": 1500,
            "unfused_multi_tensor_all_to_all_calls": 0,
            "sequence_output_all_gather_calls": 3,
        },
    )
    assert audit["passed"] is True
    assert audit["issues"] == []
    assert audit["fallbacks"] == []


@pytest.mark.parametrize(
    "counter_name",
    ("packed_weight_calls", "dense_policy_calls", "dense_fallback_calls"),
)
def test_quantization_audit_rejects_any_dense_compute_lane(counter_name: str) -> None:
    report = {
        "requested": {"quantization": "int8"},
        "effective": {"quantization": "int8-w8a8-triton"},
        "fallbacks": [],
        "runtime": {
            "quantization": {
                "low_precision_kernel_calls": 100,
                "packed_weight_calls": 0,
                "dense_policy_calls": 0,
                "dense_fallback_calls": 0,
                "fallback_reasons": [],
            }
        },
    }
    report["runtime"]["quantization"][counter_name] = 1

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"quantization": "int8"}},
    )

    assert audit["passed"] is False
    assert any(counter_name in issue for issue in audit["issues"])


def test_vmoba_audit_accepts_complete_request_window_receipts() -> None:
    audit = benchmark._optimization_audit(
        {"denoiser": _vmoba_optimization_report()},
        expected_requests={"denoiser": {"approximate_attention": "vmoba"}},
        expected_approximate_workload=_vmoba_expected_workload(),
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


def test_lightx2v_audit_accepts_complete_multigpu_receipt(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        "--steps",
        "2",
        "--approximate-attention",
        "dynamic_sparse",
    )
    workload = benchmark._expected_approximate_workload(args)
    assert workload is not None
    report = _lightx2v_optimization_report(workload)

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={
            "denoiser": {"approximate_attention": "dynamic_sparse"}
        },
        expected_approximate_workload=workload,
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("kind", "profile"),
    (
        ("fastvideo_sla", {}),
        (
            "fastvideo_sagesla",
            {
                "fastvideo_topk_ratio": 0.4,
                "fastvideo_feature_map": "relu",
            },
        ),
    ),
)
def test_fastvideo_sla_audit_accepts_complete_projection_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    profile: dict[str, object],
) -> None:
    _stub_fastvideo_sla_preflight(monkeypatch)
    profile_args = (
        (
            "--approximate-attention-profile-json",
            json.dumps(profile),
        )
        if profile
        else ()
    )
    args = _args(
        tmp_path,
        "--steps",
        "2",
        "--approximate-attention",
        kind,
        *profile_args,
    )
    workload = benchmark._expected_approximate_workload(args)
    assert workload is not None
    report = _fastvideo_sla_optimization_report(workload)

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={
            "denoiser": {"approximate_attention": kind}
        },
        expected_approximate_workload=workload,
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    (
        ("provider", "receipt.fastvideo_sla.events"),
        ("family", "receipt.fastvideo_sla.events"),
        ("source-commit", "receipt.fastvideo_sla.events"),
        ("source-fingerprint", "receipt.fastvideo_sla.events"),
        ("source-root", "receipt.fastvideo_sla.events"),
        ("dirty", "receipt.fastvideo_sla.events"),
        ("parity", "receipt.fastvideo_sla.events"),
        ("injected", "receipt.fastvideo_sla.events"),
        ("layout", "receipt.fastvideo_sla.events"),
        ("global-projection", "receipt.fastvideo_sla.events"),
        ("layer-projection", "receipt.fastvideo_sla.events"),
        ("source-keys", "receipt.fastvideo_sla.events"),
        ("provider-fingerprint", "receipt.fastvideo_sla.events"),
        ("global-call", "receipt.fastvideo_sla.events"),
        ("layer-call", "receipt.fastvideo_sla.events"),
        ("timestep", "receipt.fastvideo_sla.events"),
        ("topk", "receipt.fastvideo_sla.events"),
        ("nested-receipt", "receipt.fastvideo_sla.events"),
        ("coverage", "receipt.event_count"),
        ("summary-projection", "receipt.fastvideo_sla.summary"),
        ("summary-layers", "receipt.fastvideo_sla.summary"),
        ("summary-calls", "receipt.fastvideo_sla.summary"),
    ),
)
def test_fastvideo_sla_audit_rejects_mutated_execution_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    issue_fragment: str,
) -> None:
    _stub_fastvideo_sla_preflight(monkeypatch)
    args = _args(
        tmp_path,
        "--steps",
        "2",
        "--approximate-attention",
        "fastvideo_sla",
    )
    workload = benchmark._expected_approximate_workload(args)
    assert workload is not None
    report = _fastvideo_sla_optimization_report(workload)
    approximate = report["runtime"]["approximate_attention"]
    fastvideo = approximate["fastvideo_sla"]
    sparse_events = [
        event
        for event in approximate["events"]
        if event["execution"] == "sparse"
    ]
    event = sparse_events[0]
    if mutation == "provider":
        event["provider_path"] = "fastvideo.attention.backends.sla.Fake"
    elif mutation == "family":
        event["provider_family"] = "fastvideo/fake"
    elif mutation == "source-commit":
        event["provider_source_commit"] = "0" * 40
    elif mutation == "source-fingerprint":
        event["provider_source_fingerprint"] = "0" * 64
    elif mutation == "source-root":
        event["provider_source_root"] = "/tmp/untrusted"
    elif mutation == "dirty":
        event["provider_source_clean"] = False
    elif mutation == "parity":
        event["reference_parity_verified"] = False
    elif mutation == "injected":
        event["injected_test_provider"] = True
    elif mutation == "layout":
        event["checkpoint_layout"] = "turbodiffusion_original"
    elif mutation == "global-projection":
        event["all_projection_weights_fingerprint"] = "d" * 64
    elif mutation == "layer-projection":
        event["projection_weight_fingerprint"] = "not-sha256"
    elif mutation == "source-keys":
        event["projection_source_keys"] = ["blocks.0.proj_l.weight"]
    elif mutation == "provider-fingerprint":
        event["provider_fingerprint"] = "d" * 64
    elif mutation == "global-call":
        sparse_events[1]["call_index"] = event["call_index"]
    elif mutation == "layer-call":
        same_layer = next(
            candidate
            for candidate in sparse_events[1:]
            if candidate["layer_idx"] == event["layer_idx"]
        )
        same_layer["layer_call_index"] = event["layer_call_index"]
    elif mutation == "timestep":
        event["current_timestep"] = 99
    elif mutation == "topk":
        event["topk_ratio"] = 0.99
    elif mutation == "nested-receipt":
        event["provider_receipt"] = {"synthetic": True}
    elif mutation == "coverage":
        approximate["events"].pop()
    elif mutation == "summary-projection":
        fastvideo["all_projection_weights_fingerprint"] = "d" * 64
    elif mutation == "summary-layers":
        fastvideo["expected_layers"].pop()
    else:
        fastvideo["provider_calls"] -= 1

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={
            "denoiser": {"approximate_attention": "fastvideo_sla"}
        },
        expected_approximate_workload=workload,
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


def test_dual_expert_lightx2v_audit_accepts_complete_per_expert_receipts(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--model-id",
        "wan2.2-t2v-a14b",
        "--steps",
        "2",
        "--approximate-attention",
        "dynamic_sparse",
    )
    workload = benchmark._expected_approximate_workload(args)
    assert workload is not None
    report = _finalized_dual_expert_report()
    report["requested"] = {"approximate_attention": "dynamic_sparse"}
    report["effective"]["approximate_attention_kernel"] = "dynamic_sparse"
    experts = report["runtime"]["dual_expert"]["experts"]
    experts["high-noise"] = _lightx2v_optimization_report(
        workload,
        branch_steps={"positive": (0,), "negative": (0,)},
        request_id="request-current",
    )
    experts["low-noise"] = _lightx2v_optimization_report(
        workload,
        branch_steps={"positive": (1,), "negative": (1,)},
        request_id="request-current",
    )

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={
            "denoiser": {"approximate_attention": "dynamic_sparse"}
        },
        expected_approximate_workload=workload,
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


def test_dual_expert_fastvideo_sla_audit_accepts_per_expert_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_fastvideo_sla_preflight(monkeypatch)
    args = _args(
        tmp_path,
        "--model-id",
        "wan2.2-t2v-a14b",
        "--steps",
        "2",
        "--approximate-attention",
        "fastvideo_sagesla",
    )
    workload = benchmark._expected_approximate_workload(args)
    assert workload is not None
    report = _finalized_dual_expert_report()
    report["requested"] = {"approximate_attention": "fastvideo_sagesla"}
    report["effective"]["approximate_attention_kernel"] = (
        "fastvideo_sagesla"
    )
    experts = report["runtime"]["dual_expert"]["experts"]
    experts["high-noise"] = _fastvideo_sla_optimization_report(
        workload,
        branch_steps={"positive": (0,), "negative": (0,)},
        request_id="request-current",
    )
    experts["low-noise"] = _fastvideo_sla_optimization_report(
        workload,
        branch_steps={"positive": (1,), "negative": (1,)},
        request_id="request-current",
    )

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={
            "denoiser": {"approximate_attention": "fastvideo_sagesla"}
        },
        expected_approximate_workload=workload,
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    (
        ("runtime-kind", "runtime reported"),
        ("provider", "real 'vmoba' provider path"),
        ("symbols", "all real VMoBA provider symbols"),
        ("missing-route", "chunk_calls.spatial > 0"),
        ("events", "non-empty VMoBA layer events"),
        ("event-provider", "malformed VMoBA events"),
        ("event-count", "one VMoBA event/chunk receipt"),
        ("fallback", "kernel_fallbacks == 0"),
    ),
)
def test_vmoba_audit_rejects_incomplete_or_false_receipts(
    mutation,
    issue_fragment,
) -> None:
    report = _vmoba_optimization_report()
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
    elif mutation == "event-count":
        approximate["sparse_calls"] = 4
        approximate["kernel_attempts"] = 4
    else:
        approximate["kernel_fallbacks"] = 1

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"approximate_attention": "vmoba"}},
        expected_approximate_workload=_vmoba_expected_workload(),
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


def _offload_runtime_report() -> dict[str, object]:
    return {
        "mode": "async-double-buffer",
        "enabled": True,
        "effective": True,
        "issues": [],
        "request": {
            "forward_calls": 30,
            "async_copy_tensors": 600,
            "synchronous_copy_tensors": 0,
            "pageable_cpu_tensors": 0,
            "peak_active_layers": 2,
        },
    }


def test_async_block_offload_semantic_gate_requires_real_overlap() -> None:
    receipt = _offload_runtime_report()

    issues = benchmark._semantic_effectiveness_issues(
        "denoiser",
        "offload",
        "block",
        effective={"offload": "async-double-buffer-executed"},
        runtime={"offload": receipt},
        fallbacks=[],
    )

    assert issues == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("request", "async_copy_tensors"), 0),
        (("request", "synchronous_copy_tensors"), 1),
        (("request", "pageable_cpu_tensors"), 1),
        (("request", "peak_active_layers"), 3),
        (("effective",), False),
        (("mode",), "legacy-on-demand-wrapper"),
    ],
)
def test_async_block_offload_semantic_gate_rejects_false_receipts(
    path,
    value,
) -> None:
    receipt = _offload_runtime_report()
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    issues = benchmark._semantic_effectiveness_issues(
        "denoiser",
        "offload",
        "block",
        effective={"offload": "async-double-buffer-executed"},
        runtime={"offload": receipt},
        fallbacks=[],
    )

    assert issues


def _attention_optimization_report(*, compiled: bool = False) -> dict:
    compile_runtime = {
        "calls": 3,
        "attention_provider_graph_traces": {"flash_attention_3": 1},
    }
    return {
        "requested": {"attention": "flash_attention_3"},
        "effective": {
            "attention": "flash_attention_3",
            **({"compile": "inductor-fullgraph"} if compiled else {}),
        },
        "fallbacks": [],
        "runtime": {
            "attention_dispatch": {
                "provider_calls": {
                    "flash_attention_3": {
                        "attempts": 0 if compiled else 120,
                        "successes": 0 if compiled else 120,
                        "fallbacks": 0,
                        "errors": 0,
                        "quarantined_skips": 0,
                        "compiled_graph_traces": 0,
                    }
                }
            },
            **({"compile": compile_runtime} if compiled else {}),
        },
    }


def _fused_rope_optimization_report(
    *,
    compiled: bool = False,
    sequence_parallel: bool = False,
) -> dict:
    runtime = {
        "fused_rope": {
            "installed_blocks": 30,
            "effective": (
                "compiled-graph-traced"
                if compiled
                else "accelerated-provider-executed"
            ),
            "eager_calls": 0 if compiled else 1500,
            "compiled_graph_traces": 1 if compiled else 0,
            "provider_calls": 0 if compiled else 1500,
            "torch_fallback_calls": 0,
            "provider_failures": 0,
            "quarantined_skips": 0,
            "malformed_receipts": 0,
            "provider_paths": (
                [] if compiled else ["triton_hidden_qk_rmsnorm_rope_3d"]
            ),
        }
    }
    effective = {"fused_rope": "hidden_qk_rmsnorm_rope_3d:fp64"}
    if compiled:
        effective["compile"] = "inductor-fullgraph"
        runtime["compile"] = {"calls": 3}
    if sequence_parallel:
        runtime["sequence_parallel"] = {
            "sp_degree": 2,
            "backend": "native-ulysses",
            "wrapped_blocks": 30,
            "head_parallel": True,
            "fused_rope_calls": 1500,
            "complex_rope_calls": 0,
        }
    return {
        "requested": {"fused_rope": True},
        "effective": effective,
        "fallbacks": [],
        "runtime": runtime,
    }


@pytest.mark.parametrize(
    "report",
    (
        _fused_rope_optimization_report(),
        _fused_rope_optimization_report(compiled=True),
        _fused_rope_optimization_report(sequence_parallel=True),
    ),
)
def test_fused_rope_audit_accepts_real_eager_compile_and_sp_receipts(report) -> None:
    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"fused_rope": True}},
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("target", "field", "value", "issue_fragment"),
    (
        ("fused_rope", "installed_blocks", 0, "installed_blocks > 0"),
        ("fused_rope", "provider_calls", 0, "provider_calls == eager_calls > 0"),
        ("fused_rope", "torch_fallback_calls", 1, "torch_fallback_calls == 0"),
        ("fused_rope", "provider_paths", ["torch"], "real Triton provider path"),
        ("sequence_parallel", "fused_rope_calls", 0, "fused_rope_calls > 0"),
        ("sequence_parallel", "complex_rope_calls", 1, "complex_rope_calls == 0"),
    ),
)
def test_fused_rope_audit_rejects_unexercised_or_fallback_paths(
    target,
    field,
    value,
    issue_fragment,
) -> None:
    report = _fused_rope_optimization_report(
        sequence_parallel=target == "sequence_parallel"
    )
    report["runtime"][target][field] = value

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"fused_rope": True}},
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


def _sequence_parallel_optimization_report() -> dict:
    return {
        "requested": {"sequence_parallel": 2},
        "effective": {"sequence_parallel_degree": 2},
        "fallbacks": [],
        "runtime": {
            "sequence_parallel": {
                "sp_degree": 2,
                "backend": "native-ulysses",
                "wrapped_blocks": 30,
                "head_parallel": True,
                "fused_rope_calls": 0,
                "complex_rope_calls": 1500,
            }
        },
    }


def _sequence_parallel_collective_counters() -> dict:
    return {
        "all_to_all_calls": 3000,
        "fused_multi_tensor_all_to_all_calls": 1500,
        "unfused_multi_tensor_all_to_all_calls": 0,
        "sequence_output_all_gather_calls": 3,
    }


def test_sequence_parallel_audit_requires_real_processor_and_collectives() -> None:
    audit = benchmark._optimization_audit(
        {"denoiser": _sequence_parallel_optimization_report()},
        expected_requests={"denoiser": {"sequence_parallel": 2}},
        collective_counters=_sequence_parallel_collective_counters(),
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    (
        ("missing-counters", "measured-window collective counters"),
        ("zero-all-to-all", "all_to_all_calls > 0"),
        ("unfused-qkv", "unfused_multi_tensor_all_to_all_calls == 0"),
        ("no-rope-calls", "positive request-window RoPE execution calls"),
    ),
)
def test_sequence_parallel_audit_rejects_config_only_or_slow_fallback(
    mutation,
    issue_fragment,
) -> None:
    report = _sequence_parallel_optimization_report()
    counters = _sequence_parallel_collective_counters()
    if mutation == "missing-counters":
        counters = None
    elif mutation == "zero-all-to-all":
        counters["all_to_all_calls"] = 0
    elif mutation == "unfused-qkv":
        counters["unfused_multi_tensor_all_to_all_calls"] = 1
    else:
        report["runtime"]["sequence_parallel"]["complex_rope_calls"] = 0

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"sequence_parallel": 2}},
        collective_counters=counters,
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


def test_attention_audit_accepts_executed_eager_provider() -> None:
    audit = benchmark._optimization_audit(
        {"denoiser": _attention_optimization_report()},
        expected_requests={"denoiser": {"attention": "flash_attention_3"}},
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    ("counter", "value", "issue_fragment"),
    (
        ("successes", 0, "successes > 0"),
        ("attempts", 121, "attempts == successes"),
        ("fallbacks", 1, "fallbacks=0"),
        ("errors", 1, "errors=0"),
        ("quarantined_skips", 1, "quarantined_skips=0"),
    ),
)
def test_attention_audit_rejects_unproven_or_failed_eager_provider(
    counter,
    value,
    issue_fragment,
) -> None:
    report = _attention_optimization_report()
    report["runtime"]["attention_dispatch"]["provider_calls"][
        "flash_attention_3"
    ][counter] = value

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"attention": "flash_attention_3"}},
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


def test_attention_audit_rejects_missing_target_provider() -> None:
    report = _attention_optimization_report()
    report["runtime"]["attention_dispatch"]["provider_calls"] = {}

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"attention": "flash_attention_3"}},
    )

    assert audit["passed"] is False
    assert any("target provider" in issue for issue in audit["issues"])


def test_attention_audit_accepts_scoped_compiled_graph_receipt() -> None:
    audit = benchmark._optimization_audit(
        {"denoiser": _attention_optimization_report(compiled=True)},
        expected_requests={"denoiser": {"attention": "flash_attention_3"}},
    )

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize("missing_proof", ("calls", "graph-trace"))
def test_attention_audit_rejects_incomplete_compile_proof(missing_proof) -> None:
    report = _attention_optimization_report(compiled=True)
    compile_runtime = report["runtime"]["compile"]
    if missing_proof == "calls":
        compile_runtime["calls"] = 0
    else:
        compile_runtime["attention_provider_graph_traces"] = {}

    audit = benchmark._optimization_audit(
        {"denoiser": report},
        expected_requests={"denoiser": {"attention": "flash_attention_3"}},
    )

    assert audit["passed"] is False
    expected = "compile.calls > 0" if missing_proof == "calls" else "graph_traces"
    assert any(expected in issue for issue in audit["issues"])


def test_optimization_audit_rejects_silent_noop() -> None:
    audit = benchmark._optimization_audit(
        {
            "denoiser": {
                "requested": {"quantization": "fp8"},
                "effective": {},
                "fallbacks": [],
                "quality_tier": "exact",
                "runtime": {},
            }
        },
        expected_requests={"denoiser": {"quantization": "fp8"}},
    )
    assert audit["passed"] is False
    assert "denoiser: requested.quantization has neither effective state nor fallback" in audit[
        "issues"
    ]
    assert any("runtime.quantization evidence" in issue for issue in audit["issues"])


@pytest.mark.parametrize(
    ("component", "expected", "report", "issue_fragment"),
    (
        (
            "denoiser",
            {"fuse_qkv": True},
            {
                "requested": {"fuse_qkv": True},
                "effective": {"fuse_qkv_blocks": 0},
                "fallbacks": [],
                "runtime": {},
            },
            "fuse_qkv_blocks > 0",
        ),
        (
            "denoiser",
            {"sequence_parallel": 4},
            {
                "requested": {"sequence_parallel": 4},
                "effective": {"sequence_parallel_degree": 2},
                "fallbacks": [],
                "runtime": {},
            },
            "sequence_parallel_degree=4",
        ),
        (
            "runner",
            {"cfg_parallel": 2},
            {
                "requested": {"cfg_parallel": 2},
                "effective": {"cfg_parallel": "exact-local"},
                "fallbacks": [],
                "runtime": {
                    "cfg_parallel_collective_calls": 0,
                    "cfg_parallel_local_branch_calls": {"positive": 0, "negative": 0},
                },
            },
            "branch-per-rank",
        ),
        (
            "decoder",
            {"vae_parallel_degree": 4},
            {
                "requested": {"vae_parallel_degree": 4},
                "effective": {
                    "vae_parallel_degree": 4,
                    "vae_decode": "parallel-spatial-tiled (single-tile)",
                    "vae_spatial_tiles": 1,
                },
                "fallbacks": [
                    "vae_spatial_tiling: requested shape produced one tile; no benefit"
                ],
                "runtime": {
                    "parallel_tiled_decode_calls": 1,
                    "last_spatial_tile_count": 1,
                },
            },
            "multi-tile parallel decode",
        ),
        (
            "denoiser",
            {"quantization": "int8"},
            {
                "requested": {"quantization": "int8"},
                "effective": {"quantization": "dense"},
                "fallbacks": ["quantization: dense fallback"],
                "runtime": {
                    "quantization": {
                        "low_precision_kernel_calls": 0,
                        "dense_fallback_calls": 1,
                        "fallback_reasons": ["kernel failed"],
                    }
                },
            },
            "low_precision_kernel_calls > 0",
        ),
    ),
)
def test_optimization_audit_rejects_present_but_ineffective_runtime_state(
    component,
    expected,
    report,
    issue_fragment,
) -> None:
    audit = benchmark._optimization_audit(
        {component: report},
        expected_requests={component: expected},
    )

    assert audit["passed"] is False
    assert any(issue_fragment in issue for issue in audit["issues"])


def test_component_reports_uses_public_native_component_property() -> None:
    class Component:
        def __init__(self, name: str) -> None:
            self.name = name

        def runtime_optimization_report(self):
            return {"component": self.name}

    pipeline = SimpleNamespace(
        native_pipeline=SimpleNamespace(
            components=SimpleNamespace(
                denoiser=Component("denoiser"),
                decoder=Component("decoder"),
            )
        )
    )
    assert benchmark._component_reports(pipeline) == {
        "denoiser": {"component": "denoiser"},
        "decoder": {"component": "decoder"},
    }
