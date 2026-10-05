from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from benchmarks.inference.approximate_attention_receipts import (
    audit_approximate_attention_receipt,
)

_LIGHTX2V_COMMIT = "6fb7c1362b89d4908a9ea197bac4fbd7482ee2d5"
_LIGHTX2V_FINGERPRINT = hashlib.sha256(
    f"{_LIGHTX2V_COMMIT}\0True\0".encode()
).hexdigest()
_LIGHTX2V_PROVIDERS = {
    "dynamic_sparse": (
        "triton",
        "lightx2v.common.ops.attn.kernels.sla_kernel._attention.apply",
    ),
    "sparge": (
        "meansim_sage2",
        "spas_sage_attn.core.spas_sage2_attn_meansim_topk_cuda",
    ),
    "nbhd": (
        "magi",
        "magi_attention.functional.flex_flash_attn_func",
    ),
    "lightx2v_sla_mask": (
        "triton",
        "lightx2v.common.ops.attn.kernels.sla_kernel._attention.apply",
    ),
    "flexblock": (
        "flex_block",
        "flex_block_attn.flex_block_attn_func",
    ),
    "lightx2v_spas_sage": (
        "sage2",
        "lightx2v.common.ops.attn.utils.sparge_util.sage2_block_sparse_attn",
    ),
    "draft_attn": (
        "magi",
        "magi_attention.functional.flex_flash_attn_func",
    ),
    "radial_attn": (
        "magi",
        "magi_attention.functional.flex_flash_attn_func",
    ),
    "rainfusion_attn": (
        "flashinfer",
        "flashinfer.sparse.VariableBlockSparseAttentionWrapper.run",
    ),
    "svg_attn": (
        "flex_attention",
        "torch.nn.attention.flex_attention.flex_attention",
    ),
    "svg2_attn": (
        "flashinfer",
        "flashinfer.sparse.VariableBlockSparseAttentionWrapper.run",
    ),
    "lightx2v_svg_mask": (
        "magi",
        "magi_attention.functional.flex_flash_attn_func",
    ),
}
_LIGHTX2V_FAMILIES = {
    "dynamic_sparse": "lightx2v/dynamic-sparse",
    "sparge": "lightx2v/sparge-direct",
    "nbhd": "lightx2v/nbhd-direct",
    "lightx2v_sla_mask": "lightx2v/general-sparse",
    "flexblock": "lightx2v/general-sparse",
    "lightx2v_spas_sage": "lightx2v/general-sparse",
    "draft_attn": "lightx2v/draft-attention",
    "radial_attn": "lightx2v/radial-attention",
    "rainfusion_attn": "lightx2v/rainfusion-attention",
    "svg_attn": "lightx2v/svg-attention",
    "svg2_attn": "lightx2v/svg2-attention",
    "lightx2v_svg_mask": "lightx2v/svg-mask-general-sparse",
}
_LIGHTX2V_PROVIDER_DENSE_PATHS = {
    "draft_attn": "flash_attn.flash_attn_interface.flash_attn_varlen_func",
    "rainfusion_attn": (
        "lightx2v.common.ops.attn.flash_attn.FlashAttn3Weight.apply"
    ),
}
_FASTVIDEO_COMMIT = "1b2b2a0161bc6b3b80158d1fa6380a051c6530c7"
_FASTVIDEO_SOURCE_FINGERPRINT = hashlib.sha256(b"pinned-fastvideo-sla").hexdigest()
_FASTVIDEO_PROVIDER_FINGERPRINT = hashlib.sha256(b"fastvideo-sla-provider").hexdigest()
_FASTVIDEO_GLOBAL_PROJECTION_FINGERPRINT = hashlib.sha256(
    b"fastvideo-sla-projections"
).hexdigest()
_FASTVIDEO_SLA_PROVIDERS = {
    "fastvideo_sla": (
        "fastvideo.attention.backends.sla.SLAAttentionImpl",
        "fastvideo/sparse-linear-attention",
        0.1,
    ),
    "fastvideo_sagesla": (
        "fastvideo.attention.backends.sla.SageSLAAttentionImpl",
        "fastvideo/sage-sparse-linear-attention",
        0.5,
    ),
}


def _profile(kind: str, *, dense_steps: int) -> dict[str, Any]:
    operator = _LIGHTX2V_PROVIDERS.get(kind, (None, None))[0]
    return {
        "kind": kind,
        "sparsity": 0.8,
        "dense_steps": dense_steps,
        "window": [3, 3, 3],
        "block_tile": [4, 4, 4],
        "lightx2v_operator": operator,
        "nbhd_coefficient": [1.0, 0.5, 0.056],
        "nbhd_min_width": 1.0,
        "attnmap_frame_num": None,
        "lightx2v_per_block_mean": False,
        "lightx2v_pool_size": 128,
        "lightx2v_skip_timesteps": -1,
        "lightx2v_dense_attn_type": "flash_attn3",
        "svg_sample_mse_max_row": 10000,
        "svg_num_sampled_rows": 64,
        "svg_context_length": 0,
        "fastvideo_topk_ratio": _FASTVIDEO_SLA_PROVIDERS.get(
            kind, (None, None, None)
        )[2],
        "fastvideo_feature_map": "softmax",
    }


def _receipt(
    *,
    kind: str,
    provider: str,
    grid: tuple[int, int, int],
    total_steps: int = 3,
    dense_steps: int = 1,
    branch_steps: Mapping[str, Sequence[int]] | None = None,
    routed_steps: bool = False,
    lightx2v_skip_timesteps: int = -1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks = 30
    request_id = f"{kind}-request"
    request_epoch = 11
    normalized_branch_steps = {
        str(branch): tuple(int(step) for step in steps)
        for branch, steps in (
            branch_steps
            or {
                "positive": tuple(range(total_steps)),
                "negative": tuple(range(total_steps)),
            }
        ).items()
    }
    profile = _profile(kind, dense_steps=dense_steps)
    profile["lightx2v_skip_timesteps"] = lightx2v_skip_timesteps
    tokens = grid[0] * grid[1] * grid[2]
    tensor_contract = {
        "shape": [1, tokens, 1, 64],
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
    }
    lightx2v = _LIGHTX2V_PROVIDERS.get(kind)
    fastvideo_sla = _FASTVIDEO_SLA_PROVIDERS.get(kind)
    fastvideo_call_index = 0
    fastvideo_layer_calls = {layer: 0 for layer in range(blocks)}
    events: list[dict[str, Any]] = []
    for branch, steps in normalized_branch_steps.items():
        for step in steps:
            for layer in range(blocks):
                scheduled_dense = (
                    step < dense_steps or step >= total_steps - dense_steps
                )
                provider_dense = not scheduled_dense and (
                    (kind == "draft_attn" and layer == 0)
                    or (
                        kind == "rainfusion_attn"
                        and step < lightx2v_skip_timesteps
                    )
                )
                execution = (
                    "scheduled_dense"
                    if scheduled_dense
                    else "provider_dense"
                    if provider_dense
                    else "sparse"
                )
                provider_attempted = execution in {"sparse", "provider_dense"}
                event_provider = (
                    _LIGHTX2V_PROVIDER_DENSE_PATHS[kind]
                    if provider_dense
                    else provider
                    if execution == "sparse"
                    else None
                )
                event: dict[str, Any] = {
                    "algorithm": kind,
                    "request_id": request_id,
                    "request_epoch": request_epoch,
                    "request_local": True,
                    "branch": branch,
                    "step": step,
                    "step_index": step,
                    "total_steps": total_steps,
                    "layer_idx": layer,
                    "module_path": f"blocks.{layer}.self_attn",
                    "execution": execution,
                    "provider_attempted": provider_attempted,
                    "provider_path": event_provider,
                    "fallback_reason": None,
                    "dense_fallback_executed": False,
                    "grid_size": list(grid),
                    "input": tensor_contract,
                    "output": tensor_contract,
                    "input_shape": tensor_contract["shape"],
                    "input_device": tensor_contract["device"],
                    "input_dtype": tensor_contract["dtype"],
                    "output_shape": tensor_contract["shape"],
                    "output_device": tensor_contract["device"],
                    "output_dtype": tensor_contract["dtype"],
                }
                if provider_attempted and lightx2v is not None:
                    operator, canonical_provider = lightx2v
                    provider_family = _LIGHTX2V_FAMILIES[kind]
                    provider_fields: dict[str, Any] = {
                        "provider_family": provider_family,
                        "operator": operator,
                        "adapter_path": (
                            "lightx2v.common.ops.attn.synthetic.Provider.apply"
                        ),
                        "provider_calls": 1,
                        "reference_lightx2v_commit": _LIGHTX2V_COMMIT,
                        "provider_source_commit": _LIGHTX2V_COMMIT,
                        "provider_source_clean": True,
                        "provider_source_fingerprint": _LIGHTX2V_FINGERPRINT,
                        "provider_source_root": "/opt/lightx2v",
                        "reference_parity_verified": True,
                        "sparsity_ratio": profile["sparsity"],
                        "canonical_sparse_provider_path": canonical_provider,
                        "sparse_kernel_executed": execution == "sparse",
                        "provider_dense_executed": execution
                        == "provider_dense",
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
                        "runtime_effective": True,
                    }
                    if kind == "nbhd":
                        provider_fields.update(
                            {
                                "attnmap_frame_num": grid[0],
                                "nbhd_coefficient": profile[
                                    "nbhd_coefficient"
                                ],
                                "nbhd_min_width": profile["nbhd_min_width"],
                            }
                        )
                    event.update(provider_fields)
                elif execution == "sparse" and fastvideo_sla is not None:
                    canonical_provider, provider_family, topk_ratio = fastvideo_sla
                    fastvideo_call_index += 1
                    fastvideo_layer_calls[layer] += 1
                    event.update(
                        {
                            "provider_family": provider_family,
                            "reference_provider_path": canonical_provider,
                            "provider_fingerprint": _FASTVIDEO_PROVIDER_FINGERPRINT,
                            "injected_test_provider": False,
                            "provider_calls": 1,
                            "reference_fastvideo_commit": _FASTVIDEO_COMMIT,
                            "provider_source_commit": _FASTVIDEO_COMMIT,
                            "provider_source_clean": True,
                            "provider_source_fingerprint": (
                                _FASTVIDEO_SOURCE_FINGERPRINT
                            ),
                            "provider_source_root": "/opt/FastVideo",
                            "provider_source_file": (
                                "/opt/FastVideo/fastvideo/attention/backends/sla.py"
                            ),
                            "reference_parity_verified": True,
                            "checkpoint_layout": "fastvideo_diffusers",
                            "projection_source_keys": [
                                f"blocks.{layer}.attn1.attn_impl.proj_l.weight",
                                f"blocks.{layer}.attn1.attn_impl.proj_l.bias",
                            ],
                            "projection_weight_fingerprint": hashlib.sha256(
                                f"projection:{layer}".encode()
                            ).hexdigest(),
                            "all_projection_weights_fingerprint": (
                                _FASTVIDEO_GLOBAL_PROJECTION_FINGERPRINT
                            ),
                            "layer_prefix": f"blocks.{layer}.attn1",
                            "call_index": fastvideo_call_index,
                            "layer_call_index": fastvideo_layer_calls[layer],
                            "current_timestep": step,
                            "topk_ratio": topk_ratio,
                            "feature_map": "softmax",
                            "runtime_effective": True,
                        }
                    )
                events.append(event)

    sparse_events = [event for event in events if event["execution"] == "sparse"]
    provider_dense_events = [
        event for event in events if event["execution"] == "provider_dense"
    ]
    scheduled_events = [
        event for event in events if event["execution"] == "scheduled_dense"
    ]
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
    execution_counts = {}
    if scheduled_events:
        execution_counts["scheduled_dense"] = len(scheduled_events)
    if sparse_events:
        execution_counts["sparse"] = len(sparse_events)
    if provider_dense_events:
        execution_counts["provider_dense"] = len(provider_dense_events)
    coverage = {
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
    }
    report: dict[str, Any] = {
        "kind": kind,
        "request_id": request_id,
        "request_epoch": request_epoch,
        "request_local": True,
        "routed_steps": routed_steps,
        "finalized": True,
        "completed": True,
        "release_reason": "completed",
        "wrapped_blocks": blocks,
        "expected_layers": list(range(blocks)),
        "expected_calls": len(events),
        "runtime_effective": True,
        "kernel_attempts": len(sparse_events) + len(provider_dense_events),
        "sparse_calls": len(sparse_events),
        "provider_dense_calls": len(provider_dense_events),
        "scheduled_dense_calls": len(scheduled_events),
        "dense_fallback_calls": 0,
        "kernel_fallbacks": 0,
        "effective_kernel": kind,
        "provider_path": provider,
        "provider_paths": [provider],
        "provider_dense_path": (
            _LIGHTX2V_PROVIDER_DENSE_PATHS.get(kind)
            if provider_dense_events
            else None
        ),
        "provider_dense_paths": (
            [_LIGHTX2V_PROVIDER_DENSE_PATHS[kind]]
            if provider_dense_events
            else []
        ),
        "grid_size": list(grid),
        "branches": branches,
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
    }
    if lightx2v is not None:
        operator, canonical_provider = lightx2v
        provider_family = _LIGHTX2V_FAMILIES[kind]
        report["lightx2v"] = {
            "algorithm": kind,
            "provider_family": provider_family,
            "provider_families": [provider_family],
            "operator": operator,
            "canonical_provider_symbol": canonical_provider,
            "canonical_provider_executed": True,
            "reference_lightx2v_commit": _LIGHTX2V_COMMIT,
            "provider_source_commit": _LIGHTX2V_COMMIT,
            "provider_source_commits": [_LIGHTX2V_COMMIT],
            "provider_source_fingerprints": [_LIGHTX2V_FINGERPRINT],
            "provider_commit_complete": True,
            "provider_source_clean": True,
            "reference_parity_verified": True,
            "provider_calls": len(sparse_events) + len(provider_dense_events),
            "receipt_count": len(sparse_events) + len(provider_dense_events),
            "sparse_event_count": len(sparse_events),
            "provider_dense_event_count": len(provider_dense_events),
            "provider_dense_events": provider_dense_events,
            "events": sparse_events,
        }
    elif fastvideo_sla is not None:
        canonical_provider, provider_family, topk_ratio = fastvideo_sla
        layer_fingerprints = {
            str(layer): hashlib.sha256(
                f"projection:{layer}".encode()
            ).hexdigest()
            for layer in range(blocks)
        }
        projection_source_keys = {
            str(layer): [
                f"blocks.{layer}.attn1.attn_impl.proj_l.weight",
                f"blocks.{layer}.attn1.attn_impl.proj_l.bias",
            ]
            for layer in range(blocks)
        }
        report["fastvideo_sla"] = {
            "algorithm": kind,
            "provider_family": provider_family,
            "provider_families": [provider_family],
            "expected_layers": list(range(blocks)),
            "canonical_provider_path": canonical_provider,
            "canonical_provider_executed": True,
            "reference_fastvideo_commit": _FASTVIDEO_COMMIT,
            "provider_source_commit": _FASTVIDEO_COMMIT,
            "provider_source_commits": [_FASTVIDEO_COMMIT],
            "provider_source_fingerprints": [
                _FASTVIDEO_SOURCE_FINGERPRINT
            ],
            "provider_fingerprints": [_FASTVIDEO_PROVIDER_FINGERPRINT],
            "provider_source_root": "/opt/FastVideo",
            "provider_source_roots": ["/opt/FastVideo"],
            "provider_source_file": (
                "/opt/FastVideo/fastvideo/attention/backends/sla.py"
            ),
            "provider_source_files": [
                "/opt/FastVideo/fastvideo/attention/backends/sla.py"
            ],
            "provider_commit_complete": True,
            "provider_source_clean": True,
            "reference_parity_verified": True,
            "checkpoint_layout": "fastvideo_diffusers",
            "checkpoint_layouts": ["fastvideo_diffusers"],
            "all_projection_weights_fingerprint": (
                _FASTVIDEO_GLOBAL_PROJECTION_FINGERPRINT
            ),
            "projection_fingerprints": [
                _FASTVIDEO_GLOBAL_PROJECTION_FINGERPRINT
            ],
            "layer_projection_fingerprints": layer_fingerprints,
            "projection_source_keys": projection_source_keys,
            "topk_ratio": topk_ratio,
            "feature_map": "softmax",
            "provider_calls": len(sparse_events),
            "receipt_count": len(sparse_events),
            "sparse_event_count": len(sparse_events),
            "events": sparse_events,
        }
    return report, profile


def _audit(
    report: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    provider: str,
    grid: tuple[int, int, int],
    total_steps: int = 3,
    branch_steps: Mapping[str, Sequence[int]] | None = None,
    routed_steps: bool = False,
) -> dict[str, Any]:
    kind = str(profile["kind"])
    lightx2v = _LIGHTX2V_PROVIDERS.get(kind)
    fastvideo_sla = _FASTVIDEO_SLA_PROVIDERS.get(kind)
    return audit_approximate_attention_receipt(
        report,
        expected_kind=kind,
        allowed_provider_paths={provider},
        expected_grid=grid,
        expected_wrapped_blocks=30,
        expected_total_steps=total_steps,
        expected_routed_steps=routed_steps,
        expected_branch_steps=(
            branch_steps
            or {
                "positive": tuple(range(total_steps)),
                "negative": tuple(range(total_steps)),
            }
        ),
        expected_profile=profile,
        expected_lightx2v=(
            {
                "operator": lightx2v[0],
                "provider_symbol": lightx2v[1],
                "provider_family": _LIGHTX2V_FAMILIES[kind],
                "commit": _LIGHTX2V_COMMIT,
                "source_fingerprint": _LIGHTX2V_FINGERPRINT,
            }
            if lightx2v is not None
            else None
        ),
        expected_fastvideo_sla=(
            {
                "provider_path": fastvideo_sla[0],
                "provider_family": fastvideo_sla[1],
                "commit": _FASTVIDEO_COMMIT,
                "source_fingerprint": _FASTVIDEO_SOURCE_FINGERPRINT,
                "checkpoint_layout": "fastvideo_diffusers",
                "projection_fingerprint": (
                    _FASTVIDEO_GLOBAL_PROJECTION_FINGERPRINT
                ),
            }
            if fastvideo_sla is not None
            else None
        ),
    )


@pytest.mark.parametrize(
    ("kind", "provider", "grid"),
    (
        (
            "sta",
            "fastvideo_kernel.sliding_tile_attention",
            (18, 48, 80),
        ),
        (
            "vsa",
            "fastvideo_kernel.video_sparse_attn_bshd",
            (31, 22, 40),
        ),
    ),
)
def test_fastvideo_receipt_requires_complete_request_workload(
    kind: str,
    provider: str,
    grid: tuple[int, int, int],
) -> None:
    report, profile = _receipt(kind=kind, provider=provider, grid=grid)

    audit = _audit(report, profile=profile, provider=provider, grid=grid)

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize("kind", tuple(_FASTVIDEO_SLA_PROVIDERS))
def test_fastvideo_learned_sla_receipt_requires_checkpoint_and_provider_proof(
    kind: str,
) -> None:
    provider = _FASTVIDEO_SLA_PROVIDERS[kind][0]
    grid = (31, 22, 40)
    report, profile = _receipt(kind=kind, provider=provider, grid=grid)

    audit = _audit(report, profile=profile, provider=provider, grid=grid)

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "provider-class",
        "source-commit",
        "dirty-source",
        "source-fingerprint",
        "test-provider",
        "checkpoint-layout",
        "projection-key",
        "layer-projection-fingerprint",
        "global-projection-fingerprint",
        "duplicate-provider-call-index",
        "wrong-timestep",
        "summary-projection-fingerprint",
        "summary-provider",
    ),
)
def test_fastvideo_learned_sla_receipt_rejects_forged_evidence(
    mutation: str,
) -> None:
    kind = "fastvideo_sla"
    provider = _FASTVIDEO_SLA_PROVIDERS[kind][0]
    grid = (31, 22, 40)
    report, profile = _receipt(kind=kind, provider=provider, grid=grid)
    report = copy.deepcopy(report)
    sparse_events = [
        event for event in report["events"] if event["execution"] == "sparse"
    ]
    event = sparse_events[0]
    if mutation == "provider-class":
        event["reference_provider_path"] = f"{provider}.Fake"
    elif mutation == "source-commit":
        event["provider_source_commit"] = "0" * 40
    elif mutation == "dirty-source":
        event["provider_source_clean"] = False
    elif mutation == "source-fingerprint":
        event["provider_source_fingerprint"] = "0" * 64
    elif mutation == "test-provider":
        event["injected_test_provider"] = True
    elif mutation == "checkpoint-layout":
        event["checkpoint_layout"] = "unknown"
    elif mutation == "projection-key":
        event["projection_source_keys"][0] += ".forged"
    elif mutation == "layer-projection-fingerprint":
        event["projection_weight_fingerprint"] = "0" * 64
    elif mutation == "global-projection-fingerprint":
        event["all_projection_weights_fingerprint"] = "0" * 64
    elif mutation == "duplicate-provider-call-index":
        sparse_events[1]["call_index"] = event["call_index"]
    elif mutation == "wrong-timestep":
        event["current_timestep"] = int(event["step"]) + 1
    elif mutation == "summary-projection-fingerprint":
        report["fastvideo_sla"]["all_projection_weights_fingerprint"] = "0" * 64
    else:
        report["fastvideo_sla"]["canonical_provider_path"] = f"{provider}.Fake"

    audit = _audit(report, profile=profile, provider=provider, grid=grid)

    assert audit["passed"] is False
    assert any("fastvideo_sla" in issue for issue in audit["issues"])


@pytest.mark.parametrize("kind", tuple(_LIGHTX2V_PROVIDERS))
def test_lightx2v_receipt_requires_canonical_pinned_provider(kind: str) -> None:
    _, provider = _LIGHTX2V_PROVIDERS[kind]
    grid = (31, 22, 40)
    report, profile = _receipt(kind=kind, provider=provider, grid=grid)

    audit = _audit(report, profile=profile, provider=provider, grid=grid)

    assert audit["passed"] is True
    assert audit["issues"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "forged-counters",
        "provider-suffix",
        "wrong-scheduled-grid",
        "missing-branch",
        "duplicate-event",
        "non-finalized",
        "missing-expected-steps",
    ),
)
def test_receipt_rejects_counter_and_coverage_bypasses(mutation: str) -> None:
    provider = "fastvideo_kernel.video_sparse_attn_bshd"
    grid = (31, 22, 40)
    report, profile = _receipt(kind="vsa", provider=provider, grid=grid)
    report = copy.deepcopy(report)
    if mutation == "forged-counters":
        report["kernel_attempts"] = 1
        report["sparse_calls"] = 1
        report["scheduled_dense_calls"] = report["expected_calls"] - 1
    elif mutation == "provider-suffix":
        forged = f"{provider}.fake"
        report["provider_path"] = forged
        report["provider_paths"] = [forged]
        for event in report["events"]:
            if event["execution"] == "sparse":
                event["provider_path"] = forged
    elif mutation == "wrong-scheduled-grid":
        report["events"][0]["grid_size"] = [31, 44, 80]
    elif mutation == "missing-branch":
        report["events"][0].pop("branch")
    elif mutation == "duplicate-event":
        report["events"][-1] = copy.deepcopy(report["events"][0])
    elif mutation == "non-finalized":
        report["finalized"] = False
    else:
        report["coverage"]["branches"]["positive"].pop("expected_steps")

    audit = _audit(report, profile=profile, provider=provider, grid=grid)

    assert audit["passed"] is False
    assert audit["issues"]


def test_vsa_receipt_rejects_mixed_allowed_providers() -> None:
    provider = "fastvideo_kernel.video_sparse_attn_bshd"
    grid = (31, 22, 40)
    report, profile = _receipt(kind="vsa", provider=provider, grid=grid)
    sparse_event = next(
        event for event in report["events"] if event["execution"] == "sparse"
    )
    sparse_event["provider_path"] = "fastvideo_kernel.video_sparse_attn"

    audit = audit_approximate_attention_receipt(
        report,
        expected_kind="vsa",
        allowed_provider_paths={
            "fastvideo_kernel.video_sparse_attn",
            "fastvideo_kernel.video_sparse_attn_bshd",
        },
        expected_grid=grid,
        expected_wrapped_blocks=30,
        expected_total_steps=3,
        expected_routed_steps=False,
        expected_branch_steps={
            "positive": tuple(range(3)),
            "negative": tuple(range(3)),
        },
        expected_profile=profile,
    )

    assert audit["passed"] is False
    assert any("event_schema" in issue for issue in audit["issues"])


@pytest.mark.parametrize(
    "mutation",
    (
        "event-commit",
        "event-dirty-source",
        "event-provider-family",
        "event-fingerprint",
        "event-runtime-effective",
        "missing-provider-calls",
        "obsolete-nested-receipt",
        "summary-commit",
        "summary-dirty-source",
        "summary-fingerprint",
        "summary-provider-family",
        "summary-provider",
    ),
)
def test_lightx2v_receipt_rejects_unpinned_or_incomplete_evidence(
    mutation: str,
) -> None:
    kind = "dynamic_sparse"
    _, provider = _LIGHTX2V_PROVIDERS[kind]
    grid = (31, 22, 40)
    report, profile = _receipt(kind=kind, provider=provider, grid=grid)
    report = copy.deepcopy(report)
    sparse_event = next(
        event for event in report["events"] if event["execution"] == "sparse"
    )
    if mutation == "event-commit":
        sparse_event["provider_source_commit"] = "0" * 40
    elif mutation == "event-dirty-source":
        sparse_event["provider_source_clean"] = False
    elif mutation == "event-provider-family":
        sparse_event["provider_family"] = "lightx2v/forged"
    elif mutation == "event-fingerprint":
        sparse_event["provider_source_fingerprint"] = "0" * 64
    elif mutation == "event-runtime-effective":
        sparse_event["runtime_effective"] = False
    elif mutation == "missing-provider-calls":
        sparse_event.pop("provider_calls")
    elif mutation == "obsolete-nested-receipt":
        sparse_event["provider_receipt"] = {"runtime_effective": True}
    elif mutation == "summary-commit":
        report["lightx2v"]["provider_source_commit"] = "0" * 40
    elif mutation == "summary-dirty-source":
        report["lightx2v"]["provider_source_clean"] = False
    elif mutation == "summary-fingerprint":
        report["lightx2v"]["provider_source_fingerprints"] = ["0" * 64]
    elif mutation == "summary-provider-family":
        report["lightx2v"]["provider_family"] = "lightx2v/forged"
    else:
        report["lightx2v"]["canonical_provider_symbol"] = f"{provider}.fake"

    audit = _audit(report, profile=profile, provider=provider, grid=grid)

    assert audit["passed"] is False
    assert any("lightx2v" in issue for issue in audit["issues"])


def test_routed_receipt_uses_explicit_global_total_steps() -> None:
    provider = "fastvideo_kernel.video_sparse_attn_bshd"
    grid = (31, 22, 40)
    branch_steps = {"positive": (0, 1)}
    report, profile = _receipt(
        kind="vsa",
        provider=provider,
        grid=grid,
        total_steps=5,
        dense_steps=0,
        branch_steps=branch_steps,
        routed_steps=True,
    )

    audit = _audit(
        report,
        profile=profile,
        provider=provider,
        grid=grid,
        total_steps=5,
        branch_steps=branch_steps,
        routed_steps=True,
    )
    assert audit["passed"] is True

    for event in report["events"]:
        event["total_steps"] = 2
    forged = _audit(
        report,
        profile=profile,
        provider=provider,
        grid=grid,
        total_steps=5,
        branch_steps=branch_steps,
        routed_steps=True,
    )
    assert forged["passed"] is False
    assert any("event_schema" in issue for issue in forged["issues"])
