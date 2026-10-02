"""Contracts for the pinned FastVideo learned SLA/SageSLA adapter."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations import (
    sparse_linear_attention as sla_module,
)
from worldfoundry.base_models.diffusion_model.optimizations.sparse_linear_attention import (
    PINNED_FASTVIDEO_COMMIT,
    FastVideoSLAAdapter,
    FastVideoSLAConfig,
    FastVideoSLAContractError,
    FastVideoSLALayerSpec,
    FastVideoSLAMetadata,
    FastVideoSLASourceIdentity,
    FastVideoSLAUnavailableError,
    fastvideo_sla_checkpoint_key_patterns,
    fastvideo_sla_runtime_expectation,
    resolve_fastvideo_sla_projection_weights,
    split_fastvideo_sla_projection_weights,
)


def _specs(count: int = 2) -> tuple[FastVideoSLALayerSpec, ...]:
    return tuple(
        FastVideoSLALayerSpec(
            layer_index=index,
            num_heads=2,
            head_size=64,
        )
        for index in range(count)
    )


def _projection(layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    weight = weight.div(64 * 64).add_(layer_index + 0.25)
    bias = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32).add_(layer_index)
    return weight, bias


def _checkpoint(
    *,
    layout: str = "fastvideo_diffusers",
    count: int = 2,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for layer_index in range(count):
        weight, bias = _projection(layer_index)
        if layout == "fastvideo_diffusers":
            prefix = f"blocks.{layer_index}.attn1.attn_impl.proj_l"
        elif layout == "turbodiffusion_original":
            prefix = (
                f"blocks.{layer_index}.self_attn.attn_op.local_attn.proj_l"
            )
        else:
            raise AssertionError(layout)
        values[f"{prefix}.weight"] = weight
        values[f"{prefix}.bias"] = bias
    values["blocks.0.to_q.weight"] = torch.randn(64, 64)
    return values


def _source_identity(
    *,
    commit: str = PINNED_FASTVIDEO_COMMIT,
    clean: bool = True,
) -> FastVideoSLASourceIdentity:
    fingerprint = hashlib.sha256(f"{commit}:{clean}".encode()).hexdigest()
    return FastVideoSLASourceIdentity(
        commit=commit,
        clean=clean,
        fingerprint=fingerprint,
        root="/pinned/FastVideo",
        source_file="/pinned/FastVideo/fastvideo/attention/backends/sla.py",
    )


class _FakeProvider(nn.Module):
    def __init__(
        self,
        spec: FastVideoSLALayerSpec,
        *,
        output_mode: str = "valid",
        delay: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj_l = nn.Linear(spec.head_size, spec.head_size, dtype=torch.float32)
        self.output_mode = output_mode
        self.delay = delay
        self.calls: list[dict[str, object]] = []
        self._active_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def forward(self, q, k, v, metadata):
        with self._active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(
                {"q": q, "k": k, "v": v, "metadata": metadata}
            )
            if self.output_mode == "wrong_shape":
                return q[:, :-1]
            if self.output_mode == "wrong_dtype":
                return q.to(torch.float64)
            if self.output_mode == "not_tensor":
                return "not-a-tensor"
            if self.output_mode == "error":
                raise RuntimeError("fake provider failed")
            return q + k + v
        finally:
            with self._active_lock:
                self.active -= 1


def _adapter(
    *,
    kind: str = "fastvideo_sla",
    count: int = 2,
    checkpoint: dict[str, torch.Tensor] | None = None,
    output_mode: str = "valid",
    delay: float = 0.0,
) -> tuple[FastVideoSLAAdapter, dict[int, _FakeProvider]]:
    providers: dict[int, _FakeProvider] = {}

    def factory(
        spec: FastVideoSLALayerSpec,
        _config: FastVideoSLAConfig,
    ) -> nn.Module:
        provider = _FakeProvider(spec, output_mode=output_mode, delay=delay)
        providers[spec.layer_index] = provider
        return provider

    adapter = FastVideoSLAAdapter(
        FastVideoSLAConfig(kind=kind),
        layer_specs=_specs(count),
        checkpoint_state_dict=checkpoint or _checkpoint(count=count),
        provider_factory=factory,
        source_identity=_source_identity(),
        allow_non_cuda_for_tests=True,
    )
    return adapter, providers


def _values(
    *,
    batch: int = 2,
    sequence: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch, sequence, 2, 64)
    return tuple(torch.randn(shape, dtype=torch.float32) for _ in range(3))


@pytest.mark.parametrize(
    ("kind", "expected_kind", "expected_topk"),
    [
        ("fastvideo_sla", "fastvideo_sla", 0.1),
        ("fastvideo-sla", "fastvideo_sla", 0.1),
        ("fastvideo_sagesla", "fastvideo_sagesla", 0.5),
    ],
)
def test_config_uses_unambiguous_fastvideo_kinds(
    kind: str,
    expected_kind: str,
    expected_topk: float,
) -> None:
    config = FastVideoSLAConfig(kind=kind)
    assert config.kind == expected_kind
    assert config.topk_ratio == expected_topk


@pytest.mark.parametrize("ambiguous", ["sla", "sagesla", "lightx2v_sla_mask"])
def test_config_rejects_lightx2v_or_ambiguous_sla_names(ambiguous: str) -> None:
    with pytest.raises(ValueError, match="LightX2V SLA mask"):
        FastVideoSLAConfig(kind=ambiguous)


def test_config_validates_topk_and_feature_map() -> None:
    assert FastVideoSLAConfig(
        kind="fastvideo_sla",
        topk_ratio=0.25,
        feature_map="ELU",
    ).feature_map == "elu"
    with pytest.raises(ValueError, match="topk_ratio"):
        FastVideoSLAConfig(kind="fastvideo_sla", topk_ratio=0.0)
    with pytest.raises(ValueError, match="feature_map"):
        FastVideoSLAConfig(kind="fastvideo_sla", feature_map="identity")


@pytest.mark.parametrize(
    ("kind", "provider_class", "family"),
    [
        (
            "fastvideo_sla",
            "SLAAttentionImpl",
            "fastvideo/sparse-linear-attention",
        ),
        (
            "fastvideo_sagesla",
            "SageSLAAttentionImpl",
            "fastvideo/sage-sparse-linear-attention",
        ),
    ],
)
def test_runtime_expectation_freezes_canonical_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    provider_class: str,
    family: str,
) -> None:
    identity = _source_identity()
    monkeypatch.setattr(
        sla_module,
        "_load_upstream_bundle",
        lambda _kind: SimpleNamespace(source_identity=identity),
    )

    expectation = fastvideo_sla_runtime_expectation(kind)

    assert expectation == {
        "kind": kind,
        "provider_path": (
            "fastvideo.attention.backends.sla." + provider_class
        ),
        "provider_family": family,
        "commit": PINNED_FASTVIDEO_COMMIT,
        "source_fingerprint": identity.fingerprint,
        "source_root": identity.root,
        "source_file": identity.source_file,
    }


@pytest.mark.parametrize(
    "layout",
    ["fastvideo_diffusers", "turbodiffusion_original"],
)
def test_resolver_accepts_both_audited_checkpoint_layouts(layout: str) -> None:
    resolved = resolve_fastvideo_sla_projection_weights(
        _checkpoint(layout=layout),
        _specs(),
    )
    assert resolved.layout == layout
    assert list(resolved.projections) == [0, 1]
    assert len(resolved.fingerprint) == 64
    for layer_index, projection in resolved.projections.items():
        expected_weight, expected_bias = _projection(layer_index)
        torch.testing.assert_close(projection.weight, expected_weight)
        torch.testing.assert_close(projection.bias, expected_bias)
        assert len(projection.fingerprint) == 64


def test_checkpoint_key_patterns_document_fastvideo_and_turbo_layouts() -> None:
    assert fastvideo_sla_checkpoint_key_patterns() == {
        "fastvideo_diffusers": (
            "blocks.<layer>.attn1.attn_impl.proj_l.{weight,bias}"
        ),
        "turbodiffusion_original": (
            "blocks.<layer>.self_attn.attn_op.local_attn.proj_l.{weight,bias}"
        ),
    }


def test_splitter_preserves_dense_weights_and_exact_projection_keys() -> None:
    checkpoint = _checkpoint(count=1)
    model_state, projection_state = split_fastvideo_sla_projection_weights(
        checkpoint
    )
    assert set(model_state) == {"blocks.0.to_q.weight"}
    assert set(projection_state) == {
        "blocks.0.attn1.attn_impl.proj_l.weight",
        "blocks.0.attn1.attn_impl.proj_l.bias",
    }
    with pytest.raises(FastVideoSLAContractError, match="unsupported"):
        split_fastvideo_sla_projection_weights(
            {"blocks.0.self_attn.proj_l.weight": torch.ones(64, 64)}
        )


def test_projection_checkpoint_rejects_non_string_keys() -> None:
    checkpoint: dict[object, object] = dict(_checkpoint(count=1))
    checkpoint[7] = torch.ones(1)
    with pytest.raises(FastVideoSLAContractError, match="keys must be strings"):
        split_fastvideo_sla_projection_weights(checkpoint)
    with pytest.raises(FastVideoSLAContractError, match="keys must be strings"):
        resolve_fastvideo_sla_projection_weights(checkpoint, _specs(1))


def test_resolver_rejects_missing_and_unexpected_layer_weights() -> None:
    missing = _checkpoint(count=2)
    missing.pop("blocks.1.attn1.attn_impl.proj_l.bias")
    with pytest.raises(FastVideoSLAContractError, match="missing=.*layer=1"):
        resolve_fastvideo_sla_projection_weights(missing, _specs())

    unexpected = _checkpoint(count=3)
    with pytest.raises(FastVideoSLAContractError, match="unexpected.*layers"):
        resolve_fastvideo_sla_projection_weights(unexpected, _specs())


def test_resolver_rejects_duplicate_aliases_mixed_layouts_and_unknown_keys() -> None:
    duplicate = _checkpoint(count=1)
    weight, bias = _projection(0)
    turbo = "blocks.0.self_attn.attn_op.local_attn.proj_l"
    duplicate[f"{turbo}.weight"] = weight
    duplicate[f"{turbo}.bias"] = bias
    with pytest.raises(FastVideoSLAContractError, match="duplicate.*mappings"):
        resolve_fastvideo_sla_projection_weights(duplicate, _specs(1))

    mixed = _checkpoint(count=2)
    mixed.pop("blocks.1.attn1.attn_impl.proj_l.weight")
    mixed.pop("blocks.1.attn1.attn_impl.proj_l.bias")
    mixed_weight, mixed_bias = _projection(1)
    mixed_prefix = "blocks.1.self_attn.attn_op.local_attn.proj_l"
    mixed[f"{mixed_prefix}.weight"] = mixed_weight
    mixed[f"{mixed_prefix}.bias"] = mixed_bias
    with pytest.raises(FastVideoSLAContractError, match="may not mix"):
        resolve_fastvideo_sla_projection_weights(mixed, _specs())

    unknown = _checkpoint(count=1)
    unknown["blocks.0.self_attn.proj_l.weight"] = weight
    with pytest.raises(FastVideoSLAContractError, match="unsupported.*keys"):
        resolve_fastvideo_sla_projection_weights(unknown, _specs(1))


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "zero", "non_tensor"])
def test_resolver_rejects_malformed_or_untrained_projection_weights(
    failure: str,
) -> None:
    checkpoint: dict[str, object] = dict(_checkpoint(count=1))
    weight_key = "blocks.0.attn1.attn_impl.proj_l.weight"
    if failure == "shape":
        checkpoint[weight_key] = torch.ones(63, 64)
        match = "shape"
    elif failure == "nonfinite":
        checkpoint[weight_key] = torch.full((64, 64), float("nan"))
        match = "non-finite"
    elif failure == "zero":
        checkpoint[weight_key] = torch.zeros(64, 64)
        match = "all zero"
    else:
        checkpoint[weight_key] = object()
        match = "must be a Tensor"
    with pytest.raises(FastVideoSLAContractError, match=match):
        resolve_fastvideo_sla_projection_weights(checkpoint, _specs(1))


def test_adapter_loads_learned_weights_calls_provider_and_receipts() -> None:
    adapter, providers = _adapter()
    q, k, v = _values()
    attempts: list[bool] = []

    result = adapter(
        1,
        q,
        k,
        v,
        current_timestep=7,
        on_provider_attempt=lambda: attempts.append(True),
    )

    torch.testing.assert_close(result.output, q + k + v)
    expected_weight, expected_bias = _projection(1)
    torch.testing.assert_close(providers[1].proj_l.weight, expected_weight)
    torch.testing.assert_close(providers[1].proj_l.bias, expected_bias)
    assert attempts == [True]
    assert len(providers[1].calls) == 1
    metadata = providers[1].calls[0]["metadata"]
    assert metadata == FastVideoSLAMetadata(current_timestep=7, topk_ratio=0.1)
    receipt = result.receipt
    assert receipt["algorithm"] == "fastvideo_sla"
    assert receipt["provider_family"] == "fastvideo/sparse-linear-attention"
    assert receipt["reference_provider_path"].endswith(".SLAAttentionImpl")
    assert receipt["injected_test_provider"] is True
    assert receipt["reference_parity_verified"] is False
    assert receipt["reference_fastvideo_commit"] == PINNED_FASTVIDEO_COMMIT
    assert receipt["provider_source_commit"] == PINNED_FASTVIDEO_COMMIT
    assert receipt["provider_source_clean"] is True
    assert receipt["checkpoint_layout"] == "fastvideo_diffusers"
    assert receipt["projection_source_keys"] == [
        "blocks.1.attn1.attn_impl.proj_l.weight",
        "blocks.1.attn1.attn_impl.proj_l.bias",
    ]
    assert len(receipt["provider_fingerprint"]) == 64
    assert len(receipt["projection_weight_fingerprint"]) == 64
    assert len(receipt["all_projection_weights_fingerprint"]) == 64
    assert receipt["call_index"] == 1
    assert receipt["layer_call_index"] == 1
    assert receipt["q"] == {
        "shape": [2, 8, 2, 64],
        "device": "cpu",
        "dtype": "torch.float32",
        "contiguous": True,
    }
    assert receipt["output"] == receipt["q"]
    assert receipt["runtime_effective"] is True
    assert json.dumps(receipt, sort_keys=True)
    assert adapter.runtime_report()["layer_calls"] == {0: 0, 1: 1}


def test_parent_dtype_conversion_preserves_fp32_projection_master() -> None:
    adapter, providers = _adapter(count=1)
    expected_weight, expected_bias = _projection(0)

    adapter.to(dtype=torch.bfloat16)

    assert providers[0].proj_l.weight.dtype is torch.float32
    assert providers[0].proj_l.bias.dtype is torch.float32
    torch.testing.assert_close(providers[0].proj_l.weight, expected_weight)
    torch.testing.assert_close(providers[0].proj_l.bias, expected_bias)


def test_sagesla_is_a_distinct_fastvideo_provider_family() -> None:
    adapter, providers = _adapter(kind="fastvideo_sagesla", count=1)
    q, k, v = _values(batch=1)

    result = adapter(0, q, k, v, current_timestep=0)

    metadata = providers[0].calls[0]["metadata"]
    assert metadata.topk_ratio == 0.5
    assert result.receipt["algorithm"] == "fastvideo_sagesla"
    assert (
        result.receipt["provider_family"]
        == "fastvideo/sage-sparse-linear-attention"
    )
    assert result.receipt["reference_provider_path"].endswith(
        ".SageSLAAttentionImpl"
    )


def test_injected_provider_seam_is_explicit_and_still_requires_pinned_source() -> None:
    def factory(spec, _config):
        return _FakeProvider(spec)

    kwargs = {
        "config": FastVideoSLAConfig(kind="fastvideo_sla"),
        "layer_specs": _specs(1),
        "checkpoint_state_dict": _checkpoint(count=1),
        "provider_factory": factory,
    }
    with pytest.raises(ValueError, match="test-only seam"):
        FastVideoSLAAdapter(
            **kwargs,
            source_identity=_source_identity(),
        )
    with pytest.raises(ValueError, match="source_identity"):
        FastVideoSLAAdapter(
            **kwargs,
            allow_non_cuda_for_tests=True,
        )
    with pytest.raises(FastVideoSLAUnavailableError, match="expected"):
        FastVideoSLAAdapter(
            **kwargs,
            source_identity=_source_identity(commit="0" * 40),
            allow_non_cuda_for_tests=True,
        )
    with pytest.raises(FastVideoSLAUnavailableError, match="clean=False"):
        FastVideoSLAAdapter(
            **kwargs,
            source_identity=_source_identity(clean=False),
            allow_non_cuda_for_tests=True,
        )


@pytest.mark.parametrize(
    ("output_mode", "match"),
    [
        ("wrong_shape", "returned shape"),
        ("wrong_dtype", "changed device/dtype"),
        ("not_tensor", "expected Tensor"),
    ],
)
def test_adapter_rejects_malformed_provider_outputs(
    output_mode: str,
    match: str,
) -> None:
    adapter, _ = _adapter(count=1, output_mode=output_mode)
    q, k, v = _values(batch=1)

    with pytest.raises(FastVideoSLAContractError, match=match):
        adapter(0, q, k, v, current_timestep=0)

    report = adapter.runtime_report()
    assert report["attempts"] == 1
    assert report["calls"] == 0
    assert report["errors"] == 1


def test_provider_exception_fails_closed_without_dense_output() -> None:
    adapter, _ = _adapter(count=1, output_mode="error")
    q, k, v = _values(batch=1)

    with pytest.raises(FastVideoSLAUnavailableError, match="fake provider failed"):
        adapter(0, q, k, v, current_timestep=0)

    assert adapter.runtime_report()["errors"] == 1


def test_adapter_enforces_strict_bshd_input_contract() -> None:
    adapter, _ = _adapter(count=1)
    q, k, v = _values(batch=1)
    with pytest.raises(FastVideoSLAContractError, match="BSHD"):
        adapter(0, q[0], k[0], v[0], current_timestep=0)
    with pytest.raises(FastVideoSLAContractError, match="shapes must match"):
        adapter(0, q, k[:, :-1], v, current_timestep=0)
    with pytest.raises(FastVideoSLAContractError, match="dtypes must match"):
        adapter(0, q, k.to(torch.float16), v, current_timestep=0)
    noncontiguous = torch.randn(1, 8, 2, 128)[..., ::2]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(FastVideoSLAContractError, match="contiguous"):
        adapter(0, noncontiguous, k, v, current_timestep=0)
    with pytest.raises(FastVideoSLAContractError, match="was not installed"):
        adapter(9, q, k, v, current_timestep=0)
    with pytest.raises(FastVideoSLAContractError, match="current_timestep"):
        adapter(0, q, k, v, current_timestep=-1)


def test_production_contract_requires_cuda_bfloat16() -> None:
    adapter, _ = _adapter(count=1)
    adapter._allow_non_cuda_for_tests = False
    q, k, v = _values(batch=1)
    with pytest.raises(FastVideoSLAUnavailableError, match="requires CUDA"):
        adapter(0, q, k, v, current_timestep=0)


def test_projection_fingerprint_changes_with_learned_weights() -> None:
    first = resolve_fastvideo_sla_projection_weights(
        _checkpoint(count=1),
        _specs(1),
    )
    changed = _checkpoint(count=1)
    changed["blocks.0.attn1.attn_impl.proj_l.weight"][0, 0] += 1.0
    second = resolve_fastvideo_sla_projection_weights(changed, _specs(1))
    assert first.fingerprint != second.fingerprint
    assert (
        first.projections[0].fingerprint
        != second.projections[0].fingerprint
    )


def test_source_identity_is_cached_and_forward_never_runs_subprocess(
    monkeypatch,
    tmp_path,
) -> None:
    checkout = tmp_path / "FastVideo"
    source_file = checkout / "fastvideo" / "attention" / "backends" / "sla.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# pinned SLA provider\n", encoding="utf-8")
    (checkout / ".git").mkdir()
    module = SimpleNamespace(__file__=str(source_file))
    calls: list[tuple[str, ...]] = []

    def fake_run(args, **_kwargs):
        calls.append(tuple(args))
        if "rev-parse" in args:
            return SimpleNamespace(
                returncode=0,
                stdout=PINNED_FASTVIDEO_COMMIT + "\n",
            )
        if "status" in args:
            return SimpleNamespace(returncode=0, stdout="")
        raise AssertionError(args)

    monkeypatch.setattr(sla_module.subprocess, "run", fake_run)
    sla_module._SOURCE_IDENTITIES.clear()
    first = sla_module._provider_source_identity(module)
    second = sla_module._provider_source_identity(module)
    assert first == second
    assert first.commit == PINNED_FASTVIDEO_COMMIT
    assert first.clean is True
    assert len(calls) == 2

    adapter, _ = _adapter(count=1)
    monkeypatch.setattr(
        sla_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess reached attention hot path")
        ),
    )
    q, k, v = _values(batch=1)
    adapter(0, q, k, v, current_timestep=0)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("kind", "class_name"),
    [
        ("fastvideo_sla", "SLAAttentionImpl"),
        ("fastvideo_sagesla", "SageSLAAttentionImpl"),
    ],
)
def test_production_loader_selects_exact_upstream_class(
    monkeypatch,
    kind: str,
    class_name: str,
) -> None:
    provider_type = type(class_name, (nn.Module,), {})
    provider_type.__module__ = "fastvideo.attention.backends.sla"
    metadata_type = type("SLAAttentionMetadata", (), {})
    fake_module = SimpleNamespace(
        SLAAttentionImpl=(
            provider_type
            if class_name == "SLAAttentionImpl"
            else type(
                "SLAAttentionImpl",
                (nn.Module,),
                {"__module__": "fastvideo.attention.backends.sla"},
            )
        ),
        SageSLAAttentionImpl=(
            provider_type
            if class_name == "SageSLAAttentionImpl"
            else type(
                "SageSLAAttentionImpl",
                (nn.Module,),
                {"__module__": "fastvideo.attention.backends.sla"},
            )
        ),
        SLAAttentionMetadata=metadata_type,
    )
    monkeypatch.setattr(
        sla_module.importlib,
        "import_module",
        lambda name: fake_module
        if name == "fastvideo.attention.backends.sla"
        else pytest.fail(name),
    )
    monkeypatch.setattr(
        sla_module,
        "_provider_source_identity",
        lambda _module: _source_identity(),
    )

    bundle = sla_module._load_upstream_bundle(kind)

    assert bundle.provider_type is provider_type
    assert (
        f"{bundle.provider_type.__module__}.{bundle.provider_type.__name__}"
        == f"fastvideo.attention.backends.sla.{class_name}"
    )


def test_runtime_counters_are_safe_under_parallel_forwards() -> None:
    adapter, providers = _adapter(count=1, delay=0.01)
    q, k, v = _values(batch=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda index: adapter(
                    0,
                    q,
                    k,
                    v,
                    current_timestep=index,
                ),
                range(8),
            )
        )

    assert sorted(result.receipt["call_index"] for result in results) == list(
        range(1, 9)
    )
    assert sorted(
        result.receipt["layer_call_index"] for result in results
    ) == list(range(1, 9))
    assert providers[0].max_active > 1
    assert adapter.runtime_report() == {
        "kind": "fastvideo_sla",
        "provider_fingerprint": adapter._provider_fingerprint,
        "checkpoint_layout": "fastvideo_diffusers",
        "all_projection_weights_fingerprint": adapter._checkpoint.fingerprint,
        "expected_layers": [0],
        "attempts": 8,
        "calls": 8,
        "errors": 0,
        "layer_attempts": {0: 8},
        "layer_calls": {0: 8},
        "layer_errors": {0: 0},
        **_source_identity().to_receipt(),
    }
