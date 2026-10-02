"""Contracts for the fail-closed LightX2V sparse-provider adapters."""

from __future__ import annotations

import subprocess
from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.optimizations import sparse_mask_attention as sparse_module
from worldfoundry.base_models.diffusion_model.optimizations.sparse_mask_attention import (
    PINNED_LIGHTX2V_COMMIT,
    LightX2VSparseAdapter,
    LightX2VSparseConfig,
    LightX2VSparseUnavailableError,
    parse_lightx2v_sparse,
)


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply(self, q, k, v, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, **kwargs})
        return (q + k + v).reshape(q.shape[0], -1)


def _values(batch: int = 2):
    shape = (batch, 24, 3, 8)
    return tuple(torch.randn(shape) for _ in range(3))


@pytest.mark.parametrize(
    ("selected", "expected_kind", "expected_operator"),
    [
        ("dynamic", "dynamic_sparse", "triton"),
        ("sparge", "sparge", "meansim_sage2"),
        ("nbhd", "nbhd", "magi"),
        ("sla", "lightx2v_sla_mask", "triton"),
        ("flex-block", "flexblock", "flex_block"),
        ("sage_sla", "lightx2v_spas_sage", "sage2"),
    ],
)
def test_parse_kinds_and_published_defaults(
    selected: str,
    expected_kind: str,
    expected_operator: str,
) -> None:
    config = parse_lightx2v_sparse(selected)
    assert config.kind == expected_kind
    assert config.operator == expected_operator


def test_parse_mapping_is_strict_and_validates_combinations() -> None:
    config = parse_lightx2v_sparse(
        {
            "kind": "sla",
            "operator": "sage3",
            "sparsity_ratio": 0.75,
            "per_block_mean": True,
        }
    )
    assert config == LightX2VSparseConfig(
        kind="sla",
        operator="sage3",
        sparsity_ratio=0.75,
        per_block_mean=True,
    )
    with pytest.raises(ValueError, match="unknown.*fields"):
        parse_lightx2v_sparse({"kind": "sla", "typo": True})
    with pytest.raises(ValueError, match="invalid for flexblock"):
        LightX2VSparseConfig(kind="flexblock", operator="sage2")
    with pytest.raises(ValueError, match="sparsity_ratio"):
        LightX2VSparseConfig(kind="sla", sparsity_ratio=1.0)


def test_adapter_batches_upstream_apply_and_returns_auditable_receipt() -> None:
    provider = _FakeProvider()
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="dynamic_sparse"),
        provider=provider,
        provider_path="lightx2v.fake.DynamicSparse.apply",
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=2)

    result = adapter(q, k, v, grid=(2, 3, 4), softmax_scale=0.125)

    torch.testing.assert_close(result.output, q + k + v)
    assert len(provider.calls) == 2
    assert all(call["max_seqlen_q"] == 24 for call in provider.calls)
    assert all(call["max_seqlen_kv"] == 24 for call in provider.calls)
    assert all(call["softmax_scale"] == 0.125 for call in provider.calls)
    assert all("cu_seqlens_q" not in call for call in provider.calls)
    assert all("cu_seqlens_kv" not in call for call in provider.calls)
    assert result.receipt == {
        "algorithm": "dynamic_sparse",
        "provider_family": "lightx2v/dynamic-sparse",
        "operator": "triton",
        "provider_path": "lightx2v.fake.DynamicSparse.apply",
        "canonical_sparse_provider_path": "lightx2v.fake.DynamicSparse.apply",
        "adapter_path": f"{_FakeProvider.__module__}._FakeProvider.apply",
        "provider_calls": 2,
        "reference_lightx2v_commit": PINNED_LIGHTX2V_COMMIT,
        "provider_source_commit": None,
        "provider_source_clean": None,
        "provider_source_fingerprint": None,
        "provider_source_root": None,
        "grid": [2, 3, 4],
        "sparsity_ratio": 0.8,
        "provider_config": {
            "sparsity_ratio": 0.8,
            "operator": "triton",
            "nbhd_coefficient": [1.0, 0.5, 0.056],
            "nbhd_min_width": 1.0,
            "attnmap_frame_num": 2,
            "per_block_mean": False,
            "pool_size": 128,
            "skip_timesteps": -1,
            "dense_attn_type": "flash_attn3",
            "svg_sample_mse_max_row": 10000,
            "svg_num_sampled_rows": 64,
            "svg_context_length": 0,
        },
        "input": {
            "shape": [2, 24, 3, 8],
            "device": "cpu",
            "dtype": "torch.float32",
        },
        "output": {
            "shape": [2, 24, 3, 8],
            "device": "cpu",
            "dtype": "torch.float32",
        },
        "runtime_effective": True,
        "execution": "sparse",
        "sparse_kernel_executed": True,
        "provider_dense_executed": False,
        "capability_contract": {
            "preflight_passed": None,
            "cuda_compute_capability": None,
            "provider_class": f"{_FakeProvider.__module__}._FakeProvider.apply",
            "required_symbols": [],
            "injected_test_provider": True,
        },
        "reference_parity_verified": False,
    }


def test_prepare_materializes_provider_without_executing_apply() -> None:
    provider = _FakeProvider()
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="dynamic_sparse"),
        provider=provider,
        provider_path="lightx2v.fake.DynamicSparse.apply",
        allow_non_cuda_for_tests=True,
    )

    receipt = adapter.prepare(device="cpu", heads=3, head_dim=8)

    assert receipt["preflight_completed"] is True
    assert receipt["provider_materialized"] is True
    assert receipt["provider_path"] == "lightx2v.fake.DynamicSparse.apply"
    assert receipt["capability_contract"]["injected_test_provider"] is True
    assert len(adapter._providers) == 1
    assert provider.calls == []


def test_prepare_defers_grid_local_provider_after_preflight() -> None:
    provider = _FakeProvider()
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="nbhd"),
        provider=provider,
        allow_non_cuda_for_tests=True,
    )

    receipt = adapter.prepare(device="cpu", heads=3, head_dim=8)

    assert receipt["preflight_completed"] is True
    assert receipt["provider_materialized"] is False
    assert receipt["deferred_reason"] == "complete runtime grid required"
    assert not adapter._providers
    assert provider.calls == []


def test_adapter_rejects_grid_input_and_provider_contract_mismatches() -> None:
    q, k, v = _values(batch=1)
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="sla"),
        provider=_FakeProvider(),
        allow_non_cuda_for_tests=True,
    )
    with pytest.raises(ValueError, match="contains 25 tokens"):
        adapter(q, k, v, grid=(1, 5, 5))
    with pytest.raises(ValueError, match="shapes must match"):
        adapter(q, k[:, :-1], v)

    malformed = SimpleNamespace(apply=lambda *_args, **_kwargs: torch.zeros(3, 4))
    malformed_adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="sla"),
        provider=malformed,
        provider_path="fake.malformed",
        allow_non_cuda_for_tests=True,
    )
    with pytest.raises(RuntimeError, match="returned shape"):
        malformed_adapter(q, k, v)


def test_adapter_requires_cuda_bfloat16_outside_contract_tests() -> None:
    q, k, v = _values(batch=1)
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="sla"),
        provider=_FakeProvider(),
    )
    with pytest.raises(LightX2VSparseUnavailableError, match="require CUDA"):
        adapter(q, k, v)


@pytest.mark.parametrize(
    ("kind", "provider_attributes"),
    (
        (
            "dynamic_sparse",
            {"BLKK": 128, "topk": 1.0 - 0.9},
        ),
        (
            "lightx2v_spas_sage",
            {
                "mask_generator": SimpleNamespace(
                    k_block_size=128,
                    topk_ratio=1.0 - 0.9,
                )
            },
        ),
    ),
)
def test_sla_zero_block_topk_fails_before_provider_kernel(
    kind: str,
    provider_attributes: dict[str, object],
) -> None:
    provider = _FakeProvider()
    for name, value in provider_attributes.items():
        setattr(provider, name, value)
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(
            kind=kind,
            operator="sage2",
            sparsity_ratio=0.9,
        ),
        provider=provider,
        allow_non_cuda_for_tests=True,
    )
    shape = (1, 128, 1, 1)
    q, k, v = (torch.randn(shape) for _ in range(3))

    with pytest.raises(
        LightX2VSparseUnavailableError,
        match=r"select zero key blocks.*empty LUT",
    ):
        adapter(q, k, v, grid=(2, 8, 8))

    assert provider.calls == []


def test_sla_supported_geometry_reaches_provider_kernel() -> None:
    provider = _FakeProvider()
    provider.BLKK = 128
    provider.topk = 1.0 - 0.9
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(
            kind="dynamic_sparse",
            operator="sage2",
            sparsity_ratio=0.9,
        ),
        provider=provider,
        allow_non_cuda_for_tests=True,
    )
    shape = (1, 1408, 1, 1)
    q, k, v = (torch.randn(shape) for _ in range(3))

    result = adapter(q, k, v, grid=(2, 16, 44))

    torch.testing.assert_close(result.output, q + k + v)
    assert len(provider.calls) == 1


def test_nbhd_frame_state_is_grid_local_and_must_match() -> None:
    provider = _FakeProvider()
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(
            kind="nbhd",
            attnmap_frame_num=2,
            nbhd_coefficient=(1.0, 0.5),
        ),
        provider=provider,
        provider_path="lightx2v.fake.NBHD.apply",
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=1)
    result = adapter(q, k, v, grid=(2, 3, 4))
    assert result.receipt["attnmap_frame_num"] == 2
    assert result.receipt["nbhd_coefficient"] == [1.0, 0.5]
    with pytest.raises(ValueError, match="attnmap_frame_num/grid mismatch"):
        adapter(q, k, v, grid=(3, 2, 4))


def test_missing_upstream_dependency_fails_closed(monkeypatch) -> None:
    def unavailable(_config, _frame_num):
        raise LightX2VSparseUnavailableError("pinned provider unavailable")

    monkeypatch.setattr(sparse_module, "_build_provider", unavailable)
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="dynamic_sparse"),
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=1)
    with pytest.raises(
        LightX2VSparseUnavailableError,
        match="pinned provider unavailable",
    ):
        adapter(q, k, v)


def test_provider_output_cannot_change_dtype_or_device() -> None:
    class WrongDtype:
        def apply(self, q, _k, _v, **_kwargs):
            return q.to(torch.float64)

    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="sla"),
        provider=WrongDtype(),
        provider_path="fake.wrong_dtype",
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=1)
    with pytest.raises(RuntimeError, match="changed device/dtype"):
        adapter(q, k, v)


def test_nbhd_provider_cache_is_partitioned_by_complete_grid() -> None:
    provider = _FakeProvider()
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="nbhd"),
        provider=provider,
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=1)

    adapter(q, k, v, grid=(3, 2, 4))
    adapter(q, k, v, grid=(3, 4, 2))
    adapter(q, k, v, grid=(3, 2, 4))

    assert len(adapter._providers) == 2
    assert len(provider.calls) == 3
    with pytest.raises(ValueError, match="complete 3D grid"):
        adapter(q, k, v)


def test_general_nbhd_flashinfer_fixes_constructor_and_isolates_mask_cache(
    monkeypatch,
) -> None:
    class BrokenFlashinferOperator:
        def __init__(
            self,
            q_block_size=128,
            k_block_size=128,
            operator_setting=None,
        ):
            raise AssertionError(
                "upstream positional GeneralSparse construction must be bypassed"
            )

    class GlobalNbhdMask:
        seqlen = None
        mask = None

        def __init__(
            self,
            q_block_size=128,
            k_block_size=128,
            sparse_setting=None,
            attnmap_frame_num=None,
        ):
            self.q_block_size = q_block_size
            self.k_block_size = k_block_size
            self.sparse_setting = dict(sparse_setting or {})
            self.attnmap_frame_num = attnmap_frame_num

        def __call__(self, q, _k):
            if q.shape[0] == GlobalNbhdMask.seqlen:
                return GlobalNbhdMask.mask
            GlobalNbhdMask.seqlen = q.shape[0]
            GlobalNbhdMask.mask = torch.full(
                (1, 1, 1, 1),
                self.attnmap_frame_num,
                dtype=torch.int32,
            )
            return GlobalNbhdMask.mask

        def reorg(self, q, k, v):
            return q, k, v

        def restore(self, output):
            return output

    class FakeGeneralSparse:
        def __init__(self):
            self._setup_operator()
            self._setup_mask_generator()

        def _setup_operator(self):
            raise AssertionError("configured provider must replace this method")

        def _setup_mask_generator(self):
            self.mask_generator = GlobalNbhdMask(
                self.operator.q_block_size,
                self.operator.k_block_size,
                self.sparse_setting,
                self.attnmap_frame_num,
            )

        def apply(self, q, _k, _v, **_kwargs):
            return q

    class FakeFlashinferWrapper:
        def __init__(self, _workspace, *, backend):
            self.backend = backend

        def plan(self, **_kwargs):
            pass

        def run(self, q, _k, _v):
            return q

    def fake_import_symbol(_module_name, symbol):
        if symbol == "GeneralSparseAttnWeight":
            return FakeGeneralSparse
        if symbol == "FlashinferOperator":
            return BrokenFlashinferOperator
        raise AssertionError(symbol)

    monkeypatch.setattr(sparse_module, "_import_symbol", fake_import_symbol)
    monkeypatch.setattr(
        sparse_module,
        "_flashinfer_wrapper_type",
        lambda: FakeFlashinferWrapper,
    )
    config = LightX2VSparseConfig(kind="nbhd", operator="flashinfer")

    provider_three = sparse_module._build_general_provider(config, 3)
    provider_four = sparse_module._build_general_provider(config, 4)

    assert provider_three.operator.q_block_size == 128
    assert provider_three.operator.k_block_size == 128
    assert provider_three.operator.operator_setting == {}
    assert provider_three._worldfoundry_adapter_receipt == {
        "compatibility_adaptations": [
            "flashinfer_constructor_and_device_plan_isolation",
            "nbhd_provider_local_mask_cache",
        ],
        "state_isolation": "device/grid/request-safe",
    }
    values = torch.randn(24, 2, 8)
    assert provider_three.mask_generator(values, values).item() == 3
    assert provider_four.mask_generator(values, values).item() == 4
    # A second same-length call must recover provider_three's private mask,
    # not the last value written by provider_four's upstream class cache.
    assert provider_three.mask_generator(values, values).item() == 3
    assert GlobalNbhdMask.seqlen is None
    assert GlobalNbhdMask.mask is None


def test_provider_source_identity_is_cached_outside_attention_hot_path(
    monkeypatch,
    tmp_path,
) -> None:
    checkout = tmp_path / "LightX2V"
    source_dir = checkout / "lightx2v"
    source_dir.mkdir(parents=True)
    (checkout / ".git").mkdir()
    source_file = source_dir / "fake_provider.py"
    source_file.write_text("# pinned provider\n", encoding="utf-8")

    class LightX2VBase:
        def apply(self, q, _k, _v, **_kwargs):
            return q

    LightX2VBase.__module__ = "lightx2v.fake_provider"

    class ConfiguredProvider(LightX2VBase):
        pass

    calls: list[tuple[str, ...]] = []

    def fake_run(args, **_kwargs):
        calls.append(tuple(args))
        if "rev-parse" in args:
            return SimpleNamespace(
                returncode=0,
                stdout=PINNED_LIGHTX2V_COMMIT + "\n",
            )
        if "status" in args:
            return SimpleNamespace(returncode=0, stdout="")
        raise AssertionError(args)

    monkeypatch.setattr(
        sparse_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(source_file)),
    )
    monkeypatch.setattr(sparse_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sparse_module,
        "_build_provider",
        lambda _config, _frame_num: ConfiguredProvider(),
    )
    sparse_module._PROVIDER_SOURCE_IDENTITIES.clear()
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="dynamic_sparse"),
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=1)

    first = adapter(q, k, v, grid=(2, 3, 4))
    second = adapter(q, k, v, grid=(2, 3, 4))
    assert first.receipt["provider_source_commit"] == PINNED_LIGHTX2V_COMMIT
    assert first.receipt["provider_source_clean"] is True
    assert first.receipt["reference_parity_verified"] is True
    assert second.receipt["provider_source_fingerprint"] == first.receipt[
        "provider_source_fingerprint"
    ]
    assert len(calls) == 2
    assert sum("rev-parse" in call for call in calls) == 1
    assert sum("status" in call for call in calls) == 1


def test_provider_source_identity_retries_transient_shared_fs_timeout(
    monkeypatch,
    tmp_path,
) -> None:
    checkout = tmp_path / "LightX2V"
    source_dir = checkout / "lightx2v"
    source_dir.mkdir(parents=True)
    (checkout / ".git").mkdir()
    source_file = source_dir / "fake_provider.py"
    source_file.write_text("# pinned provider\n", encoding="utf-8")

    class LightX2VBase:
        pass

    LightX2VBase.__module__ = "lightx2v.fake_provider"
    calls: list[tuple[tuple[str, ...], float]] = []

    def fake_run(args, **kwargs):
        calls.append((tuple(args), kwargs["timeout"]))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if "rev-parse" in args:
            return SimpleNamespace(
                returncode=0,
                stdout=PINNED_LIGHTX2V_COMMIT + "\n",
            )
        if "status" in args:
            return SimpleNamespace(returncode=0, stdout="")
        raise AssertionError(args)

    monkeypatch.setattr(
        sparse_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(source_file)),
    )
    monkeypatch.setattr(sparse_module.subprocess, "run", fake_run)
    sparse_module._PROVIDER_SOURCE_IDENTITIES.clear()

    identity = sparse_module._provider_source_identity(LightX2VBase())

    assert identity["provider_source_commit"] == PINNED_LIGHTX2V_COMMIT
    assert identity["provider_source_clean"] is True
    assert len(calls) == 3
    assert sum("rev-parse" in call for call, _timeout in calls) == 2
    assert sum("status" in call for call, _timeout in calls) == 1
    assert all(
        timeout == sparse_module._PROVIDER_GIT_TIMEOUT_SECONDS
        for _call, timeout in calls
    )


@pytest.mark.parametrize(
    ("selected", "expected_kind", "expected_operator"),
    [
        ("draft", "draft_attn", "magi"),
        ("radial", "radial_attn", "magi"),
        ("rainfusion", "rainfusion_attn", "flashinfer"),
        ("svg", "svg_attn", "flex_attention"),
        ("svg2", "svg2_attn", "flashinfer"),
        ("svg_mask_generator", "lightx2v_svg_mask", "magi"),
    ],
)
def test_parse_additional_pinned_provider_families(
    selected: str,
    expected_kind: str,
    expected_operator: str,
) -> None:
    config = parse_lightx2v_sparse(selected)
    assert config.kind == expected_kind
    assert config.operator == expected_operator


def test_draft_and_rainfusion_provider_dense_phases_are_not_sparse_successes() -> None:
    q, k, v = _values(batch=1)
    draft = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="draft"),
        provider=_FakeProvider(),
        provider_path="fake.draft.apply",
        allow_non_cuda_for_tests=True,
    )
    first_layer = draft(q, k, v, grid=(2, 3, 4), layer_idx=0, step_index=0)
    later_layer = draft(q, k, v, grid=(2, 3, 4), layer_idx=1, step_index=0)
    assert first_layer.receipt["execution"] == "provider_dense"
    assert first_layer.receipt["sparse_kernel_executed"] is False
    assert first_layer.receipt["provider_dense_executed"] is True
    assert later_layer.receipt["execution"] == "sparse"

    rainfusion = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="rainfusion", skip_timesteps=2),
        provider=_FakeProvider(),
        provider_path="fake.rainfusion.apply",
        allow_non_cuda_for_tests=True,
    )
    warmup = rainfusion(
        q,
        k,
        v,
        grid=(2, 3, 4),
        layer_idx=0,
        step_index=1,
        request_key="request-a:positive",
    )
    sparse = rainfusion(
        q,
        k,
        v,
        grid=(2, 3, 4),
        layer_idx=0,
        step_index=2,
        request_key="request-a:positive",
    )
    assert warmup.receipt["execution"] == "provider_dense"
    assert sparse.receipt["execution"] == "sparse"


def test_request_stateful_provider_cache_is_partitioned_and_bounded() -> None:
    adapter = LightX2VSparseAdapter(
        LightX2VSparseConfig(kind="svg2"),
        provider=_FakeProvider(),
        allow_non_cuda_for_tests=True,
    )
    q, k, v = _values(batch=1)
    for request_index in range(sparse_module._MAX_PROVIDER_REQUESTS + 3):
        adapter(
            q,
            k,
            v,
            grid=(2, 3, 4),
            layer_idx=0,
            step_index=0,
            request_key=f"request-{request_index}",
        )
    assert len(adapter._providers) == sparse_module._MAX_PROVIDER_REQUESTS


def test_flashinfer_runtime_reuses_one_uint8_128m_workspace_per_device_backend(
    monkeypatch,
) -> None:
    allocations: list[tuple[int, torch.dtype, torch.device]] = []

    class FakeWorkspace:
        def __init__(self, size: int, device: torch.device) -> None:
            self._size = size
            self.device = device

        def numel(self) -> int:
            return self._size

        def element_size(self) -> int:
            return 1

    class FakeWrapper:
        def __init__(self, workspace, *, backend):
            self.workspace = workspace
            self.backend = backend

        def plan(self, **_kwargs):
            pass

        def run(self, q, _k, _v):
            return q

    def fake_empty(size, *, dtype, device):
        normalized_device = torch.device(device)
        allocations.append((size, dtype, normalized_device))
        return FakeWorkspace(size, normalized_device)

    monkeypatch.setattr(sparse_module.torch, "empty", fake_empty)
    monkeypatch.setattr(sparse_module, "_device_context", lambda _device: nullcontext())
    monkeypatch.setattr(sparse_module, "_FLASHINFER_RUNTIMES", {})
    device = torch.device("cuda:3")

    first = sparse_module._flashinfer_runtime(FakeWrapper, device, backend="auto")
    second = sparse_module._flashinfer_runtime(FakeWrapper, device, backend="auto")
    fa2 = sparse_module._flashinfer_runtime(FakeWrapper, device, backend="fa2")

    assert first is second
    assert first is not fa2
    assert allocations == [
        (sparse_module._FLASHINFER_WORKSPACE_BYTES, torch.uint8, device),
        (sparse_module._FLASHINFER_WORKSPACE_BYTES, torch.uint8, device),
    ]
    assert first.wrapper.backend == "auto"
    assert fa2.wrapper.backend == "fa2"


def test_svg2_reuses_wrapper_but_replans_every_dynamic_mask(monkeypatch) -> None:
    class FakeSvg2:
        centroids_init = True

        def __init__(self):
            self.config = {}

    class FakeWrapperType:
        pass

    module = SimpleNamespace(
        Svg2AttnWeight=FakeSvg2,
        flashinfer=SimpleNamespace(
            sparse=SimpleNamespace(
                VariableBlockSparseAttentionWrapper=FakeWrapperType
            )
        ),
    )
    monkeypatch.setattr(
        sparse_module.importlib,
        "import_module",
        lambda name: module
        if name == "lightx2v.common.ops.attn.svg2_attn"
        else (_ for _ in ()).throw(AssertionError(name)),
    )

    class RuntimeWrapper:
        def __init__(self):
            self.plans: list[dict[str, object]] = []
            self.runs = 0

        def plan(self, **kwargs):
            self.plans.append(kwargs)

        def run(self, q, _k, _v):
            self.runs += 1
            return q

    wrapper = RuntimeWrapper()
    runtime = SimpleNamespace(wrapper=wrapper, lock=RLock())
    monkeypatch.setattr(
        sparse_module,
        "_flashinfer_runtime",
        lambda wrapper_type, _device, *, backend: runtime
        if wrapper_type is FakeWrapperType and backend == "auto"
        else (_ for _ in ()).throw(AssertionError((wrapper_type, backend))),
    )

    provider = sparse_module._build_direct_provider(
        LightX2VSparseConfig(kind="svg2"),
        frame_num=2,
    )
    assert provider.centroids_init is False
    assert provider._worldfoundry_adapter_receipt == {
        "compatibility_adaptations": [
            "request_local_kmeans_centroids",
            "device_local_flashinfer_wrapper_reuse",
            "flashinfer_uint8_128m_workspace",
        ],
        "state_isolation": "device/grid/request-safe",
    }

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rows = torch.full((1, 2, 2), 2, dtype=torch.int32)
    cols = rows.clone()
    first_mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    second_mask = first_mask.clone()
    second_mask[..., 0, 1] = False

    first = provider.dynamic_block_sparse_fwd_flashinfer(
        q, k, v, first_mask, rows, cols, is_cpu=True
    )
    second = provider.dynamic_block_sparse_fwd_flashinfer(
        q, k, v, second_mask, rows, cols, is_cpu=True
    )

    torch.testing.assert_close(first, q)
    torch.testing.assert_close(second, q)
    assert wrapper.runs == 2
    assert len(wrapper.plans) == 2
    assert torch.equal(wrapper.plans[0]["block_mask_map"], first_mask.reshape(2, 2, 2))
    assert torch.equal(wrapper.plans[1]["block_mask_map"], second_mask.reshape(2, 2, 2))
