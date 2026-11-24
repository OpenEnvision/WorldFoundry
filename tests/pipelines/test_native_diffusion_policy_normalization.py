from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.core.distributed.sequence_parallel import runtime
from worldfoundry.core.model_loading.policy import QuantizationMode, QuantizationPolicy
from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline


@pytest.fixture
def native_loader(monkeypatch: pytest.MonkeyPatch) -> Mock:
    loader = Mock(return_value=SimpleNamespace(model_id=Wan2p2Pipeline.MODEL_ID))
    monkeypatch.setattr(NativeDiffusionPipeline, "from_pretrained", loader)
    return loader


@pytest.fixture
def parallel_runtime(monkeypatch: pytest.MonkeyPatch) -> Mock:
    ensure = Mock(return_value=torch.device("cuda:0"))
    monkeypatch.setattr(runtime, "ensure_parallel_runtime", ensure)
    return ensure


@pytest.mark.parametrize("mode_key", ["quantization", "quantization_mode"])
def test_native_diffusion_flat_quantization_companions(native_loader: Mock, mode_key: str) -> None:
    options: dict[str, Any] = {mode_key: "fp8"}
    Wan2p2Pipeline.from_pretrained(
        model_path="/unused",
        device="cpu",
        **options,
        quantization_exclude=["head"],
        quantization_compute_dtype="bf16",
        fp8_min_features=2048,
        fp8_scaling="rowwise",
        keep_dense_fallback=False,
        use_fast_accum=True,
    )
    policy = native_loader.call_args.kwargs["policy"].quantization
    assert policy.mode is QuantizationMode.FP8
    assert policy.exclude == ("head",)
    assert policy.compute_dtype is torch.bfloat16
    assert dict(policy.options) == {
        "min_features": 2048,
        "scaling": "rowwise",
        "keep_dense_fallback": False,
        "use_fast_accum": True,
    }


@pytest.mark.parametrize("fp8_scaling, expected", [("rowwise", "rowwise"), (None, "tensorwise")])
def test_native_diffusion_flat_quantization_overrides_without_mutation(
    native_loader: Mock, fp8_scaling: str | None, expected: str
) -> None:
    options = {
        "quantization": {
            "mode": "fp8",
            "exclude": ["old_head"],
            "compute_dtype": "fp16",
            "min_features": 1024,
            "options": {"scaling": "dynamic", "use_fast_accum": False},
        },
        "quantization_mode": "int8",
        "quantization_exclude": ["head"],
        "quantization_compute_dtype": None,
        "fp8_min_features": 2048,
        "fp8_scaling": fp8_scaling,
        "scaling": "tensorwise",
    }
    original = deepcopy(options)
    Wan2p2Pipeline.from_pretrained(model_path=options, device="cpu")
    policy = native_loader.call_args.kwargs["policy"].quantization
    assert policy.mode is QuantizationMode.FP8
    assert policy.exclude == ("head",)
    assert policy.compute_dtype is torch.float16
    assert dict(policy.options) == {"min_features": 2048, "scaling": expected, "use_fast_accum": False}
    assert options == original


def test_native_diffusion_typed_quantization_preserves_nested_options(native_loader: Mock) -> None:
    quantization = QuantizationPolicy(
        mode=QuantizationMode.FP8, exclude=("head",), options={"min_features": 1024, "scaling": "rowwise"}
    )
    Wan2p2Pipeline.from_pretrained(
        model_path="/unused",
        device="cpu",
        quantization=quantization,
        fp8_min_features=2048,
        fp8_scaling=None,
        scaling=None,
    )
    policy = native_loader.call_args.kwargs["policy"].quantization
    assert policy.exclude == ("head",)
    assert dict(policy.options) == {"min_features": 2048, "scaling": "rowwise"}
    assert quantization.options["min_features"] == 1024


@pytest.mark.parametrize("key", ["cfg_parallel", "cfg_parallel_degree"])
@pytest.mark.parametrize("value", [True, 2, "2"])
def test_native_diffusion_cfg_parallel_uses_two_ranks(
    native_loader: Mock, parallel_runtime: Mock, key: str, value: object
) -> None:
    options: dict[str, Any] = {key: value}
    Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cuda", **options)
    parallel_runtime.assert_called_once_with(1, 2)
    policy = native_loader.call_args.kwargs["policy"]
    assert policy.device == torch.device("cuda:0")
    assert policy.options["cfg_parallel"] == 2
    assert "cfg_parallel_degree" not in policy.options


@pytest.mark.parametrize("key", ["cfg_parallel", "cfg_parallel_degree"])
@pytest.mark.parametrize("value", [False, 0, 1])
def test_native_diffusion_disabled_cfg_needs_no_cuda(
    native_loader: Mock, parallel_runtime: Mock, key: str, value: object
) -> None:
    options: dict[str, Any] = {key: value}
    Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cpu", **options)
    parallel_runtime.assert_not_called()
    native_loader.assert_called_once()


@pytest.mark.parametrize("primary, alias", [(True, 2), (2, True), (False, 1), (0, False), (1, 0)])
def test_native_diffusion_cfg_aliases_compare_normalized_degrees(
    native_loader: Mock, parallel_runtime: Mock, primary: object, alias: object
) -> None:
    Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cuda", cfg_parallel=primary, cfg_parallel_degree=alias)
    if primary is True or primary == 2:
        parallel_runtime.assert_called_once_with(1, 2)
    else:
        parallel_runtime.assert_not_called()


@pytest.mark.parametrize("primary, alias", [(True, 1), (False, 2), (0, 2), (2, False)])
def test_native_diffusion_cfg_conflicting_aliases_rejected(
    native_loader: Mock, parallel_runtime: Mock, primary: object, alias: object
) -> None:
    with pytest.raises(ValueError, match="cfg_parallel and cfg_parallel_degree must match"):
        Wan2p2Pipeline.from_pretrained(
            model_path="/unused", device="cuda", cfg_parallel=primary, cfg_parallel_degree=alias
        )
    parallel_runtime.assert_not_called()
    native_loader.assert_not_called()


@pytest.mark.parametrize("key", ["cfg_parallel", "cfg_parallel_degree"])
@pytest.mark.parametrize("value", [-1, 3])
def test_native_diffusion_invalid_cfg_degree_rejected(
    native_loader: Mock, parallel_runtime: Mock, key: str, value: int
) -> None:
    options: dict[str, Any] = {key: value}
    with pytest.raises(ValueError, match="cfg_parallel must be 1 or 2"):
        Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cuda", **options)
    parallel_runtime.assert_not_called()
    native_loader.assert_not_called()


@pytest.mark.parametrize("key", ["cfg_parallel", "cfg_parallel_degree"])
@pytest.mark.parametrize("value", [True, 2])
def test_native_diffusion_real_cfg_parallel_rejects_cpu(
    native_loader: Mock, parallel_runtime: Mock, key: str, value: object
) -> None:
    options: dict[str, Any] = {key: value}
    with pytest.raises(ValueError, match="requires a CUDA pipeline device"):
        Wan2p2Pipeline.from_pretrained(model_path="/unused", device="cpu", **options)
    parallel_runtime.assert_not_called()
    native_loader.assert_not_called()


def test_native_diffusion_cfg_composes_with_sp_and_vae(native_loader: Mock, parallel_runtime: Mock) -> None:
    Wan2p2Pipeline.from_pretrained(
        model_path="/unused", device="cuda", cfg_parallel=True, sequence_parallel=2, vae_parallel=4
    )
    parallel_runtime.assert_called_once_with(2, 2)
    policy = native_loader.call_args.kwargs["policy"]
    assert policy.options["sequence_parallel"] == 2
    assert policy.options["vae_parallel"] == 4
