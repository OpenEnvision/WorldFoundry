"""Guard that the native diffusion loader compiles through the persistent cache.

The loader's ``policy.compile`` branch must route through
``compile_callable_cached`` (persistent Inductor/Triton disk cache + per-instance
variant reuse) rather than a bare ``torch.compile``. Compiling the bound forward
keeps the concrete checkpoint-compatible model type; returning OptimizedModule
would fail component factory type validation. A full ``load`` needs a real
checkpoint, so here we (1) assert the loader source wires the cache API and
(2) lock the cache contract the loader relies on, all on CPU.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from worldfoundry.base_models.diffusion_model.loaders import module as loader_module
from worldfoundry.base_models.diffusion_model.loaders.checkpoints import CheckpointSpec
from worldfoundry.base_models.diffusion_model.loaders.module import (
    ModuleLoadSpec,
    NativeModuleLoader,
    _compile_policy_from_runtime,
)
from worldfoundry.core.model_loading.policy import RuntimePolicy
from worldfoundry.runtime.compile_cache import CompilePolicy, compile_callable_cached

safetensors = pytest.importorskip("safetensors.torch")


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _checkpoint(tmp_path) -> CheckpointSpec:
    safetensors.save_file(_Tiny().state_dict(), str(tmp_path / "model.safetensors"))
    return CheckpointSpec(source=str(tmp_path), files=("model.safetensors",))


def test_loader_uses_persistent_compile_cache_not_bare_compile() -> None:
    source = inspect.getsource(loader_module.NativeModuleLoader.load)
    # The compile branch must delegate to the cached helper...
    assert "compile_callable_cached" in source
    # ...and must not fall back to a bare torch.compile on the module.
    assert "torch.compile(module)" not in source


def test_compile_cache_disabled_returns_original() -> None:
    m = torch.nn.Linear(16, 16)
    forward = m.forward
    out = compile_callable_cached(forward, policy=CompilePolicy(enabled=False))
    assert out is forward


def test_compile_cache_variant_reuse_contract() -> None:
    # The loader relies on same-options calls not creating new variants. On CPU
    # torch.compile is a light wrapper; the variant dict must still be reused.
    m = torch.nn.Linear(16, 16)
    first = compile_callable_cached(m.forward, policy=CompilePolicy(enabled=True), namespace="diffusion-modules")
    variants = getattr(m, "_worldfoundry_compiled_variants", None)
    assert isinstance(variants, dict) and len(variants) == 1
    second = compile_callable_cached(m.forward, policy=CompilePolicy(enabled=True), namespace="diffusion-modules")
    assert first is second
    assert len(m._worldfoundry_compiled_variants) == 1


def test_loader_installs_compile_wrapper_and_audits_resolved_config(tmp_path) -> None:
    loaded = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=_Tiny),
        _checkpoint(tmp_path),
        RuntimePolicy(
            compile=True,
            options={
                "compile_mode": "max-autotune-no-cudagraphs",
                "compile_fullgraph": True,
                "compile_dynamic": False,
            },
        ),
    )

    assert isinstance(loaded, _Tiny)
    assert loaded._worldfoundry_compile_config == {
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": True,
        "dynamic": False,
        "options": {},
    }
    snapshot = loaded._worldfoundry_applied_optimizations.to_optimization_snapshot()
    assert snapshot.requested["compile"] is True
    assert snapshot.effective["compile"] == "compile-wrapper-installed (lazy)"
    assert snapshot.effective["compile_config"] == loaded._worldfoundry_compile_config
    assert loaded._worldfoundry_compile_runtime == {
        "wrapper_installed": True,
        "calls": 0,
        "failures": 0,
        "last_error": None,
        "request_calls": 0,
        "request_failures": 0,
        "request_last_error": None,
        "attention_provider_graph_traces": {},
    }


def test_external_cuda_graph_selects_inductor_no_cudagraph_mode() -> None:
    compile_policy, options = _compile_policy_from_runtime(
        RuntimePolicy(compile=True, options={"cuda_graph": True})
    )
    assert compile_policy.mode == "max-autotune-no-cudagraphs"
    assert options == {}


def test_external_cuda_graph_rejects_nested_inductor_cudagraph_mode() -> None:
    with pytest.raises(ValueError, match="nested CUDA Graph"):
        _compile_policy_from_runtime(
            RuntimePolicy(
                compile=True,
                options={"cuda_graph": True, "compile_mode": "reduce-overhead"},
            )
        )


def test_custom_compile_options_disable_mode_preset() -> None:
    compile_policy, options = _compile_policy_from_runtime(
        RuntimePolicy(
            compile=True,
            options={"compile_options": {"epilogue_fusion": True}},
        )
    )
    assert compile_policy.mode is None
    assert options == {"epilogue_fusion": True}


@pytest.mark.parametrize(
    ("options", "error"),
    (
        ({"compile_mode": "turbo"}, ValueError),
        ({"compile_dynamic": "yes"}, TypeError),
        ({"compile_fullgraph": 1}, TypeError),
        ({"compile_options": ["epilogue_fusion"]}, TypeError),
        (
            {
                "compile_mode": "default",
                "compile_options": {"epilogue_fusion": True},
            },
            ValueError,
        ),
    ),
)
def test_compile_policy_rejects_malformed_public_options(options, error) -> None:
    with pytest.raises(error):
        _compile_policy_from_runtime(RuntimePolicy(compile=True, options=options))
