"""Selection, reversible ownership and real scaled-mm execution contracts."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale import SanaMS
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations import precision
from worldfoundry.base_models.diffusion_model.optimizations.plugins import (
    diffusion_acceleration_registry,
    install_diffusion_accelerations,
)
from worldfoundry.core.acceleration.plugins import PreparedAcceleration
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    quantization_runtime_report,
    reset_quantization_runtime_window,
    set_low_precision_enabled,
)
from worldfoundry.core.model_loading.policy import OffloadPolicy, QuantizationPolicy, RuntimePolicy


def _wan():
    torch.manual_seed(71)
    return WanModel(
        dim=128,
        in_dim=2,
        ffn_dim=256,
        out_dim=2,
        text_dim=64,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=1,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()


def _sana():
    return SanaMS(
        input_size=4,
        patch_size=1,
        in_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=2,
        caption_channels=16,
        model_max_length=4,
        pred_sigma=False,
        use_pe=False,
        attn_type="linear",
        linear_head_dim=16,
        ffn_type="mlp",
        mlp_ratio=2,
        cross_attn_type="vanilla",
    ).eval()


def _options(*include, **kwargs):
    return {
        "selective_fp8": {
            "include": list(include or ("blocks.*.ffn.0", "blocks.*.ffn.2")),
            "min_features": 16,
            **kwargs,
        }
    }


def test_selected_cpu_fallback_and_exact_uninstall_restore_original_objects():
    model = _wan()
    original = model.blocks[0].ffn[0]
    keys = tuple(model.state_dict())
    value = torch.randn(2, 9, 128)
    with torch.no_grad():
        expected = model.blocks[0].ffn(value)
        session = install_diffusion_accelerations(model, _options())
        replacement = model.blocks[0].ffn[0]
        assert isinstance(replacement, Float8Linear)
        assert replacement.weight.data_ptr() == original.weight.data_ptr()
        assert type(model.blocks[0].self_attn.q) is nn.Linear
        torch.testing.assert_close(model.blocks[0].ffn(value), expected, rtol=0, atol=0)
    receipt = session.report()
    assert receipt["execution_verified"] is False
    assert receipt["installed"][0]["execution"] == "runtime-pending"
    report = quantization_runtime_report(model)
    assert report["low_precision_kernel_calls"] == 0
    assert report["dense_fallback_calls"] == 2
    reset_quantization_runtime_window(model)
    assert quantization_runtime_report(model)["dense_fallback_calls"] == 0
    session.uninstall()
    assert model.blocks[0].ffn[0] is original
    assert tuple(model.state_dict()) == keys
    assert "_apply" not in model.__dict__
    assert "train" not in model.__dict__
    with torch.no_grad():
        torch.testing.assert_close(model.blocks[0].ffn(value), expected, rtol=0, atol=0)


def test_sana_native_full_forward_remains_exact_on_dense_fallback():
    model = _sana()
    inputs = (torch.randn(1, 4, 4, 4), torch.tensor([500.0]), torch.randn(1, 1, 4, 16))
    with torch.no_grad():
        expected = model(*inputs)
        session = install_diffusion_accelerations(model, _options("blocks.*.attn.qkv", "blocks.*.mlp.fc1"))
        actual = model(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert quantization_runtime_report(model)["dense_fallback_calls"] == 2
    session.uninstall()


@pytest.mark.parametrize(
    "overrides",
    [
        {"include": []},
        {"include": "blocks.*.ffn.0"},
        {"include": [""]},
        {"include": ["blocks.*.ffn.0", "blocks.*.ffn.0"]},
        {"min_features": 0},
        {"scaling": "unscaled"},
        {"use_fast_accum": "true"},
        {"unknown": True},
    ],
)
def test_invalid_config_has_no_mutations(overrides):
    model = _wan()
    original = model.blocks[0].ffn[0]
    options = _options()["selective_fp8"] | overrides
    with pytest.raises((ValueError, TypeError)):
        install_diffusion_accelerations(model, {"selective_fp8": options})
    assert model.blocks[0].ffn[0] is original
    assert not hasattr(model, "_worldfoundry_accelerations")


@pytest.mark.parametrize("include", [("blocks.*.missing",), ("blocks.*.ffn.0", "missing"), ("time_embedding.0",)])
def test_unmatched_or_unsupported_paths_fail(include):
    with pytest.raises(ValueError, match="eligible|outside supported"):
        install_diffusion_accelerations(_wan(), _options(*include))


def test_exclude_selects_only_requested_layers():
    model = _wan()
    install_diffusion_accelerations(model, _options("blocks.*.ffn.*", exclude=["*.ffn.2"]))
    assert isinstance(model.blocks[0].ffn[0], Float8Linear)
    assert type(model.blocks[0].ffn[2]) is nn.Linear


@pytest.mark.parametrize(
    "policy",
    [
        RuntimePolicy(compile=True),
        RuntimePolicy(options={"cuda_graph": True}),
        RuntimePolicy(options={"sequence_parallel": True}),
        RuntimePolicy(options={"static_cross_kv": True}),
        RuntimePolicy(options={"feature_cache": "firstblock"}),
        RuntimePolicy(quantization=QuantizationPolicy(mode="fp8")),
        RuntimePolicy(offload=OffloadPolicy(mode="block")),
    ],
)
def test_unqualified_combinations_fail_before_mutation(policy):
    model = _wan()
    original = model.blocks[0].ffn[0]
    with pytest.raises(ValueError, match="conflicts|coexist|resident"):
        install_diffusion_accelerations(model, _options(), policy)
    assert model.blocks[0].ffn[0] is original


def test_plain_native_eval_graph_required():
    with pytest.raises(ValueError, match="checkpoint graph"):
        install_diffusion_accelerations(nn.Sequential(nn.Linear(128, 128)).eval(), _options())
    model = _wan().train()
    with pytest.raises(ValueError, match="eval"):
        install_diffusion_accelerations(model, _options())


@pytest.mark.parametrize(
    "kind",
    [
        "hook",
        "custom_forward",
        "parametrization",
        "shared_weight",
        "shared_storage",
        "module_alias",
        "nonfinite",
    ],
)
def test_unsafe_projection_graphs_are_rejected(kind):
    model = _wan()
    original = model.blocks[0].self_attn.q
    if kind == "hook":
        original.register_forward_pre_hook(lambda module, args: args)
    elif kind == "custom_forward":
        original.forward = lambda value: value
    elif kind == "parametrization":
        nn.utils.parametrize.register_parametrization(original, "weight", nn.Identity())
        model.eval()
    elif kind == "shared_weight":
        model.blocks[0].self_attn.k.weight = original.weight
    elif kind == "shared_storage":
        model.blocks[0].self_attn.k.weight = nn.Parameter(original.weight.detach().view_as(original.weight))
    elif kind == "module_alias":
        model.alias = original
    else:
        with torch.no_grad():
            original.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="hooks|customized|plain|tied|shared|nonfinite"):
        install_diffusion_accelerations(model, _options("blocks.*.self_attn.q"))
    assert model.blocks[0].self_attn.q is original


def test_placement_and_training_are_explicitly_fixed_until_uninstall():
    model = _wan()
    session = install_diffusion_accelerations(model, _options())
    dtypes = [parameter.dtype for parameter in model.parameters()]
    with pytest.raises(RuntimeError, match="placement"):
        model.to(dtype=torch.float16)
    assert [parameter.dtype for parameter in model.parameters()] == dtypes
    with pytest.raises(RuntimeError, match="no-grad"):
        model.blocks[0].ffn(torch.randn(2, 128))
    with pytest.raises(RuntimeError, match="training"):
        model.blocks[0].ffn[0].train()
    flags = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match="training"):
        model.train()
    assert [module.training for module in model.modules()] == flags
    assert model.eval() is model and model.train(False) is model
    assert [module.training for module in model.modules()] == flags
    with pytest.raises(ValueError, match="boolean"):
        model.train("true")
    assert [module.training for module in model.modules()] == flags
    session.uninstall()
    assert "train" not in model.__dict__
    model.to(dtype=torch.float16).train()


def test_source_weight_mutation_cannot_leave_stale_quantized_execution():
    model = _wan()
    original = model.blocks[0].ffn[0]
    session = install_diffusion_accelerations(model, _options())
    with torch.no_grad():
        original.weight.add_(0.1)
        with pytest.raises(RuntimeError, match="source weights changed"):
            model.blocks[0].ffn(torch.randn(2, 128))
    session.uninstall()
    assert model.blocks[0].ffn[0] is original


def test_runtime_compile_guard_precedes_dense_and_fp8_execution(monkeypatch):
    model = _wan()
    session = install_diffusion_accelerations(model, _options())
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with torch.no_grad(), pytest.raises(RuntimeError, match="torch.compile"):
        model.blocks[0].ffn(torch.randn(2, 128))
    assert quantization_runtime_report(model)["low_precision_kernel_calls"] == 0
    assert quantization_runtime_report(model)["dense_fallback_calls"] == 0
    session.uninstall()


def test_custom_attention_processor_and_approximate_provider_are_rejected():
    model = _wan()
    model.blocks[0].self_attn.processor = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="default attention processor"):
        install_diffusion_accelerations(model, _options("blocks.*.self_attn.q"))
    model = _wan()
    model.blocks[0].self_attn.attn.attention_backend = "sol_attn"
    with pytest.raises(ValueError, match="approximate attention backends"):
        install_diffusion_accelerations(model, _options())


def test_constructor_failure_and_late_plugin_failure_preserve_original_graph(monkeypatch):
    model = _wan()
    originals = tuple(model.blocks[0].ffn)
    factory = precision._FixedPlacementFloat8Linear
    calls = 0

    def fail_second(source, config):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("quantization allocation failed")
        return factory(source, config)

    monkeypatch.setattr(precision, "_FixedPlacementFloat8Linear", fail_second)
    with pytest.raises(RuntimeError, match="allocation"):
        install_diffusion_accelerations(model, _options())
    assert tuple(model.blocks[0].ffn) == originals
    monkeypatch.setattr(precision, "_FixedPlacementFloat8Linear", factory)
    registry = diffusion_acceleration_registry()

    def failure(model, options, policy):
        def activate():
            raise RuntimeError("later installation failed")

        return PreparedAcceleration("fail", frozenset(), activate)

    registry.register("fail", failure)
    with pytest.raises(RuntimeError, match="later installation"):
        registry.install(model, _options() | {"fail": True})
    assert tuple(model.blocks[0].ffn) == originals
    assert "_apply" not in model.__dict__
    assert "train" not in model.__dict__


def test_cross_kv_composition_conflict_is_checked_in_both_orders():
    fp8 = _options("blocks.*.cross_attn.k", "blocks.*.cross_attn.v")
    kv = {"wan_cross_kv_fusion": True}
    for options in (fp8 | kv, kv | fp8):
        model = _wan()
        with pytest.raises(ValueError, match="conflicts on seams"):
            install_diffusion_accelerations(model, options)
        assert type(model.blocks[0].cross_attn.k) is nn.Linear
        assert not hasattr(model, "_worldfoundry_accelerations")


@pytest.mark.parametrize("context", [None, RuntimePolicy()])
@pytest.mark.parametrize(
    "other",
    [
        {"easycache": {"threshold": 0.0}},
        {"sana_block_fusion": {"repack_tokens": True}},
        {"attention_policy": {"self": "sol"}},
    ],
)
def test_actual_manual_options_drive_order_independent_composition_gates(context, other):
    fp8 = _options("blocks.*.attn.qkv") if "sana_block_fusion" in other else _options()
    for options in (fp8 | other, other | fp8):
        model = _sana() if "sana_block_fusion" in other else _wan()
        original_keys = tuple(model.state_dict())
        # Provider resolution is not needed: reject the combined policy before
        # the attention adapter imports an optional package in either ordering.
        if "attention_policy" in other and next(iter(options)) == "attention_policy":
            with pytest.raises((ValueError, RuntimeError), match="Sol|sol|unavailable|usable|conflicts|BF16"):
                install_diffusion_accelerations(model, options, context)
        else:
            with pytest.raises(ValueError, match="conflicts"):
                install_diffusion_accelerations(model, options, context)
        assert tuple(model.state_dict()) == original_keys
        assert not hasattr(model, "_worldfoundry_accelerations")


def test_direct_registry_none_context_still_rejects_fp8_and_cache_before_mutation():
    for options in (_options() | {"easycache": {"threshold": 0.0}}, {"easycache": {"threshold": 0.0}} | _options()):
        model = _wan()
        with pytest.raises(ValueError, match="conflicts on seams"):
            diffusion_acceleration_registry().install(model, options)
        assert type(model.blocks[0].ffn[0]) is nn.Linear


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_scaled_mm_receipts_layout_and_exact_removal(monkeypatch):
    from worldfoundry.core.acceleration.quantization import linear

    model = _wan().to(device="cuda", dtype=torch.bfloat16)
    if not linear._fp8_hardware_eligible(torch.device("cuda")):
        pytest.skip("requires FP8 scaled-mm hardware")
    # Verify correctness on a small allocation; this is not a speed claim.
    monkeypatch.setattr(linear, "_fp8_min_gemm_work", lambda: 0.0)
    value = torch.randn(2, 128, 257, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    original = model.blocks[0].ffn[0]
    with torch.no_grad():
        expected = model.blocks[0].ffn(value)
        session = install_diffusion_accelerations(model, _options())
        reset_quantization_runtime_window(model)
        actual = model.blocks[0].ffn(value)
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert torch.isfinite(actual).all() and relative_l2 < 0.06
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        report = quantization_runtime_report(model)
        assert report["low_precision_kernel_calls"] == 2
        assert report["dense_fallback_calls"] == 0
        assert set_low_precision_enabled(model, False) == 2
        reset_quantization_runtime_window(model)
        torch.testing.assert_close(model.blocks[0].ffn(value), expected, rtol=0, atol=0)
        assert quantization_runtime_report(model)["low_precision_kernel_calls"] == 0
        assert quantization_runtime_report(model)["dense_fallback_calls"] == 2
        # A caller can start capture after installation; reject it before
        # either the FP8 operator or the retained dense module is entered.
        reset_quantization_runtime_window(model)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="CUDA Graph capture"):
            model.blocks[0].ffn(value)
        assert quantization_runtime_report(model)["low_precision_kernel_calls"] == 0
        assert quantization_runtime_report(model)["dense_fallback_calls"] == 0
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        session.uninstall()
        assert model.blocks[0].ffn[0] is original
        torch.testing.assert_close(model.blocks[0].ffn(value), expected, rtol=0, atol=0)
