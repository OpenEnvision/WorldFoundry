"""Native Wan cross-KV parity, retained checkpoints and reversible lifecycle."""

import copy
from dataclasses import replace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention
from worldfoundry.base_models.diffusion_model.optimizations.kv_fusion import prepare_wan_cross_kv_fusion
from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import install_static_cross_kv_cache
from worldfoundry.core.model_loading.policy import RuntimePolicy


def _attention(image=False, device="cpu", dtype=torch.float32):
    torch.manual_seed(9)
    attention = CrossAttention(128, 4, has_image_input=image).to(device=device, dtype=dtype).eval()
    attention.attn.set_attention_backend("torch")
    return attention


def _inputs(attention):
    x = torch.randn(2, 7, 128, device=attention.q.weight.device, dtype=attention.q.weight.dtype)
    context = torch.randn(2, 267 if attention.has_image_input else 10, 128, device=x.device, dtype=x.dtype)
    return x, context


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("static_cache", [False, True])
def test_disabled_fusion_preserves_dense_projection_normalization_order(image, static_cache):
    attention = _attention(image)
    if static_cache:
        install_static_cross_kv_cache(attention)
    x, context = _inputs(attention)
    names = ["k", "norm_k", "v"]
    if image:
        names += ["k_img", "norm_k_img", "v_img"]
    calls = []
    handles = [
        getattr(attention, name).register_forward_hook(lambda module, args, output, name=name: calls.append(name))
        for name in names
    ]
    try:
        with torch.inference_mode():
            attention(x, context)
        expected = names[3:] + names[:3] if image and static_cache else names
        assert calls == expected
        if static_cache:
            calls.clear()
            with torch.inference_mode():
                attention(x, context)
            assert not calls
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("image", [False, True])
def test_native_output_checkpoint_and_uninstall(image):
    attention = _attention(image)
    keys = tuple(attention.state_dict())
    parameter_ids = {name: id(parameter) for name, parameter in attention.named_parameters()}
    x, context = _inputs(attention)
    with torch.inference_mode():
        expected = attention(x, context)
        session = install_diffusion_accelerations(
            attention, {"wan_cross_kv_fusion": {"min_tokens": 0, "include_image": True}}
        )
        actual = attention(x, context)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert tuple(attention.state_dict()) == keys
    assert {name: id(parameter) for name, parameter in attention.named_parameters()} == parameter_ids
    report = session.report()["installed"][0]
    assert report["runtime"]["text_packed_calls"] == 1
    assert report["runtime"]["image_packed_calls"] == int(image)
    state = attention._worldfoundry_cross_kv_fusion
    session.uninstall()
    assert not state.routes
    assert not hasattr(attention, "_worldfoundry_cross_kv_fusion")
    with torch.inference_mode():
        torch.testing.assert_close(attention(x, context), expected, rtol=0, atol=0)


def test_parameter_mutation_and_checkpoint_assignment_refresh_derived_weights():
    attention = _attention()
    x, context = _inputs(attention)
    session = install_diffusion_accelerations(attention, {"wan_cross_kv_fusion": {"min_tokens": 0}})
    with torch.no_grad():
        attention.k.weight.add_(0.01)
        attention.v.bias.add_(0.3)
        reference = copy.deepcopy(attention)
        del reference._worldfoundry_cross_kv_fusion
        torch.testing.assert_close(attention(x, context), reference(x, context), rtol=2e-5, atol=2e-6)
        replacement = _attention()
        attention.load_state_dict(replacement.state_dict(), strict=True, assign=True)
        torch.testing.assert_close(attention(x, context), replacement(x, context), rtol=2e-5, atol=2e-6)
    assert session.report()["installed"][0]["runtime"]["weight_refreshes"] == 2


def test_small_context_is_bitwise_dense_and_receipt_is_truthful():
    attention = _attention()
    x, context = _inputs(attention)
    with torch.inference_mode():
        expected = attention(x, context)
        session = install_diffusion_accelerations(attention, {"wan_cross_kv_fusion": True})
        torch.testing.assert_close(attention(x, context), expected, rtol=0, atol=0)
    receipt = session.report()["installed"][0]["runtime"]
    assert receipt["text_packed_calls"] == 0 and receipt["dense_projection_pairs"] == 1


def test_static_cache_misses_use_fusion_and_removal_invalidates_rounded_entries():
    attention = _attention(True)
    reference = copy.deepcopy(attention)
    cache = install_static_cross_kv_cache(attention)
    session = install_diffusion_accelerations(
        attention, {"wan_cross_kv_fusion": {"min_tokens": 0, "include_image": True}}
    )
    x, context = _inputs(attention)
    with torch.inference_mode():
        first = attention(x, context)
        torch.testing.assert_close(first, reference(x, context), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(attention(x, context), first, rtol=0, atol=0)
        # Unknown kwargs retain the old processor bypass contract.
        torch.testing.assert_close(attention(x, context, custom_flag=True), first, rtol=0, atol=0)
    assert cache.report()["hits"] == 1 and cache.report()["bypasses"] == 1
    receipt = session.report()["installed"][0]["runtime"]
    assert receipt["text_packed_calls"] == 2 and receipt["image_packed_calls"] == 2
    session.uninstall()
    assert not cache._entries
    with torch.inference_mode():
        torch.testing.assert_close(attention(x, context), reference(x, context), rtol=0, atol=0)


def test_populated_static_cache_must_be_invalidated_before_install():
    attention = _attention()
    cache = install_static_cross_kv_cache(attention)
    x, context = _inputs(attention)
    with torch.inference_mode():
        attention(x, context)
    with pytest.raises(ValueError, match="invalidate existing"):
        prepare_wan_cross_kv_fusion(attention, {}, None)
    cache.invalidate()
    install_diffusion_accelerations(attention, {"wan_cross_kv_fusion": True})


@pytest.mark.parametrize("options", [{"min_tokens": -1}, {"min_tokens": True}, {"include_image": 1}])
def test_invalid_options_do_not_mutate_model(options):
    attention = _attention()
    with pytest.raises((TypeError, ValueError)):
        prepare_wan_cross_kv_fusion(attention, options, None)
    assert not hasattr(attention, "_worldfoundry_cross_kv_fusion")


def test_hooked_or_custom_processors_are_rejected_before_activation():
    attention = _attention()
    hook = attention.k.register_forward_hook(lambda module, args, output: output + 1)
    with pytest.raises(ValueError, match="unhooked"):
        prepare_wan_cross_kv_fusion(attention, {}, None)
    hook.remove()
    attention.k.forward = lambda value: value
    with pytest.raises(ValueError, match="unhooked"):
        prepare_wan_cross_kv_fusion(attention, {}, None)
    del attention.k.forward
    attention.processor = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="native default"):
        prepare_wan_cross_kv_fusion(attention, {}, None)


@pytest.mark.parametrize("key", ["cuda_graph", "sequence_parallel", "sp_degree", "device_map", "approximate_attention"])
def test_unvalidated_combinations_are_rejected(key):
    with pytest.raises(ValueError, match="unvalidated"):
        prepare_wan_cross_kv_fusion(_attention(), {}, RuntimePolicy(options={key: True}))
    with pytest.raises(ValueError, match="compile"):
        prepare_wan_cross_kv_fusion(_attention(), {}, replace(RuntimePolicy(), compile=True))


def test_grad_enabled_execution_fails_explicitly():
    attention = _attention()
    install_diffusion_accelerations(attention, {"wan_cross_kv_fusion": {"min_tokens": 0}})
    with pytest.raises(RuntimeError, match="no_grad"):
        attention(*_inputs(attention))


def test_dtype_conversion_refreshes_nonpersistent_packed_weights():
    attention = _attention()
    session = install_diffusion_accelerations(attention, {"wan_cross_kv_fusion": {"min_tokens": 0}})
    attention.to(dtype=torch.bfloat16)
    reference = copy.deepcopy(attention)
    del reference._worldfoundry_cross_kv_fusion
    x, context = _inputs(attention)
    with torch.inference_mode():
        torch.testing.assert_close(attention(x, context), reference(x, context), rtol=0.025, atol=0.005)
    assert attention._worldfoundry_cross_kv_fusion.routes[False].weight.dtype == torch.bfloat16
    assert session.report()["installed"][0]["runtime"]["weight_refreshes"] == 1


def test_projection_hook_added_after_install_is_never_silently_bypassed():
    attention = _attention()
    install_diffusion_accelerations(attention, {"wan_cross_kv_fusion": {"min_tokens": 0}})
    attention.v.register_forward_hook(lambda module, args, output: output + 1)
    with torch.inference_mode(), pytest.raises(ValueError, match="unhooked"):
        attention(*_inputs(attention))


def test_inference_parameters_without_mutation_versions_are_rejected():
    with torch.inference_mode():
        attention = _attention()
    with pytest.raises(ValueError, match="versioned"):
        prepare_wan_cross_kv_fusion(attention, {}, None)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_real_native_cross_attention_is_finite_and_close():
    attention = _attention(True, "cuda", torch.bfloat16)
    x, context = _inputs(attention)
    with torch.inference_mode():
        expected = attention(x, context)
        session = install_diffusion_accelerations(
            attention, {"wan_cross_kv_fusion": {"min_tokens": 0, "include_image": True}}
        )
        actual = attention(x, context)
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert difference.item() < 0.005
    assert session.report()["installed"][0]["runtime"]["text_packed_calls"] == 1
