"""CPU-only tests for miscellaneous core-compute fixes.

Covers:
- CC-03: device availability stays lazy and NPU setup is explicit.
- CC-11: RoPE and block KV cache resolve an omitted device at runtime.
- CC-36: Sparse3DCache optional capacity bound (default stays unbounded).
- CC-18: metric_sync.init_distributed raises informative errors instead of
  NameError on incomplete launcher environments.
- CC-33: TF32 configuration writes the real torch attribute paths.
- CC-32: SDPA patch install/uninstall round-trip.
- CC-26: DynamicSwapInstaller double-install keeps the real backup class.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch

from worldfoundry.core.geometry.warp import Sparse3DCache


def test_device_import_has_no_availability_or_npu_configuration_side_effect(monkeypatch):
    import importlib

    import worldfoundry.core.execution.device as device

    calls = {"cuda": 0, "npu": 0}

    def cuda_available() -> bool:
        calls["cuda"] += 1
        return False

    monkeypatch.setattr(torch.cuda, "is_available", cuda_available)
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "is_available", lambda: calls.__setitem__("npu", calls["npu"] + 1) or False)

    importlib.reload(device)
    assert calls == {"cuda": 0, "npu": 0}
    assert device.is_cuda_available() is False
    assert calls["cuda"] == 1


def test_deprecated_device_availability_constant_is_lazy(monkeypatch):
    import worldfoundry.core.execution.device as device

    calls = 0

    def available() -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(device, "is_cuda_available", available)
    assert "IS_CUDA_AVAILABLE" not in vars(device)
    with pytest.warns(DeprecationWarning, match="is_cuda_available"):
        assert device.IS_CUDA_AVAILABLE is True
    assert calls == 1


def test_rope_and_kv_cache_default_to_runtime_selected_device(monkeypatch):
    from worldfoundry.core.attention.cache import kvcache
    from worldfoundry.core.attention.rotary import rope

    selected = torch.device("cpu")
    monkeypatch.setattr(rope, "get_current_torch_device", lambda: selected)
    monkeypatch.setattr(kvcache, "get_current_torch_device", lambda: selected)

    frequencies = rope._compute_freqs(4)
    rotary = rope.RotaryPositionEmbedding3D(head_dim=12, len_h=1, len_w=1, len_t=1)
    cache = kvcache.BlockKVCache(
        k_shape=(1, 2, 4),
        v_shape=(1, 2, 4),
        seq_dim=1,
        chunk_size=1,
        window_size=2,
        dtype=torch.float32,
    )

    assert frequencies.device == selected
    assert rotary.device == selected
    assert cache.device == selected
    assert cache._k.device == selected


def test_native_attention_restores_shared_context_parallel_options(monkeypatch):
    from worldfoundry.core.attention.backends import native

    options = types.SimpleNamespace(enable_load_balance=True, rotate_method="original")
    fake_attention_module = types.ModuleType("torch.distributed.tensor.experimental._attention")
    fake_attention_module._cp_options = options
    fake_attention_module.set_rotate_method = lambda method: setattr(options, "rotate_method", method)
    monkeypatch.setattr(native.importlib, "import_module", lambda _name: fake_attention_module)
    monkeypatch.setattr(native, "_context_parallel", object())
    monkeypatch.setattr(
        native,
        "DeviceMesh",
        types.SimpleNamespace(from_group=lambda group, device_type: (group, device_type)),
    )
    monkeypatch.setattr(native, "_CP_OPTIONS_USERS", 0)
    monkeypatch.setattr(native, "_CP_OPTIONS_SNAPSHOT", None)

    first = native.NativeAttention(backend="math")
    second = native.NativeAttention(backend="math")
    first.set_context_parallel_group("first")
    second.set_context_parallel_group("second")
    assert options.enable_load_balance is False
    assert options.rotate_method == "allgather"

    first.clear_context_parallel_group()
    assert options.enable_load_balance is False
    second.set_context_parallel_group(None)
    assert options.enable_load_balance is True
    assert options.rotate_method == "original"
    assert native._CP_OPTIONS_USERS == 0


def test_vram_probe_interval_throttles_driver_queries(monkeypatch):
    from worldfoundry.core.vram import layers

    monkeypatch.setenv("WORLDFOUNDRY_VRAM_CHECK_INTERVAL", "3")
    monkeypatch.setattr(layers, "is_npu_available", lambda: False)
    calls = 0

    def mem_get_info(_device):
        nonlocal calls
        calls += 1
        return 7 * 1024**3, 8 * 1024**3

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    module = layers.AutoTorchModule(computation_device="cuda:0", vram_limit=2.0)
    assert [module.check_free_vram() for _ in range(7)] == [True] * 7
    assert calls == 3


def test_disk_offload_keeps_buffers_but_releases_direct_parameters():
    from worldfoundry.core.vram.layers import AutoWrappedModule

    class BufferedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.register_buffer("scratch", torch.ones(2), persistent=False)
            self.child = torch.nn.Linear(2, 2)

    source = BufferedModule()
    wrapper = AutoWrappedModule(source, offload_dtype="disk", name="buffered", disk_map={})
    wrapper.offload_to_disk(source)

    assert source.weight.device.type == "meta"
    assert source.child.weight.device.type == "meta"
    assert source.child.bias.device.type == "meta"
    assert source.scratch.device.type == "cpu"


def test_temporary_module_deepcopy_warns_once_per_module_class(caplog):
    from worldfoundry.core.vram import layers

    class ExpensiveTemporary(torch.nn.Linear):
        pass

    layers._DEEPCOPY_WARNING_CLASSES.discard(ExpensiveTemporary)
    wrapper = layers.AutoWrappedModule(ExpensiveTemporary(2, 2))
    with caplog.at_level(logging.WARNING, logger="worldfoundry.core.vram.layers"):
        wrapper.cast_to(wrapper.module, torch.float32, "cpu")
        wrapper.cast_to(wrapper.module, torch.float32, "cpu")
    messages = [record.getMessage() for record in caplog.records if "deep-copies ExpensiveTemporary" in record.getMessage()]
    assert len(messages) == 1


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch build has no FP8 dtype")
def test_fp8_linear_scales_large_weights_without_saturation(monkeypatch):
    from worldfoundry.core.vram.layers import AutoWrappedLinear

    source = torch.nn.Linear(2, 2, bias=False)
    wrapper = AutoWrappedLinear(
        source,
        computation_dtype=torch.float8_e4m3fn,
        computation_device="cpu",
    )
    observed: dict[str, torch.Tensor] = {}

    def scaled_mm(a, b, *, scale_a, scale_b, bias, out_dtype):
        del bias, out_dtype
        observed.update(a=a, b=b, scale_a=scale_a, scale_b=scale_b)
        return (a.float() @ b.float()) * scale_a * scale_b

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    inputs = torch.tensor([[900.0, -450.0]], dtype=torch.float32)
    weights = torch.tensor([[900.0, 450.0], [-896.0, 224.0]], dtype=torch.float32)
    actual = wrapper.fp8_linear(inputs, weights)

    assert torch.all(observed["scale_b"] > 1.0)
    reconstructed_weight = observed["b"].float().T * observed["scale_b"].T
    torch.testing.assert_close(reconstructed_weight, weights, rtol=0.03, atol=1.0)
    torch.testing.assert_close(actual, inputs @ reconstructed_weight.T, rtol=1e-6, atol=1e-3)


def test_litema_fuses_same_device_updates_and_validates_restore(monkeypatch):
    from worldfoundry.core.nn.blocks.ema import LitEma

    model = torch.nn.Linear(3, 2)
    ema = LitEma(model, decay=0.5, use_num_upates=False)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)

    real_foreach_lerp = torch._foreach_lerp_
    calls = 0

    def observed_foreach_lerp(shadows, parameters, weight):
        nonlocal calls
        calls += 1
        return real_foreach_lerp(shadows, parameters, weight)

    monkeypatch.setattr(torch, "_foreach_lerp_", observed_foreach_lerp)
    ema(model)
    assert calls == 1
    shadows = dict(ema.named_buffers())
    for name, parameter in model.named_parameters():
        expected = initial[name].lerp(parameter, 0.5)
        torch.testing.assert_close(shadows[ema.m_name2s_name[name]], expected)

    ema.store(model.parameters())
    with pytest.raises(ValueError, match="parameter count"):
        ema.restore([next(model.parameters())])


def test_sparse3d_cache_unbounded_by_default():
    cache = Sparse3DCache(downsample=1)
    for index in range(64):
        cache.add_precomputed(points=torch.zeros(1, 2, 2, 3), latent_index=index)
    assert len(cache) == 64


def test_sparse3d_cache_evicts_oldest_when_bounded():
    cache = Sparse3DCache(downsample=1, max_entries=4)
    for index in range(10):
        cache.add_precomputed(points=torch.full((1, 2, 2, 3), float(index)), latent_index=index)
    assert len(cache) == 4
    assert cache._latent_indices == [6, 7, 8, 9]
    assert cache._frame_ids == [6, 7, 8, 9]
    assert float(cache._world_points[0][0, 0, 0, 0]) == 6.0


def test_sparse3d_cache_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        Sparse3DCache(max_entries=0)


def test_sparse3d_cache_clear():
    cache = Sparse3DCache()
    cache.add_precomputed(points=torch.zeros(1, 2, 2, 3), latent_index=0)
    cache.clear()
    assert len(cache) == 0
    assert (
        cache.retrieve(
            target_world_to_camera=torch.eye(4).unsqueeze(0),
            target_intrinsic=torch.eye(3).unsqueeze(0),
            target_hw=(8, 8),
            count=1,
        )
        == []
    )


def test_sparse3d_cache_chunked_projection_matches_single_batch():
    caches = [
        Sparse3DCache(downsample=1, projection_batch_size=1),
        Sparse3DCache(downsample=1, projection_batch_size=8),
    ]
    candidates = (
        torch.tensor([[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]]),
        torch.tensor([[[[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]]]),
    )
    for cache in caches:
        for index, points in enumerate(candidates):
            cache.add_precomputed(points=points, latent_index=index, frame_id=10 + index)

    kwargs = {
        "target_world_to_camera": torch.eye(4).unsqueeze(0),
        "target_intrinsic": torch.eye(3).unsqueeze(0),
        "target_hw": (1, 4),
        "count": 2,
    }
    chunked = caches[0].retrieve(**kwargs)
    single_batch = caches[1].retrieve(**kwargs)
    assert chunked == single_batch == [(1, 11), (0, 10)]


def test_metric_sync_init_distributed_raises_on_incomplete_torchrun_env(monkeypatch):
    from worldfoundry.core.distributed.collectives import metric_sync

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        metric_sync.init_distributed()


def test_configure_torch_backends_touches_real_tf32_attributes():
    from worldfoundry.core.execution import inference as core_inference

    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=False)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        core_inference._configure_torch_backends(matmul_precision="high", enable_tf32=True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        # A dead attribute on the module object must not be (re)created.
        assert "allow_tf32" not in vars(torch.backends.cuda)
        # Explicit highest precision wins over the TF32 default.
        core_inference._configure_torch_backends(matmul_precision="highest", enable_tf32=True)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def test_sdpa_patch_install_uninstall_roundtrip(monkeypatch):
    import torch.nn.functional as F

    from worldfoundry.core.execution import inference as core_inference

    monkeypatch.delenv("WORLDFOUNDRY_ATTENTION_BACKEND", raising=False)
    original = F.scaled_dot_product_attention
    assert not getattr(original, "_worldfoundry_core_sdpa", False)
    try:
        core_inference.install_worldfoundry_inference_infra(patch_sdpa=True)
        patched = F.scaled_dot_product_attention
        assert getattr(patched, "_worldfoundry_core_sdpa", False)
        assert core_inference.inference_infra_state().sdpa_patched is True

        # The patched function must still compute correct attention.
        torch.manual_seed(0)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        torch.testing.assert_close(patched(q, k, v), original(q, k, v), rtol=1e-5, atol=1e-6)

        core_inference.uninstall_worldfoundry_inference_infra()
        assert F.scaled_dot_product_attention is original
        assert core_inference.inference_infra_state().sdpa_patched is False
        assert core_inference.inference_infra_state().installed is False

        # Context manager: patched inside, restored outside.
        core_inference.install_worldfoundry_inference_infra(patch_sdpa=True)
        with core_inference.worldfoundry_inference_infra_disabled():
            assert F.scaled_dot_product_attention is original
        assert getattr(F.scaled_dot_product_attention, "_worldfoundry_core_sdpa", False)
    finally:
        core_inference.uninstall_worldfoundry_inference_infra()
        F.scaled_dot_product_attention = original


def test_dynamic_swap_installer_double_install_keeps_real_class():
    from worldfoundry.core.vram.memory import DynamicSwapInstaller

    module = torch.nn.Linear(3, 3)
    DynamicSwapInstaller.install_model(module, device="cpu")
    assert module.__dict__["forge_backup_original_class"] is torch.nn.Linear
    DynamicSwapInstaller.install_model(module, device="cpu")
    assert module.__dict__["forge_backup_original_class"] is torch.nn.Linear
    DynamicSwapInstaller.uninstall_model(module)
    assert module.__class__ is torch.nn.Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="layerwise offload requires CUDA")
def test_layerwise_offload_handle_disable_restores_model():
    from worldfoundry.core.vram.layerwise_offload import enable_layerwise_cpu_offload

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    model = TinyModel()
    reference_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    handle = enable_layerwise_cpu_offload(model, device="cuda:0")
    assert handle.enabled and handle.layer_count == 3
    with torch.no_grad():
        out_offloaded = model(torch.randn(2, 8, device="cuda:0"))
    assert out_offloaded.shape == (2, 8)

    assert handle.disable() is True
    assert handle.enabled is False
    assert not getattr(model, "_worldfoundry_layerwise_cpu_offload", False)
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu"
        assert parameter.shape == reference_state[name].shape
        torch.testing.assert_close(parameter.detach().cpu(), reference_state[name], rtol=0, atol=0)

    # Plain CPU forward works again with no hooks left behind.
    with torch.no_grad():
        x = torch.randn(2, 8)
        expected = x
        for layer in model.layers:
            expected = layer(expected)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

    # Second disable is a no-op.
    assert handle.disable() is False
