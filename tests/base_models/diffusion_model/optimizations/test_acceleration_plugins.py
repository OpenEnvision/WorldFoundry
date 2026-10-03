"""Plugins exercise real native seams, request isolation and removal."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.denoisers.sana import SanaDenoiser
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WanDenoiser, _build_wan22_dual_expert_denoiser
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale import SanaMS
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention, SelfAttention, WanModel
from worldfoundry.base_models.diffusion_model.optimizations.plugins import (
    acceleration_policy,
    diffusion_acceleration_registry,
    install_diffusion_accelerations,
    validate_acceleration_installation,
)
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.core.model_loading.policy import RuntimePolicy


def _model():
    torch.manual_seed(9)
    model = SanaMS(
        input_size=4,
        patch_size=1,
        in_channels=4,
        hidden_size=24,
        depth=2,
        num_heads=3,
        caption_channels=12,
        model_max_length=4,
        pred_sigma=False,
        use_pe=False,
        attn_type="linear",
        linear_head_dim=8,
        ffn_type="mlp",
        cross_attn_type="vanilla",
    ).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.1)
    return model


def _input(request="a", branch="positive", step=0):
    return DenoiserInput(
        latents=torch.ones(1, 4, 4, 4),
        timestep=torch.tensor([500.0]),
        next_timestep=torch.tensor([400.0]),
        conditioning={"context": torch.ones(1, 1, 4, 12), "context_mask": torch.ones(1, 4)},
        step_index=step,
        total_steps=6,
        branch=branch,
        request_id=request,
    )


def test_zero_threshold_keeps_native_output_and_state_dict_exact():
    model = _model()
    keys = tuple(model.state_dict())
    dense = SanaDenoiser(model)
    with torch.inference_mode():
        expected = dense(_input()).sample
        dense.end_request("a")
        session = install_diffusion_accelerations(
            model, {"sana_block_fusion": True, "easycache": {"threshold": 0.0}}, RuntimePolicy()
        )
        candidate = SanaDenoiser(model)
        for step in range(6):
            torch.testing.assert_close(candidate(_input(step=step)).sample, expected, rtol=0, atol=0)
    report = candidate.feature_cache_report("a")
    assert report["hits"] == 0 and report["dense_block_calls"] == 12
    assert tuple(model.state_dict()) == keys
    with pytest.raises(RuntimeError, match="active requests"):
        session.uninstall()
    candidate.end_request("a")
    assert candidate.feature_cache_lifecycle_report()["live_requests"] == 0
    session.uninstall()
    assert candidate._feature_cache_config is None
    assert all(block._worldfoundry_block_fusion is None for block in model.blocks)


def test_requests_and_cfg_branches_do_not_share_residuals():
    model = _model()
    install_diffusion_accelerations(model, {"easycache": {"threshold": 0.1}})
    denoiser = SanaDenoiser(model)
    expected = {}
    with torch.inference_mode():
        for step in range(3):
            for request in ("a", "b"):
                for branch in ("negative", "positive"):
                    item = _input(request, branch, step)
                    offset = (1 if request == "b" else 0) + (2 if branch == "negative" else 0)
                    item = replace(item, latents=item.latents + offset)
                    output = denoiser(item).sample
                    key = (request, branch)
                    if step == 0:
                        expected[key] = output.clone()
                    torch.testing.assert_close(output, expected[key], rtol=1e-5, atol=1e-6)
    for request in ("a", "b"):
        report = denoiser.feature_cache_report(request)
        assert report["hits"] == 2 and report["request_local"]
        assert set(report["branches"]) == {"negative", "positive"}
        denoiser.end_request(request)
    assert denoiser.feature_cache_lifecycle_report()["live_requests"] == 0


def test_scoped_attention_does_not_modify_other_scope():
    model = torch.nn.Module()
    model.self_attn = SelfAttention(128, 1)
    model.cross_attn = CrossAttention(128, 1)
    session = install_diffusion_accelerations(model, {"attention_policy": {"self": "torch"}}, RuntimePolicy())
    assert model.self_attn.attn.attention_backend == "torch"
    assert model.cross_attn.attn.attention_backend is None
    session.uninstall()
    assert model.self_attn.attn.attention_backend is None


def test_scoped_attention_rejects_empty_target_without_installation():
    model = _model()
    with pytest.raises(ValueError, match="found no native Wan"):
        install_diffusion_accelerations(model, {"attention_policy": {"self": "torch"}})
    assert not hasattr(model, "_worldfoundry_accelerations")


@pytest.mark.parametrize("backend", ["sol_attn", "sage_attention", "sage_attention_3", "cudnn_fp8"])
@pytest.mark.parametrize("cache_first", [False, True])
def test_cache_and_approximate_scopes_conflict_in_direct_registry_without_context(monkeypatch, backend, cache_first):
    from worldfoundry.core.attention.backends import probe

    monkeypatch.setattr(probe, "resolve_attention_backend", lambda *args: backend)
    model = WanModel(
        dim=128,
        in_dim=2,
        ffn_dim=256,
        out_dim=2,
        text_dim=64,
        freq_dim=16,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=1,
        eps=1e-6,
        has_image_input=False,
    ).eval()
    options = {"easycache": {"threshold": 0.1}, "attention_policy": {"self": backend}}
    if not cache_first:
        options = dict(reversed(tuple(options.items())))
    with pytest.raises(ValueError, match="conflicts on seams"):
        diffusion_acceleration_registry().install(model, options)
    assert not hasattr(model, "_worldfoundry_accelerations")
    assert getattr(model, "_worldfoundry_easycache_config", None) is None
    assert model.blocks[0].self_attn.attn.attention_backend is None


def test_conflicting_cache_and_compile_fail_before_installation():
    model = _model()
    for policy in (
        RuntimePolicy(compile=True),
        RuntimePolicy(options={"cuda_graph": True}),
        RuntimePolicy(options={"feature_cache": "firstblock"}),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            install_diffusion_accelerations(model, {"sana_block_fusion": True, "easycache": {"threshold": 0.1}}, policy)
        assert not hasattr(model, "_worldfoundry_accelerations")
        assert all(block._worldfoundry_block_fusion is None for block in model.blocks)


def test_dense_request_after_removal_has_no_stale_receipt():
    model = _model()
    denoiser = SanaDenoiser(model)
    session = install_diffusion_accelerations(model, {"easycache": {"threshold": 0.1}})
    with torch.inference_mode():
        denoiser(_input("cached"))
        denoiser.end_request("cached")
        session.uninstall()
        denoiser(_input("dense"))
    report = denoiser.feature_cache_report()
    assert report["request_id"] == "dense" and report["events"] == 0 and not report["enabled"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sana_cuda_default_fusion_keeps_native_strides_and_output():
    model = _model().to(device="cuda", dtype=torch.bfloat16)
    denoiser = SanaDenoiser(model)
    item = _input()
    item = replace(
        item,
        latents=item.latents.cuda().bfloat16(),
        conditioning={key: value.cuda().bfloat16() for key, value in item.conditioning.items()},
    )
    strides = []
    hook = model.blocks[0].register_forward_pre_hook(lambda block, args: strides.append(args[0].stride()))
    try:
        with torch.inference_mode():
            expected = denoiser(item).sample
            denoiser.end_request("a")
            session = install_diffusion_accelerations(
                model, {"sana_block_fusion": {"backend": "triton", "min_elements": 0}}
            )
            receipt = {}
            with kernel_dispatch_receipt_scope(receipt):
                actual = denoiser(item).sample
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert strides[0] == strides[1] and strides[0][-1] != 1
        assert any(row["op"] == "scale_shift" and row["accelerated"] for row in receipt["dispatches"])
        denoiser.end_request("a")
        session.uninstall()
    finally:
        hook.remove()


@pytest.mark.parametrize("threshold", [0.0, 0.1])
def test_real_wan_blocks_use_request_cache_and_zero_threshold_is_exact(threshold):
    torch.manual_seed(9)
    model = WanModel(
        dim=12,
        in_dim=2,
        ffn_dim=24,
        out_dim=2,
        text_dim=8,
        freq_dim=4,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=2,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()
    denoiser = WanDenoiser(model, compute_dtype=torch.float32)
    item = replace(_input(), latents=torch.ones(1, 2, 1, 2, 2), conditioning={"context": torch.ones(1, 2, 8)})
    with torch.inference_mode():
        expected = denoiser(item).sample
        denoiser.end_request("a")
        session = install_diffusion_accelerations(model, {"easycache": {"threshold": threshold}})
        for step in range(6):
            actual = denoiser(replace(item, step_index=step)).sample
            torch.testing.assert_close(
                actual, expected, rtol=0 if threshold == 0 else 1e-5, atol=0 if threshold == 0 else 1e-6
            )
    report = denoiser.feature_cache_report("a")
    assert report["algorithm"] == "easycache" and report["request_local"]
    assert report["hits"] == (0 if threshold == 0 else 3)
    assert report["dense_block_calls"] + report["skipped_block_calls"] == 12
    denoiser.end_request("a")
    session.uninstall()


def test_component_options_override_runtime_and_cannot_enable_training_plugins():
    context = SimpleNamespace(
        policy=RuntimePolicy(options={"accelerations": {"easycache": {"threshold": 0.1}}, "cuda_graph": True}),
        component_options={"accelerations": {"sana_block_fusion": True}, "cuda_graph": False},
        purpose="inference",
    )
    policy = acceleration_policy(context)
    assert policy.options["accelerations"] == {"sana_block_fusion": True}
    assert policy.options["cuda_graph"] is False
    context.policy = RuntimePolicy()
    context.purpose = "training"
    with pytest.raises(ValueError, match="accelerations"):
        acceleration_policy(context)


def test_unsupported_denoiser_cannot_silently_accept_plugin_options():
    context = SimpleNamespace(
        policy=RuntimePolicy(options={"accelerations": {"sana_block_fusion": True}}),
        component_options={},
        purpose="inference",
    )
    with pytest.raises(ValueError, match="did not install"):
        validate_acceleration_installation(SimpleNamespace(), context)
    supported = SimpleNamespace(
        runtime_optimization_report=lambda: {
            "effective": {"accelerations": {"installed": [{"name": "sana_block_fusion"}], "execution_verified": False}}
        }
    )
    validate_acceleration_installation(supported, context)


@pytest.mark.parametrize("option", [{"cuda_graph": True}, {"cfg_parallel": 2}, {"cfg_gate_step": 0.5}])
def test_dual_expert_easycache_unsupported_combinations_rejected_before_loading(option):
    context = SimpleNamespace(
        policy=RuntimePolicy(options={"accelerations": {"easycache": {"threshold": 0.1}}, **option}),
        component_options={},
        purpose="inference",
    )
    with pytest.raises(ValueError, match="dual-expert EasyCache conflicts"):
        _build_wan22_dual_expert_denoiser(context, config={})


def test_assembler_enforces_support_and_component_training_policy():
    from worldfoundry.base_models.diffusion_model.assembly import NativeDiffusionAssembler
    from worldfoundry.base_models.diffusion_model.components import (
        ComponentKey,
        ComponentKind,
        ComponentSpec,
        ExecutionSpec,
    )
    from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe

    class Denoiser:
        def __call__(self, value):
            return value

    builds = []
    key = ComponentKey(ComponentKind.DENOISER)

    def factory(context):
        builds.append(context)
        return Denoiser()

    recipe = NativeDiffusionRecipe(
        model_id="unsupported-test",
        components=(ComponentSpec(key, factory),),
        execution=ExecutionSpec(bindings={"denoiser": key}),
    )
    assembler = NativeDiffusionAssembler()
    options = {key: {"accelerations": {"sana_block_fusion": True}}}
    with pytest.raises(ValueError, match="did not install"):
        assembler.build_components(recipe, purpose="inference", component_options=options)
    builds.clear()
    with pytest.raises(ValueError, match="accelerations"):
        assembler.build_components(recipe, purpose="training", component_options=options)
    assert not builds


@pytest.mark.parametrize("options", [{"tau": -1}, {"kv_splits": 2}, {"thresh_type": "unknown"}])
def test_invalid_sol_policy_does_not_partially_install(monkeypatch, options):
    from worldfoundry.core.attention.backends import probe

    monkeypatch.setattr(probe, "resolve_attention_backend", lambda *args: "sol_attn")
    model = torch.nn.Module()
    model.self_attn = SelfAttention(128, 1)
    with pytest.raises(ValueError):
        install_diffusion_accelerations(model, {"attention_policy": {"self": {"backend": "sol_attn", **options}}})
    assert model.self_attn.attn.attention_backend is None and not hasattr(model, "_worldfoundry_accelerations")
