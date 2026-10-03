"""New schedules preserve canonical math, checkpoint identity and removal."""

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
from worldfoundry.core.acceleration.quantization.calibration import save_calibration
from worldfoundry.core.acceleration.quantization.svdquant import calibrate_svdquant
from worldfoundry.core.model_loading.policy import RuntimePolicy


def _model(image=False):
    return WanModel(
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
        has_image_input=image,
    ).eval()


@pytest.mark.parametrize("fusion", ["none", "kv", "qkv"])
@pytest.mark.parametrize("image", [False, True])
def test_mha_canonical_paths_restore(fusion, image):
    model = _model(image)
    block = model.blocks[0]
    x = torch.randn(2, 11, 128)
    context = torch.randn(2, 300 if image else 17, 128)
    freqs = torch.polar(torch.ones(11, 1, 64), torch.randn(11, 1, 64))
    with torch.no_grad():
        before = block.self_attn(x, freqs)
        cross_before = block.cross_attn(x, context)
    keys = list(model.state_dict())
    session = install_diffusion_accelerations(
        model,
        {
            "optimized_mha": {
                "self": {"fusion": fusion},
                "cross": {"fusion": fusion},
            }
        },
        RuntimePolicy(device="cpu", dtype=torch.float32),
    )
    with torch.no_grad():
        torch.testing.assert_close(block.self_attn(x, freqs), before, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(block.cross_attn(x, context), cross_before, atol=3e-6, rtol=3e-5)
    assert list(model.state_dict()) == keys
    report = session.report()["installed"][0]["runtime"]
    assert all(value["calls"] == 1 for value in report.values())
    assert report["blocks.0.cross_attn"]["sdpa_calls"] == (2 if image else 1)
    from benchmarks.inference.plugin_diagnostics import qualify_execution
    from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WanDenoiser

    runtime = WanDenoiser(model, compute_dtype=torch.float32, manage_autocast=False)
    assert qualify_execution(
        {"optimized_mha": {"self": {"fusion": fusion}, "cross": {"fusion": fusion}}},
        {"denoiser": runtime.runtime_optimization_report()},
    )["passed"]
    session.reset_request_window()
    assert all(value["calls"] == 0 for value in session.report()["installed"][0]["runtime"].values())
    with pytest.raises(RuntimeError, match="placement"):
        model.to(torch.float64)
    with pytest.raises(RuntimeError, match="training"):
        model.train()
    session.uninstall()
    with torch.no_grad():
        torch.testing.assert_close(block.self_attn(x, freqs), before, atol=0, rtol=0)
        torch.testing.assert_close(block.cross_attn(x, context), cross_before, atol=0, rtol=0)


def test_svdquant_artifact_installation_restore_and_checkpoint_guard(tmp_path):
    model = _model()
    original = model.blocks[0].ffn[0]
    state = calibrate_svdquant(original, torch.ones(128), rank=16)
    path = tmp_path / "state.pt"
    save_calibration(path, kind="svdquant", states={"blocks.0.ffn.0": state}, metadata={"seed": 7})
    expected = original.weight.detach().clone()
    session = install_diffusion_accelerations(model, {"svdquant": {"artifact": path}})
    with pytest.raises(RuntimeError, match="serializ"):
        model.state_dict()
    with pytest.raises(RuntimeError, match="training"):
        model.train()
    session.uninstall()
    assert model.blocks[0].ffn[0] is original
    torch.testing.assert_close(model.state_dict()["blocks.0.ffn.0.weight"], expected, atol=0, rtol=0)
    with torch.no_grad():
        original.weight.add_(0.01)
    with pytest.raises(ValueError, match="source weights"):
        install_diffusion_accelerations(model, {"svdquant": {"artifact": path}})
    assert not hasattr(model, "_worldfoundry_accelerations")


@pytest.mark.parametrize("first", ["optimized_mha", "attention_policy"])
def test_mha_and_scoped_attention_conflict_before_mutation(first):
    model = _model()
    original = model.blocks[0].self_attn.processor
    options = {"optimized_mha": {"self": {}}, "attention_policy": {"self": "torch"}}
    if first == "attention_policy":
        options = dict(reversed(tuple(options.items())))
    with pytest.raises(ValueError, match="conflict"):
        install_diffusion_accelerations(model, options)
    assert model.blocks[0].self_attn.processor is original
