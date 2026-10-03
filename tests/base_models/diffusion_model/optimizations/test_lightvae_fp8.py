"""Calibrated convolution replacement preserves causal state and codec ownership."""

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import load_wan_video_codec
from worldfoundry.base_models.diffusion_model.optimizations.lightvae_fp8 import (
    install_lightvae_fp8,
    lightvae_encoder_convolutions,
)
from worldfoundry.core.acceleration.plugins import acceleration_runtime_scope
from worldfoundry.core.acceleration.quantization.calibration import ChannelObserver, save_calibration
from worldfoundry.core.acceleration.quantization.fp8_conv import calibrate_fp8_convolution


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_matched_codec_quantized_encoder_restores_cache_checkpoint_and_owner(tmp_path):
    checkpoint = Path("/tmp/worldfoundry-lightvae21/lightvaew2_1.pth")
    if not checkpoint.is_file():
        pytest.skip("matched LightVAE21 checkpoint required")
    codec = load_wan_video_codec(checkpoint, variant="lightvae-wan21")
    modules = lightvae_encoder_convolutions(codec.vae)
    selected = {name: module for name, module in modules.items() if name in {"model.encoder.conv1", "model.conv1"}}
    assert len(selected) == 2
    originals = {name: (module, type(module), module.forward.__func__) for name, module in selected.items()}
    keys = tuple(codec.vae.state_dict())
    pixels = torch.rand(1, 3, 9, 16, 16, device="cuda").mul(2).sub(1)
    with ChannelObserver(selected, channel_dim=1) as observer, torch.inference_mode():
        baseline = codec.encode(pixels)
        decoded = codec.decode(baseline)
    observer.validate()
    states = {name: calibrate_fp8_convolution(module, observer.maxima[name]) for name, module in selected.items()}
    artifact = tmp_path / "encoder.pt"
    save_calibration(artifact, kind="lightvae_fp8", states=states, metadata={"test": "causal-cache"})
    session = install_lightvae_fp8(codec.vae, artifact)
    session.bind_runtime(codec)
    assert tuple(codec.vae.state_dict()) == keys
    for name, (module, cls, forward) in originals.items():
        assert selected[name] is module and type(module) is cls and module.forward.__func__ is forward
    with torch.inference_mode():
        first = codec.encode(pixels)
        receipt = codec.runtime_optimization_report()["fp8_encoder"]
        second = codec.encode(pixels)
        assert torch.equal(first, second)
        # Request counters reset; they must not accumulate across encode calls.
        assert codec.runtime_optimization_report()["fp8_encoder"]["kernel_calls"] == receipt["kernel_calls"]
        assert receipt["kernel_calls"] >= 4 and receipt["clipped_input_operands"] == 0
        torch.testing.assert_close(codec.decode(baseline), decoded, rtol=0, atol=0)
    for operation in (lambda: codec.vae.to("cpu"), lambda: codec.vae.load_state_dict({}), codec.vae.train):
        with pytest.raises(RuntimeError, match="uninstall"):
            operation()
    with pytest.raises(RuntimeError, match="injected"):
        with acceleration_runtime_scope(codec.vae, codec):
            with pytest.raises(RuntimeError, match="active requests"):
                session.uninstall()
            raise RuntimeError("injected forward error")
    session.uninstall()
    assert all("_conv_forward" not in module.__dict__ for module in selected.values())
    with torch.inference_mode():
        torch.testing.assert_close(codec.encode(pixels), baseline, rtol=0, atol=0)
    # The public loader installs the same checkpoint-bound artifact.
    loaded = load_wan_video_codec(checkpoint, variant="lightvae-wan21", fp8_calibration=artifact)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.encode(pixels), first, rtol=0, atol=0)
