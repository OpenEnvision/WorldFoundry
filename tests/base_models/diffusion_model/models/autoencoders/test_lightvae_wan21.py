"""Matched LightVAE admission, independent native parity and honest reporting."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.components import (
    BuildPurpose,
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import component
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE, WanVideoVAE38
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import WanVAE_
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.light_21 import (
    Wan21LightVAE,
    convert_lightvae_wan21_state_dict,
)
from worldfoundry.core.model_loading.policy import OffloadPolicy, RuntimePolicy


def _meta_state(module_class=Wan21LightVAE):
    with torch.device("meta"):
        return module_class().state_dict()


def _context(*, purpose=BuildPurpose.INFERENCE, variant="lightvae-wan21", **options):
    return ComponentBuildContext(
        "wan-lightvae-test",
        ComponentKey(ComponentKind.DECODER),
        RuntimePolicy(device="cpu"),
        {"weights": CheckpointSpec(source="/nonexistent/matching-checkpoint.pth")},
        component_options={"variant": variant, **options},
        purpose=purpose,
    )


def test_default_teacher_and_explicit_student_geometry_remain_distinct():
    with torch.device("meta"):
        teacher = WanVideoVAE(16, None)
        student = Wan21LightVAE()
    assert teacher.model.encoder.conv1.out_channels == 96
    assert student.model.encoder.conv1.out_channels == 24
    assert student.z_dim == teacher.z_dim == 16
    assert student.upsampling_factor == teacher.upsampling_factor == 8
    assert len(student.model.encoder.downsamples) == 11
    assert tuple(student.state_dict()) == tuple(teacher.state_dict())
    assert len(student.state_dict()) == 194
    with pytest.raises(ValueError, match="16 latent"):
        Wan21LightVAE(z_dim=48)


def test_strict_converter_accepts_only_all_matched_student_keys_and_shapes():
    student = _meta_state()
    raw = {key.removeprefix("model."): value for key, value in student.items()}
    assert tuple(convert_lightvae_wan21_state_dict(raw)) == tuple(student)
    assert tuple(convert_lightvae_wan21_state_dict({"model_state": raw})) == tuple(student)
    for malformed in ({}, {**raw, "extra": torch.empty(1)}, {**raw, "encoder.conv1.weight": torch.empty(1)}):
        with pytest.raises(ValueError, match="does not match"):
            convert_lightvae_wan21_state_dict(malformed)
    with pytest.raises(ValueError, match="floating-point"):
        convert_lightvae_wan21_state_dict({"not_tensor": "metadata"})
    with pytest.raises(ValueError, match="duplicate"):
        convert_lightvae_wan21_state_dict({**raw, "model.encoder.conv1.weight": raw["encoder.conv1.weight"]})


@pytest.mark.parametrize("teacher_class", [WanVideoVAE, WanVideoVAE38])
def test_teacher_and_48_channel_checkpoint_rejected(teacher_class):
    with pytest.raises(ValueError, match="teacher Wan2.1 or Wan2.2"):
        convert_lightvae_wan21_state_dict(_meta_state(teacher_class))


@pytest.mark.parametrize("purpose", [BuildPurpose.TRAINING, BuildPurpose.ROLLOUT, BuildPurpose.REWARD])
def test_non_inference_student_build_rejected_before_checkpoint_read(purpose):
    with pytest.raises(ValueError, match="inference builds only"):
        component.build_wan_video_decoder(_context(purpose=purpose))


def test_wan22_and_unknown_variant_rejected_before_loading_or_preview():
    with pytest.raises(ValueError, match="48-channel"):
        component.build_wan_video_vae38_decoder(_context(preview_decoder_path="/unused/preview.pth"))
    with pytest.raises(ValueError, match="unsupported Wan codec variant"):
        component.build_wan_video_decoder(_context(variant="arbitrary-student"))


@pytest.mark.parametrize(
    "options",
    [
        {"vae_decode_autocast": "bf16"},
        {"vae_channels_last": True},
        {"vae_channels_last_3d": True},
        {"cuda_graph": True},
        {"device_map": "balanced"},
    ],
)
def test_unvalidated_student_options_rejected_before_loading(options):
    policy = RuntimePolicy(device="cpu", dtype=torch.float32, options=options)
    with pytest.raises(ValueError, match="unvalidated"):
        component._load_wan_video_decoder(
            CheckpointSpec(source="/unused/checkpoint.pth"),
            policy,
            module_class=WanVideoVAE,
            variant="lightvae-wan21",
        )


@pytest.mark.parametrize("kwargs", [{"tiled": True}, {"temporal_chunk_size": 2}, {"parallel_degree": 2}])
def test_student_tiling_and_parallel_are_explicitly_unvalidated(kwargs):
    with pytest.raises(ValueError, match="unvalidated"):
        component._load_wan_video_decoder(
            CheckpointSpec(source="/unused/checkpoint.pth"),
            RuntimePolicy(device="cpu"),
            module_class=WanVideoVAE,
            variant="lightvae-wan21",
            **kwargs,
        )


def test_student_compile_offload_and_lower_precision_rejected():
    baseline = RuntimePolicy(device="cpu")
    for policy in (replace(baseline, compile=True), replace(baseline, offload=OffloadPolicy(mode="component"))):
        with pytest.raises(ValueError, match="unvalidated"):
            component._load_wan_video_decoder(
                CheckpointSpec(source="/unused/checkpoint.pth"),
                policy,
                module_class=WanVideoVAE,
                variant="lightvae-wan21",
            )
    with pytest.raises(ValueError, match="FP32"):
        component.load_wan_video_codec(
            "/unused/checkpoint.pth", device="cpu", dtype=torch.bfloat16, variant="lightvae-wan21"
        )


@pytest.fixture
def matched_checkpoint():
    path = Path(os.environ.get("WORLDFOUNDRY_LIGHTVAE21_CHECKPOINT", "/tmp/worldfoundry-lightvae21/lightvaew2_1.pth"))
    if not path.is_file():
        pytest.skip("provide the matched public student via WORLDFOUNDRY_LIGHTVAE21_CHECKPOINT")
    return path


@pytest.fixture
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matched_checkpoint_encode_decode_matches_independent_reference(
    matched_checkpoint,
    small_cpu_threads,
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    codec = component.load_wan_video_codec(matched_checkpoint, device=device, variant="lightvae-wan21")
    state = torch.load(matched_checkpoint, map_location="cpu", weights_only=True)
    reference = WanVAE_(dim=24, z_dim=16, temperal_downsample=[False, True, True]).eval()
    reference.load_state_dict(state, strict=True)
    reference.to(device=device, dtype=torch.float32)
    scale = [value.to(device=device) for value in codec.vae.scale]
    torch.manual_seed(42)
    images = torch.rand(1, 3, 5, 16, 16, device=device).mul(2).sub(1)
    with torch.inference_mode():
        expected_latents = reference.encode(images, scale)
        actual_latents = codec.encode(images)
        assert actual_latents.shape == (1, 16, 2, 2, 2)
        torch.testing.assert_close(actual_latents, expected_latents, rtol=1e-5, atol=1e-6)
        expected_video = reference.decode(expected_latents, scale).clamp(-1, 1)
        actual_video = codec.decode(expected_latents)
        assert actual_video.shape == images.shape
        torch.testing.assert_close(actual_video, expected_video, rtol=1e-5, atol=1e-6)
        # A second complete call must start with a fresh causal cache.
        torch.testing.assert_close(codec.decode(expected_latents), actual_video, rtol=0, atol=0)
    report = codec.runtime_optimization_report()
    assert report["quality_tier"] == "algorithmically-approximate"
    assert report["effective"]["vae_variant"] == "lightvae-wan21"
    assert report["runtime"]["lightvae_encode_calls"] == 1
    assert report["runtime"]["lightvae_decode_calls"] == 1
    assert report["runtime"]["lifetime"]["lightvae_decode_calls"] == 2


def test_first_and_body_encoder_cache_match_independent_reference(matched_checkpoint, small_cpu_threads):
    codec = component.load_wan_video_codec(matched_checkpoint, device="cpu", variant="lightvae-wan21")
    reference = WanVAE_(dim=24, z_dim=16, temperal_downsample=[False, True, True]).eval()
    reference.load_state_dict(torch.load(matched_checkpoint, map_location="cpu", weights_only=True), strict=True)
    native = codec.vae.model
    native.clear_cache()
    reference.clear_cache()
    torch.manual_seed(43)
    with torch.inference_mode():
        for frames in (1, 4):
            pixels = torch.randn(1, 3, frames, 16, 16)
            native._enc_conv_idx = [0]
            reference._enc_conv_idx = [0]
            actual, native._enc_feat_map, native._enc_conv_idx = native.encoder(
                pixels,
                feat_cache=native._enc_feat_map,
                feat_idx=native._enc_conv_idx,
            )
            expected = reference.encoder(pixels, feat_cache=reference._enc_feat_map, feat_idx=reference._enc_conv_idx)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert native._enc_conv_idx == reference._enc_conv_idx
            for actual_cache, expected_cache in zip(native._enc_feat_map, reference._enc_feat_map, strict=True):
                if actual_cache is None:
                    assert expected_cache is None
                else:
                    torch.testing.assert_close(actual_cache, expected_cache, rtol=1e-5, atol=1e-6)


def test_first_and_body_decoder_cache_match_independent_reference(matched_checkpoint, small_cpu_threads):
    codec = component.load_wan_video_codec(matched_checkpoint, device="cpu", variant="lightvae-wan21")
    reference = WanVAE_(dim=24, z_dim=16, temperal_downsample=[False, True, True]).eval()
    reference.load_state_dict(torch.load(matched_checkpoint, map_location="cpu", weights_only=True), strict=True)
    native = codec.vae.model
    native.clear_cache()
    reference.clear_cache()
    torch.manual_seed(44)
    with torch.inference_mode():
        for output_frames in (1, 4, 4):
            projected_latent = torch.randn(1, 16, 1, 2, 2)
            native._conv_idx = [0]
            reference._conv_idx = [0]
            actual, native._feat_map, native._conv_idx = native.decoder(
                projected_latent,
                feat_cache=native._feat_map,
                feat_idx=native._conv_idx,
            )
            expected = reference.decoder(projected_latent, feat_cache=reference._feat_map, feat_idx=reference._conv_idx)
            assert actual.shape == (1, 3, output_frames, 16, 16)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert native._conv_idx == reference._conv_idx
            for actual_cache, expected_cache in zip(native._feat_map, reference._feat_map, strict=True):
                if isinstance(actual_cache, torch.Tensor):
                    torch.testing.assert_close(actual_cache, expected_cache, rtol=1e-5, atol=1e-6)
                else:
                    assert actual_cache == expected_cache


def test_component_factory_uses_explicit_checkpoint_binding_and_rejects_outer_autocast(
    matched_checkpoint,
    small_cpu_threads,
):
    context = replace(_context(), checkpoints={"weights": CheckpointSpec(source=str(matched_checkpoint))})
    codec = component.build_wan_video_decoder(context)
    assert codec.runtime_optimization_report()["effective"]["vae_decode"] == "pending"
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="outer autocast"):
        codec.decode(torch.zeros(1, 16, 1, 2, 2))
    assert codec.runtime_optimization_report()["runtime"]["lightvae_decode_calls"] == 0
