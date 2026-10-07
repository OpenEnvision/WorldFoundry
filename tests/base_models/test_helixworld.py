from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.helixworld import HelixWorldDenoiser
from worldfoundry.base_models.diffusion_model.models.encoders.helixworld import HelixWorldConditioner
from worldfoundry.base_models.diffusion_model.models.encoders.helixworld_controls import (
    action_id,
    camera_poses,
    expand_action_plan,
    prepare_video_control,
)
from worldfoundry.base_models.diffusion_model.models.initializers.ltx import LTXMultiStageLatentInitializer
from worldfoundry.base_models.diffusion_model.models.networks.ltx.helixworld import (
    HelixWorldAttention,
    HelixWorldModel,
    VideoControlCondition,
)
from worldfoundry.base_models.diffusion_model.runners.joint_chunked import JointChunkedDiffusionRunner
from worldfoundry.base_models.diffusion_model.runners.multistage import MultiStageComponents
from worldfoundry.base_models.diffusion_model.schedulers.clean_sample import CleanSampleNoiseScheduler
from worldfoundry.core.attention.cache.context import ContextAttentionCache


def _request(**kwargs):
    return DiffusionRequest(
        prompt="A forest path.", height=32, width=32, num_frames=185,
        sampling=SamplingConfig(num_inference_steps=4, seed=42),
        inputs={"images": Image.new("RGB", (48, 32)), "actions": "W:4,A:4,S:4,D:4,right:7",
                "audio_prompt": "Footsteps.", "av_prompt": "Footsteps follow the camera."},
        **kwargs,
    )


def test_action_plan_and_camera_coordinates():
    actions = expand_action_plan("W:5,right:5,stop", 15)
    assert [action_id(action) for action in actions] == [9] * 5 + [1] * 5 + [0] * 5
    assert action_id("W+right") == 10
    assert action_id("A+down") == 40
    assert action_id("W+S") == 0
    first = camera_poses(["W", "D"], "first_person")
    torch.testing.assert_close(first[:, :3, 3], torch.tensor([[0., 0., 0.], [0., 0., .08], [.08, 0., .08]]))
    third = camera_poses(["D"], "third_person")
    assert float(third[1, 0, 3] - third[0, 0, 3]) == pytest.approx(-.08)
    with pytest.raises(ValueError, match="latent transitions"):
        expand_action_plan("W:5", 15)


def test_control_file_roundtrip_preserves_token_masks(tmp_path):
    request = _request()
    control = prepare_video_control(request, device=torch.device("cpu"), dtype=torch.float32)
    assert control.action_ids.shape == (1, 24)
    torch.testing.assert_close(control.camera_w2c[0, 0], torch.eye(4))
    centers = torch.linalg.solve(control.camera_w2c[0, :, :3, :3], -control.camera_w2c[0, :, :3, 3, None]).squeeze(-1)
    step = torch.quantile(torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1), .75)
    assert float(step) == pytest.approx(.03)
    payload = {name: getattr(control, name).clone() for name in (
        "camera_intrinsics", "camera_w2c", "camera_valid_mask", "action_ids", "action_valid_mask",
    )}
    payload["camera_valid_mask"][:, 3] = False
    payload["action_valid_mask"][:, 3] = False
    path = tmp_path / "controls.pt"
    torch.save(payload, path)
    loaded = prepare_video_control(
        replace(request, inputs={**request.inputs, "control_path": path}), device=torch.device("cpu"), dtype=torch.float32,
    )
    for name, expected in payload.items():
        torch.testing.assert_close(getattr(loaded, name), expected)
    torch.testing.assert_close(loaded.camera_projection[0, 3], torch.eye(4))
    with pytest.raises(ValueError, match="one sample"):
        prepare_video_control(replace(request, prompt=("first", "second")), device=torch.device("cpu"), dtype=torch.float32)


def test_triplet_caption_uses_shared_text_conditioner():
    captured = []

    def encode(request, **_kwargs):
        captured.append(request.prompts)
        return Conditioning(positive={"text": "encoded"})

    conditioner = HelixWorldConditioner(SimpleNamespace(encode=encode))
    result = conditioner.encode(_request(), device=torch.device("cpu"), dtype=torch.float32)
    assert captured == [("Video description:\nA forest path.\n\nAudio description:\nFootsteps.\n\nJoint audio-visual description:\nFootsteps follow the camera.",)]
    assert result.positive["text"] == "encoded"
    assert result.shared["video_control"].action_ids.shape == (1, 24)


def test_camera_cache_matches_dense_attention_and_retains_invalid_keys():
    torch.manual_seed(3)
    attention = HelixWorldAttention(query_dim=8, heads=2, dim_head=4, apply_gated_attention=True).eval()
    x = torch.randn(2, 7, 8)
    intrinsics = torch.eye(3).repeat(2, 7, 1, 1)
    poses = torch.eye(4).repeat(2, 7, 1, 1)
    poses[:, :, 2, 3] = torch.arange(7) * .1
    mask = torch.tensor([[True, False, True, True, False, True, True]]).expand(2, -1)
    control = VideoControlCondition(intrinsics, poses, mask, torch.zeros(2, 7, dtype=torch.long), mask)
    cache = ContextAttentionCache()
    with torch.no_grad():
        expected = attention(x, camera_control=control)[:, 4:]
        attention(x[:, :4], camera_control=control.token_slice(0, 4), cache=cache.writing())
        actual = attention(x[:, 4:], camera_control=control.token_slice(4, 7), cache=cache)
        torch.testing.assert_close(actual, expected)
        perturbation_mask = torch.tensor([1., 0.]).view(2, 1, 1)
        perturbed = attention(x[:, 4:], camera_control=control.token_slice(4, 7), cache=cache,
                              perturbation_mask=perturbation_mask)
        bypass = attention(x[:, 4:], all_perturbed=True)
        torch.testing.assert_close(perturbed[0], actual[0])
        torch.testing.assert_close(perturbed[1], bypass[1])
    assert all(store.length == 4 for store in cache.stores.values())
    fresh = ContextAttentionCache()
    assert not fresh.stores and not fresh.key_masks


def test_chunked_native_model_replays_bounded_history_and_isolates_requests():
    torch.manual_seed(11)
    model = HelixWorldModel(
        num_attention_heads=2, attention_head_dim=24, num_layers=1, cross_attention_dim=48,
        audio_num_attention_heads=2, audio_attention_head_dim=24, audio_cross_attention_dim=48,
        apply_gated_attention=True, cross_attention_adaln=True,
    ).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=.05)
    denoiser = HelixWorldDenoiser(model, compute_dtype=torch.float32)
    events = []

    def record(model_input):
        cache = model_input.conditioning["attention_cache"]
        control = model_input.conditioning["video_control"]
        events.append((cache, cache.write, int(control.action_ids[0, 0]),
                       float(model_input.modalities["video"].positions[:, 0, :, 1].max())))
        return denoiser(model_input)

    conditioner = SimpleNamespace(encode=lambda request, **kwargs: Conditioning(
        positive={"video_context": torch.zeros(1, 3, 48), "audio_context": torch.zeros(1, 3, 48)},
        shared={"video_control": prepare_video_control(request, **kwargs)},
    ))
    initializer = LTXMultiStageLatentInitializer(
        lambda image: image.new_full((1, 128, 1, 1, 1), .25), stage_divisors=(1,), image_resize_mode="center_crop",
    )
    components = MultiStageComponents(
        denoiser=record, conditioner=conditioner, latent_initializer=initializer,
        schedulers=(CleanSampleNoiseScheduler((1., .9, .7, .4, 0.)),), processor=None,
        decoder=SimpleNamespace(decode_modalities=lambda states, request: {
            "video": states["video"].latent, "audio": states["audio"].latent,
        }),
    )
    runner = JointChunkedDiffusionRunner(
        model_id="helixworld", components=components, stage_steps=(4,), device="cpu", dtype=torch.float32,
        history_chunks=3, prefix_chunks=1, sampling_seed_offset=1_000_003, token_conditions={"video_control": "video"},
    )
    first = runner.run(_request())
    assert first.metadata["chunks"] == 6
    assert torch.isfinite(first.sample).all() and torch.isfinite(first.artifacts["audio"]).all()
    torch.testing.assert_close(first.latents[:, :1], torch.full((1, 1, 128), .25))
    last_cache = events[-1][0]
    assert [event[2] for event in events if event[0].stores is last_cache.stores and event[1]] == [0, 18, 27]
    assert max(event[3] for event in events) <= 121 / 24 + 1e-6
    assert last_cache.stores[model.transformer_blocks[0].attn1].length == 12
    prior_caches = {id(event[0]): event[0] for event in events}
    events.clear()
    repeated = runner.run(_request())
    torch.testing.assert_close(repeated.latents, first.latents, atol=0, rtol=0)
    torch.testing.assert_close(repeated.artifacts["audio"], first.artifacts["audio"], atol=0, rtol=0)
    assert prior_caches.keys().isdisjoint({id(event[0]) for event in events})
    changed = runner.run(replace(_request(), sampling=SamplingConfig(num_inference_steps=4, seed=43)))
    assert not torch.equal(changed.latents, first.latents)
