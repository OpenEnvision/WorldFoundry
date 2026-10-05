"""Small real tensors for shared model loading, causal sampling and v1.5 I/O."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.camera_21 import WanVAE_
from worldfoundry.base_models.diffusion_model.models.denoisers.wan_causal_world import WanCausalWorldDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_camera_21 import CausalWanModel
from worldfoundry.base_models.diffusion_model.runners.inspatio_world import InspatioCausalRollout, sample_noise_like
from worldfoundry.base_models.three_dimensions.point_clouds.depth_warper import DepthWarper
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_io import (
    decode_depth,
    decode_video_depth_frame,
    encode_depth,
    encode_video_frame,
    iter_mask_latents,
)
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_render import select_chunk_top3
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_runtime import InspatioWorldV15Runtime
from worldfoundry.synthesis.visual_generation.inspatio_world.v15_scene import (
    load_scene,
    padded_frame_count,
    stage_direct_input,
)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return CausalWanModel(in_dim=36, dim=32, ffn_dim=64, freq_dim=16,
                          text_dim=32, text_len=4, num_heads=4, num_layers=1).eval()


def test_checkpoint_loads_without_wan_base_dit_and_rejects_missing_keys(tmp_path):
    torch.manual_seed(8)
    model = tiny_model()
    (tmp_path / "config.json").write_text(json.dumps(dict(model.config)))
    state = {"model." + key: value for key, value in model.state_dict().items()}
    weights = tmp_path / "checkpoint.safetensors"
    save_file(state, str(weights))
    loaded = WanCausalWorldDenoiser.from_pretrained(weights, tmp_path, device="cpu", dtype=torch.float32)
    assert set(loaded.model.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    del state["model.patch_embedding.weight"]
    save_file(state, str(weights))
    with pytest.raises(RuntimeError, match="Missing key"):
        WanCausalWorldDenoiser.from_pretrained(weights, tmp_path, device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_schedule_and_flow_conversion_match_released_equations(dtype):
    denoiser = WanCausalWorldDenoiser(tiny_model())
    sigma = torch.linspace(1.0, 0.0, 1001)[:-1]
    sigma = 5 * sigma / (1 + 4 * sigma)
    torch.testing.assert_close(denoiser.scheduler.sigmas, sigma, rtol=0, atol=0)
    torch.manual_seed(6)
    flow = torch.randn(4, 16, 4, 4).to(dtype)
    noisy = torch.randn_like(flow)
    times = torch.tensor([1000., 750., 500., 250.])
    index = (sigma.double()[None] * 1000 - times[:, None]).abs().argmin(dim=1)
    expected = (noisy.double() - sigma.double()[index, None, None, None] * flow.double()).to(dtype)
    actual = denoiser._convert_flow_pred_to_x0(flow, noisy, times)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("permuted", [False, True])
def test_request_noise_preserves_official_draws_layout_and_global_rng(dtype, permuted):
    source = torch.empty(1, 16, 6, 4, 4, dtype=dtype)
    if permuted:
        source = source.permute(0, 2, 1, 3, 4)
    torch.manual_seed(42)
    expected = torch.randn_like(source)
    state = torch.get_rng_state().clone()
    actual = sample_noise_like(source, torch.Generator().manual_seed(42))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    assert torch.equal(state, torch.get_rng_state())


def make_rollout():
    denoiser = WanCausalWorldDenoiser(tiny_model().to(torch.bfloat16))
    text = torch.randn(1, 4, 32).to(torch.bfloat16)
    config = SimpleNamespace(denoising_step_list=[1000, 750, 500, 250],
                             warp_denoising_step=True, num_frame_per_block=3)
    runner = InspatioCausalRollout(config, generator=denoiser,
                                  text_encoder=lambda text_prompts: {"prompt_embeds": text})
    runner.frame_seq_length = 4
    return runner, text


@torch.no_grad()
def test_two_block_rollout_matches_independent_upstream_equations_and_rng():
    torch.manual_seed(5)
    runner, text = make_rollout()
    shape = (1, 6, 16, 4, 4)
    noise, source, render = (torch.randn(shape).to(torch.bfloat16) for _ in range(3))
    mask = torch.randn(1, 6, 4, 4, 4).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(19)
    before = torch.get_rng_state().clone()
    actual = runner.inference(noise, ["room"], source, render, mask, decode=False, noise_generator=generator)
    assert torch.equal(before, torch.get_rng_state())

    # Independent reference uses the public v1.5 block equations, without the
    # shared denoise helper. It detects context ordering and RoPE/cache offsets.
    rng = torch.Generator().manual_seed(19)
    caches = [{name: torch.zeros(1, 24, 4, 8, dtype=torch.bfloat16) for name in ("k", "v")}]
    results = []
    for start in (0, 3):
        ref = source[:, start:start + 3]
        zeros = torch.zeros_like(ref)
        context = torch.cat((ref, zeros[:, :, :4], zeros), dim=2)
        if results:
            previous = results[-1]
            context = torch.cat((context, torch.cat((previous, zeros[:, :, :4], zeros), dim=2)), dim=1)
        control = torch.cat((mask[:, start:start + 3], render[:, start:start + 3]), dim=2)
        def forward(value, time, size, offset):
            return runner.generator(value, {"prompt_embeds": text}, torch.full((1, 3), time),
                                    caches, (0, size), control, offset)
        forward(context, 0, -1, 0)
        sample = noise[:, start:start + 3]
        for index, time in enumerate(runner.denoising_step_list):
            _, clean = forward(sample, time, context.shape[1] * 4, 6)
            if index < 3:
                next_time = runner.denoising_step_list[index + 1]
                step = (runner.scheduler.timesteps[None] - next_time).abs().argmin(dim=1)
                sigma = runner.scheduler.sigmas[step].reshape(1, 1, 1, 1)
                clean_flat = clean.flatten(0, 1)
                fresh = torch.randn(clean_flat.shape, generator=rng, dtype=torch.bfloat16)
                sample = ((1 - sigma) * clean_flat + sigma * fresh).to(torch.bfloat16).unflatten(0, (1, 3))
        results.append(clean)
    torch.testing.assert_close(actual, torch.cat(results, dim=1), rtol=0, atol=0)
    repeated = runner.inference(noise, ["room"], source, render, mask, decode=False,
                                noise_generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    runner._initialize_kv_cache(2, torch.float64, "cpu")
    assert runner.kv_cache1[0]["k"].shape == (2, 24, 4, 8)
    assert runner.kv_cache1[0]["k"].dtype == torch.float64
    with pytest.raises(ValueError, match="mask latent shapes"):
        runner.inference(noise, ["room"], source, render, mask[:, :3], decode=False)


@torch.no_grad()
def test_cached_vae_preserves_one_shot_pixels_and_clears_failed_encode():
    torch.manual_seed(4)
    model = WanVAE_(dim=8, z_dim=2, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                    temperal_downsample=[False, True, True]).eval()
    video = torch.randn(1, 3, 9, 16, 16)
    scale = [torch.tensor([.1, .2]), torch.tensor([1.5, 2.0])]
    expected = model.encode(video, scale)
    model.clear_cache()
    actual = torch.cat([model.cached_encode(chunk, scale) for chunk in
                        (video[:, :, :1], video[:, :, 1:5], video[:, :, 5:])], dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected_pixels = model.decode(expected, scale)
    model.clear_cache()
    pixels = torch.cat([model.cached_decode(chunk, scale) for chunk in actual.split(1, dim=2)], dim=2)
    torch.testing.assert_close(pixels, expected_pixels, rtol=1e-5, atol=1e-5)
    runtime = InspatioWorldV15Runtime(device="cpu")
    runtime.vae = SimpleNamespace(model=model, scale=scale)
    def failed_chunks():
        yield torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
        raise RuntimeError("read failure")
    with pytest.raises(RuntimeError, match="read failure"):
        runtime._encode_condition(failed_chunks())
    assert all(item is None for item in model._enc_feat_map + model._feat_map)


def test_depth_bytes_masks_and_block_padding():
    depth = np.linspace(.2, 5, 16, dtype=np.float32).reshape(4, 4)
    encoded = encode_depth(depth, .2, 5)
    expected = decode_depth(encoded, .2, 5)
    bgr = encode_video_frame(encoded)[..., ::-1].copy()
    np.testing.assert_array_equal(decode_video_depth_frame(bgr, .2, 5), expected)
    np.testing.assert_allclose(expected, depth, atol=5/65535, rtol=0)
    assert [padded_frame_count(count) for count in (1, 8, 9, 10, 21, 22)] == [9, 9, 9, 21, 21, 33]
    chunks = [torch.full((1, 3, 16, 16), 255, dtype=torch.uint8), torch.zeros(4, 3, 16, 16, dtype=torch.uint8)]
    masks = torch.cat(list(iter_mask_latents(iter(chunks), "cpu", torch.bfloat16)), dim=1)
    assert masks.shape == (1, 2, 4, 2, 2)
    assert torch.all(masks[:, 0] == 1) and torch.all(masks[:, 1] == -1)


def test_four_view_selection_keeps_three_distinct_source_views():
    choices, views = select_chunk_top3([[10, 9, 0, 1], [10, 0, 8, 1], [10, 0, 0, 4]])
    assert views == [[0, 1, 2]] and choices == [1, 2, 0]
    assert select_chunk_top3([[1] * 4] * 3) == ([0, 1, 2], [[0, 1, 2]])
    with pytest.raises(ValueError, match="three latents"):
        select_chunk_top3([[0] * 4])


def test_depth_warper_identity_preserves_rgb_and_holes():
    image = torch.rand(1, 3, 4, 6) * 2 - 1
    depth = torch.ones(1, 1, 4, 6)
    source_mask = torch.ones_like(depth)
    flow = torch.zeros(1, 2, 4, 6)
    warper = DepthWarper()
    warped, mask = warper.bilinear_splatting(image, source_mask, depth, flow, None, is_image=True)
    torch.testing.assert_close(warped, image, rtol=1e-6, atol=1e-6)
    assert torch.all(mask == 1)
    source_mask[:, :, 1, 2] = 0
    warped, mask = warper.bilinear_splatting(image, source_mask, depth, flow, None, is_image=True)
    assert mask[0, 0, 1, 2] == 0 and torch.all(warped[0, :, 1, 2] == -1)


def prepared_image_scene(tmp_path, views=1):
    (tmp_path / "input").mkdir()
    (tmp_path / "depth").mkdir()
    for index in range(views):
        Image.new("RGB", (832, 480), (index * 20, 128, 60)).save(tmp_path / f"input/view_{index:02d}.png")
        name = "depth.png" if views == 1 else f"depth_{index:02d}.png"
        Image.fromarray(np.full((480, 832), 32768, dtype=np.uint16)).save(tmp_path / "depth" / name)
    (tmp_path / "input/prompt.txt").write_text("room")
    (tmp_path / "depth/metadata.txt").write_text("0 5\n")
    np.savetxt(tmp_path / "depth/source_intrinsics.txt", np.repeat(np.eye(3)[None], views, axis=0).reshape(-1, 9))
    np.savetxt(tmp_path / "depth/source_tcw.txt", np.repeat(np.eye(4)[None], views, axis=0).reshape(-1, 16))
    np.savetxt(tmp_path / "input/target_tcw.txt", np.repeat(np.eye(4)[None], 9, axis=0).reshape(-1, 16))
    (tmp_path / "scene.json").write_text(json.dumps({"kind": "image", "views": views, "frames": 9,
                                                   "valid_frames": 8, "fps": 16, "resolution": [480, 832],
                                                   "target_intrinsics": "estimated_source_view_00_fixed",
                                                   "depth_encoding": "uint16_minmax_v1"}))
    return tmp_path


@pytest.mark.parametrize("views", [1, 4])
def test_prepared_scene_checks_real_rgb_depth_and_camera_counts(tmp_path, views):
    scene = prepared_image_scene(tmp_path, views)
    assert load_scene(scene)["views"] == views
    before = (scene / "input/prompt.txt").read_bytes()
    assert load_scene(scene, prompt="override")["text"] == "override"
    assert (scene / "input/prompt.txt").read_bytes() == before
    np.savetxt(scene / "input/target_tcw.txt", np.eye(4).reshape(1, 16))
    with pytest.raises(ValueError, match="count mismatch"):
        load_scene(scene)


def test_prepared_scene_rejects_nonfinite_camera_and_wrong_depth_dtype(tmp_path):
    scene = prepared_image_scene(tmp_path)
    Image.new("RGB", (832, 480)).save(scene / "depth/depth.png")
    with pytest.raises(ValueError, match="uint16"):
        load_scene(scene)
    target = np.eye(4).reshape(1, 16).repeat(9, axis=0)
    target[0, 0] = np.nan
    np.savetxt(scene / "input/target_tcw.txt", target)
    with pytest.raises(ValueError, match="finite"):
        load_scene(scene)


@pytest.mark.parametrize("views", [1, 4])
def test_direct_images_preserve_view_order_and_write_replayable_metadata(tmp_path, views):
    trajectory = tmp_path / "target.txt"
    np.savetxt(trajectory, np.tile(np.eye(4), (9, 1, 1)).reshape(-1, 16))
    images = [Image.new("RGB", (832, 480), (index * 50, 128, 70)) for index in range(views)]
    directory = tmp_path / "prepared"
    record = stage_direct_input(directory, images=images, prompt="room", traj_txt_path=trajectory)
    assert record["views"] == views and record["valid_frames"] == 9
    assert (directory / "input/target_tcw.txt").read_bytes() == trajectory.read_bytes()
    assert json.loads((directory / "scene.json").read_text())["resolution"] == [480, 832]
    for index in range(views):
        with Image.open(directory / f"input/view_{index:02d}.png") as actual:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(images[index]))


def test_direct_image_rejects_invalid_inputs_before_model_loading(tmp_path):
    trajectory = tmp_path / "target.txt"
    np.savetxt(trajectory, np.tile(np.eye(4), (9, 1, 1)).reshape(-1, 16))
    image = Image.new("RGB", (832, 480))
    for index, (options, match) in enumerate([
        ({"images": [image, image]}, "one or four"),
        ({"images": image, "videos": "v.mp4"}, "images or videos"),
        ({"images": Image.new("RGB", (32, 32))}, "832x480"),
        ({"images": image, "prompt": ""}, "prompt"),
    ]):
        with pytest.raises(ValueError, match=match):
            stage_direct_input(tmp_path / f"invalid-{index}", **{"prompt": "room", "traj_txt_path": trajectory, **options})
    np.savetxt(trajectory, np.tile(np.eye(4), (10, 1, 1)).reshape(-1, 16))
    with pytest.raises(ValueError, match="three-latent"):
        stage_direct_input(tmp_path / "invalid-multiview", images=[image] * 4, prompt="room", traj_txt_path=trajectory)


def test_prepared_scene_allows_official_empty_prompt(tmp_path):
    scene = prepared_image_scene(tmp_path)
    (scene / "input/prompt.txt").write_text("")
    assert load_scene(scene)["text"] == ""
