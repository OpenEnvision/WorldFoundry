from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.optimizations.camera_condition import (
    CameraConditionCache,
    camera_features,
)


def _gamecraft_model(text_projection="linear"):
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.gamecraft.models import (
        HYVideoDiffusionTransformer,
    )

    args = SimpleNamespace(
        text_projection=text_projection,
        text_states_dim=16,
        text_states_dim_2=8,
        use_attention_mask=True,
    )
    return HYVideoDiffusionTransformer(
        args,
        hidden_size=64,
        num_heads=4,
        depth_double_blocks=1,
        depth_single_blocks=1,
        rope_dim_list=[8, 4, 4],
        camera_down_coef=2,
    ).eval()


def _wan_model(control_net=False, **overrides):
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dualcamctrl import WanControlNet

    kwargs = dict(
        dim=96,
        in_dim=4,
        ffn_dim=192,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        add_control_adapter=True,
        in_dim_control_adapter=6,
    )
    kwargs.update(overrides)
    if control_net:
        kwargs.setdefault("control_block_index", [])
        return WanControlNet(**kwargs).eval()
    return WanModel(**kwargs).eval()


def test_request_camera_features_reuse_and_condition_replacement():
    encoder = torch.nn.Conv2d(3, 4, 1).eval()
    calls = []
    handle = encoder.register_forward_hook(lambda *_: calls.append(1))
    first = CameraConditionCache(torch.randn(2, 3, 4, 4))
    second = CameraConditionCache(torch.randn(2, 3, 4, 4))
    keys = tuple(encoder.state_dict())
    with torch.no_grad():
        expected = camera_features(encoder, first.source)
        for _ in range(5):
            assert torch.equal(camera_features(encoder, first), expected)
        assert first.features.data_ptr() == camera_features(encoder, first).data_ptr()
        assert not torch.equal(camera_features(encoder, second), first.features)
    handle.remove()
    assert len(calls) == 3
    assert tuple(encoder.state_dict()) == keys
    assert first.nbytes == first.features.numel() * first.features.element_size()


@pytest.mark.parametrize("batch", [2, 4])
def test_cfg_split_camera_cache_retains_each_batch_slice(batch):
    from worldfoundry.base_models.diffusion_model.optimizations.wan.cfg_skip import cfg_skip

    class Model(torch.nn.Module):
        cfg_skip_ratio = 0.5
        num_inference_steps = 4
        current_steps = 0

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 4, 1)

        @cfg_skip()
        def forward(self, x, camera, cond_flag=True):
            return x + camera_features(self.encoder, camera)

    model = Model().eval()
    source = torch.randn(batch, 3, 4, 4)
    values = torch.randn(batch, 4, 4, 4)
    condition = CameraConditionCache(source)
    calls = []
    handle = model.encoder.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        for step in range(4):
            model.current_steps = step
            assert torch.equal(model(values, source), model(values, condition))
    handle.remove()
    assert len(calls) == (8 if batch == 2 else 12)
    assert len(condition._batches) == 2
    cached_batch_count = batch if batch == 2 else batch + batch // 2
    assert condition.nbytes == cached_batch_count * 4 * 4 * 4 * source.element_size()


def test_camera_features_training_remains_differentiable():
    encoder = torch.nn.Conv2d(3, 4, 1)
    source = torch.randn(1, 3, 4, 4, requires_grad=True)
    camera_features(encoder, source).sum().backward()
    assert source.grad is not None
    assert encoder.weight.grad is not None
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval no-grad"):
        camera_features(encoder, CameraConditionCache(source))
    encoder.eval()
    with pytest.raises(RuntimeError, match="eval no-grad"):
        camera_features(encoder, CameraConditionCache(source))


@pytest.mark.parametrize("control_net", [False, True])
@pytest.mark.parametrize("batch,camera_batch", [(1, 1), (2, 2), (2, 1)])
def test_wan_camera_adapter_patchify_matches_raw(control_net, batch, camera_batch):
    model = _wan_model(control_net)
    latent = torch.randn(batch, 4, 2, 4, 4)
    camera = torch.randn(camera_batch, 6, 2, 32, 32)
    condition = CameraConditionCache(camera)
    keys = tuple(model.state_dict())
    with torch.no_grad():
        raw, grid = model.patchify(latent, camera)
        cached, cached_grid = model.patchify(latent, condition)
        assert torch.equal(raw, cached)
        assert grid == cached_grid
        assert raw.shape[0] == batch
        expected = model.patch_embedding(latent) + model.control_adapter(camera)
        assert torch.equal(raw, expected.flatten(2).transpose(1, 2).contiguous())
        assert torch.equal(
            model.patchify(latent + 1, camera)[0],
            model.patchify(latent + 1, condition)[0],
        )
    assert tuple(model.state_dict()) == keys


def test_new_camera_request_uses_updated_weights():
    encoder = torch.nn.Conv2d(3, 4, 1).eval()
    source = torch.randn(1, 3, 4, 4)
    with torch.no_grad():
        previous = camera_features(encoder, CameraConditionCache(source)).clone()
        encoder.bias.add_(1)
        current = CameraConditionCache(source)
        assert torch.equal(camera_features(encoder, current), encoder(source))
        assert not torch.equal(current.features, previous)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_camera_feature_autocast(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("CPU camera path uses BF16 autocast")
    encoder = torch.nn.Conv2d(3, 4, 1).to(device).eval()
    source = torch.randn(2, 3, 4, 4, device=device)
    condition = CameraConditionCache(source)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        expected = encoder(source)
        assert torch.equal(camera_features(encoder, condition), expected)
        assert condition.features.dtype == dtype
        assert torch.equal(camera_features(encoder, condition), expected)


@pytest.mark.parametrize("latent_frames", [9, 10, 18])
def test_gamecraft_camera_features_match_raw(latent_frames):
    model = _gamecraft_model()
    camera = torch.randn(2, 33, 6, 8, 8)
    condition = CameraConditionCache(camera)
    calls = []
    handle = model.camera_net.register_forward_hook(lambda *_: calls.append(1))

    def encode(value):
        return model._encode_camera_condition(value, latent_frames)

    with torch.no_grad():
        raw = camera_features(model.camera_net, camera, encode=encode)
        before = len(calls)
        for _ in range(4):
            cached = camera_features(model.camera_net, condition, encode=encode)
            assert torch.equal(raw, cached)
        assert cached.shape == (2, latent_frames * 4, 64)
        assert len(calls) == before * 2
    handle.remove()


@pytest.mark.parametrize("latent_frames", [9, 10, 18])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gamecraft_full_forward_camera_reuse(latent_frames, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _gamecraft_model("single_refiner").to(device)
    kwargs = dict(
        x=torch.randn(2, 4, latent_frames, 4, 4, device=device),
        text_states=torch.randn(2, 4, 16, device=device),
        text_mask=torch.ones(2, 4, dtype=torch.bool, device=device),
        text_states_2=torch.randn(2, 8, device=device),
        freqs_cos=torch.ones(latent_frames * 4, 16, device=device),
        freqs_sin=torch.zeros(latent_frames * 4, 16, device=device),
    )
    camera = torch.randn(2, 33, 6, 8, 8, device=device)
    cache = CameraConditionCache(camera)
    keys = tuple(model.state_dict())
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        model.camera_net.final_proj.weight.normal_(std=0.01)
        model.final_layer.linear.weight.normal_(std=0.01)
        model.final_layer.adaLN_modulation[1].weight.normal_(std=0.01)
        for timestep in (1000, 500):
            t = torch.full((2,), timestep, dtype=torch.float32, device=device)
            raw = model(t=t, cam_latents=camera, **kwargs)["x"]
            cached = model(t=t, cam_latents=cache, **kwargs)["x"]
            assert torch.equal(raw, cached)
    assert tuple(model.state_dict()) == keys


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_wan_fun_full_forward_camera_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.video_x_fun.transformer import (
        WanTransformer3DModel,
    )

    model = (
        WanTransformer3DModel(
            model_type="t2v",
            in_dim=4,
            out_dim=4,
            dim=64,
            ffn_dim=128,
            freq_dim=16,
            text_dim=16,
            text_len=8,
            num_heads=4,
            num_layers=1,
            add_control_adapter=True,
            in_dim_control_adapter=6,
            downscale_factor_control_adapter=2,
        )
        .to(device)
        .eval()
    )
    kwargs = dict(
        x=torch.randn(2, 4, 3, 4, 4, device=device),
        context=torch.randn(2, 4, 16, device=device),
        seq_len=12,
    )
    camera = torch.randn(2, 6, 3, 8, 8, device=device)
    cache = CameraConditionCache(camera)
    keys = tuple(model.state_dict())
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        model.head.head.weight.normal_(std=0.01)
        for timestep in (1000, 500):
            t = torch.full((2,), timestep, dtype=torch.float32, device=device)
            raw = model(t=t, y_camera=camera, **kwargs)
            cached = model(t=t, y_camera=cache, **kwargs)
            assert torch.equal(raw, cached)
    assert tuple(model.state_dict()) == keys


def test_dualcamctrl_full_denoiser_reuses_only_camera_stem():
    pytest.importorskip("ftfy")
    from worldfoundry.synthesis.visual_generation.dualcamctrl.native_pipeline import model_fn_wan_video

    model = _wan_model(True, control_block_index=[0], rgb_inject_blocks=[0], depth_inject_blocks=[0])
    camera = torch.randn(1, 6, 2, 32, 32)
    cache = CameraConditionCache(camera)
    kwargs = dict(
        dit=model,
        latents=torch.randn(1, 4, 2, 4, 4),
        control_latents=torch.randn(1, 4, 2, 4, 4),
        context=torch.randn(1, 4, 16),
        return_control_latents=True,
    )
    control_calls, camera_calls = [], []
    handles = [
        model.control_blocks[0].register_forward_hook(lambda *_: control_calls.append(1)),
        model.control_adapter.register_forward_hook(lambda *_: camera_calls.append(1)),
    ]
    with torch.no_grad():
        model.control_zero_inits[0].weight.normal_(std=0.01)
        model.head.head.weight.normal_(std=0.01)
        for timestep in (1000, 500):
            t = torch.tensor([timestep], dtype=torch.float32)
            raw = model_fn_wan_video(timestep=t, control_camera_latents_input=camera, **kwargs)
            cached = model_fn_wan_video(timestep=t, control_camera_latents_input=cache, **kwargs)
            assert all(torch.equal(left, right) for left, right in zip(raw, cached))
    for handle in handles:
        handle.remove()
    assert len(camera_calls) == 5
    assert len(control_calls) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_camera_cache_with_sequential_cpu_offload():
    from accelerate import cpu_offload

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 4, 1)

        def forward(self, values, camera):
            return values + camera_features(self.encoder, camera)

    model = Model().eval()
    cpu_offload(model, execution_device=torch.device("cuda"))
    values = torch.randn(2, 4, 4, 4, device="cuda")
    camera = torch.randn(2, 3, 4, 4, device="cuda")
    condition = CameraConditionCache(camera)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw = model(values, camera)
        cached = model(values, condition)
        assert torch.equal(raw, cached)
        assert model.encoder.weight.device.type == "meta"
        assert torch.equal(model(values + 1, camera), model(values + 1, condition))
