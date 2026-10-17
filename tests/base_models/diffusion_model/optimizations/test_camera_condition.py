"""Camera preparation preserves inference arithmetic and request ownership."""

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.camera_condition import (
    apply_camera_modulation,
    camera_condition_for_block,
    prepare_wan_camera_condition,
)


def camera_model():
    torch.manual_seed(3)
    model = nn.Module()
    model.patch_size = (1, 2, 2)
    model.patch_embedding_wancamctrl = nn.Linear(24, 16)
    model.c2ws_hidden_states_layer1 = nn.Linear(16, 16)
    model.c2ws_hidden_states_layer2 = nn.Linear(16, 16)
    model.blocks = nn.ModuleList([nn.Module() for _ in range(2)])
    for block in model.blocks:
        for name in ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer"):
            setattr(block, name, nn.Linear(16, 16))
    return model.eval()


def test_chunk_camera_cache_matches_original_and_preserves_inputs():
    model = camera_model()
    original = {"c2ws_plucker_emb": [torch.randn(1, 6, 2, 4, 4)]}
    hidden = torch.randn(1, 8, 16)
    keys = tuple(model.state_dict())
    counts = [0, 0]

    def count(index):
        def hook(*_):
            counts[index] += 1

        return hook

    handles = [block.cam_scale_layer.register_forward_hook(count(index)) for index, block in enumerate(model.blocks)]
    with torch.no_grad():
        prepared = prepare_wan_camera_condition(model, original)
        assert counts == [1, 1]
        for index, block in enumerate(model.blocks):
            bound = camera_condition_for_block(prepared, index)
            cached = apply_camera_modulation(block, hidden, bound)
            for _ in range(5):
                torch.testing.assert_close(apply_camera_modulation(block, hidden, bound), cached, rtol=0, atol=0)
            assert counts[index] == 1
            raw = apply_camera_modulation(block, hidden, {"c2ws_plucker_emb": prepared["c2ws_plucker_emb"]})
            torch.testing.assert_close(cached, raw, rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    assert tuple(original) == ("c2ws_plucker_emb",)
    assert original["c2ws_plucker_emb"][0].shape == (1, 6, 2, 4, 4)
    assert tuple(model.state_dict()) == keys
    assert "_camera_modulation" not in prepared


def test_two_requests_and_changed_camera_do_not_share_prepared_state():
    model = camera_model()
    hidden = torch.randn(1, 8, 16)
    with torch.no_grad():
        a = prepare_wan_camera_condition(model, {"c2ws_plucker_emb": [torch.zeros(1, 6, 2, 4, 4)]})
        expected = apply_camera_modulation(model.blocks[0], hidden, camera_condition_for_block(a, 0)).clone()
        b = prepare_wan_camera_condition(model, {"c2ws_plucker_emb": [torch.ones(1, 6, 2, 4, 4)]})
        other = apply_camera_modulation(model.blocks[0], hidden, camera_condition_for_block(b, 0))
        assert not torch.equal(expected, other)
        torch.testing.assert_close(
            apply_camera_modulation(model.blocks[0], hidden, camera_condition_for_block(a, 0)), expected, rtol=0, atol=0
        )
        assert a["_camera_modulations"][0][0].data_ptr() != b["_camera_modulations"][0][0].data_ptr()


def test_camera_preparation_is_inference_only_and_uncached_training_remains_valid():
    model = camera_model()
    condition = {"c2ws_plucker_emb": [torch.zeros(1, 6, 2, 4, 4)]}
    with pytest.raises(RuntimeError, match="no-grad"):
        prepare_wan_camera_condition(model, condition)
    with torch.no_grad():
        prepared = prepare_wan_camera_condition(model, condition)
    with pytest.raises(RuntimeError, match="inference-only"):
        apply_camera_modulation(model.blocks[0], torch.randn(1, 8, 16), camera_condition_for_block(prepared, 0))
    hidden = torch.randn(1, 8, 16, requires_grad=True)
    apply_camera_modulation(
        model.blocks[0], hidden, {"c2ws_plucker_emb": prepared["c2ws_plucker_emb"]}
    ).sum().backward()
    assert hidden.grad is not None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_action22_full_forward_prepared_camera_matches_raw_with_padding():
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_22 import WanModel

    torch.manual_seed(7)
    model = (
        WanModel(
            model_type="ti2v",
            in_dim=4,
            out_dim=4,
            dim=64,
            ffn_dim=128,
            freq_dim=16,
            text_dim=16,
            text_len=8,
            num_heads=4,
            num_layers=2,
            action_config={"blocks": []},
            use_memory=True,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.05)
    x = torch.randn(1, 4, 2, 4, 4, device="cuda", dtype=torch.bfloat16)
    camera = torch.randn(1, 1536, 2, 4, 4, device="cuda", dtype=torch.bfloat16)
    context = [torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)]
    timestep = torch.tensor([500.0], device="cuda")
    keys = tuple(model.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = model(x, timestep, context, 12, plucker_emb=camera, predict_latent_idx=(0, 2))
        prepared = model.prepare_camera_condition(camera, token_length=12)
        cached = model(x, timestep, context, 12, plucker_emb=prepared, predict_latent_idx=(0, 2))
    torch.testing.assert_close(cached[0], original[0], rtol=0, atol=0)
    assert tuple(model.state_dict()) == keys
