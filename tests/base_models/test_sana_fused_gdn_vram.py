from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks as sana_blocks
import worldfoundry.base_models.diffusion_model.models.networks.sana.sana_multi_scale_video_camctrl as sana_camctrl
import worldfoundry.base_models.diffusion_model.models.networks.sana.sana_v2v_attn_blocks as sana_attention
from worldfoundry.base_models.diffusion_model.models.networks.sana.ops import (
    _execution_weight_bias,
    _prepare_fused_gdn_inputs,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_gdn_blocks import GDN
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_gdn_blocks_triton import (
    _project_materialized_camera_qkv,
    _triton_gdn_is_supported,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_gdn_camctrl_blocks import (
    _camera_execution_weight_bias,
    _prepare_cam_qkv_softmax,
)
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_v2v_attn_blocks import (
    V2VAfterRoPEGatedSoftmaxAttention,
)


class _FakeManagedModule(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Module,
        computation: torch.nn.Module,
        *,
        tuple_result: bool = False,
    ) -> None:
        super().__init__()
        self.module = source
        self._computation = computation
        self._tuple_result = tuple_result

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias

    def computation(self):
        if self._tuple_result:
            return self._computation.weight, self._computation.bias
        return self._computation


class _FakeManagedConv(_FakeManagedModule):
    kernel_size = (3,)
    stride = (1,)
    padding = (1,)
    dilation = (1,)
    groups = 1


def test_camctrl_caption_packing_matches_cross_attention_backend() -> None:
    assert sana_camctrl._xformers_available is sana_blocks._xformers_available


def test_triton_31_uses_torch_gdn_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_FORCE_SANA_TRITON_GDN", raising=False)

    assert not _triton_gdn_is_supported("3.1.0")
    assert _triton_gdn_is_supported("3.2.0")


def test_execution_weight_bias_accepts_biasless_norm() -> None:
    norm = torch.nn.RMSNorm(4)

    weight, bias = _execution_weight_bias(norm)

    assert weight is norm.weight
    assert bias is None


def test_camera_execution_weight_bias_uses_materialized_vram_wrapper() -> None:
    source = torch.nn.Linear(2, 2, device="meta")
    computation = torch.nn.Linear(2, 2)
    managed = _FakeManagedModule(source, computation)

    weight, bias = _camera_execution_weight_bias(managed)

    assert weight is computation.weight
    assert bias is computation.bias


def test_triton_camera_qkv_uses_materialized_vram_wrapper_weights() -> None:
    q = torch.nn.Linear(2, 2)
    k = torch.nn.Linear(2, 2, bias=False)
    v = torch.nn.Linear(2, 2)
    managed = (
        _FakeManagedModule(torch.nn.Linear(2, 2, device="meta"), q),
        _FakeManagedModule(
            torch.nn.Linear(2, 2, bias=False, device="meta"),
            k,
        ),
        _FakeManagedModule(torch.nn.Linear(2, 2, device="meta"), v),
    )
    inputs = torch.randn(1, 3, 2)

    projected = _project_materialized_camera_qkv(inputs, *managed)

    torch.testing.assert_close(projected, torch.cat((q(inputs), k(inputs), v(inputs)), dim=-1))


def test_softmax_camera_qkv_uses_materialized_vram_wrapper_weights() -> None:
    channels = 2
    owner = SimpleNamespace(
        q_proj_cam=_FakeManagedModule(
            torch.nn.Linear(channels, channels, device="meta"),
            torch.nn.Linear(channels, channels),
        ),
        k_proj_cam=_FakeManagedModule(
            torch.nn.Linear(channels, channels, device="meta"),
            torch.nn.Linear(channels, channels),
        ),
        v_proj_cam=_FakeManagedModule(
            torch.nn.Linear(channels, channels, device="meta"),
            torch.nn.Linear(channels, channels),
        ),
        conv_q_cam=None,
        conv_k_cam=None,
        conv_v_cam=None,
        q_norm_cam=torch.nn.Identity(),
        k_norm_cam=torch.nn.Identity(),
        cam_heads=1,
        cam_head_dim=channels,
        _stabilize_cam_transforms=lambda **values: (
            values["q_cam_trans"],
            values["k_cam_trans"],
            values["v_cam_trans"],
        ),
    )
    def identity(value):
        return value

    q, k, v, _ = _prepare_cam_qkv_softmax(
        owner,
        torch.ones(1, 2, channels),
        (2, 1, 1),
        torch.empty(0),
        None,
        prope_fns=(identity, identity, identity),
    )

    assert q.shape == k.shape == v.shape == (1, 1, channels, 2)


def test_fused_gdn_uses_materialized_vram_wrapper_weights() -> None:
    channels = 2
    q = _FakeManagedModule(
        torch.nn.Conv1d(channels, channels, 1, device="meta"),
        torch.nn.Conv1d(channels, channels, 1, bias=False),
    )
    k = _FakeManagedModule(
        torch.nn.Conv1d(channels, channels, 1, device="meta"),
        torch.nn.Conv1d(channels, channels, 1, bias=False),
    )
    v = _FakeManagedModule(
        torch.nn.Linear(channels, channels, device="meta"),
        torch.nn.Linear(channels, channels, bias=False),
        tuple_result=True,
    )
    owner = SimpleNamespace(
        heads=1,
        dim=channels,
        q=q,
        k=k,
        v=v,
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        _compute_frame_gates=lambda x, hw: (
            torch.ones(x.shape[0], hw[0], hw[1] * hw[2], 1),
            torch.ones(x.shape[0], 1, hw[0]),
        ),
    )

    prepared = _prepare_fused_gdn_inputs(owner, torch.ones(1, 2, channels), (1, 1, 2))

    assert prepared.qkv.device.type == "cpu"
    assert prepared.qkv.shape == (1, 2, 3, 1, channels)


def test_frame_gates_use_materialized_vram_wrapper_weights() -> None:
    channels = 2
    owner = SimpleNamespace(
        heads=1,
        beta_proj=_FakeManagedModule(
            torch.nn.Linear(channels, 1, device="meta"),
            torch.nn.Linear(channels, 1),
        ),
        gate_proj=_FakeManagedModule(
            torch.nn.Linear(channels, 1, device="meta"),
            torch.nn.Linear(channels, 1),
        ),
        dt_bias=torch.zeros(1),
        A_log=torch.zeros(1),
    )

    beta, decay = GDN._compute_frame_gates(owner, torch.ones(1, 2, channels), (2, 1, 1))

    assert beta.shape == (1, 1, 2, 1)
    assert decay.shape == (1, 1, 2)


def test_after_rope_attention_uses_materialized_temporal_conv_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sana_attention, "_flash_attn_available", False)
    monkeypatch.setattr(sana_attention, "_xformers_available", False)
    attention = V2VAfterRoPEGatedSoftmaxAttention(
        in_dim=2,
        out_dim=2,
        heads=1,
        dim=2,
        use_output_gate=False,
    )
    attention.q = _FakeManagedConv(
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False, device="meta"),
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False),
    )
    attention.k = _FakeManagedConv(
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False, device="meta"),
        torch.nn.Conv1d(2, 2, 3, padding=1, bias=False),
    )

    output = attention(torch.ones(1, 2, 2), HW=(2, 1, 1))

    assert output.shape == (1, 2, 2)
