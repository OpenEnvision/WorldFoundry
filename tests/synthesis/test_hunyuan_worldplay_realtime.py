# SPDX-FileCopyrightText: WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Public HY-WorldPlay runtime hooks and shared placement behavior."""

import builtins
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from PIL import Image

from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import (
    HunyuanWorldPlayPipeline,
)
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay import commons, realtime

pytestmark = pytest.mark.unit


def test_worldplay_normalizes_typed_cli_interaction_tokens() -> None:
    assert HunyuanWorldPlayPipeline._normalize_pose(("w-2", "right-1")) == "w-2,right-1"
    assert HunyuanWorldPlayPipeline._normalize_pose("w-2") == "w-2"


def test_canonical_cached_attention_runs_without_flash_padding_helpers(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"flash_attn", "flash_attn.bert_padding"}:
            raise ModuleNotFoundError("No module named 'flash_attn'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15 import camera_attention

    monkeypatch.setattr(camera_attention, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    source = torch.linspace(-1, 1, 12).reshape(1, 3, 1, 4)
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            source.transpose(1, 2), source.transpose(1, 2), source.transpose(1, 2)
        )
        .transpose(1, 2)
        .reshape(1, 3, 4)
    )
    result, cache = camera_attention.sequence_parallel_attention_txt(source, source, source, 0, 0, cache_txt=True)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache["k_txt"], source.transpose(1, 2), rtol=0, atol=0)


def test_manual_offload_skips_models_managed_by_group_offload(monkeypatch) -> None:
    from diffusers.hooks import group_offloading

    class TrackingModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moves = []

        def to(self, device, *args, **kwargs):
            self.moves.append(torch.device(device))
            return self

    grouped = TrackingModule()
    manual = TrackingModule()
    monkeypatch.setattr(
        group_offloading,
        "_is_group_offload_enabled",
        lambda model: model is grouped,
    )

    with commons.auto_offload_model([grouped, manual], torch.device("cuda")):
        assert grouped.moves == []
        assert manual.moves == [torch.device("cuda")]

    assert grouped.moves == []
    assert manual.moves == [torch.device("cuda"), torch.device("cpu")]


def test_realtime_camera_history_grows_by_native_latent_blocks() -> None:
    session = realtime.HunyuanWorldPlayRealtimeSession.__new__(realtime.HunyuanWorldPlayRealtimeSession)
    session.operator = SimpleNamespace(
        forward_speed=0.08,
        yaw_speed_deg=3.0,
        pitch_speed_deg=3.0,
    )
    session._motions = []
    session._latents = None

    viewmats, intrinsics, actions = session._camera_conditions(["forward"], None)
    assert viewmats.shape == (1, 4, 4, 4)
    assert intrinsics.shape == (1, 4, 3, 3)
    assert actions.shape == (1, 4)

    session._latents = torch.empty(1, 1, 4, 1, 1)
    viewmats, intrinsics, actions = session._camera_conditions(["camera_r"], None)
    assert viewmats.shape == (1, 8, 4, 4)
    assert intrinsics.shape == (1, 8, 3, 3)
    assert actions.shape == (1, 8)


def test_pipeline_reports_hy_worldplay_native_realtime_spec() -> None:
    runtime = SimpleNamespace(
        model=SimpleNamespace(execution_device="cpu", target_dtype=torch.float32),
        _realtime_lock=Lock(),
    )
    synthesis = SimpleNamespace(runtime=runtime)
    operator = SimpleNamespace()
    pipeline = HunyuanWorldPlayPipeline(
        synthesis_model=cast(Any, synthesis),
        operators=operator,
        device="cpu",
    )
    assert pipeline.prepare_realtime() == {
        "realtime_spec": {
            "fps": 24,
            "first_chunk_frames": 13,
            "steady_chunk_frames": 16,
            "controls": [
                "forward",
                "backward",
                "left",
                "right",
                "camera_up",
                "camera_down",
                "camera_l",
                "camera_r",
            ],
            "transport": "in-memory-rgb",
            "stateful": True,
        }
    }


def test_pipeline_materializes_requested_output(monkeypatch, tmp_path) -> None:
    videos = torch.zeros(1, 3, 5, 2, 2)

    def synthesis(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(videos=videos)

    pipeline = HunyuanWorldPlayPipeline(
        synthesis_model=cast(Any, synthesis),
        operators=SimpleNamespace(
            forward_speed=0.08,
            yaw_speed_deg=3.0,
            pitch_speed_deg=3.0,
            infer_video_length=lambda _: 125,
        ),
        device="cpu",
    )
    monkeypatch.setattr(
        pipeline,
        "process",
        lambda **_: {
            "reference_image": object(),
            "operator_condition": {
                "viewmats": torch.zeros(4, 4, 4),
                "Ks": torch.zeros(4, 3, 3),
                "action": torch.zeros(4),
            },
        },
    )
    writes = []
    monkeypatch.setattr(pipeline, "save_video", lambda video, path, fps: writes.append((video, path, fps)))
    output_path = tmp_path / "worldplay.mp4"

    result = pipeline(
        prompt="test",
        images=Image.new("RGB", (2, 2)),
        interactions="w-4",
        output_path=output_path,
        fps=12,
        return_dict=True,
    )

    assert result["artifact_path"] == str(output_path.resolve())
    assert result["num_output_frames"] == 5
    assert writes == [(videos, str(output_path.resolve()), 12)]
