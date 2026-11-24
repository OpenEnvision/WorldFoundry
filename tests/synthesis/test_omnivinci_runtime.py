# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the public OmniVinci local-checkpoint route."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from transformers import GenerationConfig

from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.modeling_vila import VILAForCausalLM
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime
from worldfoundry.synthesis.visual_generation.omnivinci.omnivinci_runtime.auto_processor import VILAProcessor
from worldfoundry.synthesis.visual_generation.omnivinci.omnivinci_runtime.media import _load_video

pytestmark = pytest.mark.unit


def test_omnivinci_route_loads_in_tree_classes_and_keeps_request_and_artifact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "input.png"
    video = tmp_path / "input.mp4"
    image.touch()
    video.touch()
    input_ids = torch.tensor([[3, 4]])
    inputs = SimpleNamespace(input_ids=input_ids, media={"video": [torch.zeros(9, 3, 8, 8)]}, media_config={})
    batch_decode = Mock(return_value=["local answer"])
    processor = Mock(config=SimpleNamespace(), tokenizer=SimpleNamespace(batch_decode=batch_decode))
    processor.return_value = inputs
    processor.apply_chat_template.side_effect = lambda conversation, **_: conversation
    embeddings = torch.ones((1, 5, 4), dtype=torch.float32)
    mask = torch.ones((1, 5), dtype=torch.bool)
    decoder = torch.nn.Embedding(16, 4, dtype=torch.float16)

    def generate(**kwargs: Any) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        assert kwargs["inputs_embeds"].dtype == decoder.weight.dtype
        assert kwargs["inputs_embeds"].device == decoder.weight.device
        assert kwargs["attention_mask"] is mask
        assert kwargs["generation_config"].max_new_tokens == 23
        return torch.tensor([[3, 4, 7, 8]])

    model = Mock(config=SimpleNamespace(), llm_model_embed_tokens=decoder, default_generation_config=GenerationConfig())
    model._embed.return_value = embeddings, None, mask
    model.llm.generate.side_effect = generate
    processor_loader = Mock(return_value=processor)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(VILAProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(VILAForCausalLM, "from_pretrained", model_loader)
    runtime = OfficialVideoRuntime.from_model_id("omnivinci")
    result = runtime._run_omnivinci(
        prompt="describe clip",
        output_path=tmp_path / "result.mp4",
        image_path=image,
        video_path=video,
        checkpoint_path=tmp_path / "checkpoint",
        extra={"max_video_frames": 9, "load_audio_in_video": True, "max_new_tokens": 23},
    )
    processor_loader.assert_called_once_with(str(tmp_path / "checkpoint"), local_files_only=True)
    model_loader.assert_called_once_with(
        str(tmp_path / "checkpoint"), torch_dtype=torch.float16, device_map="auto", local_files_only=True
    )
    assert model.config.num_video_frames == processor.config.num_video_frames == 9
    assert model.config.load_audio_in_video is processor.config.load_audio_in_video is True
    assert batch_decode.call_args.args[0].tolist() == [[7, 8]]
    assert batch_decode.call_args.kwargs == {"skip_special_tokens": True}
    artifact = json.loads(Path(result["artifact_path"]).read_text())
    assert artifact == {
        "prompt": "describe clip",
        "text": "local answer",
        "image_path": str(image),
        "video_path": str(video),
    }
    model.eval.assert_called_once()


@pytest.mark.parametrize(
    ("extra", "error", "message"),
    [
        ({"max_video_frames": 0}, ValueError, "between 1 and 128"),
        ({"max_video_frames": 129}, ValueError, "between 1 and 128"),
        ({"load_audio_in_video": "true"}, TypeError, "must be a bool"),
    ],
)
def test_omnivinci_route_validates_options_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict, error: type[Exception], message: str
) -> None:
    loader = Mock(side_effect=AssertionError("loading must not run"))
    monkeypatch.setattr(VILAProcessor, "from_pretrained", loader)
    with pytest.raises(error, match=message):
        OfficialVideoRuntime.from_model_id("omnivinci")._run_omnivinci(
            prompt="describe clip",
            output_path=tmp_path / "result.json",
            image_path=None,
            video_path=None,
            checkpoint_path=tmp_path / "checkpoint",
            extra=extra,
        )
    loader.assert_not_called()


def test_video_audio_interleave_keeps_released_segment_indices_and_releases_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from worldfoundry.base_models.llm_mllm_core.mllm.omnivinci.configuration_vila import VILAConfig
    from worldfoundry.synthesis.visual_generation.omnivinci.omnivinci_runtime import media

    class Capture:
        def __init__(self) -> None:
            self.index = 0
            self.released = False

        def get(self, key: int) -> float:
            return 62.0 if key == media.cv2.CAP_PROP_FRAME_COUNT else 2.0

        def set(self, key: int, value: int) -> None:
            self.index = value

        def grab(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.full((8, 8, 3), self.index, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setattr(media.cv2, "VideoCapture", lambda _: capture)
    audio_info = {"audio_start_sec": 0.0, "audio_end_sample_sec": 31.0}
    monkeypatch.setattr(media, "_load_speech", lambda *_: (np.zeros(16000 * 60, dtype=np.float32), audio_info))
    config = VILAConfig(num_video_frames=8, load_audio_in_video=True)
    frames, audio, info = _load_video("clip.mp4", config)
    assert capture.released
    assert len(frames) == 8
    assert audio is not None
    assert info["video_frame_times"] == [0.0, 4.5, 8.5, 13.0, 17.5, 22.0, 26.0, 30.5]
    assert info["segment_vis_indices_list"] == [[0, 1, 2, 3, 4, 5, 6], [7]]
    assert info["segment_aud_indices_list"] == [[0, 7980], [7980, 8246]]
    assert info["has_audio"] is True


def test_omnivinci_control_plane_imports_stay_lazy() -> None:
    script = """
import sys
import worldfoundry.base_models.llm_mllm_core.mllm.omnivinci
import worldfoundry.synthesis.visual_generation.omnivinci.omnivinci_runtime
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime
runtime = OfficialVideoRuntime.from_model_id('omnivinci')
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'trust_remote_code' not in runtime.runtime
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
