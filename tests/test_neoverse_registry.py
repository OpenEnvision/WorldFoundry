import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.models.layers import (
    attention as attention_module,
)
from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.models.layers.attention import (
    Attention,
)
from worldfoundry.operators.neoverse_operator import NeoVerseOperator
from worldfoundry.pipelines.neoverse.pipeline_neoverse import NeoVersePipeline
from worldfoundry.studio.inference import catalog as studio_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hyworldmirror_attention_accepts_flash_attention_v3_tuple(monkeypatch):
    monkeypatch.setattr(attention_module, "_USE_FLASH_ATTN_V3", True)
    monkeypatch.setattr(
        attention_module,
        "flash_attn_func_v3",
        lambda query, key, value: (query, torch.zeros(1)),
    )

    attention = Attention(dim=8, num_heads=2)
    query = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    output = attention._apply_attention(query, query, query)

    assert isinstance(output, torch.Tensor)
    assert output.shape == query.shape
    torch.testing.assert_close(output, query)


def test_neoverse_workspace_and_inference_contract_honor_hfd_root(monkeypatch, tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    model_root = checkpoint_root / "Yuppie1204--NeoVerse"
    model_root.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(checkpoint_root))

    entry = studio_catalog.find_entry("neoverse")
    assert entry.default_model_ref == str(model_root)
    default_video = Path(entry.default_input_path)
    assert default_video.is_file()
    assert default_video.relative_to(REPO_ROOT).as_posix() == (
        "worldfoundry/data/test_cases/neoverse/videos/robot.mp4"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    payload = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from worldfoundry.runtime.inference_catalog import get_model_inference_spec; "
                "spec = get_model_inference_spec('neoverse'); "
                "variant = spec.variant(); "
                "print(json.dumps({'checkpoint': variant.checkpoints[0].uri, "
                "'model_path': variant.load_kwargs['model_path']}))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    resolved = json.loads(payload)
    assert resolved == {
        "checkpoint": str(model_root),
        "model_path": str(model_root),
    }


def test_neoverse_operator_video_and_image_perception():
    assert NeoVersePipeline.__name__ == "NeoVersePipeline"

    operator = NeoVerseOperator(frames_per_action=20)
    operator.get_interaction(["forward", "camera_left", "right"])
    processed = operator.process_interaction()
    operator.delete_last_interaction()

    assert processed["num_frames"] == 61
    assert processed["keyframes"][0] == {0: [{"static": {}}]}
    assert processed["actions"] == ["forward", "camera_l", "right"]

    image = Image.new("RGB", (640, 384), color="white")
    perception = operator.process_perception(image)
    assert perception["input_frames"][0].size == (560, 336)
    assert perception["static_scene"] is True

    demo_video = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/data/test_cases/neoverse/videos/robot.mp4"
    )
    video_perception = operator.process_perception(
        str(demo_video),
        height=64,
        width=96,
        num_frames=3,
    )
    assert len(video_perception["input_frames"]) == 3
    assert video_perception["input_frames"][0].size == (96, 64)
    assert video_perception["static_scene"] is False


class _FakeNeoVerseOperator:
    zoom_ratio = 1.0
    trajectory_mode = "relative"

    def __init__(self):
        self.perception_call = None

    def process_perception(self, images, **kwargs):
        self.perception_call = {"images": images, "kwargs": kwargs}
        return {
            "input_frames": [Image.new("RGB", (96, 64), color="white")],
            "static_scene": kwargs["static_scene"],
        }

    def get_interaction(self, interaction):
        self.interaction = interaction

    def process_interaction(self):
        return {
            "actions": [],
            "predefined_trajectory": "tilt_up",
            "trajectory_file": None,
            "trajectory_data": None,
            "keyframes": None,
            "num_frames": 81,
            "trajectory_mode": "relative",
            "trajectory_name": "tilt_up",
            "zoom_ratio": 1.0,
            "angle": 15,
            "distance": 0,
            "orbit_radius": 0,
            "use_first_frame": True,
        }

    def delete_last_interaction(self):
        pass


class _FakeNeoVerseSynthesis:
    height = 64
    width = 96

    def __init__(self):
        self.predict_call = None

    def predict(self, **kwargs):
        self.predict_call = kwargs
        return {"video": ["frame"]}


def test_neoverse_pipeline_accepts_workspace_video_alias_without_forcing_static_scene():
    operator = _FakeNeoVerseOperator()
    synthesis = _FakeNeoVerseSynthesis()
    pipeline = NeoVersePipeline(operator=operator, synthesis_model=synthesis)

    result = pipeline(
        video_path="/tmp/robot.mp4",
        predefined_trajectory="tilt_up",
        num_frames=81,
        use_first_frame=True,
        static_scene=False,
        return_dict=True,
    )

    assert result["video"] == ["frame"]
    assert operator.perception_call["images"] == "/tmp/robot.mp4"
    assert operator.perception_call["kwargs"]["num_frames"] == 81
    assert operator.perception_call["kwargs"]["static_scene"] is False
    assert synthesis.predict_call["static_scene"] is False
    assert synthesis.predict_call["use_first_frame"] is True
