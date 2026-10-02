import numpy as np
from PIL import Image

from worldfoundry.pipelines.fantasy_world.pipeline_fantasy_world_wan21 import (
    FantasyWorldWan21Pipeline,
)


class DummyFantasyWorldSynthesis:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        return {
            "frames": frames,
            "video": frames,
            "prediction": {"dummy": True},
            "generated_video_path": None,
            "pointcloud_path": None,
            "scene_name": kwargs.get("scene_name"),
            "variant": "wan21",
        }


def _make_pose(tx=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    return pose


pipe = FantasyWorldWan21Pipeline(
    synthesis_model=DummyFantasyWorldSynthesis(),
    device="cpu",
)

ref_image = Image.new("RGB", (64, 64), color=(32, 64, 96))
poses = [_make_pose(0.0), _make_pose(0.1)]
poses_second = [_make_pose(0.2), _make_pose(0.3)]

first = pipe.stream(
    images=ref_image,
    camera_poses=poses,
    prompt="first",
    reset_memory=True,
)
second = pipe.stream(prompt="second")
third = pipe.stream(camera_poses=poses_second, prompt="third")

assert first.shape == (2, 8, 8, 3)
assert second.shape == (2, 8, 8, 3)
assert third.shape == (2, 8, 8, 3)
assert len(pipe.memory_module.all_frames) == 6
assert len(pipe.synthesis_model.calls) == 3
assert pipe.synthesis_model.calls[-1]["camera_source"] == poses_second

print("fantasy world stream smoke passed")
