import numpy as np
from PIL import Image

from worldfoundry.pipelines.fantasy_world.pipeline_fantasy_world import (
    FantasyWorldWan21Pipeline,
)
from worldfoundry.pipelines.fantasy_world.pipeline_fantasy_world import (
    FantasyWorldWan22Pipeline,
)


class DummyFantasyWorldSynthesis:
    def __init__(self, variant: str):
        self.variant = variant
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
            "variant": self.variant,
        }


def _make_pose(tx=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    return pose


ref_image = Image.new("RGB", (64, 64), color=(64, 128, 192))
end_image = Image.new("RGB", (64, 64), color=(192, 128, 64))
camera_poses = [_make_pose(0.0), _make_pose(0.2)]
K = np.array(
    [
        [500.0, 0.0, 31.5],
        [0.0, 500.0, 31.5],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


wan21_synthesis = DummyFantasyWorldSynthesis("wan21")
wan21_pipe = FantasyWorldWan21Pipeline(synthesis_model=wan21_synthesis, device="cpu")
wan21_result = wan21_pipe(
    images=ref_image,
    camera_poses=camera_poses,
    K=K,
    prompt="A loft interior",
    return_dict=True,
)
assert wan21_result["variant"] == "wan21"
assert wan21_result["frames"].shape == (2, 8, 8, 3)
assert wan21_synthesis.calls[-1]["camera_source"] == camera_poses
assert wan21_synthesis.calls[-1]["K"] is K


wan22_synthesis = DummyFantasyWorldSynthesis("wan22")
wan22_pipe = FantasyWorldWan22Pipeline(synthesis_model=wan22_synthesis, device="cpu")
wan22_result = wan22_pipe(
    images=ref_image,
    end_image=end_image,
    camera_data={"cameras_interp": camera_poses},
    K=K,
    prompt="A camera flythrough",
    return_dict=True,
)
assert wan22_result["variant"] == "wan22"
assert wan22_synthesis.calls[-1]["end_image"].size == end_image.size
assert "cameras_interp" in wan22_synthesis.calls[-1]["camera_source"]


print("fantasy world pipeline smoke passed")
