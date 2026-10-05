import numpy as np
from PIL import Image

from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline


class DummyGen3CSynthesis:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        return {
            "frames": frames,
            "video": frames,
            "generated_video_path": None,
            "output_dir": kwargs.get("output_dir"),
            "scene_name": kwargs.get("scene_name"),
            "trajectory": kwargs.get("trajectory"),
            "fps": kwargs.get("fps", 24),
        }


pipe = Gen3CPipeline(
    synthesis_model=DummyGen3CSynthesis(),
    device="cpu",
)

ref_image = Image.new("RGB", (64, 64), color=(32, 64, 96))

first = pipe.stream(
    images=ref_image,
    interactions=["forward"],
    prompt="first",
    reset_memory=True,
)
second = pipe.stream(
    interactions=["camera_r"],
    prompt="second",
)

assert first.shape == (2, 8, 8, 3)
assert second.shape == (2, 8, 8, 3)
assert len(pipe.memory_module.all_frames) == 4
assert len(pipe.synthesis_model.calls) == 2
assert pipe.synthesis_model.calls[0]["trajectory"] == "zoom_in"
assert pipe.synthesis_model.calls[1]["trajectory"] == "clockwise"

print("gen3c stream smoke passed")
