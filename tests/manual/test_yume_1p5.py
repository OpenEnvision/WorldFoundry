import os

if __name__ != "__main__" and os.getenv("WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    import pytest

    pytest.skip("YUME-1.5 demo inference is opt-in; set WORLDFOUNDRY_RUN_HEAVY_MODEL_TESTS=1.", allow_module_level=True)

import torch
import numpy as np
from diffusers.utils import export_to_video
from PIL import Image
from decord import VideoReader
from worldfoundry.pipelines.yume.pipeline_yume_1p5 import Yume1p5Pipeline


pretrained_model_path = os.getenv("YUME15_MODEL_PATH", "stdstu123/Yume-5B-720P")
prompt = os.getenv("YUME15_PROMPT", "A fire-breathing dragon appeared.")
image_path = os.getenv(
    "YUME15_IMAGE_PATH",
    "./worldfoundry/data/test_cases/studio_demo/00/image.jpg",
)
video_path = os.getenv("YUME15_VIDEO_PATH") or None
if os.getenv("YUME15_TASK_TYPE") == "t2v":
    image_path = None
    video_path = None
elif os.getenv("YUME15_TASK_TYPE") == "v2v":
    image_path = None
    if video_path is None:
        raise ValueError("YUME15_TASK_TYPE=v2v requires YUME15_VIDEO_PATH.")
interactions = [item.strip() for item in os.getenv("YUME15_INTERACTIONS", "forward,camera_l").split(",") if item.strip()]
interaction_speeds = [float(item.strip()) for item in os.getenv("YUME15_INTERACTION_SPEEDS", "100,4").split(",")]
interaction_distances = [
    None if item.strip().lower() in {"none", "null", ""}
    else float(item.strip())
    for item in os.getenv("YUME15_INTERACTION_DISTANCES", "4,None").split(",")
]
seed = int(os.getenv("YUME15_SEED", "43"))
size = os.getenv("YUME15_SIZE", "704*1280")


# Determine task type and prepare inputs
if image_path is not None and video_path is None:
    task_type = "i2v"

    assert not os.path.isdir(image_path), "`image_path` must point to a single image file, not a directory."
    assert os.path.exists(image_path), f"Image file not found: {image_path}"

    images = Image.open(image_path)
    if images.mode == 'RGBA':
        background = Image.new('RGB', images.size, (0, 0, 0))
        background.paste(images, mask=images.split()[3])
        images = background
    else:
        images = images.convert("RGB")
    videos = None

elif video_path is not None and image_path is None:
    task_type = "v2v"

    assert video_path.endswith(".mp4"), f"`video_path` must point to a .mp4 file, got: {video_path}"
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    video_reader = VideoReader(video_path)
    assert len(video_reader) > 0, f"Failed to read video or video is empty: {video_path}"

    # configure frame sampling
    total_frames_target = 33
    start_idx = 0

    # sample frames from the video
    target_times = np.arange(total_frames_target) / 30
    original_indices = np.round(target_times * 30).astype(int)
    batch_index = [idx + start_idx for idx in original_indices]
    if len(batch_index) < total_frames_target:
        batch_index = batch_index[:total_frames_target]

    videos = [Image.fromarray(video_reader[idx].asnumpy()) for idx in batch_index]
    images = None

elif image_path is None and video_path is None:
    task_type = "t2v"

    assert prompt, "Prompt must be provided for t2v."
    images = None
    videos = None

else:
    raise ValueError("Only one of `image_path` or `video_path` can be provided, not both.")

assert interactions, "Interactions must be provided when using video input."
assert len(interactions) == len(interaction_speeds) == len(interaction_distances), "interactions, interaction_speeds, and interaction_distances must have the same length"

pipeline = Yume1p5Pipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
    weight_dtype=torch.bfloat16,
    fsdp=os.getenv("YUME15_FSDP", "1").lower() in {"1", "true", "yes"}
)

output_video = pipeline(
    prompt=prompt,
    interactions=interactions,
    interaction_speeds=interaction_speeds,
    interaction_distances=interaction_distances,
    images=images, # None or one PIL image
    videos=videos, # None or list of PIL images from one video
    size=size,
    seed=seed,
    task_type=task_type,
    num_euler_timesteps=int(os.getenv("YUME15_NUM_EULER_TIMESTEPS", "4")),
)

if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    output_path = os.getenv("YUME15_OUTPUT_PATH", "./yume_1p5_demo.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export_to_video(output_video, output_path, fps=int(os.getenv("YUME15_FPS", "16")))
    print(f"Yume 1.5 video saved to: {output_path}")
