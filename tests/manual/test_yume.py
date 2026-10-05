import os
import torch
import numpy as np
from diffusers.utils import export_to_video
from PIL import Image
from decord import VideoReader
from worldfoundry.pipelines.yume.pipeline_yume import YumePipeline


pretrained_model_path = "stdstu123/Yume-I2V-540P"
prompt = "A fire-breathing dragon appeared." # needed for t2v
image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"  # needed for i2v, set None for t2v or v2v
video_path = None # needed for v2v, set None for t2v or i2v
interactions = ["forward", "camera_l"]  # list, e.g., ["forward", "camera_l", "forward", "camera_r"]
interaction_speeds=[100, 4] # camera movement speed: xxx meters per second; camera rotation speed: xxx
interaction_distances=[4, None] # camera movement distance: xxx; camera rotation distance: None
seed = 43
size = '544*960' # e.g., '544*960', '960*544'
sampling_method = "ode"  # "ode" (default) or "sde"


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

assert interactions, "Interactions must be provided."
assert len(interactions) == len(interaction_speeds) == len(interaction_distances), "interactions, interaction_speeds, and interaction_distances must have the same length"

pipeline = YumePipeline.from_pretrained(
    model_path=pretrained_model_path,
    device="cuda",
    weight_dtype=torch.bfloat16,
    fsdp=True
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
    sampling_method=sampling_method,
)

if torch.distributed.get_rank() == 0:
    export_to_video(output_video, "./yume_demo.mp4", fps=16)
    print("Video saved successfully.")
