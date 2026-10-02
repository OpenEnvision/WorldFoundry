import os
from PIL import Image
from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline

image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")
prompt = "A cozy snowy fairy-tale village with thatched cottages covered in thick snow."
interaction_signal = ["forward", "camera_l", "camera_r"]
video_sync_path = "tencent/HunyuanVideo-1.5"
action_ckpt = "tencent/HY-WorldPlay"

output_path = "./outputs"
os.makedirs(output_path, exist_ok=True)

pipeline = HunyuanWorldPlayPipeline.from_pretrained(
    model_path=action_ckpt,
    mode="480p_i2v",
    required_components = {"video_model_path": video_sync_path},
    enable_offloading=True,
    device="cuda"
)
output = pipeline(
    prompt=prompt,
    images=input_image,
    interactions=interaction_signal,
    forward_speed=0.08,
    yaw_speed_deg=3.0,
    pitch_speed_deg=3.0,
)

save_video_path = os.path.join(output_path, "hunyuan_worldplay_demo.mp4")
HunyuanWorldPlayPipeline.save_video(output.videos, save_video_path)
