import sys
sys.path.append("..")
from worldfoundry.pipelines.wan.pipeline_wan_2p5 import Wan2p5Pipeline
from PIL import Image
import os
import requests


def save_video(url: str, save_path: str, chunk_size: int = 1024 * 1024) -> None:
    """下载远程文件到本地"""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)


image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
image = Image.open(image_path).convert('RGB')
test_prompt = "An old-fashioned European village with thatched roofs on the houses."
output_dir = "./output/wan25"

wan25_pipeline = Wan2p5Pipeline.api_init()

result = wan25_pipeline(
    prompt=test_prompt,
    images=image,  # 提供图像则自动使用 i2av
)

# 从结果中获取 video_url 并下载
video_url = result.get('video_url')
task_type = result.get('task_type')

if video_url:
    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"wan25_{task_type}.mp4"
    video_file_path = os.path.join(output_dir, video_filename)
    save_video(video_url, video_file_path)
else:
    raise ValueError("未获取到视频URL，无法下载")
