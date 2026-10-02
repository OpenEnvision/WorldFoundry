import os

from PIL import Image

from worldfoundry.pipelines.wan.pipeline_wan_2p6 import Wan2p6Pipeline


def main():
    api_key = os.getenv("WAN26_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    if not api_key:
        print("Please set WAN26_API_KEY or DASHSCOPE_API_KEY before running this example.")
        return

    endpoint = os.getenv(
        "WAN26_ENDPOINT",
        os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/api/v1"),
    )

    image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
    image = Image.open(image_path).convert("RGB")

    pipeline = Wan2p6Pipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="A cinematic drive through a rain-soaked neon city at dusk.",
        images=image,
        task_type="i2av",
        wait=True,
        output_path="./output/wan26/wan26_i2av.mp4",
    )
    print(result)


if __name__ == "__main__":
    main()
