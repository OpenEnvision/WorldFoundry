import os

from PIL import Image

from worldfoundry.pipelines.minimax.pipeline_hailuo_2p3 import Hailuo2p3Pipeline


def main():
    api_key = os.getenv("MINIMAX_API_KEY", "")
    if not api_key:
        print("Please set MINIMAX_API_KEY before running this example.")
        return

    endpoint = os.getenv("MINIMAX_ENDPOINT", "https://api.minimax.io/v1")

    image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
    image = Image.open(image_path).convert("RGB")

    pipeline = Hailuo2p3Pipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="A retro-futuristic city street with light traffic and glowing signs.",
        images=image,
        wait=True,
        output_path="./output/hailuo/hailuo_2p3_i2av.mp4",
    )
    print(result)


if __name__ == "__main__":
    main()
