import os

from PIL import Image

from worldfoundry.pipelines.kling.pipeline_kling_api import KlingApiPipeline


def main():
    api_key = os.getenv("KLING_API_KEY", "")
    if not api_key:
        print("Please set KLING_API_KEY before running this example.")
        return

    endpoint = os.getenv("KLING_ENDPOINT", "https://api.klingapi.com")
    image_url = os.getenv("KLING_IMAGE_URL", "")
    if image_url:
        image_input = image_url
    else:
        image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
        image_input = Image.open(image_path).convert("RGB")

    pipeline = KlingApiPipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="A cinematic aerial shot moving above a coastal city at sunset.",
        images=image_input,
        task_type="i2av",
        wait=True,
        output_path="./output/kling_api/kling_i2av.mp4",
    )
    print(result)


if __name__ == "__main__":
    main()
