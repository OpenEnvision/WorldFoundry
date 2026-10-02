import os

from PIL import Image

from worldfoundry.pipelines.runway.pipeline_runway_gen4p5 import RunwayGen4p5Pipeline


def main():
    api_key = os.getenv("RUNWAY_API_KEY", "")
    if not api_key:
        print("Please set RUNWAY_API_KEY before running this example.")
        return

    endpoint = os.getenv("RUNWAY_ENDPOINT", "https://api.dev.runwayml.com/v1")

    image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
    image = Image.open(image_path).convert("RGB")

    pipeline = RunwayGen4p5Pipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="A calm drone shot moving over a mountain village at sunrise.",
        images=image,
        wait=True,
        output_path="./output/runway/runway_gen45_i2av.mp4",
    )
    print(result)


if __name__ == "__main__":
    main()
