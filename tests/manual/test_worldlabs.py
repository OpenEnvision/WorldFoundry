import os

from PIL import Image

from worldfoundry.pipelines.worldlabs.pipeline_worldlabs import WorldLabsPipeline


def main():
    api_key = os.getenv("WORLDLABS_API_KEY", os.getenv("WLT_API_KEY", ""))
    if not api_key:
        print("Please set WORLDLABS_API_KEY or WLT_API_KEY before running this example.")
        return

    endpoint = os.getenv("WORLDLABS_ENDPOINT", "https://api.worldlabs.ai")

    image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
    image = Image.open(image_path).convert("RGB")

    pipeline = WorldLabsPipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="A peaceful village square with cobblestone paths and warm evening light.",
        images=image,
        prompt_type="image",
        model="marble-1.1",
        wait=True,
        output_path="./output/worldlabs/world.json",
        download_assets=False,
    )
    print(result)


if __name__ == "__main__":
    main()
