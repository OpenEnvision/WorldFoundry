import os

from worldfoundry.pipelines.luma.pipeline_luma_ray2 import LumaRay2Pipeline


def main():
    api_key = os.getenv("LUMA_API_KEY", "")
    if not api_key:
        print("Please set LUMA_API_KEY before running this example.")
        return

    image_url = os.getenv("LUMA_FIRST_FRAME_URL", "")
    if not image_url:
        print("Please set LUMA_FIRST_FRAME_URL to a public image URL before running this example.")
        return

    endpoint = os.getenv("LUMA_ENDPOINT", "https://api.lumalabs.ai/dream-machine/v1")

    pipeline = LumaRay2Pipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="A handheld cinematic shot moving through a narrow market alley.",
        images=image_url,
        wait=True,
        output_path="./output/luma/luma_ray2_i2av.mp4",
    )
    print(result)


if __name__ == "__main__":
    main()
