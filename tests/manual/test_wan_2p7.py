import os

from worldfoundry.pipelines.wan.pipeline_wan_2p7 import Wan2p7Pipeline


def main():
    api_key = os.getenv("WAN27_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    if not api_key:
        print("Please set WAN27_API_KEY or DASHSCOPE_API_KEY before running this example.")
        return

    first_frame_url = os.getenv("WAN27_FIRST_FRAME_URL", "")
    if not first_frame_url:
        print("Please set WAN27_FIRST_FRAME_URL to a public image URL before running this example.")
        return

    endpoint = os.getenv(
        "WAN27_ENDPOINT",
        os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/api/v1"),
    )

    pipeline = Wan2p7Pipeline.api_init(
        endpoint=endpoint,
        api_key=api_key,
    )

    result = pipeline(
        prompt="The camera slowly pushes forward while the city lights begin to glow.",
        images=first_frame_url,
        wait=True,
        output_path="./output/wan27/wan27_i2av.mp4",
    )
    print(result)


if __name__ == "__main__":
    main()
