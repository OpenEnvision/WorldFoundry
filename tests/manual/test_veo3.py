import sys
sys.path.append("..")

import requests
from PIL import Image

from worldfoundry.pipelines.veo.pipeline_veo3 import Veo3Pipeline


image_path = "./worldfoundry/data/test_cases/test_image_case1/ref_image.png"
image = Image.open(image_path).convert('RGB')
input_prompt = "An old-fashioned European village with thatched roofs on the houses."

veo3_pipeline = Veo3Pipeline.api_init()

result = veo3_pipeline(
    images=image,
    prompt=input_prompt
)

print(result)

# download video from result 由于目前仅支持三方api，暂时没有实现统一的下载路径
