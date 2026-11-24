import json
import os
from io import BytesIO

import requests
from PIL import Image


def _add_to_res_file(eval_res_file, curr_res_compare_dict):
    dirname = os.path.dirname(eval_res_file)
    os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(eval_res_file):
        with open(eval_res_file, "w") as f:
            json.dump([curr_res_compare_dict], f, indent=4)
        return
    else:
        with open(eval_res_file) as f:
            all_data = json.load(f)
        all_data.append(curr_res_compare_dict)
        with open(eval_res_file, "w") as f:
            json.dump(all_data, f, indent=4)


def _ans_formatted(ans_scores, num_aspect):
    NUM_ASPECT = num_aspect
    new_ans_scores = []
    if len(ans_scores) > NUM_ASPECT:
        new_ans_scores = ans_scores[:NUM_ASPECT]
    else:
        new_ans_scores = ans_scores + [0] * (NUM_ASPECT - len(ans_scores))

    for idx in range(len(new_ans_scores)):
        if new_ans_scores[idx] not in [1, 2, 3, 4]:
            new_ans_scores[idx] = 0
    return new_ans_scores


GENERATION_QUERY_TEMPLATE = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos, 
please watch the following frames of a given video and see the text prompt for generating the video, 
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a number from [1,2,3,4], 
in which '1' means 'Bad', '2' means 'Average', '3' means 'Good', 
'4' means 'Real' or 'Perfect' (the video is like a real video)
Here is an output example:
visual quality: 4
temporal consistency: 4
dynamic degree: 3
text-to-video alignment: 1
factual consistency: 2

For this video, the text prompt is "${source}",
all the frames of video are as follows: 
"""


def label_query_template():
    return GENERATION_QUERY_TEMPLATE


REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


def regression_query_template():
    return REGRESSION_QUERY_PROMPT


def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = image_file.convert("RGB") if isinstance(image_file, Image.Image) else load_image(image_file)
        out.append(image)
    return out


def merge_images(image_links: list = []):  # noqa: B006 - Read-only default; retain the public signature.
    """Merge multiple images into one image

    Args:
        image_links (List, optional): List of image links. Defaults to [].

    Returns:
        [type]: [description]
    """
    if len(image_links) == 0:
        return None
    images = load_images(image_links)
    if len(images) == 1:
        return images[0]
    widths, heights = zip(*(i.size for i in images), strict=False)
    average_height = sum(heights) // len(heights)
    for i, im in enumerate(images):
        # scale in proportion
        images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
    widths, heights = zip(*(i.size for i in images), strict=False)
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
    x_offset = 0
    for i, im in enumerate(images):
        if i > 0:
            # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
            new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
            x_offset += 1
            new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
            x_offset += 8
            new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
            x_offset += 1
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im
