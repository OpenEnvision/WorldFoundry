import logging
import os
import re
from datetime import datetime

import fire
import numpy as np
import torch
from datasets import load_dataset
from mantis.models.idefics2 import Idefics2ForSequenceClassification
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from utils_conv import conv_templates
from utils_tools import _add_to_res_file, regression_query_template

CONV_TEMPLATE = conv_templates["idefics_2"]
NUM_ASPECT = 5
ROUND_DIGIT = 3
MAX_NUM_FRAMES = 16
BENCH_NAMES = ["video_feedback", "eval_crafter", "vbench", "genaibench"]
REGRESSION_QUERY_TEMPLATE = regression_query_template()


def _read_video_frames(
    frame_paths: list[str],
    max_frames: int,
):
    total_frames = len(frame_paths)
    indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)
    selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
    return np.stack(selected_frames)


def _model_output(
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    video_prompt: str,
    frames_path_list: list[str],
):
    video_frames = _read_video_frames(frames_path_list, MAX_NUM_FRAMES)
    frames = [Image.fromarray(x) for x in video_frames]
    eval_prompt = REGRESSION_QUERY_TEMPLATE.format(text_prompt=video_prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)

    if not eval_prompt:
        print("Please provide a prompt")
        return
    if not [frames]:
        frames = None

    flatten_images = []
    for x in [frames]:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)
    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    num_aspects = logits.shape[-1]

    aspect_scores = []
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(), ROUND_DIGIT))
    return aspect_scores


def main(
    model_repo_name: str = "TIGER-Lab/VideoScore",
    data_repo_name: str = "TIGER-Lab/VideoScore-Bench",
    frames_dir: str = "../data/video_feedback/test",
    name_postfixs: list[str] = ["video_feedback"],  # noqa: B006 - Read-only default; retain the public signature.
    result_file: str = "./eval_results/video_feedback/eval_video_feedback_videoscore.json",
    bench_name: str = "video_feedback",
):
    """
    evalualte VideoScore model on VideoScore-Bench which contains four benchmarks, save results to 'result_file'
    and calculate spearman correlation coefficient between human-annotated references and model output.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    date_time = datetime.now().strftime("%m-%d %H:%M:%S")
    log_file = f"./logs/{bench_name}/eval_videoscore_on_{bench_name}_{date_time}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    processor = AutoProcessor.from_pretrained(model_repo_name, torch_dtype=torch.bfloat16)
    model = Idefics2ForSequenceClassification.from_pretrained(model_repo_name, torch_dtype=torch.bfloat16).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("processor and model loaded")

    for source in name_postfixs:
        test_data = load_dataset(data_repo_name, name=source, split="test")

        curr_frames_dir = f"{frames_dir}/frames_{source}"

        for idx, item in tqdm(enumerate(test_data)):
            vid = item["id"]
            frame_path_list = [f"{curr_frames_dir}/{vid}/{img}" for img in item["images"]]

            human_text = item["conversations"][0]["value"]
            bot_text = item["conversations"][1]["value"]

            video_prompt = human_text.split('text prompt is "')[1].split('",\n')[0]
            assert bench_name in BENCH_NAMES, "benchmark name is not supported"
            if bench_name == "video_feedback":
                ref_scores = [int(item) for item in re.findall(r": (\d+)", bot_text)]
            else:
                ref_scores = item["score_list"]

            ans_scores = _model_output(model, processor, video_prompt, frame_path_list)

            logger.info(f"{idx} {vid} {ans_scores}")
            curr_compare_dict = {"id": vid, "text": video_prompt, "ref": f"{ref_scores}", "ans": f"{ans_scores}"}
            _add_to_res_file(result_file, curr_compare_dict)


if __name__ == "__main__":
    fire.Fire(main)
