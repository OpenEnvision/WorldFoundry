import argparse
import csv
import json
import os
import random
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.conversation import conv_templates
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.model.builder import load_pretrained_model
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.utils import disable_torch_init
from worldfoundry.core.media.codecs.video import write_video_torchvision as write_video
from worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.asset_paths import asset_path


class Video_preprocess:
    def __init__(self):
        pass

    def extract_frames(self, video_path, num_frames=16):
        frames = []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames", total_frames)
        if total_frames <= num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def rgb_to_yuv(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def frames_to_video(self, frames, output_path, fps=8):
        yuv_frames = [self.rgb_to_yuv(frame) for frame in frames]
        video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
        write_video(output_path, video_tensor, fps, video_codec="h264", options={"crf": "18"})

    def convert_video(self, input_path, output_path, num_frames):
        frames = self.extract_frames(input_path, num_frames=num_frames)
        self.frames_to_video(frames, output_path)

    def merge_grid(self, image_list):
        # Concatenate the images horizontally
        row1 = np.concatenate((image_list[0], image_list[1]), axis=1)
        row2 = np.concatenate((image_list[2], image_list[3]), axis=1)
        row3 = np.concatenate((image_list[4], image_list[5]), axis=1)
        # Concatenate the rows vertically
        grid = np.concatenate((row1, row2, row3), axis=0)
        return grid

    def read_video_path(self, video_path):
        if os.path.isdir(video_path):  # if video_path is a list of videos
            video = os.listdir(video_path)
        elif os.path.isfile(video_path):  # else if video_path is a single video
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        video.sort()
        return video, video_path

    def convert_video_to_frames(self, video_path, num_frames=16):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to {num_frames} frames from path:", video_path)

        output_path = os.path.join(os.path.dirname(video_path), "frames", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            vid_id = v.split(".")[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path, num_frames=num_frames)
            for frame_count, frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f"{vid_id}_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, frame)
        print("finish converting from path: ", video_path)
        print("video frames stored in: ", output_path)
        return output_path

    def convert_video_to_standard_video(self, video_path, num_frames):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to video with 16 frames from path:", video_path)

        output_path = os.path.join(os.path.dirname(video_path), "video_standard", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            v_mp4 = v.split(".")[0] + ".mp4"
            self.convert_video(os.path.join(video_path, f"{v}"), os.path.join(output_path, f"{v_mp4}"), num_frames)
        print("finish converting from path: ", video_path)
        print("standard video stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path, num_image=6):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to image grid with 6 frames from path:", video_path)

        output_path = os.path.join(os.path.dirname(video_path), "image_grid", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)

        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path)
            frame_indices = np.linspace(
                0, len(frames) - 1, num_image, dtype=int
            )  # take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            grid_filename = os.path.join(output_path, f"{vid_id}.png")
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path


def load_image(image_file):
    if image_file.startswith(("http", "https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def extract_json(string):
    # Find the start and end positions of the JSON part
    start = string.find("{")
    end = string.rfind("}") + 1

    # Extract the JSON part from the string
    json_part = string[start:end]

    # Load the JSON part as a dictionary
    try:
        json_data = json.loads(json_part)
    except json.JSONDecodeError:
        # Handle the case when the JSON part is not valid
        print("Invalid JSON part")
        return None

    return json_data


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_model(args):
    # preprocess: video 2 grid
    image_grid_path = args.image_grid_path
    if image_grid_path is None:
        video_path = args.video_path
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(video_path)

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, f"{args.t2v_model}_object_interactions_score.csv")

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as csvreader:
            reader = csv.reader(csvreader)
            lines = list(reader)  # Read all lines into a list
            line_count = len(lines)  # Count the number of lines
    else:
        line_count = 0

    with open(csv_path, "a", newline="") as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        if line_count == 0:
            # Write the header row
            csv_writer.writerow(
                [
                    "name",
                    "prompt",
                    "seed0_answer1",
                    "seed0_answer2",
                    "seed0_answer3",
                    "seed0_score",
                    "seed1_answer1",
                    "seed1_answer2",
                    "seed1_answer3",
                    "seed1_score",
                    "seed2_answer1",
                    "seed2_answer2",
                    "seed2_answer3",
                    "seed2_score",
                    "seed_score",
                    "Score",
                ]
            )

        grid_images = [f for f in os.listdir(image_grid_path) if f[0].isdigit()]
        grid_images = sorted(grid_images)

        print(len(grid_images))

        evaluated = max(line_count - 1, 0)

        for i in range(evaluated, len(grid_images)):
            grid_image_name = grid_images[i]

            num = int(grid_image_name[0:4]) - 1

            this_prompt = prompts[num]["prompt"]

            image_files = [os.path.join(image_grid_path, grid_images[i])]
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

            Q1 = "The provided image arranges 6 key frames from an AI generated video in a 3-row by 2-column grid layout.  Describe the video, focusing on the interactions between the characters or objects that appear throughout the frames."
            qs1 = Q1

            qs1 = DEFAULT_IMAGE_TOKEN + "\n" + qs1

            conv_mode = "chatml_direct"

            args.conv_mode = conv_mode

            conv_init = conv_templates[args.conv_mode].copy()
            conv_init.append_message(conv_init.roles[0], qs1)
            conv_init.append_message(conv_init.roles[1], None)
            prompt_init = conv_init.get_prompt()

            input_ids_init = (
                tokenizer_image_token(prompt_init, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            outputs_1 = []
            outputs_2 = []
            outputs_3 = []
            scores_tmp = []

            for iteration in range(3):
                set_seed(args.seed + iteration)

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs1)
                conv.append_message(conv.roles[1], None)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids_init,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,  # 1
                        max_new_tokens=args.max_new_tokens,  # 512
                        use_cache=True,
                    )

                output_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                outputs_1.append(output_1)
                conv.messages[-1][-1] = output_1

                Q2 = f"To evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please carefully answer the following question.\n \
Question: \n \
A: All the objects involved in the interaction are clearly portrayed in the video. \n \
B: Some objects involved in the interaction are not depicted very clear. \n \
C: Some objects involved in the interaction are missing. \n \
D: None of the objects involved in the interaction are present. \n \
Select the most suitable option according to the video and your previous description. \
Provide your answer in JSON format with the following keys: option (e.g., B), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., A)."

                qs2 = Q2
                conv.append_message(conv.roles[0], qs2)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature,  # 0.2
                        top_p=args.top_p,
                        num_beams=args.num_beams,  # 1
                        max_new_tokens=args.max_new_tokens,  # 512
                        use_cache=True,
                    )

                output_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                outputs_2.append(output_2)
                conv.messages[-1][-1] = output_2

                json_obj_2 = extract_json(output_2)
                try:
                    option_value_2 = json_obj_2["adjust"]
                except BaseException:
                    option_value_2 = "bad reply 1"
                print("option_value_2 ", option_value_2)

                Q3_A = f"To evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please carefully examine the interaction and select the most suitable option.\n \
A: The interaction process is clearly shown, with the objects involved engaging in a dynamic manner. The outcome logically follow from the preceding actions and aligns accurately with what the prompt indicated, if mentioned. \n \
B: The interaction process is mostly clear, with objects engaging actively. The outcome generally aligns with what the prompt indicated, if mentioned. \n \
C: The interaction process is somewhat clear, but the objects show limited engagement. The development of process might be unclear, and while the outcome aligns with the prompt but with no previous actions. \n \
D: The interaction process is unclear, with minimal engagement from the objects. There is little to no development of the process. The outcome, if mentioned in prompt, is vague. \n \
E: The interaction process is virtually nonexistent, with no visible engagement from the objects. The outcome in the prompt, if mentioned, is absent or completely irrelevant.\n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

                Q3_B = f"Following the previous question, to evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please select the most suitable option.\n \
A: the interaction of the clearly presented object(s) is highly dynamic and compensates effectively for the unclear ones, with a logical outcome that implies the interaction. \n \
B: the interaction of the clearly presented object(s) is weak, with confusing relationship to the unclear ones and an irrelevant outcome to the prompt.  \n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

                Q3_C = f"Following the previous question, to evaluate if this prompt '{this_prompt}' is correctly portrayed in the video, please select the most suitable option.\n \
A: the interaction of the clearly presented object(s) is highly dynamic and compensates effectively for the missing ones, with a logical outcome that implies the interaction. \n \
B: the interaction of the clearly presented object(s) is weak, with confusing relationship to the missing ones and an irrelevant outcome to the prompt.  \n \
Provide your answer in a JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., B)."

                if option_value_2 == "A":
                    Q3 = Q3_A
                    ask_Q3 = True
                elif option_value_2 == "B":
                    Q3 = Q3_B
                    ask_Q3 = True
                elif option_value_2 == "C":
                    Q3 = Q3_C
                    ask_Q3 = True
                elif option_value_2 == "D":
                    score_tmp = 1
                    ask_Q3 = False
                else:
                    ask_Q3 = False
                    score_tmp = "bad reply"

                output_3 = ""
                if ask_Q3:
                    qs3 = Q3
                    conv.append_message(conv.roles[0], qs3)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_tensor,
                            image_sizes=image_sizes,
                            do_sample=args.temperature > 0,
                            temperature=args.temperature,  # 0.2
                            top_p=args.top_p,
                            num_beams=args.num_beams,  # 1
                            max_new_tokens=args.max_new_tokens,  # 512
                            use_cache=True,
                        )

                    output_3 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                    json_obj_3 = extract_json(output_3)
                    try:
                        option_value_3 = json_obj_3["adjust"]
                    except BaseException:
                        option_value_3 = "bad reply 2"
                    print("option_value_3 ", option_value_3)

                    if option_value_2 == "A":
                        if option_value_3 == "A":
                            score_tmp = 10
                        elif option_value_3 == "B":
                            score_tmp = 9
                        elif option_value_3 == "C":
                            score_tmp = 8
                        elif option_value_3 == "D":
                            score_tmp = 7
                        elif option_value_3 == "E":
                            score_tmp = 6
                        else:
                            score_tmp = "bad reply ?"
                            print("reply wrong format")

                    elif option_value_2 == "B":
                        if option_value_3 == "A":
                            score_tmp = 5
                        elif option_value_3 == "B":
                            score_tmp = 3
                        else:
                            score_tmp = "bad reply ?"
                            print("reply wrong format")

                    elif option_value_2 == "C":
                        if option_value_3 == "A":
                            score_tmp = 4
                        elif option_value_3 == "B":
                            score_tmp = 2
                        else:
                            score_tmp = "bad reply ?"
                            print("reply wrong format")

                outputs_3.append(output_3)

                scores_tmp.append(score_tmp)
                print("score for", grid_images[i], score_tmp)

            int_flag = 0
            for score in scores_tmp:
                if not isinstance(score, int):
                    int_flag = 1
            score_avg = sum(scores_tmp) / len(scores_tmp) if int_flag == 0 else "bad reply"

            csv_writer.writerow(
                [
                    grid_image_name,
                    this_prompt,
                    outputs_1[0],
                    outputs_2[0],
                    outputs_3[0],
                    scores_tmp[0],
                    outputs_1[1],
                    outputs_2[1],
                    outputs_3[1],
                    scores_tmp[1],
                    outputs_1[2],
                    outputs_2[2],
                    outputs_3[2],
                    scores_tmp[2],
                    scores_tmp,
                    score_avg,
                ]
            )

            csvfile.flush()

        return csv_path


def model_score(csv_path):
    with open(csv_path) as file:
        reader = csv.reader(file)
        lines = list(reader)
        score = 0
        cnt = 0
        for line in lines[1:]:
            try:
                score_tmp = (float(line[-1]) - 1) / 9
                score += score_tmp
                cnt += 1
            except BaseException:
                continue

        score = score / cnt
        print("number of images evaluated: ", cnt, " object interactions model score: ", score)

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["score: ", score])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--output-path", type=str, default="../csv_output_interactions", help="path to store the video scores"
    )
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default=str(asset_path("meta_data/object_interactions.json")),
        help="path of txt file with input prompts and meta data",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="path to videos",
    )
    parser.add_argument(
        "--t2v-model",
        type=str,
        required=True,
        help="model name",
    )

    parser.add_argument(
        "--image_grid_path",
        type=str,
        default=None,
        help="image grid path",
    )
    args = parser.parse_args()

    csv_path = eval_model(args)
    model_score(csv_path)
