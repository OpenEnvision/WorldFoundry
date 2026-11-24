import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import worldfoundry.base_models.perception_core.detection.grounding_dino.util.transforms as T
from worldfoundry.base_models.perception_core.detection.grounding_dino.models import build_model
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
    checkpoint_path as grounding_dino_checkpoint_path,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
    config_path as grounding_dino_config_path,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.slconfig import SLConfig
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.vl_utils import (
    create_positive_map_from_span,
)
from worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.asset_paths import asset_path
from worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.grounded_sam_metrics.video_io import (
    write_video,
)


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


def plot_boxes_to_image(image_pil, tgt, prompt_objs):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    objs = [label.split("(")[0] for label in labels]
    if len(prompt_objs) == 1:
        color1 = (255, 0, 0)
    elif len(prompt_objs) == 2:
        color1 = (255, 0, 0)
        color2 = (0, 0, 255)
    else:
        print("wrong objects")

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label, obj in zip(boxes, labels, objs, strict=False):
        if obj == prompt_objs[0]:  # obj1
            color = color1
            # print("color1")
        elif len(prompt_objs) == 2:
            if obj == prompt_objs[1]:  # obj2
                color = color2
                # print("color2")
        else:
            print("wrong objects")
            color = (0, 0, 0)

        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        xc = int(box[0])
        yc = int(box[1])
        s = 3
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        draw.ellipse((xc - s, yc - s, xc + s, yc + s), fill=color)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None
):
    assert text_threshold is not None or token_spans is not None, (
        "text_threshould and token_spans should not be None at the same time!"
    )
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        # sky
        prob = []
        for logit, _box in zip(logits_filt, boxes_filt, strict=False):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            # sky
            prob.append(logit.max().item())

    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(model.tokenizer(text_prompt), token_span=token_spans).to(  # noqa: F821 - Upstream given-phrase mode; the runner never passes --token_spans.
            image.device
        )  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for token_span, logit_phr in zip(token_spans, logits_for_phrases, strict=False):
            # get phrase
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases, prob


def calculate_iou(box0, box1):  # xywh
    # Calculate IoU
    # from xywh to xyxy
    [box0_xmin, box0_ymin] = box0[:2] - box0[2:] / 2
    [box0_xmax, box0_ymax] = box0[:2] + box0[2:] / 2
    [box1_xmin, box1_ymin] = box1[:2] - box1[2:] / 2
    [box1_xmax, box1_ymax] = box1[:2] + box1[2:] / 2

    x_overlap = max(0, min(box0_xmax, box1_xmax) - max(box0_xmin, box1_xmin))
    y_overlap = max(0, min(box0_ymax, box1_ymax) - max(box0_ymin, box1_ymin))

    intersection = x_overlap * y_overlap
    box0_area = box0[2] * box0[3]
    box1_area = box1[2] * box1[3]
    union = box0_area + box1_area - intersection

    IoU = intersection / union  # intersection over union
    IoMinA = intersection / min(box0_area, box1_area)  # intersection over the smaller box area
    IoU = IoU.item()
    IoMinA = IoMinA.item()
    return IoU, IoMinA


def filter_box(boxes, phrases, probs, iou_threshold):
    new_boxes = []
    new_probs = []
    new_phrases = []
    for i in range(len(boxes)):
        flag = 0
        for j in range(len(new_boxes)):
            IoU, _ = calculate_iou(boxes[i], new_boxes[j])
            if IoU > iou_threshold:
                flag = 1
                if probs[i] > new_probs[j]:
                    new_boxes[j] = boxes[i]
                    new_probs[j] = probs[i]
                    new_phrases[j] = phrases[i]
                break
        if flag == 0:
            new_boxes.append(boxes[i])
            new_probs.append(probs[i])
            new_phrases.append(phrases[i])

    return new_boxes, new_phrases, new_probs


def combine_frame(input_csv, output_csv):
    score_total = 0
    cnt = 0
    with open(input_csv) as file:
        reader = csv.reader(file)
        lines = list(reader)

        batch_size = 16
        num_vid = (len(lines) - 1) / batch_size
        if num_vid != int(num_vid):
            print("error: number of lines WRONG")

        score_vid_1 = []
        id = []
        score_frame_1 = []

        for i in range(int(num_vid)):
            batch = lines[i * batch_size + 1 : (i + 1) * batch_size + 1]
            # Process the batch of lines
            frame_score_1 = []

            id.append(batch[0][0])
            for line in batch:
                frame_score_1.append(float(line[-1]))

            score_tmp = sum(frame_score_1) / 16
            score_vid_1.append(score_tmp)
            score_frame_1.append(frame_score_1)
            cnt += 1
            score_total += score_tmp

    score_avg = score_total / cnt

    ####################################
    # socre_obj_1 = score_vid_1[0:20]
    # score_one_obj_2 = score_vid_1[20:50]
    # score_one_obj_3 = score_vid_1[50:80]
    # score_one_obj_4 = score_vid_1[80:110]
    # score_one_obj_5 = score_vid_1[110:140]
    # score_one_obj_678 = score_vid_1[140:160]
    # score_two_obj_low = score_vid_1[160:175]
    # score_two_obj_mid = score_vid_1[175:190]
    # score_two_obj_high = score_vid_1[190:200]

    # socre_obj_1 = sum(socre_obj_1) / len(socre_obj_1)
    # score_one_obj_2 = sum(score_one_obj_2) / len(score_one_obj_2)
    # score_one_obj_3 = sum(score_one_obj_3) / len(score_one_obj_3)
    # score_one_obj_4 = sum(score_one_obj_4) / len(score_one_obj_4)
    # score_one_obj_5 = sum(score_one_obj_5) / len(score_one_obj_5)
    # score_one_obj_678 = sum(score_one_obj_678) / len(score_one_obj_678)
    # score_two_obj_low = sum(score_two_obj_low) / len(score_two_obj_low)
    # score_two_obj_mid = sum(score_two_obj_mid) / len(score_two_obj_mid)
    # score_two_obj_high = sum(score_two_obj_high) / len(score_two_obj_high)
    ####################################

    print("number of videos evaluated: ", cnt, " numeracy model score: ", score_avg)

    score_vid_1 = ["Score_1", *score_vid_1]
    id = ["id", *id]
    score_frame_1 = ["Score_frame_1", *score_frame_1]

    if len(score_vid_1) != len(score_frame_1) != len(id):
        print("counting error")

    with open(output_csv, "w") as output_file:
        writer = csv.writer(output_file)
        for i in range(len(id)):
            # Append data to the end of each row
            row = [id[i], score_frame_1[i], score_vid_1[i]]
            # Write the modified row to the new file
            writer.writerow(row)

    with open(output_csv, "a", newline="") as file:
        writer = csv.writer(file)

        # writer.writerow(["socre_obj_1: ",socre_obj_1])
        # writer.writerow(["score_one_obj_2: ",score_one_obj_2])
        # writer.writerow(["score_one_obj_3: ",score_one_obj_3])
        # writer.writerow(["score_one_obj_4: ",score_one_obj_4])
        # writer.writerow(["score_one_obj_5: ",score_one_obj_5])
        # writer.writerow(["score_one_obj_678: ",score_one_obj_678])
        # writer.writerow(["score_two_obj_low: ",score_two_obj_low])
        # writer.writerow(["score_two_obj_mid: ",score_two_obj_mid])
        # writer.writerow(["score_two_obj_high: ",score_two_obj_high])
        writer.writerow(["score: ", score_avg])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument(
        "--config_file", "-c", type=str, default=str(grounding_dino_config_path()), help="path to config file"
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default=str(grounding_dino_checkpoint_path()),
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output_numeracy", help="directory to save the output images"
    )
    parser.add_argument("--output-path", type=str, default="csv_numeracy", help="path to store the video scores")
    parser.add_argument("--iou_threshold", type=float, default=0.9, help="threshold to filter out the duplicated boxes")
    parser.add_argument("--box_threshold", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument(
        "--token_spans",
        type=str,
        default=None,
        help="The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ",
    )
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default=str(asset_path("meta_data/generative_numeracy.json")),
        help="path to the meta data",
    )
    parser.add_argument("--video-path", type=str, required=True, help="path to the input videos")

    parser.add_argument("--frame-folder", type=str, default=None, help="path to the frame folder")
    parser.add_argument("--t2v-model", required=True, type=str)

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = os.path.join(args.output_dir, args.t2v_model)  # change the path of the output image directory
    output_path = args.output_path  # change the path of the output score csv
    video_path = args.video_path  # change the path of the input video folder
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    iou_threshold = args.iou_threshold
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    vid2frame = Video_preprocess()
    frame_folder = vid2frame.convert_video_to_frames(video_path) if args.frame_folder is None else args.frame_folder

    videos = os.listdir(frame_folder)
    videos.sort()  # sort

    csv_path = f"{output_path}/{args.t2v_model}_numeracy_frame.csv"
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
            csv_writer.writerow(["video_name", "image_name", "prompt", "objects", "numbers", "actual_numbers", "score"])

        evaluated = max(line_count - 1, 0)
        for i in range(evaluated, len(videos)):
            os.makedirs(os.path.join(output_dir, videos[i]), exist_ok=True)
            video_path = os.path.join(frame_folder, videos[i])
            images = os.listdir(video_path)
            images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # sort

            video_name = videos[i]
            num = int(video_name[0:4]) - 1

            prompt = prompts[num]["prompt"]
            objects = prompts[num]["objects"]
            numbers = prompts[num]["numbers"]
            objs = objects.split(",")
            nums = numbers.split(",")

            if len(objs) != len(nums):
                print("video ", i, " parse wrong, objects and quantities not match")
                break
            for j in range(len(objs)):
                objs[j] = objs[j].strip()
                try:
                    nums[j] = nums[j].strip()
                    nums[j] = int(nums[j])
                except BaseException:
                    print("object number not int")
            print(objs, nums)

            for image_name in images:
                # load image
                image_path = os.path.join(frame_folder, videos[i], image_name)
                image_pil, image = load_image(image_path)

                if token_spans is not None:
                    text_threshold = None
                    print("Using token_spans. Set the text_threshold to None.")

                # run model
                objs_json = {}
                image_with_box = image_pil
                all_box = []
                all_prob = []
                all_phrase = []
                for j in range(len(objs)):
                    boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                        model,
                        image,
                        objs[j],
                        box_threshold,
                        text_threshold,
                        cpu_only=args.cpu_only,
                        token_spans=eval(f"{token_spans}"),
                    )
                    size = image_pil.size

                    all_box = [*all_box, boxes_filt_0]
                    all_prob = all_prob + prob_0
                    all_phrase = all_phrase + pred_phrases_0

                all_box = torch.cat(all_box, dim=0)
                all_box, all_phrase, all_prob = filter_box(all_box, all_phrase, all_prob, iou_threshold)

                # plot boxes & probs on image
                pred_dict_0 = {
                    "boxes": all_box,
                    "size": [size[1], size[0]],  # H,W
                    "labels": all_phrase,
                }
                image_with_box = plot_boxes_to_image(image_with_box, pred_dict_0, objs)[0]

                if len(objs) == 1:
                    real_obj1_num = len(
                        [phrase.split("(")[1] for phrase in all_phrase if phrase.split("(")[0] == objs[0]]
                    )
                    real_obj_num = [real_obj1_num]
                    a = 0
                    if nums[0] == real_obj1_num:
                        a += 1

                elif len(objs) > 1:
                    real_obj1_num = len(
                        [phrase.split("(")[1] for phrase in all_phrase if phrase.split("(")[0] == objs[0]]
                    )
                    real_obj2_num = len(
                        [phrase.split("(")[1] for phrase in all_phrase if phrase.split("(")[0] == objs[1]]
                    )
                    real_obj_num = [real_obj1_num, real_obj2_num]
                    a = 0
                    if nums[0] == real_obj1_num:
                        a += 0.5
                    if nums[1] == real_obj2_num:
                        a += 0.5

                csv_writer.writerow([videos[i], image_name, prompt, objs, nums, real_obj_num, a])
                csvfile.flush()
                image_with_box.save(os.path.join(output_dir, videos[i], f"{image_name.split('.')[0]}.jpg"))

    combine_frame(
        f"{output_path}/{args.t2v_model}_numeracy_frame.csv", f"{output_path}/{args.t2v_model}_numeracy_video.csv"
    )
