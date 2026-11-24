import argparse
import csv
import json
import os

import cv2
import matplotlib.pyplot as plt
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
from worldfoundry.base_models.perception_core.segment.sam_v1 import (
    SamPredictor,
    sam_model_registry,
)
from worldfoundry.base_models.perception_core.segment.sam_v1.paths import checkpoint_path as sam_v1_checkpoint_path
from worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.asset_paths import asset_path
from worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.grounded_sam_metrics.compbench_run_depth import (
    run_depth,
)


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


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []

    prob = []
    for logit, _box in zip(logits_filt, boxes_filt, strict=False):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        prob.append(logit.max().item())

    return boxes_filt, pred_phrases, prob


# 2d functions
def plot_boxes_to_image(image_pil, tgt, prompt_objs, color1, color2):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    objs = [label.split("(")[0] for label in labels]

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label, obj in zip(boxes, labels, objs, strict=False):
        print(obj)
        print(prompt_objs)
        if obj == prompt_objs[0]:  # obj1
            color = color1
        elif obj == prompt_objs[1]:  # obj2
            color = color2
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


def clean_boxes(boxes, size):
    H = size[1]
    W = size[0]
    clean_boxes = []
    m = 1
    for i in range(len(boxes)):
        box = boxes[i]
        box = box * torch.Tensor([W, H, W, H])
        clean_boxes.append(box)
    if len(clean_boxes) == 0:
        m = 0
    return clean_boxes, m


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


def filter_box(boxes, phrases, probs, iou_threshold=0.9):
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


def spatial_judge(box0, box1, spatial):
    # box:[xc,yc,w,h]
    correct_spatial = 0
    IoU = 0
    IoMinA = 0

    IoU, IoMinA = calculate_iou(box0, box1)

    centre_0 = [box0[0].item(), box0[1].item()]
    centre_1 = [box1[0].item(), box1[1].item()]
    dw = centre_0[0] - centre_1[0]  # x0-x1
    dh = centre_0[1] - centre_1[1]  # y0-y1

    if dw < 0.0 and abs(dw) > abs(dh):
        actual_spatial = "left"
    elif dw > 0.0 and abs(dw) > abs(dh):
        actual_spatial = "right"
    elif dh < 0.0 and abs(dh) > abs(dw):
        actual_spatial = "above"
    elif dh > 0.0 and abs(dh) > abs(dw):
        actual_spatial = "under"
    else:
        actual_spatial = ""

    if (
        actual_spatial == spatial
        or (spatial == "on" and actual_spatial == "above")
        or (spatial == "below" and actual_spatial == "under")
    ):
        correct_spatial = 1
    else:
        correct_spatial = 0

    return actual_spatial, correct_spatial, centre_0, centre_1, IoU, IoMinA


def pick_max_2d(total_score_1_list, record_all_correct_spatial):
    max1 = max(total_score_1_list)
    ind1 = total_score_1_list.index(max1)
    best_box = record_all_correct_spatial[ind1]
    score = best_box["spatial_score_1"]
    selected_box_0 = best_box["box0"]
    selected_box_1 = best_box["box1"]
    selected_label = best_box["label"]
    return score, selected_box_0, selected_box_1, selected_label


# 3d functions
def show_mask(mask, ax, my_color, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else my_color
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def intersection_judge(box0, box1):
    # box:[xc,yc,w,h]
    IoU, IoMinA = calculate_iou(box0, box1)
    return IoU, IoMinA


def pick_max_3d(total_score_1_list, record_all_good_spatial):
    max1 = max(total_score_1_list)
    ind1 = total_score_1_list.index(max1)
    best_box = record_all_good_spatial[ind1]
    score = best_box["spatial_score_1"]
    mask0 = best_box["mask0"]
    mask1 = best_box["mask1"]
    return score, mask0, mask1


def combine_frame(input_csv, output_csv):
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
                my_score_1 = float(line[-1])

                if my_score_1 < -1:  # =-2
                    my_score_1 = 0
                elif my_score_1 < 0:  # =-1
                    my_score_1 = 0.2
                elif my_score_1 == 0:  # =0
                    my_score_1 = 0.4
                elif my_score_1 > 0:
                    my_score_1 = (my_score_1 * 0.6) + 0.4
                frame_score_1.append(my_score_1)

            score_vid_1.append(sum(frame_score_1) / 16)
            score_frame_1.append(frame_score_1)

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

    return output_csv


def combine_csv_and_cal_model_score(csv_2d, csv_3d, output_file):
    with open(csv_2d) as file:
        reader = csv.reader(file)
        lines_2d = list(reader)
    with open(csv_3d) as file2:
        reader2 = csv.reader(file2)
        lines_3d = list(reader2)

    lines = lines_2d[1:] + lines_3d[1:]
    lines = sorted(lines, key=lambda x: int(x[0]))  # Sort by the first element
    lines = [["id", "score_frame", "Score"], *lines]
    score = []
    print(lines)
    for line in lines[1:]:
        score.append(float(line[-1]))

    score = sum(score) / len(score)

    with open(output_file, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)
        writer.writerow(["Score: ", score])


def spatial_2d(args):
    config_file = args.config  # change the path of the model config file
    checkpoint_path = args.grounded_checkpoint  # change the path of the model
    output_dir = os.path.join(args.output_dir, args.t2v_model)  # change the path of the output image directory
    output_path = args.output_path  # change the path of the output score csv
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold_2d
    device = args.device

    # load model
    model = load_model(config_file, checkpoint_path, device=device)

    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    frame_folder = args.frame_folder
    videos = os.listdir(frame_folder)
    videos.sort()  # sort

    with open(f"{output_path}/{args.t2v_model}_2dframe.csv", "w", newline="") as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["video_name", "image_name", "prompt", "object_1", "object_2", "score"])

        for i in range(len(videos)):
            video_name = videos[i]
            num = int(video_name[0:4]) - 1

            prompt = prompts[num]["prompt"]
            spatial = prompts[num]["spatial"]  # A is on the left of B
            phrase_0 = prompts[num]["object_1"]
            phrase_1 = prompts[num]["object_2"]

            if spatial not in ["left", "right", "above", "on", "under", "below", "in front of", "behind"]:
                print(spatial, "spatial not included!!!, index: ", videos[i])
                break

            if spatial in ["left", "right", "above", "on", "under", "below"]:
                os.makedirs(os.path.join(output_dir, videos[i]), exist_ok=True)
                video_path = os.path.join(frame_folder, videos[i])
                images = os.listdir(video_path)
                images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # sort

                for image_name in images:
                    image_path = os.path.join(frame_folder, videos[i], image_name)
                    image_pil, image = load_image(image_path)

                    # run model
                    boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                        model, image, phrase_0, box_threshold, text_threshold, device=device
                    )

                    boxes_filt_1, pred_phrases_1, prob_1 = get_grounding_output(
                        model, image, phrase_1, box_threshold, text_threshold, device=device
                    )
                    size = image_pil.size

                    all_box = torch.cat((boxes_filt_0, boxes_filt_1), dim=0)
                    all_prob = prob_0 + prob_1
                    all_phrase = pred_phrases_0 + pred_phrases_1
                    all_box, all_phrase, all_prob = filter_box(
                        all_box, all_phrase, all_prob, iou_threshold=iou_threshold
                    )

                    clean_prob_0 = [
                        all_prob[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_0
                    ]
                    clean_prob_1 = [
                        all_prob[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_1
                    ]
                    clean_boxes_0 = [
                        all_box[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_0
                    ]
                    clean_boxes_1 = [
                        all_box[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_1
                    ]
                    clean_label_0 = [
                        all_phrase[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_0
                    ]
                    clean_label_1 = [
                        all_phrase[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_1
                    ]

                    m0 = len(clean_prob_0)
                    m1 = len(clean_prob_1)

                    record_all_correct_spatial = []
                    if m0 != 0 and m1 != 0:
                        for ii in range(len(clean_boxes_0)):
                            for jj in range(len(clean_boxes_1)):
                                # correct_spatial, good_spatial,centre_0,centre_1, IoU, IoMinA, XIoMinX, YIoMinY = spatial_judge(clean_boxes_0[ii],clean_boxes_1[jj],spatial)
                                actual_spatial, correct_spatial, centre_0, centre_1, IoU, IoMinA = spatial_judge(
                                    clean_boxes_0[ii], clean_boxes_1[jj], spatial
                                )
                                if correct_spatial == 1:
                                    spatial_score_1 = 1 - IoU
                                    prob_score_A = 0.5 * clean_prob_0[ii] + 0.5 * clean_prob_1[jj]
                                    total_score_1 = 0.5 * spatial_score_1 + 0.5 * prob_score_A

                                    info = {}
                                    info["name"] = f"{ii}_{jj}"
                                    info["box0"] = clean_boxes_0[ii]
                                    info["box1"] = clean_boxes_1[jj]
                                    info["total_score_1"] = total_score_1
                                    info["spatial_score_1"] = spatial_score_1
                                    info["label"] = [clean_label_0[ii], clean_label_1[jj]]

                                    record_all_correct_spatial.append(info)

                        if len(record_all_correct_spatial) != 0:
                            total_score_1_list = []

                            for candidate_box in record_all_correct_spatial:
                                total_score_1_list.append(candidate_box["total_score_1"])

                            score_1, selected_box_0, selected_box_1, selected_label = pick_max_2d(
                                total_score_1_list, record_all_correct_spatial
                            )  # correct spatial relationship, score = 1-IoU

                        else:
                            score_1 = 0  # wrong spatial relationship

                    elif (m0 == 0 and m1 != 0) or (m0 != 0 and m1 == 0):  # 1 object missing
                        score_1 = -1
                    elif m0 == 0 and m1 == 0:  # both objects missing
                        score_1 = -2

                    csv_writer.writerow([videos[i], image_name, prompt, m0, m1, score_1])
                    csvfile.flush()

                    # visualize pred
                    # plot all detected boxes
                    pred_dict_0 = {
                        "boxes": all_box,
                        "size": [size[1], size[0]],  # H,W
                        "labels": all_phrase,
                    }
                    color1 = (255, 200, 200)
                    color2 = (150, 200, 255)
                    image_with_box = plot_boxes_to_image(image_pil, pred_dict_0, [phrase_0, phrase_1], color1, color2)[
                        0
                    ]

                    if len(record_all_correct_spatial) != 0:
                        # plot selected correct boxes
                        pred_dict_1 = {
                            "boxes": [selected_box_0, selected_box_1],
                            "size": [size[1], size[0]],  # H,W
                            "labels": selected_label,
                        }
                        color1 = (255, 0, 0)
                        color2 = (0, 0, 255)
                        image_with_box = plot_boxes_to_image(
                            image_with_box, pred_dict_1, [phrase_0, phrase_1], color1, color2
                        )[0]

                    os.makedirs(os.path.join(output_dir, videos[i]), exist_ok=True)
                    image_with_box.save(os.path.join(output_dir, videos[i], image_name))

        output_csv = combine_frame(
            f"{output_path}/{args.t2v_model}_2dframe.csv", f"{output_path}/{args.t2v_model}_2dvideo.csv"
        )
        return output_csv


def spatial_3d(args):
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    use_sam_hq = args.use_sam_hq
    depth_folder = os.path.join(args.depth_folder, args.t2v_model)

    output_dir = os.path.join(args.output_dir, args.t2v_model)  # change the path of the output image directory
    output_path = args.output_path  # change the path of the output score csv
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold_3d
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    if use_sam_hq:
        raise NotImplementedError(
            "SAM-HQ is not integrated as a WorldFoundry base model. "
            "Run this metric with the default SAM v1 checkpoint instead."
        )
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    frame_folder = args.frame_folder
    videos = os.listdir(frame_folder)
    videos.sort()  # sort

    with open(f"{output_path}/{args.t2v_model}_3dframe.csv", "w", newline="") as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["video_name", "image_name", "prompt", "object_1", "object_2", "score"])

        for i in range(len(videos)):
            video_name = videos[i]
            num = int(video_name[0:4]) - 1

            spatial = prompts[num]["spatial"]
            if spatial in ["in front of", "behind"]:
                prompt = prompts[num]["prompt"]
                phrase_0 = prompts[num]["object_1"]  # A is on the left of B
                phrase_1 = prompts[num]["object_2"]

                images = os.listdir(os.path.join(frame_folder, videos[i]))
                images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

                for image_name in images:
                    image_path = os.path.join(frame_folder, videos[i], image_name)

                    # load image
                    image_pil, image_loded = load_image(image_path)

                    depth_path = os.path.join(depth_folder, videos[i], image_name)

                    boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                        model, image_loded, phrase_0, box_threshold, text_threshold, device=device
                    )
                    boxes_filt_1, pred_phrases_1, prob_1 = get_grounding_output(
                        model, image_loded, phrase_1, box_threshold, text_threshold, device=device
                    )
                    size = image_pil.size

                    all_box = torch.cat((boxes_filt_0, boxes_filt_1), dim=0)
                    all_prob = prob_0 + prob_1
                    all_phrase = pred_phrases_0 + pred_phrases_1
                    all_box, all_phrase, all_prob = filter_box(
                        all_box, all_phrase, all_prob, iou_threshold=iou_threshold
                    )

                    clean_prob_0 = [
                        all_prob[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_0
                    ]
                    clean_prob_1 = [
                        all_prob[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_1
                    ]
                    clean_boxes_0 = [
                        all_box[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_0
                    ]
                    clean_boxes_1 = [
                        all_box[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_1
                    ]
                    clean_label_0 = [
                        all_phrase[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_0
                    ]
                    clean_label_1 = [
                        all_phrase[j] for j, phrase in enumerate(all_phrase) if phrase.split("(")[0] == phrase_1
                    ]

                    boxes_filt_0 = torch.stack(clean_boxes_0, dim=0) if len(clean_boxes_0) > 0 else torch.tensor([])

                    boxes_filt_1 = torch.stack(clean_boxes_1, dim=0) if len(clean_boxes_1) > 0 else torch.tensor([])

                    m0 = len(clean_prob_0)
                    m1 = len(clean_prob_1)

                    # sam
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image)

                    H, W = size[1], size[0]
                    for k in range(boxes_filt_0.size(0)):
                        print(boxes_filt_0[k].size())
                        boxes_filt_0[k] = boxes_filt_0[k] * torch.Tensor([W, H, W, H])
                        boxes_filt_0[k][:2] -= boxes_filt_0[k][2:] / 2
                        boxes_filt_0[k][2:] += boxes_filt_0[k][:2]
                    boxes_filt_0 = boxes_filt_0.cpu()

                    for k in range(boxes_filt_1.size(0)):
                        boxes_filt_1[k] = boxes_filt_1[k] * torch.Tensor([W, H, W, H])
                        boxes_filt_1[k][:2] -= boxes_filt_1[k][2:] / 2
                        boxes_filt_1[k][2:] += boxes_filt_1[k][:2]
                    boxes_filt_1 = boxes_filt_1.cpu()

                    transformed_boxes_0 = predictor.transform.apply_boxes_torch(boxes_filt_0, image.shape[:2]).to(
                        device
                    )
                    transformed_boxes_1 = predictor.transform.apply_boxes_torch(boxes_filt_1, image.shape[:2]).to(
                        device
                    )

                    if m0 != 0:
                        masks_0, _, _ = predictor.predict_torch(  # masks_0[0]:[1,320,576]
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes_0.to(device),
                            multimask_output=False,
                        )

                    if m1 != 0:
                        masks_1, _, _ = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes_1.to(device),
                            multimask_output=False,
                        )

                    record_all_correct_spatial = []
                    if m0 != 0 and m1 != 0:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image)
                        for box, label in zip(boxes_filt_0, clean_label_0, strict=False):
                            show_box(box.numpy(), plt.gca(), label)
                        for box, label in zip(boxes_filt_1, clean_label_1, strict=False):
                            show_box(box.numpy(), plt.gca(), label)

                        for ii in range(len(clean_boxes_0)):
                            for jj in range(len(clean_boxes_1)):
                                print(jj)
                                print(boxes_filt_1.size())
                                IoU, IoMinA = intersection_judge(clean_boxes_0[ii], clean_boxes_1[jj])
                                if IoU != 0:
                                    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                                    height, width = depth_map.shape

                                    mask_image_0 = (masks_0[ii].cpu().numpy().squeeze() * 255).astype(np.uint8)
                                    obj1_seg = cv2.bitwise_and(depth_map, depth_map, mask=mask_image_0)
                                    d1 = np.sum(obj1_seg) / cv2.countNonZero(mask_image_0)

                                    mask_image_1 = (masks_1[jj].cpu().numpy().squeeze() * 255).astype(np.uint8)
                                    obj2_seg = cv2.bitwise_and(depth_map, depth_map, mask=mask_image_1)
                                    d2 = np.sum(obj2_seg) / cv2.countNonZero(mask_image_1)

                                    if (not 0 <= d1 <= 255) or (not 0 <= d2 <= 255):
                                        print("d1 wrong value")
                                    if spatial == "in front of":
                                        if d1 > d2:
                                            prob_score = 0.5 * prob_0[ii] + 0.5 * prob_1[jj]
                                            spatial_score_1 = IoU
                                            total_score_1 = 0.5 * prob_score + 0.5 * spatial_score_1

                                            seg_save_path = os.path.join(
                                                output_dir, videos[i], image_name.split(".")[0], f"obj1_seg_{ii}.png"
                                            )
                                            cv2.imwrite(seg_save_path, obj1_seg)
                                            seg_save_path = os.path.join(
                                                output_dir, videos[i], image_name.split(".")[0], f"obj2_seg_{jj}.png"
                                            )
                                            cv2.imwrite(seg_save_path, obj2_seg)

                                            info = {}
                                            info["name"] = f"{ii}_{jj}"
                                            info["box0"] = clean_boxes_0[ii]
                                            info["box1"] = clean_boxes_1[jj]
                                            info["total_score_1"] = total_score_1
                                            info["spatial_score_1"] = spatial_score_1
                                            info["mask0"] = masks_0[ii]
                                            info["mask1"] = masks_1[jj]

                                            record_all_correct_spatial.append(info)

                                    elif spatial == "behind" and d1 < d2:
                                        prob_score = 0.5 * prob_0[ii] + 0.5 * prob_1[jj]
                                        spatial_score_1 = IoU
                                        total_score_1 = 0.5 * prob_score + 0.5 * spatial_score_1

                                        seg_save_path = os.path.join(
                                            output_dir, videos[i], image_name.split(".")[0], f"obj1_seg_{ii}.png"
                                        )
                                        cv2.imwrite(seg_save_path, obj1_seg)
                                        seg_save_path = os.path.join(
                                            output_dir, videos[i], image_name.split(".")[0], f"obj2_seg_{jj}.png"
                                        )
                                        cv2.imwrite(seg_save_path, obj2_seg)

                                        info = {}
                                        info["name"] = f"{ii}_{jj}"
                                        info["box0"] = clean_boxes_0[ii]
                                        info["box1"] = clean_boxes_1[jj]
                                        info["total_score_1"] = total_score_1
                                        info["spatial_score_1"] = spatial_score_1
                                        info["mask0"] = masks_0[ii]
                                        info["mask1"] = masks_1[jj]

                                        record_all_correct_spatial.append(info)

                        if len(record_all_correct_spatial) != 0:
                            total_score_1_list = []
                            for candidate_box in record_all_correct_spatial:
                                total_score_1_list.append(candidate_box["total_score_1"])
                            score_1, mask0, mask1 = pick_max_3d(total_score_1_list, record_all_correct_spatial)

                        else:
                            score_1 = 0

                    elif (m0 == 0 and m1 != 0) or (m0 != 0 and m1 == 0):
                        score_1 = -1
                    elif m0 == 0 and m1 == 0:
                        score_1 = -2

                    csv_writer.writerow([videos[i], image_name, prompt, m0, m1, score_1])
                    csvfile.flush()

                    if len(record_all_correct_spatial) != 0:
                        color0 = np.array([150 / 255, 150 / 255, 255 / 255, 0.6])
                        show_mask(mask0.cpu().numpy(), plt.gca(), color0, random_color=False)
                        color1 = np.array([255 / 255, 150 / 255, 150 / 255, 0.6])
                        show_mask(mask1.cpu().numpy(), plt.gca(), color1, random_color=False)

                    plt.axis("off")
                    os.makedirs(os.path.join(output_dir, videos[i], image_name.split(".")[0]), exist_ok=True)
                    plt.savefig(
                        os.path.join(output_dir, videos[i], image_name.split(".")[0], "grounded_sam_output.jpg"),
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=0.0,
                    )

    output_csv = combine_frame(
        f"{output_path}/{args.t2v_model}_3dframe.csv", f"{output_path}/{args.t2v_model}_3dvideo.csv"
    )
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WorldFoundry Grounded-SAM spatial metric", add_help=True)
    # 2d args
    parser.add_argument("--config", type=str, default=str(grounding_dino_config_path()), help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default=str(grounding_dino_checkpoint_path()), help="path to checkpoint file"
    )
    parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")

    parser.add_argument(
        "--iou_threshold_2d", type=float, default=0.9, help="threshold to filter out the duplicated boxes"
    )

    # 3d args
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default=str(sam_v1_checkpoint_path("vit_h")), help="path to sam checkpoint file"
    )
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")

    parser.add_argument("--depth_folder", type=str, required=True)
    parser.add_argument(
        "--iou_threshold_3d", type=float, default=0.95, help="threshold to filter out the duplicated boxes"
    )
    parser.add_argument("--output-path", type=str, default="csv_spatial", help="path to store the video scores")
    parser.add_argument("--read-prompt-file", type=str, default=str(asset_path("meta_data/spatial_relationships.json")))
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--frame_folder", type=str)
    parser.add_argument("--t2v-model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_spatial/", help="directory to save the output images")

    args = parser.parse_args()

    frame_folder = run_depth(args.video_path, args.t2v_model, args.depth_folder, args.read_prompt_file)

    args.frame_folder = frame_folder
    csv_2d = spatial_2d(args)
    csv_3d = spatial_3d(args)
    output_csv = f"{args.output_path}/{args.t2v_model}_spatial_score.csv"
    combine_csv_and_cal_model_score(csv_2d, csv_3d, output_csv)
