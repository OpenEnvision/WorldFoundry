import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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
from worldfoundry.base_models.perception_core.segment.sam_v1 import SamPredictor, sam_model_registry
from worldfoundry.base_models.perception_core.segment.sam_v1.paths import checkpoint_path as sam_v1_checkpoint_path
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
        print("start converting video to video of fps=8 from path:", video_path)

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

    logits_list = []
    for logit, _box in zip(logits_filt, boxes_filt, strict=False):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        logits_list.append(logit.max().item())

    return boxes_filt, pred_phrases, logits_list


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 255 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    # value = 0  # 0 for background
    value = 1

    mask_img = torch.ones(mask_list.shape[-2:])
    for _idx, mask in enumerate(mask_list):
        # mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        mask_img[mask.cpu().numpy()[0] == True] = value - 1  # noqa: E712 - Elementwise NumPy mask comparison.
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "mask_background.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list, strict=False):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    with open(os.path.join(output_dir, "mask.json"), "w") as f:
        json.dump(json_data, f)


def save_mask_foreground(output_dir, mask, obj_prompt):
    # value = 0  # 0 for background
    value = 0

    mask_img = torch.zeros(mask.shape[-2:])
    mask_img[mask.cpu().numpy()[0] == True] = value + 1  # noqa: E712 - Elementwise NumPy mask comparison.
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy(), cmap="gray")
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"mask_foreground_{obj_prompt}.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0
    )


def foreground_background_mask(args):
    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    use_sam_hq = args.use_sam_hq
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    output_dir = os.path.join(args.output_dir, args.t2v_model)
    video_path = args.video_path
    total_frame = int(args.total_frame)
    fps = int(args.fps)

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

    vid_process = Video_preprocess()
    frame_folder = vid_process.convert_video_to_frames(video_path, num_frames=1)

    model_frames = total_frame // fps * 8  # number of frames to be extracted from the original video
    print(f"{args.t2v_model} extracted frames: {model_frames}")
    stardard_video_path = vid_process.convert_video_to_standard_video(
        video_path, model_frames
    )  # take ~8 frames per second and recombine to a video of fps=8

    videos = os.listdir(frame_folder)
    videos.sort()

    with open(args.read_prompt_file) as json_data:
        prompts = json.load(json_data)

    for k in range(len(videos)):
        video_name = videos[k]
        num = int(video_name[0:4]) - 1

        _prompt = prompts[num]["prompt"]
        object_1 = prompts[num]["object_1"]  # A is on the left of B
        object_2 = prompts[num]["object_2"]
        d_1 = prompts[num]["d_1"]
        d_2 = prompts[num]["d_2"]

        # print(object_1,object_2)
        directions = ["left", "right", "up", "down", ""]
        if d_1 not in directions[:4] or d_2 not in directions:
            print(d_1, d_2, " direction not included!!!, index: ", k)
            break
        if object_2 != "":
            background_prompt = object_1 + " . " + object_2
            object_to_detect = [object_1, object_2]
        else:
            background_prompt = object_1
            object_to_detect = [object_1]

        os.makedirs(os.path.join(output_dir, videos[k]), exist_ok=True)

        image_name = videos[k] + "_000000.png"
        image_path = os.path.join(frame_folder, videos[k], image_name)

        # load image
        image_pil, image_loded = load_image(image_path)

        # mask_foreground
        for obj_prompt in object_to_detect:
            boxes_filt, pred_phrases, probs = get_grounding_output(
                model, image_loded, obj_prompt, box_threshold, text_threshold, device=device
            )
            if boxes_filt.shape[0] == 0:
                continue
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )

            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
            for box, label in zip(boxes_filt, pred_phrases, strict=False):
                show_box(box.numpy(), plt.gca(), label)

            plt.axis("off")
            plt.savefig(
                os.path.join(output_dir, videos[k], f"grounded_sam_output_{obj_prompt}.jpg"),
                bbox_inches="tight",
                dpi=300,
                pad_inches=0.0,
            )

            m = max(probs)
            ind = probs.index(m)
            mask_max_prob = masks[ind]
            save_mask_foreground(os.path.join(output_dir, videos[k]), mask_max_prob, obj_prompt)

        # mask_background
        boxes_filt, pred_phrases, probs = get_grounding_output(
            model, image_loded, background_prompt, box_threshold, text_threshold, device=device
        )
        if boxes_filt.shape[0] == 0:
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
        for box, label in zip(boxes_filt, pred_phrases, strict=False):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, videos[k], "grounded_sam_output_background.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        save_mask_data(os.path.join(output_dir, videos[k]), masks, boxes_filt, pred_phrases)  # save background

    print("standard video path: ", stardard_video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WorldFoundry Grounded-SAM motion-mask metric", add_help=True)
    parser.add_argument("--config", type=str, default=str(grounding_dino_config_path()), help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default=str(grounding_dino_checkpoint_path()), help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default=str(sam_v1_checkpoint_path("vit_h")), help="path to sam checkpoint file"
    )
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")

    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--t2v-model", type=str, required=True)
    parser.add_argument("--total_frame", type=str, required=True)
    parser.add_argument("--fps", type=str, required=True)
    parser.add_argument("--read-prompt-file", type=str, default=str(asset_path("meta_data/motion_binding.json")))
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="output_motion_binding_seg",
        help="output directory of 1st frame segmentations",
    )
    args = parser.parse_args()

    foreground_background_mask(args)
