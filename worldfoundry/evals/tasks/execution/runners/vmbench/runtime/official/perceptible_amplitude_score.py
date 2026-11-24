import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.perception_core.detection.grounding_dino.models import build_model
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
    checkpoint_path as grounding_dino_checkpoint_path,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
    config_path as grounding_dino_config_path,
)

# Grounding DINO
from worldfoundry.base_models.perception_core.detection.grounding_dino.util import transforms as T
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.misc import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.slconfig import SLConfig

# segment anything
from worldfoundry.base_models.perception_core.segment.sam_v1 import SamPredictor, sam_model_registry
from worldfoundry.base_models.perception_core.segment.sam_v1.paths import checkpoint_path as sam_checkpoint_path

# Co-Tracker
from worldfoundry.base_models.perception_core.tracking.cotracker import CoTrackerPredictor
from worldfoundry.base_models.perception_core.tracking.cotracker.paths import (
    checkpoint_path as cotracker_checkpoint_path,
)


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None, None, None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        print("Error reading frames from the video")
        return None, None, None, None

    # take the first frame as the query image
    frame_rgb = frames[0]
    image_pil = Image.fromarray(frame_rgb)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)

    return image_pil, image, frame_rgb, np.stack(frames)


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
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
    for logit, _box in zip(logits_filt, boxes_filt, strict=False):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def calculate_motion_degree(keypoints, video_width, video_height):
    """
    Calculate the normalized motion amplitude for each batch sample

    Parameters:
    keypoints: torch.Tensor, shape [batch_size, 49, 792, 2]
    video_width: int, width of the video
    video_height: int, height of the video

    Returns:
    motion_amplitudes: torch.Tensor, shape [batch_size], containing the normalized motion amplitude for each batch sample
    """

    # Calculate the length of the video diagonal
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2, dtype=torch.float32))

    # Compute the Euclidean distance between adjacent frames
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)  # shape [batch_size, 48, 792]

    # Normalize the distances (divide by the diagonal length)
    normalized_distances = distances / diagonal

    # Sum the normalized distances to get the total normalized motion distance for each keypoint
    total_normalized_distances = torch.sum(normalized_distances, dim=1)  # shape [batch_size, 792]

    # Compute the normalized motion amplitude for each batch sample (mean of total normalized motion distance for all points)
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)  # shape [batch_size]

    return motion_amplitudes


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--meta_info_path", type=str, required=True, help="path to meta info json")
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=False,
        help="text prompt",
        default="person. dog. cat. horse. car. ball. robot. bird. bicycle. motorcycle. surfboard. skateboard. bucket. bat. basketball. "
        "racket. kitten. puppy. fish. laptop. umbrella. wheelchair. drone. scooter. rollerblades. truck. bus. skier. snowboard. "
        "sled. kayak. canoe. sailboat. guitar. piano. drum. violin. trumpet. saxophone. clarinet. flute. accordion. telescope. "
        "microscope. treadmill. rope. ladder. swing. tugboat. train.",
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--grid_size", type=int, default=30, help="Regular grid size")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # model cfg
    config_file = os.environ.get("WORLDFOUNDRY_VMBENCH_GROUNDING_DINO_CONFIG", str(grounding_dino_config_path()))
    grounded_checkpoint = os.environ.get(
        "WORLDFOUNDRY_VMBENCH_GROUNDING_DINO_CKPT", str(grounding_dino_checkpoint_path())
    )
    bert_base_uncased_path = os.environ.get(
        "WORLDFOUNDRY_GROUNDING_DINO_BERT_BASE_UNCASED", ".cache/google-bert/bert-base-uncased"
    )
    sam_version = "vit_h"
    sam_checkpoint = os.environ.get("WORLDFOUNDRY_SAM_VIT_H_CKPT", str(sam_checkpoint_path("vit_h")))
    cotracker_checkpoint = os.environ.get("WORLDFOUNDRY_COTRACKER2_CKPT", str(cotracker_checkpoint_path()))

    meta_info_path = args.meta_info_path
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    grid_size = args.grid_size
    device = args.device

    # load model
    grounding_model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # initialize SAM
    sam_predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    # intialize Co-Tracker
    cotracker_model = CoTrackerPredictor(
        checkpoint=cotracker_checkpoint,
        v2=False,
        offline=True,
        window_len=60,
    ).to(device)

    # load meta info json
    with open(args.meta_info_path) as f:
        meta_infos = json.load(f)

    for meta_info in tqdm(meta_infos, desc="Motion Degree: Grounded SAM Segmentation"):
        image_pil, image, image_array, video = load_video(meta_info["filepath"])

        text_prompt = meta_info["subject_noun"] + "."

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            grounding_model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        # no detect object
        if boxes_filt.shape[0] == 0:
            print(f"can not detect {text_prompt} in {meta_info['prompt']}")
        else:
            sam_predictor.set_image(image_array)

            # convert boxes into xyxy format
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            # run sam model
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(device),
                multimask_output=False,
            )

        # load the input video frame by frame
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_width, video_height = video.shape[-1], video.shape[-2]
        video = video.to(device)

        if boxes_filt.shape[0] != 0:
            background_mask = torch.any(~masks, dim=0).to(torch.uint8) * 255
        else:
            background_mask = torch.ones((1, video_height, video_width), dtype=torch.uint8, device=device) * 255

        # eval background (camera) motion degree
        background_mask = background_mask.unsqueeze(0)
        pred_tracks, pred_visibility = cotracker_model(
            video, grid_size=grid_size, grid_query_frame=0, backward_tracking=True, segm_mask=background_mask
        )

        background_motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()
        # meta_info['motion_amplitude']['camera'] = background_motion_degree.item()

        if boxes_filt.shape[0] != 0:
            subject_mask = torch.any(masks, dim=0).to(torch.uint8) * 255
            # eval subject motion degree
            subject_mask = subject_mask.unsqueeze(0)
            pred_tracks, pred_visibility = cotracker_model(
                video, grid_size=grid_size, grid_query_frame=0, backward_tracking=True, segm_mask=subject_mask
            )

            subject_motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()
            # subject_motion_degree = subject_motion_degree.item()
            if subject_motion_degree > background_motion_degree:
                subject_motion_degree = subject_motion_degree - background_motion_degree
            if not np.isnan(subject_motion_degree):
                meta_info["perceptible_amplitude_score"] = subject_motion_degree
            else:
                meta_info["perceptible_amplitude_score"] = background_motion_degree
        else:
            meta_info["perceptible_amplitude_score"] = background_motion_degree

        # save meta info per video
        with open(args.meta_info_path, "w") as f:
            json.dump(meta_infos, f, indent=4)
