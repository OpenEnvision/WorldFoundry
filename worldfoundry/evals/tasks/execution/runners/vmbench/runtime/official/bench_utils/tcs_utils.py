import numpy as np
import torch

from worldfoundry.base_models.perception_core.detection.grounding_dino.models import build_model

# Grounding DINO
from worldfoundry.base_models.perception_core.detection.grounding_dino.util import transforms as T
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.misc import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.util.slconfig import SLConfig


def transform_pil(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


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


# Get all suddenly disappeared objects
def get_disappear_objects(tracking_result):
    result = []
    first_appearances = {}  # To track the index and mask of each object's first appearance

    # First, iterate through the entire list, recording the index and mask of each object's first appearance
    for i, current_dict in enumerate(tracking_result):
        for key, value in current_dict.items():
            if key not in first_appearances:
                first_appearances[key] = {"frame": i, "mask": np.array(value["mask"]) if "mask" in value else None}

    # Then iterate through the list again, checking for object disappearances
    for i in range(len(tracking_result) - 1):
        dict1 = tracking_result[i]
        dict2 = tracking_result[i + 1]

        # Find disappeared keys
        disappeared_keys = set(dict1.keys()) - set(dict2.keys())

        # If there are disappeared keys, record them and their information
        for key in disappeared_keys:
            disappeared_object_info = {
                "object_id": key,  # Record the ID of the disappeared object
                "mask": first_appearances[key]["mask"],  # Record the mask when the object first appeared
                "first_appearance": first_appearances[key]["frame"],  # Record the index when the object first appeared
                "last_frame": i,  # Record the index when the object last appeared
            }

            result.append(disappeared_object_info)

    return result


def is_edge_vanish(pred_tracks, pred_visibility, start, width=720, height=480, visibility_ratio=0.8, point_ratio=0.5):
    # get the invisible frames
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    # filter frames after tracking start frame
    indices = indices[indices > start]
    selected_frames = pred_tracks[0, indices]  # shape: [len(filtered_indices), point_num, 2]
    # consider every edges
    left_mask = selected_frames[:, :, 0] < 0
    right_mask = selected_frames[:, :, 0] > width
    top_mask = selected_frames[:, :, 1] < 0
    bottom_mask = selected_frames[:, :, 1] > height
    # satisfy any condition
    out_of_screen_mask = left_mask | right_mask | top_mask | bottom_mask
    # calculate the ratio out of screen
    out_of_screen_ratio = out_of_screen_mask.float().mean(dim=1)
    # check out of screen ratio >= point_ratio
    valid_frames_mask = out_of_screen_ratio >= point_ratio
    # get all valid frames indices
    vanish_indices = indices[valid_frames_mask]
    edge_vanish = len(vanish_indices) > 0
    # print("Normal disappearance from screen edge")
    # print("Not disappearing from screen edge")
    return edge_vanish


def is_small_vanish(
    pred_tracks,
    pred_visibility,
    start,
    width=720,
    height=480,
    visibility_ratio=-1,
    point_ratio=0.8,
    size_threshold=0.07,
):
    # get the invisible frames
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    # filter frames after tracking start frame
    indices = indices[indices > start]
    pred_tracks = pred_tracks[0, indices]
    # record small object frames
    small_object_frames = []

    for i, frame in enumerate(pred_tracks):
        # Determine if the object is on screen
        left_mask = frame[:, 0] > 0
        right_mask = frame[:, 0] < width
        top_mask = frame[:, 1] > 0
        bottom_mask = frame[:, 1] < height
        in_screen_mask = left_mask & right_mask & top_mask & bottom_mask

        # Calculate if the object formed by key points is very small
        valid_points = frame[in_screen_mask]
        if (
            in_screen_mask.float().mean(dim=0) >= point_ratio and valid_points.shape[0] > 1
        ):  # Ensure at least two points to calculate size
            q_low = torch.quantile(valid_points, 0.1, dim=0)
            q_high = torch.quantile(valid_points, 0.9, dim=0)
            object_width = (q_high[0] - q_low[0]) / width
            object_height = (q_high[1] - q_low[1]) / height
            object_size = max(object_width, object_height)

            if object_size < size_threshold:
                small_object_frames.append(i)

    small_vanish = len(small_object_frames) > 0
    # print("Normal disappearance due to being too small")
    # print("Not disappearing due to being too small")
    return small_vanish


# Determine if there are really invisible frames
def is_vanish_detect_error(pred_tracks, pred_visibility, start, visibility_ratio=1.0):
    # get the invisible frames
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    # filter frames after tracking start frame
    indices = indices[indices > start]
    detect_error = len(indices) == 0
    # print("Error due to detection mistake")
    # print("No error due to detection mistake")
    return detect_error


# Get all suddenly appearing objects
def get_appear_objects(dict_list):
    result = []
    first_appearances = {}  # To track the index and mask of each object's first appearance

    # Iterate through the list, recording the index and mask of each object's first appearance
    for i, current_dict in enumerate(dict_list):
        for key, value in current_dict.items():
            if key not in first_appearances:
                first_appearances[key] = {"frame": i, "mask": np.array(value["mask"]) if "mask" in value else None}

    # Then iterate through the list, checking for object appearances
    for i in range(1, len(dict_list)):
        dict1 = dict_list[i - 1]
        dict2 = dict_list[i]

        # Find newly appeared keys
        appeared_keys = set(dict2.keys()) - set(dict1.keys())

        # If there are newly appeared keys, record them and their information
        for key in appeared_keys:
            appeared_object_info = {
                "object_id": key,  # Record the ID of the appeared object
                "mask": first_appearances[key]["mask"],  # Record the mask when the object first appeared
                "first_appearance": first_appearances[key]["frame"],  # Record the index when the object first appeared
            }

            result.append(appeared_object_info)

    return result


def is_edge_emerge(pred_tracks, pred_visibility, start, width=720, height=480, visibility_ratio=0.85, point_ratio=0.5):
    # filter object invisible frames
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    # filter the frames before the start
    indices = indices[indices < start]
    selected_frames = pred_tracks[0, indices]  # shape: [len(filtered_indices), 7, 2]

    # consider every edges
    left_mask = selected_frames[:, :, 0] < 0
    right_mask = selected_frames[:, :, 0] > width
    top_mask = selected_frames[:, :, 1] < 0
    bottom_mask = selected_frames[:, :, 1] > height
    # satisfy any condition
    out_of_screen_mask = left_mask | right_mask | top_mask | bottom_mask
    # calculate the ratio out of screen
    out_of_screen_ratio = out_of_screen_mask.float().mean(dim=1)
    # check out of screen ratio >= point_ratio
    valid_frames_mask = out_of_screen_ratio >= point_ratio
    # get all valid frames indices
    emerge_indices = indices[valid_frames_mask]

    edge_emerge = len(emerge_indices) > 0
    # print("Normal appearance from screen edge")
    # print("Not appearing from screen edge")

    return edge_emerge


def is_small_emerge(
    pred_tracks,
    pred_visibility,
    start,
    width=720,
    height=480,
    visibility_ratio=-1,
    point_ratio=0.8,
    size_threshold=0.03,
):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices < start]
    pred_tracks = pred_tracks[0, indices]
    small_object_frames = []

    for i, frame in enumerate(pred_tracks):
        # Determine if the object is on screen
        left_mask = frame[:, 0] > 0
        right_mask = frame[:, 0] < width
        top_mask = frame[:, 1] > 0
        bottom_mask = frame[:, 1] < height
        in_screen_mask = left_mask & right_mask & top_mask & bottom_mask

        # Calculate if the object formed by key points is very small
        valid_points = frame[in_screen_mask]
        if (
            in_screen_mask.float().mean(dim=0) >= point_ratio and valid_points.shape[0] > 1
        ):  # Ensure at least two points to calculate size
            q_low = torch.quantile(valid_points, 0.1, dim=0)
            q_high = torch.quantile(valid_points, 0.9, dim=0)
            object_width = (q_high[0] - q_low[0]) / width
            object_height = (q_high[1] - q_low[1]) / height
            object_size = max(object_width, object_height)

            if object_size < size_threshold:
                small_object_frames.append(i)

    small_emerge = len(small_object_frames) > 0
    # print("Normal appearance from being too small")
    # print("Not appearing from being too small")

    return small_emerge


# Determine if there are really invisible frames
def is_emerge_detect_error(pred_tracks, pred_visibility, start, visibility_ratio=0.8):
    # get the invisible frames
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    # filter frames before tracking start frame
    indices = indices[indices < start]

    detect_error = len(indices) == 0
    # print("Error due to detection mistake")
    # print("No error due to detection mistake")
    return detect_error
