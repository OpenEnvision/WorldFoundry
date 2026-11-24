import copy
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

# isort: split
from worldfoundry.base_models.perception_core.segment.sam2.build_sam import build_sam2, build_sam2_video_predictor

# isort: split
from worldfoundry.base_models.perception_core.segment.sam2.sam2_image_predictor import SAM2ImagePredictor

# isort: split
from worldfoundry.base_models.perception_core.segment.sam2 import (
    checkpoint_path as sam2_checkpoint_path,
)

# isort: split
from worldfoundry.base_models.perception_core.segment.sam2 import (
    config_name as sam2_config_name,
)

# isort: split
from grounded_sam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
    checkpoint_path as grounding_dino_checkpoint_path,
)
from worldfoundry.base_models.perception_core.detection.grounding_dino.paths import (
    config_path as grounding_dino_config_path,
)

# isort: split
from worldfoundry.base_models.perception_core.tracking.cotracker import CoTrackerPredictor

# isort: split
from bench_utils.tcs_utils import (
    get_appear_objects,
    get_disappear_objects,
    get_grounding_output,
    is_edge_emerge,
    is_edge_vanish,
    is_emerge_detect_error,
    is_small_emerge,
    is_small_vanish,
    is_vanish_detect_error,
    load_model,
    transform_pil,
)
from tqdm import tqdm

from worldfoundry.base_models.perception_core.tracking.cotracker.paths import (
    checkpoint_path as cotracker_checkpoint_path,
)
from worldfoundry.base_models.perception_core.tracking.cotracker.visualizer import read_video_from_path


# Read the video frames and encode them as JPEG images
def extract_frames_from_video(video_path, jpeg_quality=95):
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_names = []
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        jpg_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(jpg_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        frames.append(pil_frame)
        frame_number = f"{frame_count:04d}"
        frame_names.append(frame_number)

    video.release()
    return frames, frame_names


def object_info_to_dict(obj):
    return {
        "instance_id": obj.instance_id,
        "mask": obj.mask.cpu().numpy().tolist(),
        "class_name": obj.class_name,
        "x1": obj.x1,
        "y1": obj.y1,
        "x2": obj.x2,
        "y2": obj.y2,
        "logit": float(obj.logit),
    }


if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grounding_dino_model",
        default="IDEA-Research/grounding-dino-base",
        help="huggingface grounding dino model",
    )
    parser.add_argument(
        "--text_prompt",
        default="car. bird. person. boat. bicycle. train. cow. bear. butterfly. hippopotamus. fish. horse. dog. frog. giraffe. shark. camel. motorcycle. rhino. lion. panda. penguin. tiger. sheep.",
        help="huggingface grounding dino model",
    )
    parser.add_argument(
        "--meta_info_path",
        type=str,
        default="./bad_case/meta_info.json",
        help="path to video frames",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=24,
        help="the step to sample frames for Grounding DINO predictor",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.35,
        help="box_threshold",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.35,
        help="text_threshold",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="use cuda or cpu",
    )
    parser.add_argument("--grid_size", type=int, default=30, help="Regular grid size")

    args = parser.parse_args()

    """
    Step 1: Environment settings and model initialization
    """
    config_file = os.environ.get("WORLDFOUNDRY_VMBENCH_GROUNDING_DINO_CONFIG", str(grounding_dino_config_path()))
    grounded_checkpoint = os.environ.get(
        "WORLDFOUNDRY_VMBENCH_GROUNDING_DINO_CKPT", str(grounding_dino_checkpoint_path())
    )
    bert_base_uncased_path = os.environ.get(
        "WORLDFOUNDRY_GROUNDING_DINO_BERT_BASE_UNCASED", ".cache/google-bert/bert-base-uncased"
    )
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    device = args.device if torch.cuda.is_available() else "cpu"
    print("device", device)

    # init sam image predictor and video predictor model
    sam2_checkpoint = os.environ.get("WORLDFOUNDRY_SAM2_CKPT", str(sam2_checkpoint_path()))
    model_cfg = os.environ.get("WORLDFOUNDRY_SAM2_CONFIG", sam2_config_name())

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    # model_id = os.path.join('.cache', args.grounding_dino_model)
    # processor = AutoProcessor.from_pretrained(model_id)
    # grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    grounding_model_local = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    print("Loading Grounding DINO model successfully")

    # intialize Co-Tracker
    cotracker_checkpoint = os.environ.get("WORLDFOUNDRY_COTRACKER2_CKPT", str(cotracker_checkpoint_path()))
    cotracker_model = CoTrackerPredictor(
        checkpoint=cotracker_checkpoint,
        v2=False,
        offline=True,
        window_len=60,
    ).to(device)
    print("Loading Co-Tracker model successfully")

    # open meta info json
    with open(args.meta_info_path) as f:
        meta_infos = json.load(f)

    for meta_info in tqdm(meta_infos):
        # define the detection object
        text_prompt = meta_info["subject_noun"] + "."
        # get video filepath
        video_path = meta_info["filepath"]
        # get frames and frame names from the video
        try:
            frames, frame_names = extract_frames_from_video(video_path)
        except BaseException:
            print("read video error, sikp this video")
            continue

        # use fps as the grounding dino step
        cap = cv2.VideoCapture(video_path)
        args.step = int(cap.get(cv2.CAP_PROP_FPS)) - 1
        cap.release()

        # init video predictor state
        inference_state = video_predictor.init_state(
            video_path=video_path, offload_video_to_cpu=True, async_loading_frames=True
        )
        step = args.step  # the step to sample frames for Grounding DINO predictor

        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
        objects_count = 0
        # record tracking object'sam2_image_model information
        video_object_data = []

        """
        Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
        """
        for start_frame_idx in range(0, len(frames), step):
            # prompt grounding dino to get the box coordinates on specific frame
            image = frames[start_frame_idx]
            image_base_name = frame_names[start_frame_idx]
            mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

            # # run grounding dino
            # inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
            # with torch.no_grad():
            #     outputs = grounding_model(**inputs)

            # # process results
            # results = processor.post_process_grounded_object_detection(
            #     outputs,
            #     inputs.input_ids,
            #     box_threshold=args.box_threshold,
            #     text_threshold=args.text_threshold,
            #     target_sizes=[image.size[::-1]]
            # )

            # convert PIL image into torch tensor
            image_tensor = transform_pil(image)
            # run grounding dino model
            with torch.cuda.amp.autocast(enabled=False):
                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    grounding_model_local,
                    image_tensor,
                    text_prompt,
                    args.box_threshold,
                    args.text_threshold,
                    with_logits=False,
                    device=device,
                )
                # convert boxes into xyxy format
                size = image.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # process the detection results
            # input_boxes = results[0]["boxes"] # .cpu().numpy()
            # OBJECTS = results[0]["labels"]
            input_boxes = boxes_filt
            OBJECTS = pred_phrases
            if input_boxes.shape[0] != 0:
                # prompt SAM 2 image predictor to get the mask for the object
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                # convert the mask shape to (n, H, W)
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                """
                Step 3: Register each object's positive points to video predictor
                """

                # If you are using point prompts, we uniformly sample positive points based on the mask
                if mask_dict.promote_type == "mask":
                    mask_dict.add_new_frame_annotation(
                        mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS
                    )
                else:
                    raise NotImplementedError("SAM 2 video predictor only support mask prompts")

                """
                Step 4: Propagate the video predictor to get the segmentation results for each frame
                """
                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks, iou_threshold=0.75, objects_count=objects_count
                )
            else:
                print(f"No object detected in the frame, skip merge the frame merge {frame_names[start_frame_idx]}")
                mask_dict = sam2_masks

            if len(mask_dict.labels) == 0:
                print(f"No object detected in the frame, skip the frame {start_frame_idx}")
                video_object_data.extend([{} for _ in range(step + 1)])
                continue
            else:
                video_predictor.reset_state(inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )

                video_segments = {}  # output the following {step} frames tracking masks
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                    inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx
                ):
                    frame_masks = MaskDictionaryModel()

                    object_data = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = out_mask_logits[i] > 0.0  # .cpu().numpy()
                        object_info = ObjectInfo(
                            instance_id=out_obj_id,
                            mask=out_mask[0],
                            class_name=mask_dict.get_target_class_name(out_obj_id),
                        )
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        image_base_name = frame_names[out_frame_idx]
                        frame_masks.mask_name = f"mask_{image_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]
                        # record each object data
                        object_data[out_obj_id] = object_info_to_dict(frame_masks.labels[out_obj_id])

                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)
                    # record each frame object data
                    video_object_data.append(object_data)

        # co-tracker predict
        # load video for co-tracker
        video = read_video_from_path(meta_info["filepath"])
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_width, video_height = video.shape[-1], video.shape[-2]
        video = video.to(device)

        # get tracking result
        tracking_result = [item for i, item in enumerate(video_object_data, 1) if i % (step + 1) != 0]
        tracking_result = tracking_result[::step]

        # object disappear evaluation
        disappear_objects = get_disappear_objects(tracking_result)
        if len(disappear_objects) == 0:
            vanish_score = 1.0
            print("No disappear objects")
        else:
            # get disappear object masks
            disappear_mask_list = []
            query_frames = []
            for _idx, obj_info in enumerate(disappear_objects):
                mask = obj_info["mask"]
                frame = obj_info["first_appearance"]
                if mask.ndim != 2:
                    continue
                image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                disappear_mask_list.append(image)
                query_frames.append(frame)

            disappear_objects_count = 0
            for disappear_mask, query_frame in zip(disappear_mask_list, query_frames, strict=False):
                # load the mask from PIL image
                disappear_mask = torch.from_numpy(np.array(disappear_mask))[None, None]

                # get grid query frame
                grid_query_frame = query_frame

                pred_tracks, pred_visibility = cotracker_model(
                    video,
                    grid_size=args.grid_size,
                    grid_query_frame=grid_query_frame,
                    backward_tracking=True,
                    segm_mask=disappear_mask,
                )

                edge_vanish = is_edge_vanish(pred_tracks, pred_visibility, grid_query_frame, video_width, video_height)

                small_vanish = is_small_vanish(
                    pred_tracks, pred_visibility, grid_query_frame, video_width, video_height
                )

                disappear_detect_error = is_vanish_detect_error(pred_tracks, pred_visibility, grid_query_frame)

                if not edge_vanish and not small_vanish and not disappear_detect_error:
                    disappear_objects_count += 1
            # record disappear detection result
            vanish_score = (objects_count - disappear_objects_count) / objects_count

        # object appear evaluation
        appear_objects = get_appear_objects(tracking_result)
        if len(appear_objects) == 0:
            # if no appear item masks get 1 score
            emerge_score = 1.0
            print("No appear objects")
        else:
            # get appear object masks
            appear_mask_list = []
            query_frames = []
            for _idx, obj_info in enumerate(appear_objects):
                mask = obj_info["mask"]
                frame = obj_info["first_appearance"]
                if mask.ndim != 2:
                    continue
                image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                appear_mask_list.append(image)
                query_frames.append(frame)

            appear_objects_count = 0
            for appear_mask, query_frame in zip(appear_mask_list, query_frames, strict=False):
                # load the mask from PIL image
                appear_mask = torch.from_numpy(np.array(appear_mask))[None, None]

                # get grid query frame
                grid_query_frame = query_frame

                pred_tracks, pred_visibility = cotracker_model(
                    video,
                    grid_size=args.grid_size,
                    grid_query_frame=grid_query_frame,
                    backward_tracking=True,
                    segm_mask=appear_mask,
                )

                edge_emerge = is_edge_emerge(pred_tracks, pred_visibility, grid_query_frame, video_width, video_height)

                small_emerge = is_small_emerge(
                    pred_tracks, pred_visibility, grid_query_frame, video_width, video_height
                )

                appear_detect_error = is_emerge_detect_error(pred_tracks, pred_visibility, grid_query_frame)

                if not edge_emerge and not small_emerge and not appear_detect_error:
                    appear_objects_count += 1
            # record disappear detection result
            emerge_score = (objects_count - appear_objects_count) / objects_count

        meta_info["temporal_coherence_score"] = (vanish_score + emerge_score) / 2
        # save eval result
        with open(args.meta_info_path, "w") as f:
            json.dump(meta_infos, f, indent=4)
