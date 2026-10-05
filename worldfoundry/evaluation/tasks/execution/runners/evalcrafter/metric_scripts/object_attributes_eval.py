import os
import cv2
import numpy as np
import torch
import gc
import re
import logging
import time
import argparse
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.base_models.perception_core.segment.sam_v1.paths import checkpoint_path as sam_checkpoint_path
from worldfoundry.base_models.perception_core.tracking.aot.paths import checkpoint_path as deaot_checkpoint_path
from worldfoundry.base_models.perception_core.tracking.track_anything import SegTracker


def _evalcrafter_assets_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_EVALCRAFTER_ASSETS_ROOT", bundled_benchmark_asset("evalcrafter"))).resolve()


sam_args = {
    'sam_checkpoint': str(sam_checkpoint_path("vit_b")),
    'model_type': "vit_b",
    'generator_args': {
        'points_per_side': 16,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    },
    'gpu_id': 0,
}
aot_args = {
    'phase': 'PRE_YTB_DAV',
    'model': 'r50_deaotl',
    'model_path': str(deaot_checkpoint_path()),
    'long_term_mem_gap': 9999,
    'max_len_long_term': 9999,
    'gpu_id': 0,
}
segtracker_args = {
    'sam_gap': 10,
    'min_area': 200,
    'max_obj_num': 255,
    'min_new_obj_iou': 0.8,
}

def create_directories(path):    
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

    return path

def video_detection(io_args, segtracker_args, sam_args, aot_args, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image):    
    cap = cv2.VideoCapture(io_args['input_video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    pred_list = []
    masked_pred_list = []
    det_count_one = []
    det_count_frames = []
    frames = []

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = segtracker_args['sam_gap']
    frame_idx = 0
    frame_idx_processed = 0
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()
    
    Num = 5  # Set the value of Num as per your requirement

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if (frame_idx % Num) == 0:
                frame_idx_processed+=1
                if not ret:
                    break
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if frame_idx == 0:
                    pred_mask, _, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    segtracker.add_reference(frame, pred_mask)
                elif (frame_idx_processed % (sam_gap//Num)) == 0:
                    seg_mask, _, _ = segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    track_mask = segtracker.track(frame)
                    new_obj_mask, _ = segtracker.find_new_objs(track_mask, seg_mask)
                    if np.sum(new_obj_mask > 0) >  frame.shape[0] * frame.shape[1] * 0.4:
                        new_obj_mask = np.zeros_like(new_obj_mask)
                    pred_mask = track_mask + new_obj_mask
                    segtracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = segtracker.track(frame,update_memory=True)
                pred_list.append(pred_mask)
                frames.append(frame)

                obj_ids = np.unique(pred_mask)
                obj_ids = obj_ids[obj_ids!=0]
                det_count_frames.append(len(obj_ids))

                print("processed frame {}, obj_num {}".format(frame_idx_processed,segtracker.get_obj_num()),end='\r')
            frame_idx += 1
        cap.release()
        
    return frames, pred_list, det_count_frames 

def detect_color_hue_based(hue_value):
    if hue_value < 15:
        color = "red"
    elif hue_value < 22:
        color = "orange"
    elif hue_value < 39:
        color = "yellow"
    elif hue_value < 78:
        color = "green"
    elif hue_value < 131:
        color = "blue"
    else:
        color = "red"

    return color



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='celebrity_id_score', help="Specify the metric to be used")
    args = parser.parse_args()

    dir_videos = args.dir_videos
    metric = args.metric

    assets_root = _evalcrafter_assets_root()
    dir_prompts = assets_root / "prompts"
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    prompt_paths = [str(dir_prompts / (os.path.splitext(os.path.basename(x))[0] + '.txt')) for x in video_paths]

     # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"../../results", exist_ok=True)
    # Set up logging
    log_file_path = f"../../results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"../../results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    

    sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        }

    # For every sam_gap frames, we use SAM to find new objects and add them for tracking
    segtracker_args = {
        'sam_gap': 49, # the interval to run sam to segment new objects
        'min_area': 200, # minimal mask area to add a new mask as a new object
        'max_obj_num': 255, # maximal object number to track in a video
        'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
    }

    # Set Text args
    '''
    parameter:
        grounding_caption: Text prompt to detect objects in key-frames
        box_threshold: threshold for box 
        text_threshold: threshold for label(text)
        box_size_threshold: If the size ratio between the box and the frame is larger than the box_size_threshold, the box will be ignored. This is used to filter out large boxes.
        reset_image: reset the image embeddings for SAM
    '''
    # grounding_caption = "car.suv"
    grounding_caption = "" #must have this class in the image, otherwise go wrong
    box_threshold, text_threshold, box_size_threshold, reset_image = 0.6, 0.5, 0.5, True 

    # COCO dataset
    keywords = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] #coco classes, https://github.com/matlab-deep-learning/Object-Detection-Using-Pretrained-YOLO-v2/blob/main/+helper/coco-classes.txt
        
    

    prompt_paths_object = []
    video_paths_object = []
    det_count = []
    keyword_object = []
    prompt_object = []
    object_metrics = []

    det_num = 0
    count_score = 0
    pos_num = 0
    vid_num = 0
    last_video_name = ' '
    start_time = time.time()  # Start the timer
    logger.info(f"Total video num {len(prompt_paths)}")

    import json
    # Load the JSON data from the file
    with open(assets_root / "metadata.json", "r") as infile:
        data = json.load(infile)
    # Extract the dictionaries
    face_vid = {}
    text_vid = {}
    color_vid = {}
    count_vid = {}
    amp_vid = {}
    action_vid = {}
    for item_key, item_value in data.items():
        attributes = item_value["attributes"]
        face = attributes.get("face", "")
        text = attributes.get("text", "")
        color = attributes.get("color", "")
        count = attributes.get("count", "")
        amp = attributes.get("amp", "")
        action = attributes.get("action", "")
        if face:
            face_vid[item_key] = face
        if text:
            text_vid[item_key] = text
        if color:
            color_vid[item_key] = color
        if count:
            count_vid[item_key] = count
        if amp:
            amp_vid[item_key] = amp
        if action:
            action_vid[item_key] = action


    scores = []
    count = 0

    for i in range(len(prompt_paths)):
        with open(prompt_paths[i], 'r', encoding='utf-8') as f:
            data = f.read()
        
        if metric == 'detection_score':
            for keyword in keywords:
                num = len(re.findall(r'\b' + re.escape(keyword) + r'(s|es)?\b', data, re.IGNORECASE))      
                if num > 0:
                    video_name = os.path.splitext(os.path.basename(prompt_paths[i]))[0]
                    if video_name != last_video_name:
                        vid_num +=1
                    last_video_name = video_name
                    io_args = {
                        'input_video': os.path.join(dir_videos, f'{video_name}.mp4'),
                        'output_mask_dir': f'../../results/{metric+timestamp}/{keyword}/masks/{video_name}', # save pred masks
                        'original_video': f'../../results/{metric+timestamp}/{keyword}/original_{video_name}.mp4', 
                        'output_video': f'../../results/{metric+timestamp}/{keyword}/mask_{video_name}.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
                        'output_gif': f'../../results/{metric+timestamp}/{keyword}/gif_{video_name}.gif', # mask visualization
                    }
                    grounding_caption = keyword
                    _, pred_list, det_count_frames = video_detection(io_args, segtracker_args, sam_args, aot_args, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                    det_num += 1 
                    det_frames = [] 
                    for k in range(len(det_count_frames)):
                        if det_count_frames[k] > 0:
                            det_frames.append(1)
                        else:
                            det_frames.append(0)
                    det_frames = np.array(det_frames)
                    det_avg = np.sum(det_frames) / det_frames.shape[0]
                    
                    score = det_avg
                    scores.append(score)
                    average_score = sum(scores) / len(scores)
                    # count+=1
                    logger.info(f"Vid: {video_name},  Current {metric}: {score}, Current avg. {metric}: {average_score},  ")
                else:
                    score = None

        elif metric == 'color_score':
            video_name = os.path.splitext(os.path.basename(prompt_paths[i]))[0]
            if  video_name in color_vid.keys():
                grounding_caption = ''
                gt_color = color_vid[video_name].split()[0]
                for j in range(len(color_vid[video_name].split())-1):
                    grounding_caption += (color_vid[video_name].split()[j+1] + ' ')
                grounding_caption = grounding_caption[:-1]
                keyword = grounding_caption

                io_args = {
                'input_video': os.path.join(dir_videos, f'{video_name}.mp4'),
                'output_mask_dir': f'../../results/{metric+timestamp}/{keyword}/masks/{video_name}', # save pred masks
                'original_video': f'../../results/{metric+timestamp}/{keyword}/original_{video_name}.mp4', 
                'output_video': f'../../results/{metric+timestamp}/{keyword}/mask_{video_name}.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
                'output_gif': f'../../results/{metric+timestamp}/{keyword}/gif_{video_name}.gif', # mask visualization
            }
                # path = create_directories(io_args['output_mask_dir'])
                frames, pred_list, det_count_frames = video_detection(io_args, segtracker_args, sam_args, aot_args, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                frames_colors = []
                for k in range(len(frames)):
                    hsv_frame = cv2.cvtColor(frames[k], cv2.COLOR_RGB2HSV)
                    hsv_frame = hsv_frame[:, :, 0]
                    hsv_frame_masked = np.multiply(hsv_frame, pred_list[k])
                    avg_hue = hsv_frame_masked.sum() / np.count_nonzero(hsv_frame_masked)  # average hue component
                    detected_color = detect_color_hue_based(avg_hue)
                    if detected_color == gt_color:
                        frames_colors.append(1)
                    else:
                        frames_colors.append(0)
                frames_colors = np.array(frames_colors)
                frames_color = np.sum(frames_colors) / frames_colors.shape[0]
                score = frames_color
            else:
                score = None
        
        elif metric == 'count_score':
            video_name = os.path.splitext(os.path.basename(prompt_paths[i]))[0]
            if  video_name in count_vid.keys():
                grounding_caption = ''
                gt_count = count_vid[video_name].split()[0]
                for j in range(len(count_vid[video_name].split())-1):
                    grounding_caption += (count_vid[video_name].split()[j+1] + ' ')
                grounding_caption = grounding_caption[:-1]
                keyword = grounding_caption
                
                io_args = {
                'input_video': os.path.join(dir_videos, f'{video_name}.mp4'),
                'output_mask_dir': f'../../results/{metric+timestamp}/{keyword}/masks/{video_name}', # save pred masks
                'original_video': f'../../results/{metric+timestamp}/{keyword}/original_{video_name}.mp4', 
                'output_video': f'../../results/{metric+timestamp}/{keyword}/mask_{video_name}.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
                'output_gif': f'../../results/{metric+timestamp}/{keyword}/gif_{video_name}.gif', # mask visualization
            }
                # path = create_directories(io_args['output_mask_dir'])

                frames, pred_list, det_count_frames = video_detection(io_args, segtracker_args, sam_args, aot_args, grounding_caption, box_threshold, text_threshold, box_size_threshold, reset_image)
                
                det_num += 1 
                det_count_frames = np.array(det_count_frames).astype('float64') 
                det_count_diff_frames = np.array(np.abs(det_count_frames - float(gt_count))) /  float(gt_count) # normalize first
                det_count_diff_avg = np.sum(det_count_diff_frames) / det_count_diff_frames.shape[0]
                if det_count_diff_avg > 1:
                    det_count_diff_avg = 1
                score = 1- det_count_diff_avg
            else:
                score = None
        if score is not None and metric != 'detection_score':
            scores.append(score)
            average_score = sum(scores) / len(scores)
            # count+=1
            logger.info(f"Vid: {os.path.basename(prompt_paths[i])},  Current {metric}: {score}, Current avg. {metric}: {average_score},  ")
            
    # Calculate the average SD score across all video-text pairs
    average_score = sum(scores) / len(scores)
    logger.info(f"Final average {metric}: {average_score},  Total videos: {len(scores)}")
