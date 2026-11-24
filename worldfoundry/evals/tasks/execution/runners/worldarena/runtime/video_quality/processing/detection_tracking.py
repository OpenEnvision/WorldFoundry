import os

import av
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor

# ===== 1. SAM3 imports =====
from sam3.model_builder import build_sam3_image_model
from tqdm import tqdm


def nms_boxes(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression to remove overlapping detections.
    """
    if len(boxes) == 0:
        return [], []

    # Convert to numpy arrays
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)

    # Sort by confidence
    indices = np.argsort(scores_np)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Compute IoU
        current_box = boxes_np[current]
        other_boxes = boxes_np[indices[1:]]

        # Intersection
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Union
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_other = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_other - intersection

        iou = intersection / (union + 1e-6)

        # Keep boxes below IoU threshold
        indices = indices[1:][iou < iou_threshold]

    return boxes_np[keep].tolist(), scores_np[keep].tolist()


class GripperDetector:
    """Gripper detector; loads the model once for reuse."""

    def __init__(self, model_path=""):
        print("Loading SAM3 model...")
        try:
            # Load model once and reuse
            self.model = build_sam3_image_model(
                checkpoint_path=os.path.join(model_path, "sam3.pt"),
                bpe_path=os.path.join(model_path, "bpe_simple_vocab_16e6.txt.gz"),
            )
            self.processor = Sam3Processor(self.model)
            print("SAM3 model loaded")
        except Exception as e:
            print(f"Failed to load SAM3 model: {e}")
            raise

    def detect_frame_single_prompt(self, image, prompt="end effector"):
        """
        Detect a single image with one text prompt.
        Args:
            image: PIL.Image or numpy array
            prompt: text prompt
        Returns:
            boxes: list of boxes [[x1,y1,x2,y2], ...]
            scores: confidence list
        """
        # Convert image format
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        try:
            # Set image and encode
            inference_state = self.processor.set_image(image)

            # Set text prompt
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

            # Get detection results
            boxes = inference_state["boxes"]
            scores = inference_state["scores"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            return boxes, scores

        except Exception as e:
            print(f"Detection failed: {e}")
            return [], []

    def detect_frame(self, image, prompts=None):
        """
        Detect a single image with multiple prompts and apply NMS.
        Args:
            image: PIL.Image or numpy array
            prompts: list of prompts, defaults to ["robot arm", "end effector", "gripper"]
        Returns:
            boxes: list of boxes [[x1,y1,x2,y2], ...]
            scores: confidence list
        """
        if prompts is None:
            prompts = ["robot arm", "end effector", "gripper"]

        all_boxes = []
        all_scores = []

        # Detect for each prompt
        for prompt in prompts:
            try:
                boxes, scores = self.detect_frame_single_prompt(image, prompt)
                if len(boxes) > 0:
                    all_boxes.extend(boxes)
                    all_scores.extend(scores)
            except Exception as e:
                print(f"Prompt '{prompt}' failed: {e}")
                continue

        # Apply NMS if any detections
        if len(all_boxes) > 0:
            return nms_boxes(all_boxes, all_scores, iou_threshold=0.3)

        return [], []

    def detect_batch(self, images, prompts=None):
        """
        Batch-detect multiple images (more efficient).
        Args:
            images: list of images [PIL.Image, ...]
            prompts: list of prompts
        Returns:
            results: list of detection results [(boxes, scores), ...]
        """
        results = []
        for img in tqdm(images, desc="batch detect", unit="frame"):
            boxes, scores = self.detect_frame(img, prompts)
            results.append((boxes, scores))
        return results


def check_if_already_processed(output_video_path, output_traj_path, input_path):
    """
    Check whether a video has already been processed.
    Returns:
        True: already processed
        False: needs processing
    """
    # Check output dirs exist
    if not os.path.exists(output_video_path) or not os.path.exists(output_traj_path):
        return False

    # Check trajectory file exists and is valid
    traj_file = os.path.join(output_traj_path, "traj.npy")
    if not os.path.exists(traj_file):
        return False

    # Validate trajectory file shape
    try:
        traj_data = np.load(traj_file)
        if len(traj_data.shape) != 3 or traj_data.shape[1] != 2 or traj_data.shape[2] != 2:
            print(f"Warning: invalid trajectory shape: {traj_data.shape}")
            return False
    except Exception as e:
        print(f"Warning: failed to load trajectory: {e}")
        return False

    # Check video exists
    video_file = os.path.join(output_video_path, "video.mp4")
    if not os.path.exists(video_file):
        return False

    # Count input images
    image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # Ensure trajectory frames match image count
    if len(traj_data) != len(image_files):
        print(f"Warning: trajectory frames ({len(traj_data)}) do not match images ({len(image_files)})")
        return False

    print(f"✓ already processed: {input_path}")
    return True


def interpolate_trajectory(traj_list, max_gap=5):
    """
    Interpolate trajectory to fill missing detections.
    Args:
        traj_list: list of (x, y) or (-1, -1) for missing
        max_gap: maximum gap to interpolate
    Returns:
        interpolated trajectory list
    """
    if not traj_list:
        return traj_list

    traj_array = np.array(traj_list)
    result = traj_array.copy()

    # Find valid points (not -1)
    valid_mask = np.all(traj_array != -1, axis=1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return traj_list

    # Interpolate each missing segment
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        gap = end_idx - start_idx - 1

        # Linear interpolation if gap within limit
        if 0 < gap <= max_gap:
            start_point = traj_array[start_idx]
            end_point = traj_array[end_idx]

            # Linear interpolation
            for j in range(1, gap + 1):
                alpha = j / (gap + 1)
                interpolated_point = start_point * (1 - alpha) + end_point * alpha
                result[start_idx + j] = interpolated_point

    return result.tolist()


def smooth_trajectory(traj_list, window_size=3):
    """
    Smooth trajectory with a sliding window.
    Args:
        traj_list: trajectory list
    Returns:
        smoothed trajectory list
    """
    if len(traj_list) < window_size:
        return traj_list

    traj_array = np.array(traj_list)
    result = traj_array.copy()

    # Sliding window smoothing
    for i in range(len(traj_array)):
        start = max(0, i - window_size // 2)
        end = min(len(traj_array), i + window_size // 2 + 1)

        # Only average valid points
        window_points = traj_array[start:end]
        valid_points = window_points[np.all(window_points != -1, axis=1)]

        if len(valid_points) > 0:
            result[i] = np.mean(valid_points, axis=0)

    return result.tolist()


def process_video_with_tracking(
    input_path, output_path, detector=None, gid=None, data_type="val", saving_fps=12, force_reprocess=False
):
    """
    Use SAM3 to detect grippers and generate trajectories.
    Args:
        detector: loaded GripperDetector instance
        force_reprocess: force reprocess even if results exist
    """

    print(f"Processing: {input_path}")

    if detector is None:
        print("Error: detector must be provided")
        return False

    # ===== Create output dirs =====
    if data_type == "val":
        output_video_path = os.path.join(output_path, gid, "gripper_detection")
        output_traj_path = os.path.join(output_path, gid, "traj")
    else:
        output_video_path = os.path.join(output_path, "gripper_detection")
        output_traj_path = os.path.join(output_path, "traj")

    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_traj_path, exist_ok=True)

    # ===== Check if already processed =====
    if not force_reprocess and check_if_already_processed(output_video_path, output_traj_path, input_path):
        return True

    # ===== Prepare output video =====
    output_video_file = os.path.join(output_video_path, "video.mp4")
    output_container = av.open(output_video_file, mode="w", format="mp4")
    output_stream = output_container.add_stream("h264", rate=saving_fps)
    output_stream.width = 640
    output_stream.height = 480
    output_stream.pix_fmt = "yuv420p"

    # ===== Trajectory storage =====
    trajectory_data = []

    # Track history for visualization
    left_track_history = []  # left gripper history
    right_track_history = []  # right gripper history

    # Store last valid detections
    last_valid_left = None
    last_valid_right = None

    # Missing frame counters
    left_missing_count = 0
    right_missing_count = 0

    # Max missing frames allowed before dropping history
    MAX_MISSING_FRAMES = 10

    # ===== Gather input images =====
    image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not image_files:
        print(f"Warning: no images in {input_path}")
        output_container.close()
        return False

    print(f"Found {len(image_files)} images, start processing...")

    # ===== Preload images (optional optimization) =====
    print("Loading images into memory...")
    images = []
    for img_file in tqdm(image_files, desc="load images", unit="frame"):
        img_path = os.path.join(input_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        else:
            images.append(None)  # keep frame indices aligned

    # ===== Detection loop =====
    print("Start detecting...")
    for global_frame_idx in tqdm(range(len(images)), desc="process frames", unit="frame"):
        img = images[global_frame_idx]

        if img is None:
            # If read fails, mark both grippers missing
            left_current = (-1, -1)
            right_current = (-1, -1)

            # Append to trajectory data
            trajectory_data.append([list(left_current), list(right_current)])

            # Create blank frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            output_frame = av.VideoFrame.from_ndarray(blank_frame, format="rgb24")
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            continue

        # Prepare visualization frame
        annotated_frame = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        # ===== Detect current frame =====
        boxes, scores = detector.detect_frame(img, prompts=["robot arm", "end effector", "gripper"])

        # ===== Parse detections =====
        left_detections = []  # left gripper candidates
        right_detections = []  # right gripper candidates

        # Process each detection
        for i, (box, score) in enumerate(zip(boxes, scores, strict=False)):
            if score < 0.3:  # confidence threshold
                continue

            # Ensure box is list-like
            if isinstance(box, (np.ndarray, torch.Tensor)):
                box = box.tolist() if hasattr(box, "tolist") else list(box)

            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Normalized coordinates
            x_norm = x_center / 640.0
            y_norm = y_center / 480.0

            # Split by image center to decide left/right
            if x_center < 320:  # left half
                left_detections.append(
                    {"box": box, "score": score, "center": (x_center, y_center), "norm": (x_norm, y_norm), "index": i}
                )
            else:  # right half
                right_detections.append(
                    {"box": box, "score": score, "center": (x_center, y_center), "norm": (x_norm, y_norm), "index": i}
                )

        # ===== Handle left gripper =====
        left_current = None

        if left_detections:
            # Choose the highest-confidence left detection
            best_left = max(left_detections, key=lambda x: x["score"])
            x_norm, y_norm = best_left["norm"]
            x_center, y_center = best_left["center"]

            left_current = (float(x_norm), float(y_norm))
            last_valid_left = left_current
            left_missing_count = 0

            # Visualize left gripper (red)
            x1, y1, x2, y2 = map(int, best_left["box"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label with score
            label = f"Left: {best_left['score']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif last_valid_left is not None and left_missing_count < MAX_MISSING_FRAMES:
            # Use last valid left within tolerance
            left_current = last_valid_left
            left_missing_count += 1
        else:
            # Missing
            left_current = (-1.0, -1.0)
            left_missing_count += 1

        # ===== Handle right gripper =====
        right_current = None

        if right_detections:
            # Choose the highest-confidence right detection
            best_right = max(right_detections, key=lambda x: x["score"])
            x_norm, y_norm = best_right["norm"]
            x_center, y_center = best_right["center"]

            right_current = (float(x_norm), float(y_norm))
            last_valid_right = right_current
            right_missing_count = 0

            # Visualize right gripper (green)
            x1, y1, x2, y2 = map(int, best_right["box"])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with score
            label = f"Right: {best_right['score']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif last_valid_right is not None and right_missing_count < MAX_MISSING_FRAMES:
            # Use last valid right within tolerance
            right_current = last_valid_right
            right_missing_count += 1
        else:
            # Missing
            right_current = (-1.0, -1.0)
            right_missing_count += 1

        # ===== Update track history =====
        if left_current != (-1, -1):
            # Convert back to pixels for visualization
            x_pixel = left_current[0] * 640
            y_pixel = left_current[1] * 480
            left_track_history.append((int(x_pixel), int(y_pixel)))
        else:
            # Missing frame keeps history unchanged
            pass

        if right_current != (-1, -1):
            # Convert back to pixels for visualization
            x_pixel = right_current[0] * 640
            y_pixel = right_current[1] * 480
            right_track_history.append((int(x_pixel), int(y_pixel)))
        else:
            # Missing frame keeps history unchanged
            pass

        # ===== Draw trajectory lines =====
        # Left gripper (red)
        if len(left_track_history) > 1:
            for i in range(1, len(left_track_history)):
                cv2.line(
                    annotated_frame,
                    left_track_history[i - 1],
                    left_track_history[i],
                    (255, 0, 0),  # red in BGR
                    3,
                    cv2.LINE_AA,
                )

        # Right gripper (red)
        if len(right_track_history) > 1:
            for i in range(1, len(right_track_history)):
                cv2.line(
                    annotated_frame, right_track_history[i - 1], right_track_history[i], (255, 0, 0), 3, cv2.LINE_AA
                )

        # ===== Save trajectory data =====
        trajectory_data.append([list(left_current), list(right_current)])

        # ===== Write video frame =====
        output_frame = av.VideoFrame.from_ndarray(annotated_frame, format="rgb24")
        for packet in output_stream.encode(output_frame):
            output_container.mux(packet)

    # ===== Finalize video =====
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()

    # ===== Post-process trajectory (interpolate & smooth) =====
    print("Post-processing trajectories...")

    # Split left/right
    left_traj = [point[0] for point in trajectory_data]
    right_traj = [point[1] for point in trajectory_data]

    # Interpolate missing points
    left_traj_interp = interpolate_trajectory(left_traj, max_gap=5)
    right_traj_interp = interpolate_trajectory(right_traj, max_gap=5)

    # Smooth trajectories
    left_traj_smooth = smooth_trajectory(left_traj_interp, window_size=3)
    right_traj_smooth = smooth_trajectory(right_traj_interp, window_size=3)

    # Recombine trajectories
    final_trajectory_data = []
    for i in range(len(left_traj_smooth)):
        final_trajectory_data.append([list(left_traj_smooth[i]), list(right_traj_smooth[i])])

    trajectory_array = np.array(final_trajectory_data, dtype=np.float32)

    print(f"Trajectory shape: {trajectory_array.shape}")
    print(f"Valid left points: {np.sum(np.all(trajectory_array[:, 0] != -1, axis=1))}")
    print(f"Valid right points: {np.sum(np.all(trajectory_array[:, 1] != -1, axis=1))}")

    # Save file
    output_file = os.path.join(output_traj_path, "traj.npy")
    np.save(output_file, trajectory_array)
    print(f"Trajectory saved: {output_file}")

    return True


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--detect_gt", action="store_false", help="Whether to detect ground-truth trajectories")
    parser.add_argument(
        "--force_reprocess", action="store_true", help="Force reprocess all videos even if outputs exist"
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    # Read paths from config
    data_base = config["data"]["val_base"]
    gt_path = config["data"]["gt_path"]

    # Model path
    model_path = config.get("ckpt", {}).get("sam3_model_ckpt", "")

    print("=" * 60)
    print("SAM3 gripper detection and trajectory (improved)")
    print("Improvements:")
    print("1. Fill gaps with last valid points within tolerance")
    print("2. Trajectory lines drawn in red")
    print("3. Post-process with interpolation and smoothing")
    print(f"Force reprocess: {'yes' if args.force_reprocess else 'no'}")
    print("=" * 60)

    # ===== Load model once =====
    print("Initializing detector...")
    try:
        detector = GripperDetector(model_path=model_path)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        exit(1)

    # Counters
    processed_count = 0
    skipped_count = 0
    total_count = 0

    # Iterate tasks
    for task in sorted(os.listdir(data_base)):
        task_path = os.path.join(data_base, task)

        if not os.path.isdir(task_path):
            continue

        for episode in sorted(os.listdir(task_path)):
            if episode.endswith((".png", ".json")):
                continue

            episode_path = os.path.join(task_path, episode)

            # Process ground-truth videos
            if args.detect_gt:
                gt_episode_path = os.path.join(gt_path, task, episode)
                gt_video = os.path.join(gt_episode_path, "video")

                if os.path.exists(gt_video):
                    total_count += 1
                    print(f"\n[GT] task: {task}, episode: {episode}")
                    if process_video_with_tracking(
                        input_path=gt_video,
                        output_path=gt_episode_path,
                        detector=detector,
                        gid=None,
                        data_type="gt",
                        force_reprocess=args.force_reprocess,
                    ):
                        processed_count += 1
                    else:
                        skipped_count += 1

            # Process generated videos
            for gid in sorted(os.listdir(episode_path)):
                input_path = os.path.join(episode_path, gid, "video")

                if os.path.exists(input_path):
                    total_count += 1
                    print(f"\n[GEN] task: {task}, episode: {episode}, GID: {gid}")
                    if process_video_with_tracking(
                        input_path=input_path,
                        output_path=episode_path,
                        detector=detector,
                        gid=gid,
                        data_type="val",
                        force_reprocess=args.force_reprocess,
                    ):
                        processed_count += 1
                    else:
                        skipped_count += 1

    print("\n" + "=" * 60)
    print("Processing done!")
    print(f"Total videos: {total_count}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print("=" * 60)
