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


class Video_preprocess:
    def __init__(self):
        pass

    def extract_frames(self, video_path, num_frames=9, seed=None):
        """
        Extract consecutive frames from a video starting from a random position.

        Args:
            video_path: Path to the video file
            num_frames: Number of consecutive frames to extract (default: 9)
            seed: Random seed for reproducible frame selection
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:  # Handle case where video can't be opened or is empty
            print(f"Warning: Video {video_path} has 0 frames or could not be opened.")
            cap.release()  # Release the video capture object
            return []

        if total_frames <= num_frames:
            # If video has fewer frames than requested, take all available frames
            frame_indices = np.arange(total_frames)
        else:
            # Randomly select a starting position for consecutive frames
            if seed is not None:
                np.random.seed(seed)

            # Ensure we don't start too late (leave room for num_frames consecutive frames)
            max_start_frame = total_frames - num_frames
            start_frame = np.random.randint(0, max_start_frame + 1)
            frame_indices = np.arange(start_frame, start_frame + num_frames)

            print(
                f"Randomly selected consecutive frames {start_frame} to {start_frame + num_frames - 1} from {total_frames} total frames (seed: {seed})"
            )

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_all_frame_groups(self, video_path, num_frames=9):
        """
        Extract all possible consecutive frame groups from a video starting from the beginning.

        Args:
            video_path: Path to the video file
            num_frames: Number of consecutive frames per group (default: 9)

        Returns:
            List of frame groups, where each group contains num_frames consecutive frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Warning: Video {video_path} has 0 frames or could not be opened.")
            cap.release()
            return []

        frame_groups = []
        current_start = 0

        print(f"Extracting all {num_frames}-frame groups from {total_frames} total frames")

        while current_start + num_frames <= total_frames:
            frames = []
            # Extract frames for current group
            for i in range(current_start, current_start + num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) == num_frames:
                frame_groups.append(frames)
                print(
                    f"  Extracted group {len(frame_groups)}: frames {current_start} to {current_start + num_frames - 1}"
                )

            current_start += num_frames  # Move to next non-overlapping group

        cap.release()
        print(f"Total frame groups extracted: {len(frame_groups)}")
        return frame_groups

    def rgb_to_yuv(self, frame):
        """
        Convert a frame from RGB to YUV.
        """
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def merge_grid(self, image_list):
        """
        Merge a list of images into a 3x3 grid.
        """
        if len(image_list) != 9:
            print(f"Warning: merge_grid expects 9 images, got {len(image_list)}. Returning None or first image.")

            if not image_list:
                return np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder for no images
            if len(image_list) < 9:  # If less than 9, pad with black images or repeat
                print("Adjusting grid merging for fewer than 9 images. This might be visually unaligned.")
                while len(image_list) < 9:  # Pad with copies of the last image or black images
                    image_list.append(image_list[-1] if image_list else np.zeros_like(image_list[0]))

        try:
            # Create 3x3 grid
            row1 = np.concatenate((image_list[0], image_list[1], image_list[2]), axis=1)
            row2 = np.concatenate((image_list[3], image_list[4], image_list[5]), axis=1)
            row3 = np.concatenate((image_list[6], image_list[7], image_list[8]), axis=1)
            grid = np.concatenate((row1, row2, row3), axis=0)
        except ValueError as e:
            print(f"Error merging grid: {e}. Images might have incompatible dimensions.")
            # Fallback: return a placeholder or the first image
            return image_list[0] if image_list else np.zeros((100, 100, 3), dtype=np.uint8)
        return grid

    def read_video_path(self, video_path):
        """
        Read a video path and return a list of video files and the base video path.
        Note: Files are not sorted here, sorting should be done externally based on JSON data.
        """
        if os.path.isdir(video_path):
            video_files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
        elif os.path.isfile(video_path):
            video_files = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        else:
            print(f"Error: Video path {video_path} is neither a file nor a directory.")
            return [], ""
        # Removed automatic sorting - will be done externally with JSON data
        return video_files, video_path

    def convert_video_to_frames(self, video_path, num_frames=9, seed=None):
        """
        Convert a video to frames.
        """
        video_files, base_video_path = self.read_video_path(video_path)
        if not video_files:
            return None
        print(f"Start converting video to {num_frames} consecutive frames from path: {base_video_path}")

        output_dir_name = os.path.basename(base_video_path) if os.path.basename(base_video_path) else "videos"
        output_path = os.path.join(os.path.dirname(base_video_path), "frames", output_dir_name)
        os.makedirs(output_path, exist_ok=True)

        for v_file in video_files:
            vid_id = os.path.splitext(v_file)[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_full_path = os.path.join(base_video_path, v_file)
            frames = self.extract_frames(vid_full_path, num_frames=num_frames, seed=seed)
            if not frames:
                print(f"No frames extracted for {vid_full_path}, skipping frame saving.")
                continue
            for frame_count, frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f"{vid_id}_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, frame)
        # print("Finish converting from path: ", base_video_path)
        # print("Video frames stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path, num_image=9, seed=None):
        """
        Convert a video to a grid of images using randomly selected consecutive frames.
        """
        video_files, base_video_path = self.read_video_path(video_path)
        if not video_files:
            return None
        # print(f"Start converting video to image grid with {num_image} consecutive frames from path: {base_video_path}")

        output_dir_name = os.path.basename(base_video_path) if os.path.basename(base_video_path) else "videos"
        output_path = os.path.join(os.path.dirname(base_video_path), "image_grid", output_dir_name)
        os.makedirs(output_path, exist_ok=True)

        for v_file in video_files:
            vid_id = os.path.splitext(v_file)[0]
            vid_full_path = os.path.join(base_video_path, v_file)
            # Extract consecutive frames starting from a random position
            extracted_frames = self.extract_frames(vid_full_path, num_frames=num_image, seed=seed)
            if not extracted_frames:
                print(f"No frames extracted for {vid_full_path}, cannot create grid.")
                continue

            # Use all extracted consecutive frames for the grid
            grid_frames_selection = extracted_frames

            # Pad grid_frames_selection if it's less than 9 and merge_grid expects 9
            while len(grid_frames_selection) > 0 and len(grid_frames_selection) < 9:
                grid_frames_selection.append(grid_frames_selection[-1])  # Repeat last frame
            if not grid_frames_selection:  # If it became empty
                print(f"Frame selection resulted in empty list for video {vid_id}. Skipping grid.")
                continue

            grid_image = self.merge_grid(grid_frames_selection)
            grid_filename = os.path.join(output_path, f"{vid_id}.png")
            cv2.imwrite(grid_filename, grid_image)
        #     print(f"Created grid for {vid_id} using {len(grid_frames_selection)} consecutive frames")
        # print("Finish converting from path: ", base_video_path)
        # print("Image grid stored in: ", output_path)
        return output_path


def load_image(image_file):
    """
    Load an image from a file.
    """
    if image_file.startswith(("http", "https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    """
    Load a list of images from a list of file paths.
    """
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def extract_scores_from_output(text_output):
    """
    Extract scores from the new format:
    Quality: 4
    Realism: 3
    Relevance: 5
    Consistency: 4
    """
    scores = {}
    lines = text_output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                metric = parts[0].strip().lower()
                try:
                    score = int(parts[1].strip())
                    if 1 <= score <= 5:  # Validate score range
                        scores[metric] = score
                except ValueError:
                    continue

    # Extract individual scores
    quality_score = scores.get("quality", 0)
    realism_score = scores.get("realism", 0)
    relevance_score = scores.get("relevance", 0)
    consistency_score = scores.get("consistency", 0)

    # Create a summary string for logging
    result_summary = f"Quality:{quality_score}, Realism:{realism_score}, Relevance:{relevance_score}, Consistency:{consistency_score}"

    return result_summary, quality_score, realism_score, relevance_score, consistency_score


def extract_single_score(text_output, dimension_name):
    """
    Extract a single score from text output for a specific dimension.
    Enhanced to handle multiple formats:
    1. Same line: "Relevance: 5"
    2. Separate lines: "Relevance:\n5"
    3. Flexible matching
    """
    import re

    lines = text_output.strip().split("\n")

    # Method 1: Look for dimension name followed by colon and score on same line
    for line in lines:
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                metric = parts[0].strip().lower()
                if dimension_name.lower() in metric:
                    try:
                        score_text = parts[1].strip()
                        # Handle cases like "Quality: [score 1-5]" or just "Quality: 5"
                        score_match = re.search(r"\b([1-5])\b", score_text)
                        if score_match:
                            score = int(score_match.group(1))
                            if 1 <= score <= 5:
                                return score
                    except ValueError:
                        continue

    # Method 2: Look for dimension name with colon, then check next line for score
    for i, line in enumerate(lines):
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) >= 1:
                metric = parts[0].strip().lower()
                if dimension_name.lower() in metric:
                    # Check if score is on the same line (after colon)
                    if len(parts) == 2 and parts[1].strip():
                        try:
                            score_text = parts[1].strip()
                            score_match = re.search(r"\b([1-5])\b", score_text)
                            if score_match:
                                score = int(score_match.group(1))
                                if 1 <= score <= 5:
                                    return score
                        except ValueError:
                            pass

                    # Check next line for score
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        try:
                            score_match = re.search(r"\b([1-5])\b", next_line)
                            if score_match:
                                score = int(score_match.group(1))
                                if 1 <= score <= 5:
                                    return score
                        except ValueError:
                            continue

    # Method 3: More flexible search - look for dimension name anywhere followed by a score
    dimension_pattern = re.compile(r"\b" + re.escape(dimension_name.lower()) + r"\b", re.IGNORECASE)

    for i, line in enumerate(lines):
        if dimension_pattern.search(line):
            # Found dimension name, look for score in this line and next few lines
            search_lines = lines[i : min(i + 3, len(lines))]  # Check current line and next 2 lines
            for search_line in search_lines:
                score_match = re.search(r"\b([1-5])\b", search_line)
                if score_match:
                    score = int(score_match.group(1))
                    if 1 <= score <= 5:
                        return score

    # If no valid score found, return 0
    print(f"Warning: Could not extract {dimension_name} score from output: {text_output}")
    return 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sort_files_by_json_number(file_list, prompts_data):
    """
    Sort files based on the 'number' field in the JSON prompts_data.
    If a file doesn't have a corresponding entry, it will be placed at the end.
    """

    def get_sort_key(filename):
        # Extract filename without extension for matching
        file_id = os.path.splitext(filename)[0]

        # Find corresponding entry in prompts_data
        for item in prompts_data:
            # Try different matching strategies
            if (
                str(item.get("number", "")) == file_id
                or str(item.get("id", "")) == file_id
                or str(item.get("name", "")) == file_id
            ):
                try:
                    # Return the number as integer for proper sorting
                    return int(item.get("number", 9999))
                except (ValueError, TypeError):
                    # If number can't be converted to int, use 9999 as fallback
                    return 9999

        # If no match found, try to extract number from filename
        import re

        number_match = re.search(r"(\d+)", file_id)
        if number_match:
            try:
                return int(number_match.group(1))
            except ValueError:
                pass

        # If no number found anywhere, place at end
        return 9999

    # Sort files based on the number field
    sorted_files = sorted(file_list, key=get_sort_key)

    print("Files sorted by JSON number field:")
    for i, filename in enumerate(sorted_files[:5]):  # Show first 5 files
        sort_key = get_sort_key(filename)
        print(f"  {i + 1}. {filename} (number: {sort_key})")
    if len(sorted_files) > 5:
        print(f"  ... and {len(sorted_files) - 5} more files")

    return sorted_files


def eval_model(args):
    # Check if pre-generated image grids are provided
    use_pregenerated_grids = args.image_grid_path is not None

    if use_pregenerated_grids:
        image_grid_path = args.image_grid_path
        video_preprocess = None  # Won't need video preprocessing
    else:
        # We'll generate grids dynamically for each iteration
        video_path = args.video_path
        if not video_path or not os.path.exists(video_path):
            print(f"Error: Video path '{video_path}' is invalid or does not exist.")
            return None
        video_preprocess = Video_preprocess()
        image_grid_path = None  # Will be set per iteration

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    try:
        with open(args.read_prompt_file) as json_data_file:
            prompts_data = json.load(json_data_file)
    except FileNotFoundError:
        print(f"Error: Prompt file {args.read_prompt_file} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.read_prompt_file}.")
        return None

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    csv_filename = f"{args.t2v_model}_video_assessment_scores.csv"
    csv_path = os.path.join(output_path, csv_filename)

    line_count = 0
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as csvreader_file:
            reader = csv.reader(csvreader_file)
            lines = list(reader)
            line_count = len(lines)

    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if line_count == 0:
            header = ["grid_image_name", "original_t2v_prompt"]
            # For frame groups evaluation, we store MINIMUM results across groups/iterations
            header.extend(
                [
                    "evaluation_description",
                    "quality_result",
                    "min_quality_score",
                    "realism_result",
                    "min_realism_score",
                    "relevance_result",
                    "min_relevance_score",
                    "consistency_result",
                    "min_consistency_score",
                    "total_min_score",
                ]
            )
            header.extend(["all_group_scores_list", "final_min_score"])
            csv_writer.writerow(header)

        if use_pregenerated_grids:
            # Use pre-generated image grids (original behavior)
            grid_images_dir = image_grid_path
            if not os.path.isdir(grid_images_dir):
                print(f"Error: Image grid path '{grid_images_dir}' is not a directory.")
                return csv_path

            grid_images = [f for f in os.listdir(grid_images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            grid_images = sort_files_by_json_number(grid_images, prompts_data)
            print(f"Found {len(grid_images)} pre-generated grid images in {grid_images_dir}")

            evaluated_count = max(line_count - 1, 0)

            for i in range(evaluated_count, len(grid_images)):
                grid_image_name = grid_images[i]

                # Try to match grid_image_name to an entry in prompts_data
                video_id_from_filename = os.path.splitext(grid_image_name)[0]
                current_prompt_entry = next(
                    (item for item in prompts_data if str(item.get("id", item.get("name"))) == video_id_from_filename),
                    None,
                )

                if current_prompt_entry is None or "prompt" not in current_prompt_entry:
                    print(
                        f"Warning: Could not find matching prompt for image {grid_image_name} in {args.read_prompt_file}. Using default prompt."
                    )
                    original_t2v_prompt = "A general video."  # Fallback prompt
                else:
                    original_t2v_prompt = current_prompt_entry["explanation"]

                # print(f"\nProcessing: {grid_image_name} (Original explanation: {original_t2v_prompt})")

                # For pre-generated images, evaluate each iteration and then average (similar to old behavior)
                iteration_scores_list = []
                iteration_details = []

                for iteration in range(args.iterations):  # iterations with different seeds
                    set_seed(args.seed + iteration)
                    print(f"--- Iteration {iteration + 1}/{args.iterations} (Seed: {args.seed + iteration}) ---")

                    # Use pre-generated grid image
                    image_files = [os.path.join(grid_images_dir, grid_image_name)]
                    try:
                        images = load_images(image_files)
                        if not images or images[0] is None:
                            print(f"Warning: Could not load image {image_files[0]}. Recording error for iteration.")
                            continue
                    except Exception as e:
                        print(f"Error loading image {image_files[0]}: {e}. Recording error for iteration.")
                        continue

                    # Process images for the model
                    image_sizes = [x.size for x in images]
                    images_processed = process_images(images, image_processor, model.config)
                    if isinstance(images_processed, list):
                        images_tensor = torch.stack(images_processed)
                    else:
                        images_tensor = images_processed
                    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

                    # Get original prompt and explanation
                    original_prompt = current_prompt_entry.get("prompt", "N/A") if current_prompt_entry else "N/A"
                    explanation = original_t2v_prompt

                    # Evaluate using the separate dimension evaluation system
                    (
                        quality_score,
                        relevance_score,
                        consistency_score,
                        realism_score,
                        quality_output,
                        relevance_output,
                        consistency_output,
                        realism_output,
                    ) = evaluate_all_dimensions(
                        images_tensor,
                        image_sizes,
                        original_prompt,
                        explanation,
                        iteration,
                        args,
                        tokenizer,
                        model,
                        IMAGE_TOKEN_INDEX,
                    )

                    iteration_total_score = quality_score + realism_score + relevance_score + consistency_score
                    iteration_scores_list.append(iteration_total_score)

                    # Store details for this iteration
                    iteration_details.append(
                        {
                            "iteration": iteration + 1,
                            "quality_score": quality_score,
                            "relevance_score": relevance_score,
                            "consistency_score": consistency_score,
                            "realism_score": realism_score,
                            "total_score": iteration_total_score,
                            "quality_output": quality_output,
                            "relevance_output": relevance_output,
                            "consistency_output": consistency_output,
                            "realism_output": realism_output,
                        }
                    )

                    # print(f"  ✅ Iteration {iteration+1} RESULTS:")
                    # print(f"     📊 Quality: {quality_score}/5")
                    # print(f"     🎯 Relevance: {relevance_score}/5")
                    # print(f"     🔄 Consistency: {consistency_score}/5")
                    # print(f"     🌟 Realism: {realism_score}/5")
                    # print(f"     📈 Total Score: {iteration_total_score}/20")
                    # print("=" * 60)

                # Calculate MINIMUM scores across all iterations
                valid_scores = [s for s in iteration_scores_list if isinstance(s, (int, float))]
                if valid_scores and iteration_details:
                    final_min_score = min(valid_scores)

                    # Calculate minimum for each dimension
                    min_quality = min([d["quality_score"] for d in iteration_details])
                    min_relevance = min([d["relevance_score"] for d in iteration_details])
                    min_consistency = min([d["consistency_score"] for d in iteration_details])
                    min_realism = min([d["realism_score"] for d in iteration_details])

                    # print(f"\n🎯 FINAL MINIMUM SCORES (across {len(iteration_details)} iterations):")
                    # print(f"   📊 Min Quality: {min_quality}/5")
                    # print(f"   🎯 Min Relevance: {min_relevance}/5")
                    # print(f"   🔄 Min Consistency: {min_consistency}/5")
                    # print(f"   🌟 Min Realism: {min_realism}/5")
                    # print(f"   📉 Total Minimum Score: {final_min_score}/20")

                    # Use first iteration's details as representative
                    first_iteration = iteration_details[0]
                    row_data_for_csv = [grid_image_name, original_t2v_prompt]
                    row_data_for_csv.extend(
                        [
                            f"pre_generated_min_{len(iteration_details)}iterations",
                            first_iteration["quality_output"][:100] + "..."
                            if len(first_iteration["quality_output"]) > 100
                            else first_iteration["quality_output"],
                            min_quality,
                            first_iteration["realism_output"][:100] + "..."
                            if len(first_iteration["realism_output"]) > 100
                            else first_iteration["realism_output"],
                            min_realism,
                            first_iteration["relevance_output"][:100] + "..."
                            if len(first_iteration["relevance_output"]) > 100
                            else first_iteration["relevance_output"],
                            min_relevance,
                            first_iteration["consistency_output"][:100] + "..."
                            if len(first_iteration["consistency_output"]) > 100
                            else first_iteration["consistency_output"],
                            min_consistency,
                            final_min_score,
                        ]
                    )
                else:
                    final_min_score = "bad_scores_all_iterations"
                    row_data_for_csv = [grid_image_name, original_t2v_prompt]
                    row_data_for_csv.extend(
                        ["evaluation_error", "0:0:0:0", 0, "0:0:0:0", 0, "0:0:0:0", 0, "0:0:0:0", 0, 0]
                    )

                row_data_for_csv.extend([str(iteration_scores_list), final_min_score])
                csv_writer.writerow(row_data_for_csv)
                csvfile.flush()
                # print(f"Finished {grid_image_name}. Minimum Score across {len(iteration_details)} iterations: {final_min_score}")

        else:
            # Generate grids dynamically from video files
            video_files, base_video_path = video_preprocess.read_video_path(args.video_path)
            if not video_files:
                print(f"Error: No video files found in {args.video_path}")
                return csv_path

            # Sort video files based on JSON number field
            video_files = sort_files_by_json_number(video_files, prompts_data)

            print(f"Found {len(video_files)} video files in {base_video_path}")
            evaluated_count = max(line_count - 1, 0)

            for i in range(evaluated_count, len(video_files)):
                video_file = video_files[i]
                grid_image_name = os.path.splitext(video_file)[0] + ".png"  # Virtual grid image name

                # Try to match grid_image_name to an entry in prompts_data
                video_id_from_filename = os.path.splitext(grid_image_name)[0]
                current_prompt_entry = next(
                    (item for item in prompts_data if str(item.get("id", item.get("name"))) == video_id_from_filename),
                    None,
                )

                if current_prompt_entry is None or "prompt" not in current_prompt_entry:
                    print(
                        f"Warning: Could not find matching prompt for image {grid_image_name} in {args.read_prompt_file}. Using default prompt."
                    )
                    original_t2v_prompt = "A general video."  # Fallback prompt
                else:
                    original_t2v_prompt = current_prompt_entry["explanation"]

                print(f"\nProcessing: {grid_image_name} (Original explanation: {original_t2v_prompt})")

                # Extract all possible 9-frame groups from the video
                video_full_path = os.path.join(base_video_path, video_file)
                frame_groups = video_preprocess.extract_all_frame_groups(video_full_path, num_frames=9)

                if not frame_groups:
                    print(f"No frame groups extracted for {video_full_path}. Skipping video.")
                    # Write error entry for this video
                    row_data_for_csv = [grid_image_name, original_t2v_prompt]
                    row_data_for_csv.extend(
                        [
                            "no_frame_groups_error",
                            "0:0:0:0",
                            0,
                            "0:0:0:0",
                            0,
                            "0:0:0:0",
                            0,
                            "0:0:0:0",
                            0,
                            0,
                            "[]",
                            "no_frame_groups_error",
                        ]
                    )
                    csv_writer.writerow(row_data_for_csv)
                    csvfile.flush()
                    continue

                # Evaluate each frame group
                all_group_scores = []
                group_details = []

                for group_idx, frame_group in enumerate(frame_groups):
                    print(f"\n--- Evaluating Frame Group {group_idx + 1}/{len(frame_groups)} ---")

                    # Create grid image from this frame group
                    grid_image = video_preprocess.merge_grid(frame_group)

                    # Save grid image if save_grid_path is provided
                    if args.save_grid_path:
                        os.makedirs(args.save_grid_path, exist_ok=True)
                        grid_filename = f"{os.path.splitext(video_file)[0]}_group{group_idx + 1:02d}.png"
                        grid_filepath = os.path.join(args.save_grid_path, grid_filename)
                        cv2.imwrite(grid_filepath, grid_image)
                        print(f"    💾 Saved grid image: {grid_filepath}")

                    # Convert opencv image to PIL Image for processing
                    grid_image_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(grid_image_rgb)
                    images = [pil_image]

                    # Process images for the model
                    image_sizes = [x.size for x in images]
                    images_processed = process_images(images, image_processor, model.config)
                    if isinstance(images_processed, list):
                        images_tensor = torch.stack(images_processed)
                    else:
                        images_tensor = images_processed
                    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

                    # Get original prompt and explanation
                    original_prompt = current_prompt_entry.get("prompt", "N/A") if current_prompt_entry else "N/A"
                    explanation = original_t2v_prompt

                    # Evaluate using the separate dimension evaluation system
                    (
                        quality_score,
                        relevance_score,
                        consistency_score,
                        realism_score,
                        quality_output,
                        relevance_output,
                        consistency_output,
                        realism_output,
                    ) = evaluate_all_dimensions(
                        images_tensor,
                        image_sizes,
                        original_prompt,
                        explanation,
                        group_idx,
                        args,
                        tokenizer,
                        model,
                        IMAGE_TOKEN_INDEX,
                    )

                    group_total_score = quality_score + realism_score + relevance_score + consistency_score
                    all_group_scores.append(group_total_score)

                    # Store details for this group
                    group_details.append(
                        {
                            "group_idx": group_idx + 1,
                            "quality_score": quality_score,
                            "relevance_score": relevance_score,
                            "consistency_score": consistency_score,
                            "realism_score": realism_score,
                            "total_score": group_total_score,
                            "quality_output": quality_output,
                            "relevance_output": relevance_output,
                            "consistency_output": consistency_output,
                            "realism_output": realism_output,
                        }
                    )

                    # print(f"  ✅ Group {group_idx + 1} RESULTS:")
                    # print(f"     📊 Quality: {quality_score}/5")
                    # print(f"     🎯 Relevance: {relevance_score}/5")
                    # print(f"     🔄 Consistency: {consistency_score}/5")
                    # print(f"     🌟 Realism: {realism_score}/5")
                    # print(f"     📈 Total Score: {group_total_score}/20")
                    # print("-" * 40)

                # Calculate MINIMUM scores across all groups
                valid_scores = [s for s in all_group_scores if isinstance(s, (int, float))]
                if valid_scores:
                    final_min_score = min(valid_scores)

                    # Calculate minimum for each dimension
                    min_quality = min([d["quality_score"] for d in group_details])
                    min_relevance = min([d["relevance_score"] for d in group_details])
                    min_consistency = min([d["consistency_score"] for d in group_details])
                    min_realism = min([d["realism_score"] for d in group_details])

                    # print(f"\n🎯 FINAL MINIMUM SCORES (across {len(frame_groups)} groups):")
                    # print(f"   📊 Min Quality: {min_quality}/5")
                    # print(f"   🎯 Min Relevance: {min_relevance}/5")
                    # print(f"   🔄 Min Consistency: {min_consistency}/5")
                    # print(f"   🌟 Min Realism: {min_realism}/5")
                    # print(f"   📉 Total Minimum Score: {final_min_score}/20")
                else:
                    final_min_score = "bad_scores_all_groups"
                    min_quality = min_relevance = min_consistency = min_realism = 0

                # Prepare CSV data - using MINIMUM scores and first group's details as representative
                row_data_for_csv = [grid_image_name, original_t2v_prompt]
                if group_details:
                    first_group = group_details[0]
                    row_data_for_csv.extend(
                        [
                            f"all_frame_groups_min_{len(frame_groups)}groups",
                            first_group["quality_output"][:100] + "..."
                            if len(first_group["quality_output"]) > 100
                            else first_group["quality_output"],
                            min_quality,
                            first_group["realism_output"][:100] + "..."
                            if len(first_group["realism_output"]) > 100
                            else first_group["realism_output"],
                            min_realism,
                            first_group["relevance_output"][:100] + "..."
                            if len(first_group["relevance_output"]) > 100
                            else first_group["relevance_output"],
                            min_relevance,
                            first_group["consistency_output"][:100] + "..."
                            if len(first_group["consistency_output"]) > 100
                            else first_group["consistency_output"],
                            min_consistency,
                            final_min_score,
                        ]
                    )
                else:
                    row_data_for_csv.extend(
                        ["evaluation_error", "0:0:0:0", 0, "0:0:0:0", 0, "0:0:0:0", 0, "0:0:0:0", 0, 0]
                    )

                row_data_for_csv.extend([str(all_group_scores), final_min_score])
                csv_writer.writerow(row_data_for_csv)
                csvfile.flush()
                # print(f"Finished {grid_image_name}. Minimum Score across {len(frame_groups)} groups: {final_min_score}")
                # print("=" * 80)

    return csv_path


# Helper function for evaluation
def evaluate_all_dimensions(
    images_tensor, image_sizes, original_prompt, explanation, iteration, args, tokenizer, model, IMAGE_TOKEN_INDEX
):
    """Evaluate all 4 dimensions separately and return scores and outputs"""

    # Define evaluation prompts for each dimension
    def create_base_prompt():
        return f"""
You are an AI video quality evaluator. Analyze this 3×3 grid showing 9 CONSECUTIVE video frames arranged chronologically from left to right, top to bottom.

**CRITICAL**: These are 9 consecutive frames extracted from a continuous video sequence, NOT randomly sampled frames.

**FRAME ARRANGEMENT**:
- Row 1: Frame N → Frame N+1 → Frame N+2
- Row 2: Frame N+3 → Frame N+4 → Frame N+5
- Row 3: Frame N+6 → Frame N+7 → Frame N+8

**GENERATION CONTEXT**:
- **Prompt**: `{original_prompt}`
- **Explanation**: `{explanation}`
"""

    # Four separate evaluation prompts
    quality_prompt = (
        DEFAULT_IMAGE_TOKEN
        + create_base_prompt()
        + """
**YOUR TASK**: Evaluate the TECHNICAL QUALITY of these consecutive frames.

**Quality (1-5): Technical Excellence**
Check for artifacts, resolution, clarity, color balance, and rendering quality across all consecutive frames.
- **1**: Severe technical issues affecting most frames
- **2**: Multiple obvious flaws impacting viewing experience  
- **3**: Acceptable with minor flaws
- **4**: High quality with trivial imperfections
- **5**: Flawless professional-grade

**CRITICAL OUTPUT FORMAT REQUIREMENT**:
You MUST end your response with exactly this format (no variations allowed):

Reasoning: [Your detailed analysis of technical quality]
Quality: X

Where X is ONLY a single digit from 1 to 5. Do not add brackets, extra text, or explanations after the number.

**EXAMPLE**:
Reasoning: The frames show good resolution and color balance with minor compression artifacts.
Quality: 4
"""
    )

    relevance_prompt = (
        DEFAULT_IMAGE_TOKEN
        + create_base_prompt()
        + f"""
**YOUR TASK**: Evaluate how well the frames match the generation goals.

**Relevance (1-5): Adherence to Goals**
Compare the video sequence against the **Prompt**: `{original_prompt}` and **Explanation**: `{explanation}`.
- **1**: Completely unrelated content
- **2**: Weak connection, missing major elements
- **3**: Captures general concept, lacks details
- **4**: Accurately represents most elements
- **5**: Perfect alignment with all requirements

**CRITICAL OUTPUT FORMAT REQUIREMENT**:
You MUST end your response with exactly this format (no variations allowed):

Reasoning: [Your detailed analysis of how well it matches the prompt and explanation]
Relevance: X

Where X is ONLY a single digit from 1 to 5. Do not add brackets, extra text, or explanations after the number.

**EXAMPLE**:
Reasoning: The video accurately depicts most elements from the prompt but lacks some specific details.
Relevance: 4
"""
    )

    consistency_prompt = (
        DEFAULT_IMAGE_TOKEN
        + create_base_prompt()
        + """
**YOUR TASK**: Evaluate the TEMPORAL CONSISTENCY between consecutive frames.

**Consistency (1-5): Temporal Coherence Between Consecutive Frames**
**IMPORTANT**: Since these are consecutive frames, analyze smooth transitions and logical progression from frame to frame.
- **1**: Chaotic inconsistency - objects teleport, backgrounds change randomly, no logical flow between consecutive frames
- **2**: Major temporal disruptions - significant jumps or morphing between adjacent frames, jarring transitions
- **3**: Generally stable progression with some noticeable but minor temporal inconsistencies between frames
- **4**: Smooth temporal flow with natural progression, only very minor variations between consecutive frames
- **5**: Perfect temporal continuity - seamless, natural progression that could be from real video footage

**CRITICAL OUTPUT FORMAT REQUIREMENT**:
You MUST end your response with exactly this format (no variations allowed):

Reasoning: [Your detailed analysis of frame-to-frame consistency and temporal flow]
Consistency: X

Where X is ONLY a single digit from 1 to 5. Do not add brackets, extra text, or explanations after the number.

**EXAMPLE**:
Reasoning: The consecutive frames show smooth transitions with natural progression and minimal temporal inconsistencies.
Consistency: 4
"""
    )

    realism_prompt = (
        DEFAULT_IMAGE_TOKEN
        + create_base_prompt()
        + """
**YOUR TASK**: Evaluate the REALISM and believability of the content.

**Realism (1-5): Physical Plausibility**
Assess believability and natural appearance throughout the consecutive sequence.
- **1**: Severe physics violations, obviously fake appearance
- **2**: Multiple unnatural elements, clearly AI-generated look
- **3**: Generally plausible with some artificial aspects
- **4**: Very natural with minimal artificial tells
- **5**: Photorealistic perfection

**CRITICAL OUTPUT FORMAT REQUIREMENT**:
You MUST end your response with exactly this format (no variations allowed):

Reasoning: [Your detailed analysis of realism and believability]
Realism: X

Where X is ONLY a single digit from 1 to 5. Do not add brackets, extra text, or explanations after the number.

**EXAMPLE**:
Reasoning: The video appears very natural with realistic physics and minimal artificial elements.
Realism: 4
"""
    )

    # Function to run evaluation for a specific dimension
    def evaluate_dimension(prompt_text, dimension_name):
        conv = conv_templates[args.conv_mode if args.conv_mode is not None else "llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids_result = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = torch.tensor(input_ids_result) if isinstance(input_ids_result, list) else input_ids_result
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=256,
                use_cache=True,
            )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(f"  📊 {dimension_name} evaluation:")
        # print(f"  {output_text}")
        # print("-" * 50)
        return output_text

    # Evaluate each dimension separately
    print(f"  🔍 Evaluating 4 dimensions separately for iteration {iteration + 1}:")

    quality_output = evaluate_dimension(quality_prompt, "Quality")
    relevance_output = evaluate_dimension(relevance_prompt, "Relevance")
    consistency_output = evaluate_dimension(consistency_prompt, "Consistency")
    realism_output = evaluate_dimension(realism_prompt, "Realism")

    # Extract scores from each output
    quality_score = extract_single_score(quality_output, "quality")
    relevance_score = extract_single_score(relevance_output, "relevance")
    consistency_score = extract_single_score(consistency_output, "consistency")
    realism_score = extract_single_score(realism_output, "realism")

    return (
        quality_score,
        relevance_score,
        consistency_score,
        realism_score,
        quality_output,
        relevance_output,
        consistency_output,
        realism_output,
    )


def model_score(csv_path):
    if csv_path is None:
        print("CSV path is None, cannot calculate model score.")
        return

    try:
        with open(csv_path, newline="") as file:
            reader = csv.reader(file)
            lines = list(reader)
            if len(lines) <= 1:  # Only header or empty
                print("CSV file is empty or contains only header. No scores to average.")
                return

            scores_sum = 0
            valid_video_count = 0
            # The 'final_min_score' is the last column data before we append the grand total
            # Header length gives total columns. final_min_score is at index -1 of data rows before we append.

            for line_index, line_data in enumerate(lines[1:]):  # Skip header
                try:
                    # The last element in the row *before* this function appends anything should be final_min_score
                    final_min_score_str = line_data[-1]
                    score_tmp = float(final_min_score_str)
                    # The score per video is already an average of iterations, ranging from 4 (4*1) to 20 (4*5)
                    # We can normalize it if needed, e.g., to 0-1 range: (score_tmp - 4) / (20 - 4)
                    # For now, let's just average the raw final_min_scores
                    scores_sum += score_tmp
                    valid_video_count += 1
                except (ValueError, IndexError):
                    print(
                        f"Warning: Could not parse score from line {line_index + 2}: {line_data[-1] if line_data else 'empty_line'}"
                    )
                    continue

            if valid_video_count > 0:
                overall_average_score = scores_sum / valid_video_count
                # print(f"\nNumber of videos evaluated: {valid_video_count}")
                # print(f"Overall Average Model Score (across all videos, scale 4-20): {overall_average_score:.4f}")

                # Append the grand average score to the CSV
                with open(csv_path, "a", newline="") as outfile:
                    writer = csv.writer(outfile)
                    # Add empty cells to align with the header if necessary
                    num_cols = len(lines[0])  # Get number of columns from header
                    padding = [""] * (num_cols - 2) if num_cols > 2 else []
                    writer.writerow(["Overall Average Model Score:", f"{overall_average_score:.4f}", *padding])
            else:
                print("No valid video scores found to calculate an overall average.")

    except FileNotFoundError:
        print(f"Error: CSV file {csv_path} not found for final scoring.")
    except Exception as e:
        print(f"An error occurred in model_score: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
        help="Conversation mode for LLaVA model, e.g., 'llava_v1', 'chatml_direct'",
    )
    parser.add_argument("--sep", type=str, default=",")  # Not directly used in LLaVA calls in this version
    parser.add_argument(
        "--temperature", type=float, default=0.7
    )  # Lower for more deterministic JSON, higher for more diverse descriptions
    parser.add_argument("--top_p", type=float, default=0.9)  # Usually 1.0 for greedy if temperature is low
    parser.add_argument("--num_beams", type=int, default=1)  # 1 for sampling, >1 for beam search
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for Q1 (description)")
    # max_new_tokens for JSON answers (Q2-Q5) is set to 128 in the code.

    parser.add_argument(
        "--output-path", type=str, default="./video_assessment_results", help="Path to store the CSV scores"
    )
    # Ensure read_prompt_file has a structure like: [{"id": "video_name_without_extension", "prompt": "text prompt for video"}, ...]
    parser.add_argument(
        "--read-prompt-file",
        type=str,
        default="./prompts_metadata.json",
        help="JSON file with input prompts and video IDs/names",
    )
    parser.add_argument("--seed", type=int, default=42)  # Changed default seed

    parser.add_argument("--video-path", type=str, required=True, help="Path to videos directory or a single video file")
    parser.add_argument(
        "--t2v-model", type=str, required=True, help="Name of the text-to-video model being evaluated (for CSV naming)"
    )
    parser.add_argument(
        "--image_grid_path", type=str, default=None, help="Optional path to pre-generated image grids directory"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for each pre-generated image (ignored for dynamic video processing)",
    )
    parser.add_argument("--save_grid_path", type=str, default=None, help="Optional path to save generated grid images")

    args = parser.parse_args()

    # Create a dummy prompts_metadata.json if it doesn't exist for testing
    if not os.path.exists(args.read_prompt_file) and args.read_prompt_file == "./prompts_metadata.json":
        print(f"Creating a dummy prompt metadata file at {args.read_prompt_file} for testing purposes.")
        dummy_prompts = [
            {
                "id": "sample_video_1",
                "prompt": "A fluffy cat chasing a red laser pointer on a wooden floor.",
                "explanation": "A detailed video showing a fluffy domestic cat actively chasing a red laser pointer dot across a wooden floor surface.",
            },
            {
                "id": "sample_video_2",
                "prompt": "A futuristic cityscape at sunset with flying vehicles.",
                "explanation": "A cinematic view of a futuristic city during sunset with advanced flying vehicles moving through the sky between tall buildings.",
            },
            # Add more if you have sample videos with these names (e.g., sample_video_1.mp4)
        ]
        with open(args.read_prompt_file, "w") as f:
            json.dump(dummy_prompts, f, indent=4)
        print("Please ensure your video filenames (without extension) match the 'id' in this JSON file.")

    final_csv_path = eval_model(args)
    if final_csv_path and os.path.exists(final_csv_path):
        model_score(final_csv_path)
    else:
        print("Evaluation did not produce a valid CSV file, skipping model_score.")
