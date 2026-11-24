import argparse
import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers to use")
    parser.add_argument("--openai_api", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for the API")
    parser.add_argument("--input_dir", type=str, default="video_frames_folder", help="Directory for input video frames")
    parser.add_argument("--output_dir", type=str, default="results/temp", help="Directory for output results")
    parser.add_argument("--model_names", nargs="+", default=["test"], help="Name of the models.")
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["open", "close"],
        default="open",
        help="Specify the evaluation mode: 'open' for open-source models or 'close' for closed-source models.",
    )
    return parser.parse_args()


txt_prompt = """
Suppose you are a data rater, specialized in generating scores for time-lapse videos. You will be supplied with eight key frames extracted from a video, each with a filename labeled with its position in the video sequence. Your task is to rate the variation of this time-lapse video on a scale of 5. The scoring criteria are as follows:
1: Minimal change. The scene appears almost like a still image, with static elements remaining motionless and only minor changes in lighting or subtle movements of elements. No significant activity is noticeable.
2: Slight change. There is a small amount of movement or change in the elements of the scene, such as a few people or vehicles moving and minor changes in light or shadows. The overall variation is still minimal, with changes mostly being quantitative.
3: Moderate change. Multiple elements in the scene undergo changes, but the overall pace is slow. This includes gradual changes in daylight, moving clouds, growing plants, or occasional vehicle and pedestrian movements. The scene begins to show a transition from quantitative to qualitative change.
4: Significant change. The elements in the scene show obvious dynamic changes with a higher speed and frequency of variation. This includes noticeable changes in city traffic, crowd activities, or significant weather transitions. The scene displays a mix of quantitative and qualitative changes.
5: Dramatic change. Elements in the scene undergo continuous and rapid significant changes, creating a very rich visual effect. This includes events like sunrise and sunset, construction of buildings, and seasonal changes, making the variation process vivid and impactful. The scene exhibits clear qualitative change.

For guidance on the expected format, refer to the provided examples: 

Brief Reasoning Statement: The changes in the frames are quite significant as we see the progression of the 3D print from start to finish. The variation in the video includes the build-up of the print layer by layer, showing dynamic changes and detailed progress. The scene exhibits both quantitative changes (incremental addition of the print layers) and qualitative changes (completed print and human interaction).
"Score":{4}

Brief Reasoning Statement: The changes in the frames show a significant dynamic process of a lunar eclipse. The video captures the progression from a full moon to a full lunar eclipse and back to a more visible moon, showcasing both quantitative (shadow coverage) and qualitative (color change) changes. The visual effect is rich, with continuous and rapid significant changes in the moon's appearance.
"Score":{5}

Brief Reasoning Statement: The changes in the frames show moderate to significant dynamic activity. The light trails from boats and water traffic, as well as cloud movement, create visual changes that is quantitative. The visual effect includes ongoing changes, but they are not as rapid or continuous as some more dramatic scenes.
"Score":{3}

Attention: Do not reply outside the example template! Below are the video title and input frames:
"""

file_lock = Lock()


def get_image_filenames(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    return [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in image_extensions
    ]


def parse_video_id(filename):
    match = re.match(r"(.+)_frame\d+\.png", filename)
    return match.group(1) if match else None


def image_b64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def group_images_by_video_id(filenames):
    images_by_video = {}
    for filename in tqdm(filenames, desc="Grouping images"):
        video_id = parse_video_id(filename)
        if video_id:
            if video_id not in images_by_video:
                images_by_video[video_id] = []
            images_by_video[video_id].append(filename)

    valid_groups = {video_id: images for video_id, images in images_by_video.items() if len(images) == 8}
    return valid_groups


def create_prompts(grouped_images, image_directory, txt_prompt):
    prompts = {}
    for video_id, group in tqdm(grouped_images.items(), desc="Creating prompts"):
        prompt = [{"type": "text", "text": txt_prompt}]

        for image_name in group:
            image_path = os.path.join(image_directory, image_name.strip())
            b64_image = image_b64(image_path)
            prompt.append({"type": "text", "text": image_name.strip()})
            prompt.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}})

        prompts[video_id] = prompt
    return prompts


def has_been_processed(video_id, output_file):
    with file_lock:
        if os.path.exists(output_file):
            with open(output_file, encoding="utf-8") as f:
                data = json.load(f)
                if video_id in data:
                    print(f"Video ID {video_id} has already been processed.")
                    return True
        return False


def extract_frame_number(filename):
    return int(filename.split("_frame")[-1].split(".")[0])


def load_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as file:
            print(f"Loading existing results from {file_path}")
            return json.load(file)
    else:
        print(f"No existing results file found at {file_path}. Creating a new file.")
        with open(file_path, "w", encoding="utf-8") as file:
            empty_data = {}
            json.dump(empty_data, file)
            return empty_data


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(prompt, model_name="gpt-4o-2024-05-13"):
    if not args.base_url:
        client = OpenAI(api_key=args.openai_api)
    else:
        client = OpenAI(api_key=args.openai_api, base_url=args.base_url)
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=2048,
    )
    print(chat_completion)
    return chat_completion.choices[0].message.content


def save_output(video_id, prompt, output_file):
    if not has_been_processed(video_id, output_file):
        result = call_gpt(prompt)
        with file_lock, open(output_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data[video_id] = result
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        print(f"Processed and saved output for Video ID {video_id}")


def main(num_workers, all_prompts, output_file):
    existing_results = load_existing_results(output_file)

    unprocessed_prompts = {vid: prompt for vid, prompt in all_prompts.items() if vid not in existing_results}
    if not unprocessed_prompts:
        print("No unprocessed video IDs found. All prompts have already been processed.")
        return

    print(f"Processing {len(unprocessed_prompts)} unprocessed video IDs.")

    progress_bar = tqdm(total=len(unprocessed_prompts))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(save_output, video_id, prompt, output_file): video_id
            for video_id, prompt in unprocessed_prompts.items()
        }

        for future in as_completed(future_to_index):
            progress_bar.update(1)
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video ID {future_to_index[future]}: {e}")

    progress_bar.close()


if __name__ == "__main__":
    args = parse_args()

    num_workers = args.num_workers
    openai_api = args.openai_api
    base_url = args.base_url
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_name in tqdm(args.model_names, desc="Processing models"):
        if args.eval_type == "open":
            for part in tqdm(["1", "2", "3"], desc="Processing parts", leave=False):
                directory = os.path.join(input_dir, model_name, part)
                output_file = os.path.join(output_dir, f"{model_name}_{part}_metamorphic.json")
                group_frames_file = os.path.join(output_dir, f"{model_name}_{part}_temp_group_frames.json")

                all_prompts = {}
                all_grouped_images = {}

                filenames = get_image_filenames(directory)
                grouped_images = group_images_by_video_id(filenames)

                for video_id in grouped_images:
                    grouped_images[video_id].sort(key=extract_frame_number)

                all_grouped_images.update(grouped_images)
                prompts = create_prompts(grouped_images, directory, txt_prompt)
                all_prompts.update(prompts)

                with open(group_frames_file, "w", encoding="utf-8") as file:
                    json.dump(all_grouped_images, file)

                main(num_workers, all_prompts, output_file)
        elif args.eval_type == "close":
            directory = os.path.join(input_dir, model_name)
            output_file = os.path.join(output_dir, f"{model_name}_metamorphic.json")
            group_frames_file = os.path.join(output_dir, f"{model_name}_temp_group_frames.json")

            all_prompts = {}
            all_grouped_images = {}

            filenames = get_image_filenames(directory)
            grouped_images = group_images_by_video_id(filenames)

            for video_id in grouped_images:
                grouped_images[video_id].sort(key=extract_frame_number)

            all_grouped_images.update(grouped_images)
            prompts = create_prompts(grouped_images, directory, txt_prompt)
            all_prompts.update(prompts)

            with open(group_frames_file, "w", encoding="utf-8") as file:
                json.dump(all_grouped_images, file)

            main(num_workers, all_prompts, output_file)
