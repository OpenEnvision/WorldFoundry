import argparse
import glob
import os

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos and extract frames.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing videos.")
    parser.add_argument("--output_dir", required=True, help="Output directory for extracted frames.")
    parser.add_argument("--model_names", nargs="+", default=["test"], help="Name of the models.")
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["open", "close"],
        default="open",
        help="Specify the evaluation mode: 'open' for open-source models or 'close' for closed-source models.",
    )
    return parser.parse_args()


def resize_frame(frame, short_edge=256):
    height, width = frame.shape[:2]
    if min(height, width) <= short_edge:
        return frame
    else:
        scale = short_edge / width if height > width else short_edge / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame


def extract_frames(video_path, output_folder, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_capture = {0, total_frames - 1}
    frames_interval = (total_frames - 1) // (num_frames - 1)
    for i in range(1, num_frames - 1):
        frames_to_capture.add(i * frames_interval)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count in frames_to_capture:
            resized_frame = resize_frame(frame)
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{count}.png"
            output_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(output_path, resized_frame)
            # print(f"Saved {output_path}")

        count += 1

    cap.release()


def process_all_videos(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(folder_path) if f.endswith((".mp4", ".avi", ".mov"))]
    total_videos = len(video_files)
    skipped_videos = 0

    print(f"Total videos to check: {total_videos}")

    for filename in tqdm(video_files, desc="Processing videos", unit="video", leave=False):
        video_name = os.path.splitext(filename)[0]
        video_related_images = glob.glob(os.path.join(output_folder, f"{video_name}_frame*.png"))

        if len(video_related_images) == 8:
            print(f"Skipping {filename}, already processed.")
            skipped_videos += 1
            continue

        for img in video_related_images:
            os.remove(img)
            print(f"Deleted {img}")

        video_path = os.path.join(folder_path, filename)
        # print(f"Processing {filename}...")
        extract_frames(video_path, output_folder)

    print(f"Skipped {skipped_videos} videos that were already processed.")
    print(f"Processed {total_videos - skipped_videos} new or incomplete videos.")


def main():
    args = parse_args()

    for model_name in tqdm(args.model_names, desc="Processing models"):
        if args.eval_type == "open":
            for part in tqdm(["1", "2", "3"], desc="Processing parts", leave=False):
                input_folder = os.path.join(args.input_dir, model_name, part)
                output_folder = os.path.join(args.output_dir, model_name, part)
                process_all_videos(input_folder, output_folder)
        elif args.eval_type == "close":
            input_folder = os.path.join(args.input_dir, model_name)
            output_folder = os.path.join(args.output_dir, model_name)
            process_all_videos(input_folder, output_folder)


if __name__ == "__main__":
    main()
