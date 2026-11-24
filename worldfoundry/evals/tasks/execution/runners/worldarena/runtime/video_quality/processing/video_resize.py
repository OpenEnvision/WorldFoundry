import os

import cv2
import yaml


def resize_and_save_images(image_folder, target_size=(640, 480)):
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for fname in image_files:
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue
        if img.shape[:2] != tuple(target_size[::-1]):
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(img_path, img_resized)
        else:
            f"Keep: {img_path}"


def process_video_or_images(input_base):
    for task in sorted(os.listdir(input_base)):
        task_path = os.path.join(input_base, task)
        for episode in os.listdir(task_path):
            if episode.endswith((".png", ".json")):
                continue
            episode_path = os.path.join(task_path, episode)
            for gid in sorted(os.listdir(episode_path)):
                video_path = os.path.join(episode_path, gid, "video")
                print(f"[Image Folder] Processing {video_path}")
                resize_and_save_images(video_path)


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")

    args = parser.parse_args()
    config = load_config(args.config_path)

    data_base = config["data"]["val_base"]

    process_video_or_images(data_base)
