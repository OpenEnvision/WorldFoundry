import argparse
import json
import os


def create_new_json(json_data, video_folder):
    new_json_data = []
    for item in json_data:
        index = item["index"]
        video_filename = f"{index}.mp4"
        video_path = os.path.join(video_folder, video_filename)
        if os.path.exists(video_path):
            new_item = item.copy()
            new_item["filepath"] = os.path.abspath(video_path)
            new_json_data.append(new_item)
    return new_json_data


def main():
    parser = argparse.ArgumentParser(description="Create new JSON with filepath for existing videos")
    parser.add_argument("-i", "--input_json", default="./prompts/prompts.json", help="Path to the input JSON file")
    parser.add_argument("-v", "--video_folder", required=True, help="Path to the folder containing video files")
    parser.add_argument("-o", "--output_json", required=True, help="Path to save the new JSON file")

    args = parser.parse_args()

    video_folder = os.path.abspath(args.video_folder)

    # Read the input JSON file
    with open(args.input_json) as f:
        json_data = json.load(f)

    # Create new JSON data for existing videos
    new_json_data = create_new_json(json_data, video_folder)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # Write the new JSON to the output file
    with open(args.output_json, "w") as f:
        json.dump(new_json_data, f, indent=2)

    print(f"New JSON with {len(new_json_data)} entries saved to {args.output_json}")


if __name__ == "__main__":
    main()
