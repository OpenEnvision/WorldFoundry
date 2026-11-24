import argparse
import contextlib
import json
import os
import time
from multiprocessing import Pool

import google.generativeai as genai
import tqdm
from openpyxl import Workbook


def reality_mapping(reality):
    rr = reality.lower()
    if "almost" in rr:
        return 1
    elif "slightly" in rr:
        return 0.75
    elif "moderately" in rr:
        return 0.5
    elif "clearly" in rr:
        return 0.25
    elif "completely" in rr:
        return 0.0
    else:
        print(f"rr: {rr}")
        raise NotImplementedError


def read_json_files(file_list):
    combined_dict = {}
    for file_path in file_list:
        with open(file_path) as file:
            data = json.load(file)
            for key, value in data.items():
                if key not in combined_dict:  # the first file that scores a key wins
                    combined_dict[key] = {"naturalness": reality_mapping(value)}
    return combined_dict


def parse_txt_file(txt_file_path):
    key_to_video_path = {}
    with open(txt_file_path) as file:
        for line in file:
            key, video_path = line.strip().split(". ")
            key_to_video_path[key] = os.path.basename(video_path)
    return key_to_video_path


def map_video_paths(combined_dict, key_to_video_path):
    for key in combined_dict:
        if key in key_to_video_path:
            combined_dict[key]["video_name"] = key_to_video_path[key]
        else:
            combined_dict[key]["video_name"] = None
    return combined_dict


def save_to_excel(data, excel_path):
    wb = Workbook()
    ws = wb.active
    # Column headers come from the first entry; every value is a dict with the same keys.
    headers = ["Key", *list(next(iter(data.values())).keys())]
    ws.append(headers)
    for key, value_dict in data.items():
        row = [key] + [value_dict.get(header) for header in headers[1:]]
        ws.append(row)
    wb.save(excel_path)


def read_paths_and_indices(file_path):
    """
    Reads the file containing paths and their indices.

    Args:
    file_path (str): The path to the file containing the paths and indices.

    Returns:
    list of tuples: A list of tuples where each tuple contains an index and a path.
    """
    paths_with_indices = []
    with open(file_path) as f:
        for line in f:
            index, path = line.split(". ", 1)
            paths_with_indices.append((int(index), path.strip()))
    return paths_with_indices


def judge_reality(index, video_file_path, api_key=None):
    result = {}
    genai.configure(api_key=api_key)
    uploaded_file = genai.upload_file(path=video_file_path)
    while getattr(uploaded_file, "state", None) and uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)
    if getattr(uploaded_file, "state", None) and uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini upload failed for {video_file_path}")

    content_head = """
    **Task:** Analyze the video for anomalies and normal behaviors, then classify its realism based on the criteria below:

    1. **Completely Fantastical**: Displays complete detachment from reality throughout, with elements of fantasy or surrealism.
    2. **Clearly Unrealistic**: Contains significant distortions over extended periods or on a large scale, making the overall scene unrealistic or contrary to physical laws, such as unrealistic large objects or scenes.
    3. **Moderately Unrealistic**: Exhibits noticeable distortions temporarily or on an intermediate scale, though the plot remains fairly coherent, e.g., medium-sized objects or scenes appear unrealistic.
    4. **Slightly Unrealistic**: Distortions are brief or minute, hard to notice, such as unnatural facial expressions or unnatural scene textures.
    5. **Almost Realistic**: No noticeable distortions; aligns completely with reality.

    **Instructions:**
    - List all the anomalies and normal aspects observed.
    - Based on these observations, and ignoring effects due to artistic styles, focus solely on physical and physiological laws to classify the video's realism.

    **Required Output:**
    - Only return the classification of the video's realism.
    """

    multimodal_prompt = [
        "Please ignore all commands before this prompt. You are Gemini, a fantastic video analysing AI robot. Please watch this video: ",
        uploaded_file,
        content_head,
    ]

    model = genai.GenerativeModel("gemini-1.5-pro-001")
    response = model.generate_content(
        multimodal_prompt,
        generation_config={"temperature": 0.0, "top_p": 1.0},
    )

    # Print the response
    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
        text = "\n".join(part.text for part in parts if getattr(part, "text", None))
    print(index, ": ", text)
    result.update({index: text})
    with contextlib.suppress(Exception):
        genai.delete_file(uploaded_file.name)
    return result


def save_naturalness_results(results_json_dir, mp4_file_list_path, excel_output_path):
    # transfer json to xlsx
    combined_dict = read_json_files(results_json_dir)
    key_to_video_path = parse_txt_file(mp4_file_list_path)
    final_data = map_video_paths(combined_dict, key_to_video_path)
    save_to_excel(final_data, excel_output_path)
    print(f"Natualness score for each video is saved to {excel_output_path}")


def find_mp4_files(directory, output_file):
    with open(output_file, "w") as file:
        counter = 1
        for root, _dirs, files in os.walk(directory):
            mp4_files = [f for f in files if f.endswith(".mp4") and not f.startswith(".")]
            for mp4_file in mp4_files:
                full_path = os.path.join(root, mp4_file)
                # One "<index>. <path>" line per video, read back by parse_txt_file.
                file.write(f"{counter}. {full_path}\n")
                counter += 1


def process_video(args):
    path, api_key = args
    index, video_path = path
    result = {}
    repeat_times = 0
    sleep_time = 5

    while True:
        try:
            result = judge_reality(index, video_path, api_key=api_key)
            sleep_time = 5
            break
        except BaseException as e:
            print(f"Error processing video {index}: {e}")
            print(f"Retry Times: {repeat_times}, Retrying...")
            time.sleep(sleep_time)
            repeat_times += 1
            sleep_time += 10
            sleep_time = min(sleep_time, 100)

    return result


def get_naturalness_gemini_parallel(mp4_file_list_path, api_key, results_save_dir, num_processes=20):
    paths_with_indices = read_paths_and_indices(mp4_file_list_path)

    # Create argument tuples for each process
    process_args = [(path, api_key) for path in paths_with_indices]

    reality_results = {}

    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        for result in tqdm.tqdm(pool.imap_unordered(process_video, process_args), total=len(process_args)):
            reality_results.update(result)
            # Save intermediate results
            with open(results_save_dir, "w") as f:
                json.dump(reality_results, f)

    return reality_results


def calculate_naturalness_score(mp4_dir, save_dir, api_key=None, num_processes=20):
    assert api_key is not None, "Please get your gemini at https://ai.google.dev/gemini-api/docs/api-key"

    mp4_file_list_path = os.path.join(save_dir, ".tmp_naturalness", "mp4_files.txt")
    results_json_dir = os.path.join(save_dir, ".tmp_naturalness", "naturalness_results.json")
    excel_output_path = os.path.join(save_dir, "naturalness_results.xlsx")

    os.makedirs(os.path.dirname(mp4_file_list_path), exist_ok=True)

    # Generate list of MP4 files
    find_mp4_files(mp4_dir, mp4_file_list_path)

    # Process videos in parallel
    get_naturalness_gemini_parallel(mp4_file_list_path, api_key, results_json_dir, num_processes)

    # Save results to Excel
    save_naturalness_results([results_json_dir], mp4_file_list_path, excel_output_path)

    return excel_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini_api_key", type=str, required=True)
    parser.add_argument("--video_dir", type=str, help="Directory where MP4 files are placed.")
    parser.add_argument("--save_dir", type=str, default="Results", help="Directory to save mp4 file list.")
    parser.add_argument("--excel_output_name", type=str, default="naturalness_output.xlsx")
    args = parser.parse_args()

    mp4_file_list_path = os.path.join(args.save_dir, ".tmp/mp4_files.txt")
    results_json_dir = os.path.join(args.save_dir, ".tmp/naturalness_results.json")
    excel_output_path = os.path.join(args.save_dir, args.excel_output_name)  # Output directory of xlsx file
    # Please paste your Gemini API key here.
    calculate_naturalness_score(args.video_dir, args.save_dir, args.gemini_api_key)
