import csv
import json
import os
import time

from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-3-flash-preview"


def extract_json_from_md(md_string):
    start_index = md_string.find("json") + len("json")
    end_index = md_string.find("```", start_index)
    json_string = md_string[start_index:end_index].strip()
    return json.loads(json_string)


def filter_from_json(json):
    fields_to_keep = ["ID", "Dimension", "Question"]
    filtered_list = []
    for item in json:
        filtered_item = {field: item[field] for field in fields_to_keep if field in item}
        filtered_list.append(filtered_item)


def generate_questions(img_path, generation_prompt):
    print("Uploading img to Gemini API...")
    img_file = client.files.upload(file=img_path)
    print(f"Uploaded as: {img_file.name}")

    print("Waiting for image processing to complete (this may take a few minutes)...")
    while True:
        img_file = client.files.get(name=img_file.name)
        if img_file.state == "ACTIVE":
            print("Image processing complete!\n")
            break
        elif img_file.state == "FAILED":
            raise Exception("Image processing failed in the Gemini API.")

        print(".", end="", flush=True)
        time.sleep(5)
    prompt = f"""
    System Role:

    You are an expert LLM judger, specializing in "World Model" evaluation. Your task is to generate questions to be used to evaluate AI-generated videos against specific text instructions.


    Input Data:

    Start Frame (Ground Truth) as attached image
    Generation Prompt (will be used as input for World Model to generate videos): "{generation_prompt}"

    Task:

    Generating 24 Yes/No questions.


    Evaluation Dimensions & Constraints:

    You must generate 6 questions for each of the 4 Dimensions below.

    Instruction Following (Positive Polarity): (Did the video strictly adhere to the specific movements and events requested in the text prompt?)
    Object and Background (Negative Polarity): (Focus on the visual consistency and identity of the nearby subject and distant runner).
    Continuity of Memory (Positive Polarity): (Focus on Object Permanence: Does the model remember the subject's location/trajectory while they are out of frame?).
    Physics Adherence (Negative Polarity): (Focus on lighting, shadows, and natural movement speed/gravity).

    Output Format:

    Please only output as JSON format with the following columns: [ID, Dimension (with polarity), Question].
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img_file, prompt],
    )

    return response.text


def evaluate_video(video_path, questions):
    print("Uploading video to Gemini API...")
    video_file = client.files.upload(file=video_path)
    print(f"Uploaded as: {video_file.name}")

    print("Waiting for video processing to complete (this may take a few minutes)...")
    while True:
        video_file = client.files.get(name=video_file.name)
        if video_file.state == "ACTIVE":
            print("Video processing complete!\n")
            break
        elif video_file.state == "FAILED":
            raise Exception("Video processing failed in the Gemini API.")

        print(".", end="", flush=True)
        time.sleep(5)
    prompt = f"""
    System Role:

    You are an expert LLM judger, specializing in "World Model" evaluation. Your task is to audit AI-generated videos against specific text instructions.


    Input Data:

    Test Video as attached video
    Questions: "{questions}" 

    Task:

    Watch the Test Video and evaluate it by answering Yes/No questions.

    Output Format:

    Please only output as JSON format with the following columns: [ID, Dimension, Question, Answer (Yes/No), Verdict (Pass/Fail), Reasoning].
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[video_file, prompt],
    )

    return response.text


def evaluate_video_with_hint(video_path, questions, hint):
    print("Uploading video to Gemini API...")
    video_file = client.files.upload(file=video_path)
    print(f"Uploaded as: {video_file.name}")

    print("Waiting for video processing to complete (this may take a few minutes)...")
    while True:
        video_file = client.files.get(name=video_file.name)
        if video_file.state == "ACTIVE":
            print("Video processing complete!\n")
            break
        elif video_file.state == "FAILED":
            raise Exception("Video processing failed in the Gemini API.")

        print(".", end="", flush=True)
        time.sleep(5)
    prompt = f"""
    System Role:

    You are an expert LLM judger, specializing in "World Model" evaluation. Your task is to audit AI-generated videos against specific text instructions.


    Input Data:

    Test Video as attached video
    Questions: "{questions}" -- please only use first three columns [ID, Dimension, Question, Answer (Yes/No)] in your evaluation.
    Hint: "{hint}"

    Tasks:

    1. Watch the Test Video and evaluate it by answering Yes/No questions given.
    2. Audit your answer by reviewing the actual failures and remove the questions that you evaluate incorrectly.

    Output Format:

    Please only output as JSON format with the following columns: [ID, Dimension, Question, Answer (Yes/No), Verdict (Pass/Fail), Reasoning].
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[video_file, prompt],
    )

    return response.text


if __name__ == "__main__":
    inputs = []
    with open("failure-cases.csv", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            inputs.append(row)
    print(inputs)

    outputs = []
    output_file_path = f"questions-{time.time}.csv"

    for input_config in inputs:
        if input_config["Finished"] != "No":
            continue

        while True:
            output = {}
            output["scene"] = input_config["scene"]
            output["video_id"] = input_config["video_id"]

            folder_name = input_config["folder"]
            print(folder_name)
            questions = generate_questions(f"first-frame/{folder_name}.jpg", input_config["prompt"])
            print("Questions:\n", questions)
            questions_json = extract_json_from_md(questions)
            output["Start Questions"] = json.dumps(questions_json)

            evaluation_gt = evaluate_video(f"ground-truth/{folder_name}.mp4", json.dumps(questions_json))
            print("Evaluation for ground truth:\n", evaluation_gt)
            evaluation_gt_json = extract_json_from_md(evaluation_gt)
            output["Evaluation on ground truth"] = json.dumps(evaluation_gt_json)
            evaluation_gt_json_filtered = [eval_gt for eval_gt in evaluation_gt_json if eval_gt["Verdict"] == "Pass"]
            questions_gt_filtered = filter_from_json(evaluation_gt_json_filtered)
            fields_to_keep = ["ID", "Dimension", "Question"]
            questions_gt_filtered = [
                {field: eval_gt[field] for field in fields_to_keep if field in eval_gt}
                for eval_gt in evaluation_gt_json_filtered
            ]

            last_occurrence_index = input_config["folder"].rfind("_")
            failure_filename = (
                input_config["folder"][:last_occurrence_index]
                + "_Failure_"
                + input_config["folder"][last_occurrence_index + 1 :]
            )
            # failure_filename = input_config["folder"].replace("_", "_Failure_")
            evaluation_failure = evaluate_video(f"failure/{failure_filename}.mp4", json.dumps(questions_gt_filtered))
            print("Evaluation for failure:\n", evaluation_failure)
            evaluation_failure_json = extract_json_from_md(evaluation_failure)
            output["Evaluation on failure"] = json.dumps(evaluation_failure_json)

            evaluation_failure_revised = evaluate_video_with_hint(
                f"failure/{failure_filename}.mp4", json.dumps(questions_gt_filtered), input_config["hint"]
            )
            print("Revised evaluation for failure:\n", evaluation_failure_revised)
            evaluation_failure_revised_json = extract_json_from_md(evaluation_failure_revised)
            output["Revised Evaluation on failure"] = json.dumps(evaluation_failure_revised_json)

            evaluation_revised = [
                eval_failure
                for eval_failure, eval_failure_revised in zip(
                    evaluation_failure_json, evaluation_failure_revised_json, strict=False
                )
                if eval_failure["Verdict"] == eval_failure_revised["Verdict"]
            ]
            final_questions = [
                {field: eval_revised[field] for field in fields_to_keep if field in eval_revised}
                for eval_revised in evaluation_revised
            ]
            print("Final question:\n", json.dumps(final_questions, indent=4))
            output["Final questions"] = json.dumps(final_questions)

            dimension_count = {
                "Instruction Following (Positive Polarity)": 0,
                "Object and Background (Negative Polarity)": 0,
                "Continuity of Memory (Positive Polarity)": 0,
                "Physics Adherence (Negative Polarity)": 0,
            }
            for question in final_questions:
                dimension_count[question["Dimension"]] += 1
            success = True
            for count in dimension_count.values():
                if count < 2:
                    success = False
                    break
            if success:
                fieldnames = output.keys()
                output_file_path = "-".join([input_config["scene"], input_config["video_id"], "questions"]) + ".csv"
                with open(output_file_path, "w", newline="", encoding="utf-8") as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    dict_writer.writeheader()
                    dict_writer.writerows([output])

                outputs.append(output)
                break

    fieldnames = outputs[0].keys()
    with open(output_file_path, "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        dict_writer.writeheader()

        dict_writer.writerows(outputs)
